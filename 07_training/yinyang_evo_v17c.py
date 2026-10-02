#!/usr/bin/env python3
"""
yinyang_evo_v17d — Wild Abort (野败) breeding + Module-swap + Elitism
=================================================================
v17c 平台期诊断：fitness 0.89-0.92 震荡，区域精度异常分化
（R5 崩溃、R0 反常差、R7 反常好）→ 说明种群多样性不足，
搜索卡在局部最优。

袁隆平"野败"策略在进化系统中的映射：
1. 不单纯按 fitness 选择，还要保留"特异个体"：
   - 行为模式最异于主流的个体（可能携带独特但未被识别的结构）
   - decay/bias 配置最极端的个体（可能是在极端环境中找到的特殊策略）
2. 强制特异个体 × 精英个体杂交：
   - 50% 是常规精英×精英杂交
   - 30% 是"野败"特异个体 × 精英杂交
   - 20% 是全新随机个体（注入新基因）
3. 保留更大的 survivor pool（top 50% 而非 top 33%），防止过早过滤野败型
"""

import random, math, hashlib, json, copy, collections

N_CHANNELS = 8
MAX_NODES = 15
N_REGIONS = 9

# ========== 环境 ==========

class StableRegionPool:
    def __init__(self, seed=42):
        self.seed = seed
        self.regions = self._make_all(seed)
        self._reset()

    def _make_all(self, seed):
        rng = random.Random(seed)
        regs = []
        for i in range(N_REGIONS):
            regs.append({
                'id': i,
                'sigma': round(0.05 + 0.64 * (i / (N_REGIONS - 1)), 2),
                'proprio': round(i / (N_REGIONS - 1), 2),
                'freq': round(0.03 + 0.12 * rng.random(), 3),
                'phase': round(rng.uniform(0, 2 * math.pi), 3),
                'type': ['sin', 'logistic', 'lorenz', 'van der Pol', 'sawtooth',
                         'rossler', 'henon', 'tent', 'cubic'][i],
                '_state': 0.3,
                '_tick': 0,
                '_phase': 0.0,
            })
        return regs

    def _reset(self):
        for r in self.regions:
            r['_state'] = 0.3
            r['_tick'] = 0
            r['_phase'] = 0.0

    def reset_states(self):
        self._reset()

    def step_global(self):
        for r in self.regions:
            r['_tick'] += 1
            r['_phase'] += r['freq']

    def get_sensory(self, region_id):
        r = self.regions[region_id]
        t = r['_tick']
        ph = r['_phase']
        rng_val = random.Random(hash((self.seed, region_id, t)) % (2**31)).gauss(0, 1)
        noise = r['sigma'] * rng_val
        rt = r['type']
        s = r['_state']

        if rt == 'sin':
            val = math.sin(ph + r['phase'])
        elif rt == 'logistic':
            val = 4 * s * (1 - s) if 0 < s < 1 else 0.5
        elif rt == 'lorenz':
            val = 0.5 + 0.3 * math.tanh(s)
        elif rt == 'van der Pol':
            val = s - 0.1 * (s**3 - s)
        elif rt == 'sawtooth':
            val = (ph + r['phase']) % 1.0
        elif rt == 'rossler':
            val = 0.5 * math.sin(0.7 * ph + s)
        elif rt == 'henon':
            val = 1 - 1.4 * s**2 + 0.3 * s
        elif rt == 'tent':
            val = 2 * s if s < 0.5 else 2 * (1 - s)
        else:
            val = s**3 - s

        s_new = max(-2.0, min(2.0, val + noise))
        r['_state'] = s_new

        return [
            s_new,
            math.sin(s_new * 2.5) if abs(s_new) < 1.5 else 0.0,
            math.tanh(s_new),
            s_new ** 2 * (1 if s_new > 0 else -1),
            abs(s_new),
            math.cos(s_new * 1.7),
            (1 if s_new > 0 else -1) * math.sqrt(abs(s_new)) if abs(s_new) < 2 else 0.0,
            r['proprio'],
        ]


# ========== World Model ==========

class WorldModel:
    def __init__(self, genome):
        self.genome = genome
        n = genome['n_nodes']
        self.nodes = [0.0] * n
        self.prev_sensory = [0.0] * N_CHANNELS

    def _region_index(self, proprio_val):
        idx = round(proprio_val * (N_REGIONS - 1))
        return max(0, min(N_REGIONS - 1, idx))

    def forward(self, sensory, update=True):
        g = self.genome
        n = g['n_nodes']
        proprio = sensory[7]
        ri = self._region_index(proprio)
        decay = g['decay_table'][ri]
        bias_base = ri * N_CHANNELS

        new_nodes = [0.0] * n
        for i in range(n):
            h = 0.0
            for j in range(N_CHANNELS):
                h += g['sensory_w'][j][i] * sensory[j]
            for k in range(n):
                h += g['recurrent_w'][i][k] * self.nodes[k]
            new_nodes[i] = math.tanh(h)

        if update:
            for i in range(n):
                self.nodes[i] = (1 - decay) * new_nodes[i] + decay * self.nodes[i]

        pred = [0.0] * N_CHANNELS
        for i in range(n):
            for c in range(N_CHANNELS):
                pw = g['predict_w'][i][c]
                bias = g['predict_w_bias'][bias_base + c]
                pred[c] += self.nodes[i] * (pw + bias)

        self.prev_sensory = sensory[:]
        return pred, self.nodes[:]

    def update_weights(self, pred, actual):
        g = self.genome
        n = g['n_nodes']
        lr = g['params']['learning_rate']
        ri = self._region_index(actual[7])
        bias_base = ri * N_CHANNELS
        errors = [actual[j] - pred[j] for j in range(N_CHANNELS)]
        mse = sum(e ** 2 for e in errors) / N_CHANNELS

        for i in range(n):
            for c in range(N_CHANNELS):
                delta = lr * self.nodes[i] * errors[c]
                g['predict_w'][i][c] += delta
                g['predict_w'][i][c] = max(-3.0, min(3.0, g['predict_w'][i][c]))

        for c in range(N_CHANNELS):
            delta_bias = lr * 1.5 * errors[c]
            g['predict_w_bias'][bias_base + c] += delta_bias
            g['predict_w_bias'][bias_base + c] = max(-5.0, min(5.0, g['predict_w_bias'][bias_base + c]))

        return mse


# ========== IM ==========

class IntrinsicMotivation:
    def __init__(self):
        self.ema_error = 0.3
        self.ema_error_fast = 0.3
        self.ema_error_slow = 0.3
        self.boredom = 0.0
        self.lr_log = []
        self.region_scores = collections.defaultdict(float)
        self.drive_energy = 0.0
        self.region_history = []

    def update(self, current_mse, region_sigma, region_id, current_region):
        self.ema_error = 0.7 * self.ema_error + 0.3 * current_mse
        self.ema_error_fast = 0.3 * self.ema_error_fast + 0.7 * current_mse
        self.ema_error_slow = 0.99 * self.ema_error_slow + 0.01 * current_mse

        if self.ema_error_fast < self.ema_error_slow * 0.7:
            self.boredom += 0.003
            lr_mult = 0.5
        else:
            self.boredom *= 0.97
            lr_mult = 1.0
        self.boredom = max(0.0, min(1.0, self.boredom))
        self.lr_log.append(lr_mult)

        info_value = region_sigma * (1.0 / max(0.001, current_mse + 0.01))
        self.region_scores[current_region] = 0.95 * self.region_scores[current_region] + 0.05 * info_value

        reward = 0.0
        if region_sigma > 0.5 and current_mse > 0.01:
            reward += 0.02
        if self.boredom > 0.05:
            reward -= 0.01
        self.drive_energy = self.drive_energy * 0.99 + reward
        self.drive_energy = max(-3.0, min(5.0, self.drive_energy))
        self.region_history.append(current_region)
        return lr_mult

    def get_action_bias(self):
        bias = {}
        if self.boredom > 0.05:
            bias[2] = 0.4
            bias[1] = 0.2
        elif self.ema_error_fast > self.ema_error * 1.2:
            bias[0] = 0.2
        return bias

    def get_stats(self):
        return {'drive': round(self.drive_energy, 3),
                'boredom': round(self.boredom, 4),
                'lr': self.lr_log[-1] if self.lr_log else 1.0}


# ========== Organism ==========

class Organism:
    def __init__(self, genome=None, lineage_id=None):
        self.genome = genome if genome else self.random_genome()
        self.wm = WorldModel(self.genome)
        self.im = IntrinsicMotivation()
        self.alive = True
        self.age = 0
        self.lineage_id = lineage_id or self._gen_id()
        self.parents = []
        self._inbreeding = 1.0

    def _gen_id(self):
        return hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]

    @staticmethod
    def random_genome():
        n = random.randint(5, MAX_NODES)
        g = {
            'n_nodes': n,
            'sensory_w': [[random.uniform(-0.3, 0.3) for _ in range(n)] for _ in range(N_CHANNELS)],
            'predict_w': [[random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)] for _ in range(n)],
            'recurrent_w': [[random.uniform(-0.1, 0.1) for _ in range(n)] for _ in range(n)],
            'params': {
                'wm_decay': random.uniform(0.5, 0.95),
                'learning_rate': random.uniform(0.03, 0.12),
            },
            'predict_w_bias': [random.uniform(-0.3, 0.3) for _ in range(N_REGIONS * N_CHANNELS)],
            'decay_table': [random.uniform(0.4, 0.95) for _ in range(N_REGIONS)],
        }
        for i in range(n):
            g['recurrent_w'][i][i] = random.uniform(0.5, 1.5)
        return g

    def clone_genome(self):
        return copy.deepcopy(self.genome)


# ========== Engine ==========

def genome_distance(g1, g2):
    n1, n2 = g1['n_nodes'], g2['n_nodes']
    n = min(n1, n2)
    dist = abs(n1 - n2) * 0.3
    for j in range(N_CHANNELS):
        for i in range(n):
            dist += abs(g1['sensory_w'][j][i] - g2['sensory_w'][j][i])
    for i in range(n):
        for c in range(N_CHANNELS):
            dist += abs(g1['predict_w'][i][c] - g2['predict_w'][i][c])
    nb = min(len(g1['predict_w_bias']), len(g2['predict_w_bias']))
    for k in range(nb):
        dist += abs(g1['predict_w_bias'][k] - g2['predict_w_bias'][k]) * 0.5
    norm = N_CHANNELS * n * 2 + nb * 0.5
    return dist / norm if norm > 0 else 0.5


def module_swap_crossover(g1, g2):
    """模块化杂交：base weight 可以做 row_mix，但 bias/decay 必须整体继承"""
    n = min(g1['n_nodes'], g2['n_nodes'])
    child_n = random.choice([g1['n_nodes'], g2['n_nodes']])
    child = {
        'n_nodes': child_n,
        'sensory_w': [[0.0] * child_n for _ in range(N_CHANNELS)],
        'predict_w': [[0.0] * N_CHANNELS for _ in range(child_n)],
        'recurrent_w': [[0.0] * child_n for _ in range(child_n)],
        'params': copy.deepcopy(g1['params']),
        'predict_w_bias': [0.0] * (N_REGIONS * N_CHANNELS),
        'decay_table': [0.0] * N_REGIONS,
    }

    # Base weights: row-level 混合
    for j in range(N_CHANNELS):
        for i in range(child_n):
            src = g1 if random.random() < 0.5 else g2
            si = min(i, src['n_nodes'] - 1)
            child['sensory_w'][j][i] = src['sensory_w'][j][si]

    for i in range(child_n):
        si = min(i, g1['n_nodes'] - 1) if random.random() < 0.5 else min(i, g2['n_nodes'] - 1)
        src = g1 if random.random() < 0.5 else g2
        si = min(i, src['n_nodes'] - 1)
        for c in range(N_CHANNELS):
            child['predict_w'][i][c] = src['predict_w'][si][c]

    for i in range(child_n):
        src = g1 if random.random() < 0.5 else g2
        si = min(i, src['n_nodes'] - 1)
        for k in range(child_n):
            sk = min(k, src['n_nodes'] - 1)
            child['recurrent_w'][i][k] = src['recurrent_w'][si][sk]

    # MODULE SWAP: bias 和 decay_table 整个区域模块继承自一个父本
    intact_parent = g1 if random.random() < 0.5 else g2
    child['predict_w_bias'] = intact_parent['predict_w_bias'][:]
    child['decay_table'] = intact_parent['decay_table'][:]

    if random.random() < 0.5:
        child['params'] = copy.deepcopy(g2['params'])

    return child


def select_complementary_pair(survivors, sf):
    idx1 = random.choices(range(len(survivors)), weights=sf, k=1)[0]
    p1 = survivors[idx1]
    dists = []
    for j, p2 in enumerate(survivors):
        if j == idx1:
            dists.append(0)
        else:
            dists.append(genome_distance(p1.genome, p2.genome))
    idx2 = max(range(len(survivors)), key=lambda j: dists[j] * sf[j])
    return survivors[idx1], survivors[idx2]


def mutate_genome(g, rate=0.05):
    n = g['n_nodes']
    for j in range(N_CHANNELS):
        for i in range(len(g['sensory_w'][j])):
            if random.random() < rate:
                g['sensory_w'][j][i] += random.gauss(0, 0.08)
                g['sensory_w'][j][i] = max(-3.0, min(3.0, g['sensory_w'][j][i]))
    for i in range(len(g['predict_w'])):
        for c in range(N_CHANNELS):
            if random.random() < rate:
                g['predict_w'][i][c] += random.gauss(0, 0.08)
                g['predict_w'][i][c] = max(-3.0, min(3.0, g['predict_w'][i][c]))
    for i in range(len(g['recurrent_w'])):
        for k in range(len(g['recurrent_w'][i])):
            if random.random() < rate:
                g['recurrent_w'][i][k] += random.gauss(0, 0.04)
                g['recurrent_w'][i][k] = max(-2.0, min(2.0, g['recurrent_w'][i][k]))

    if random.random() < 0.2:
        g['params']['wm_decay'] += random.gauss(0, 0.04)
        g['params']['wm_decay'] = max(0.1, min(0.99, g['params']['wm_decay']))
    if random.random() < 0.2:
        g['params']['learning_rate'] += random.gauss(0, 0.02)
        g['params']['learning_rate'] = max(0.02, min(0.15, g['params']['learning_rate']))

    # bias 和 decay 用更低突变率
    for i in range(len(g['predict_w_bias'])):
        if random.random() < rate * 0.5:
            g['predict_w_bias'][i] += random.gauss(0, 0.1)
            g['predict_w_bias'][i] = max(-5.0, min(5.0, g['predict_w_bias'][i]))
    for i in range(len(g['decay_table'])):
        if random.random() < rate * 0.3:
            g['decay_table'][i] += random.gauss(0, 0.03)
            g['decay_table'][i] = max(0.1, min(0.99, g['decay_table'][i]))

    if random.random() < 0.03 and n < MAX_NODES:
        for j in range(N_CHANNELS):
            g['sensory_w'][j].append(random.uniform(-0.2, 0.2))
        g['predict_w'].append([random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)])
        for row in g['recurrent_w']:
            row.append(random.uniform(-0.05, 0.05))
        g['recurrent_w'].append([random.uniform(-0.05, 0.05) for _ in range(n + 1)])
        g['recurrent_w'][-1][-1] = random.uniform(0.5, 1.5)
        g['n_nodes'] = n + 1

    return g


class EvolutionaryEngine:
    def __init__(self, n_organisms=30, n_generations=200, seed=7):
        self.n = n_organisms
        self.n_gen = n_generations
        self.pool = StableRegionPool(seed=seed)
        self.population = [Organism() for _ in range(n_organisms)]
        self.history = []
        self.generation = 0

    def compute_fitness(self, org, ticks=300):
        self.pool.reset_states()
        org.wm.nodes = [0.0] * org.genome['n_nodes']
        org.wm.prev_sensory = [0.0] * N_CHANNELS
        org.im = IntrinsicMotivation()

        total_mse = 0.0
        total_info = 0.0
        region_ticks = collections.Counter()
        region_switch_count = 0
        current_region = 0

        for t in range(ticks):
            sensory = self.pool.get_sensory(current_region)
            pred, nodes = org.wm.forward(sensory, update=True)
            mse = sum((sensory[j] - pred[j]) ** 2 for j in range(N_CHANNELS)) / N_CHANNELS
            lr_mult = org.im.update(mse, self.pool.regions[current_region]['sigma'], current_region, current_region)
            org.wm.update_weights(pred, sensory)
            total_mse += mse
            region_ticks[current_region] += 1
            self.pool.step_global()

            bias = org.im.get_action_bias()
            if bias and random.random() < 0.3:
                action = max(bias, key=bias.get)
            else:
                action = random.choices([0, 1, 2], weights=[0.92, 0.06, 0.02])[0]

            old_region = current_region
            if action == 1:
                current_region = random.randint(0, N_REGIONS - 1)
            elif action == 2:
                probs = [r['sigma'] for r in self.pool.regions]
                total_p = sum(probs)
                probs = [p / total_p for p in probs]
                current_region = random.choices(range(N_REGIONS), weights=probs, k=1)[0]
            if current_region != old_region:
                region_switch_count += 1

            if 0.001 < mse < 0.5:
                total_info += 1
            org.age += 1

        avg_mse = total_mse / ticks
        fitness = 1.0 / (1.0 + avg_mse)
        diversity = len([c for c in region_ticks.values() if c > 0])
        fitness *= getattr(org, '_inbreeding', 1.0)
        return {
            'fitness': fitness, 'avg_mse': avg_mse, 'total_info': total_info,
            'diversity': diversity, 'region_ticks': dict(region_ticks),
            'region_switches': region_switch_count,
        }

    def _detect_wild_aborts(self, sorted_pairs):
        """野败识别：在排名后半的个体中找行为最异常的独特个体。

        袁隆平的野败 = 天然不育的野生稻，单独种毫无价值，
        但跟栽培稻杂交后产生产量飞跃。

        映射到本系统：
        - 排名 bottom 50% 的个体中
        - 找出 region_history 模式与种群均值差异最大的若干个体
        - 这些个体可能携带了"探索极端环境"的基因组合
        - 让它们跟精英杂交，看能不能产出突破性后代
        """
        bottom_half = sorted_pairs[len(sorted_pairs)//2:]
        if len(bottom_half) < 3:
            return []

        # 计算种群的平均行为特征：每个个体访问各区域的频率
        all_traces = []
        for p in sorted_pairs:
            org = p[0]
            if org.im.region_history:
                hist = org.im.region_history
                freq = [0] * N_REGIONS
                for r in hist:
                    freq[r] += 1
                total = sum(freq)
                if total > 0:
                    freq = [f/total for f in freq]
                all_traces.append(freq)

        if not all_traces:
            return []

        # 种群平均区域分布
        pop_avg = [0.0] * N_REGIONS
        for t in all_traces:
            for i in range(N_REGIONS):
                pop_avg[i] += t[i]
        pop_avg = [x / len(all_traces) for x in pop_avg]

        # 找区域分布最离群的 bottom-half 个体
        wild_candidates = []
        for p in bottom_half:
            org = p[0]
            if not org.im.region_history:
                continue
            freq = [0] * N_REGIONS
            for r in org.im.region_history:
                freq[r] += 1
            total = sum(freq)
            if total > 0:
                freq = [f/total for f in freq]
            # 欧氏距离 from population average
            dist = sum((freq[i] - pop_avg[i])**2 for i in range(N_REGIONS)) ** 0.5
            # 还考虑基因组的"极端度"：bias 参数的标准差
            biases = org.genome['predict_w_bias']
            bias_std = (sum((b - sum(biases)/len(biases))**2 for b in biases) / len(biases)) ** 0.5
            wild_candidates.append((p[0], dist + bias_std * 0.1))

        # 按野败分数排序，取 top 30%
        wild_candidates.sort(key=lambda x: x[1], reverse=True)
        n_pick = max(2, len(wild_candidates) // 3)
        return [wc[0] for wc in wild_candidates[:n_pick]]

    def evolve(self):
        for gen in range(self.n_gen):
            self.generation = gen
            gen_results = []

            for org in self.population:
                result = self.compute_fitness(org, ticks=800)
                org._last_result = result
                gen_results.append(result)

            sorted_pairs = sorted(zip(self.population, gen_results),
                                  key=lambda x: x[1]['fitness'], reverse=True)

            avg_fit = sum(r['fitness'] for r in gen_results) / len(gen_results)
            avg_mse = sum(r['avg_mse'] for r in gen_results) / len(gen_results)
            best_fit = sorted_pairs[0][1]['fitness']
            best_mse = sorted_pairs[0][1]['avg_mse']
            avg_boredom = sum(org.im.boredom for org in self.population) / len(self.population)
            diversity = sum(r['diversity'] for r in gen_results) / len(gen_results)

            # diversity
            div_mean = 0
            traces = [org.im.region_history[-30:] for org in self.population if org.im.region_history]
            if len(traces) >= 2:
                dists = []
                for i in range(min(15, len(traces))):
                    for j in range(i + 1, min(i + 4, len(traces))):
                        if traces[i] and traces[j]:
                            d = sum(1 for a, b in zip(traces[i], traces[j]) if a != b) / max(len(traces[i]), len(traces[j]))
                            dists.append(d)
                if dists:
                    div_mean = sum(dists) / len(dists)

            all_regions = set()
            for org in self.population:
                for r in org.im.region_history:
                    all_regions.add(r)
            region_diversity = len(all_regions)
            total_switches = sum(r['region_switches'] for r in gen_results)
            avg_drive = sum(org.im.drive_energy for org in self.population) / len(self.population)

            wild_aborts = self._detect_wild_aborts(sorted_pairs)

            if gen % 20 == 0 or gen == self.n_gen - 1:
                print(f"Gen {gen:4d} | fit={best_fit:.4f}/{avg_fit:.4f} | mse={best_mse:.4f}/{avg_mse:.4f} | "
                      f"bored={avg_boredom:.4f} | div={div_mean:.3f} | r_div={region_diversity} | sw={total_switches} | WA={len(wild_aborts)}")

            self.history.append({
                'gen': gen, 'best_fitness': best_fit, 'avg_fitness': avg_fit,
                'best_mse': best_mse, 'avg_mse': avg_mse, 'avg_boredom': avg_boredom,
                'div_mean': div_mean, 'region_diversity': region_diversity,
                'total_switches': total_switches, 'alive': len(self.population),
            })

            if best_fit >= 0.99:
                print(f"Converged at gen {gen}")
                break

            # === REPRODUCTION: ELITE + WILD ABORT (野败) + RANDOM INJECTION ===
            ELITE_COUNT = 2
            elites = [p[0].clone_genome() for p in sorted_pairs[:ELITE_COUNT]]

            # 野败识别：在低 fitness 个体中找行为最异常的
            # (已在上方计算，此处复用)

            # 扩大 survivor pool 到 top 50%
            survivor_n = max(10, self.n // 2)
            survivors = [p[0] for p in sorted_pairs[:survivor_n]]
            survivor_fits = [p[1]['fitness'] for p in sorted_pairs[:survivor_n]]
            fit_shifted = [f - min(survivor_fits) + 0.01 for f in survivor_fits]
            total_f = sum(fit_shifted)
            sf = [f / total_f for f in fit_shifted]

            # 三种交配配额
            n_elite_hybrid = max(1, int(self.n * 0.45))
            n_wild_hybrid = max(1, int(self.n * 0.25))
            n_random = max(1, int(self.n * 0.15))
            n_mutant = self.n - ELITE_COUNT - n_elite_hybrid - n_wild_hybrid - n_random

            new_pop = []
            for elite_g in elites:
                elite_org = Organism(genome=copy.deepcopy(elite_g))
                elite_org._inbreeding = 1.0
                new_pop.append(elite_org)

            elite_hybrid_count = 0
            wild_hybrid_count = 0
            random_count = 0
            mutant_count = 0
            attempts = 0

            while len(new_pop) < self.n and attempts < self.n * 50:
                attempts += 1
                if not survivors or sum(sf) <= 0:
                    new_pop.append(Organism())
                    continue

                if elite_hybrid_count < n_elite_hybrid:
                    top_n = max(4, len(survivors) // 3)
                    p1, p2 = select_complementary_pair(survivors[:top_n], sf[:top_n])
                    child_g = module_swap_crossover(p1.clone_genome(), p2.clone_genome())
                    child_g = mutate_genome(child_g, rate=0.05)
                    child = Organism(genome=child_g,
                                     lineage_id=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8])
                    child.parents = [p1.lineage_id, p2.lineage_id]
                    child._inbreeding = 1.0
                    elite_hybrid_count += 1
                    new_pop.append(child)
                elif wild_hybrid_count < n_wild_hybrid and wild_aborts:
                    wa_org = random.choice(wild_aborts)
                    elite_org = random.choices(survivors[:5], weights=sf[:5], k=1)[0]
                    child_g = module_swap_crossover(wa_org.clone_genome(), elite_org.clone_genome())
                    child_g = mutate_genome(child_g, rate=0.08)
                    child = Organism(genome=child_g,
                                     lineage_id=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8])
                    child.parents = [wa_org.lineage_id, elite_org.lineage_id]
                    child._inbreeding = 1.0
                    wild_hybrid_count += 1
                    new_pop.append(child)
                elif random_count < n_random:
                    new_pop.append(Organism())
                    random_count += 1
                elif mutant_count < n_mutant:
                    parent = random.choices(survivors, weights=sf, k=1)[0]
                    child_g = mutate_genome(parent.clone_genome(), rate=0.05)
                    child = Organism(genome=child_g,
                                     lineage_id=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8])
                    child.parents = [parent.lineage_id]
                    child._inbreeding = 1.0
                    mutant_count += 1
                    new_pop.append(child)

            self.population = new_pop

        # Final eval
        for org in self.population:
            result = self.compute_fitness(org, ticks=800)
            org._last_result = result

        return self.population, self.history


def run_experiment():
    print("=" * 70)
    print("yinyang_evo_v17d — Wild Abort + Module-swap + Elitism")
    print("=" * 70)

    engine = EvolutionaryEngine(n_organisms=20, n_generations=400, seed=11)
    pop, history = engine.evolve()

    best = max(pop, key=lambda o: getattr(o, '_last_result', {}).get('fitness', 0))
    best_fit = getattr(best, '_last_result', {}).get('fitness', 0)

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best fitness: {best_fit:.4f}")
    print(f"Best avg_mse: {getattr(best, '_last_result', {}).get('avg_mse', 'N/A')}")
    print(f"Genome n_nodes: {best.genome['n_nodes']}")
    print(f"learning_rate: {best.genome['params']['learning_rate']:.4f}")
    print(f"decay_table: {[f'{x:.3f}' for x in best.genome['decay_table']]}")

    # Regional precision (speed: warmup=40, n=20)
    print("\n--- Regional Precision ---")
    for region_id in range(N_REGIONS):
        pool = StableRegionPool(seed=42)
        pool.reset_states()
        best.wm.nodes = [0.0] * best.genome['n_nodes']
        for _ in range(40):
            s = pool.get_sensory(region_id)
            best.wm.forward(s, update=True)
            pool.step_global()
        mses = []
        for _ in range(20):
            s = pool.get_sensory(region_id)
            pred, _ = best.wm.forward(s, update=False)
            mse = sum((s[j] - pred[j]) ** 2 for j in range(N_CHANNELS)) / N_CHANNELS
            mses.append(mse)
            pool.step_global()
        avg_mse = sum(mses) / len(mses)
        bar = '#' * max(1, int(avg_mse * 8))
        print(f"  R{region_id} σ={pool.regions[region_id]['sigma']:.2f}: {avg_mse:.4f} |{bar}")

    print()
    print("History (every 10 gen):")
    for h in history[-10:]:
        print(f"  Gen {h['gen']:3d}: fit={h['best_fitness']:.4f} mse={h['best_mse']:.4f} "
              f"div={h['div_mean']:.3f} r_div={h['region_diversity']}")

    return engine, best


if __name__ == '__main__':
    run_experiment()
