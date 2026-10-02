#!/usr/bin/env python3
"""
yinyang_evo_v17b — Direct Region-Specialized WM (查表式 context 切换)
===========================================================
v17 诊断：45 个连续调制参数搜索空间太大 → 进化没找到有效值。
Test C 证实所有区域预测 = +0.000（零状态，零输出）。
Test A (update=False) 测的是零节点输出 = 信号方差，不是真正精度。

v17b 设计：
1. 查表法：proprio → region_index (round(proprio * N_REGIONS-1))
2. predict_w_bias[region_index * N_CHANNELS + c] — 每区域每通道的固定偏置
3. decay_table[region_index] — 每区域的固定 decay
4. 没有连续调制、没有额外 RNN 门控 — 直接 = 为每个区域学一组固定参数
5. Online Hebbian 保留，但偏置表本身是 offine 学好的（via 进化）
6. 精度测试改为：先预热(update=True)让节点收敛，再测(update=False)

这个设计的核心思想：
- 精准 = 查表给出的每区域偏置已经能输出正确预测
- 自主 = IM + 进化仍然驱动区域选择和多区域行为
"""

import random, math, hashlib, json, copy, collections

N_CHANNELS = 8
MAX_NODES = 15
N_REGIONS = 9

# ========== 环境 ==========

def make_regions(seed=42):
    rng = random.Random(seed)
    return [
        {'id': i,
         'sigma': round(0.05 + 0.64 * (i / (N_REGIONS - 1)), 2),
         'proprio': round(i / (N_REGIONS - 1), 2),
         'freq': round(0.03 + 0.12 * rng.random(), 3),
         'phase': round(rng.uniform(0, 2*math.pi), 3),
         'type': ['sin', 'logistic', 'lorenz', 'van der Pol', 'sawtooth',
                  'rossler', 'henon', 'tent', 'cubic'][i]}
        for i in range(N_REGIONS)
    ]

class StableRegionPool:
    """固定吸引子参数，每代 reset；step_global 每 tick 调用一次"""
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

        # 确定性噪声（seeded by region + tick）
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
        else:  # cubic
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
            r['proprio'],  # channel 7 = region identifier
        ]


# ========== World Model ==========

class WorldModel:
    """v17b: 查表式区域专用预测 + 固定衰减"""
    def __init__(self, genome):
        self.genome = genome
        n = genome['n_nodes']
        self.nodes = [0.0] * n
        self.prev_sensory = [0.0] * N_CHANNELS

    def _region_index(self, proprio_val):
        """proprio → region index (查表)"""
        idx = round(proprio_val * (N_REGIONS - 1))
        return max(0, min(N_REGIONS - 1, idx))

    def forward(self, sensory, update=True):
        g = self.genome
        n = g['n_nodes']

        proprio = sensory[7]
        ri = self._region_index(proprio)
        decay = g['decay_table'][ri]

        # Region-specific predict_w bias
        bias_base = ri * N_CHANNELS

        # 标准 RNN 更新
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

        # 区域专用预测输出: base_w + bias[region]
        pred = [0.0] * N_CHANNELS
        for i in range(n):
            for c in range(N_CHANNELS):
                pw = g['predict_w'][i][c]
                bias = g['predict_w_bias'][bias_base + c]
                pred[c] += self.nodes[i] * (pw + bias)

        self.prev_sensory = sensory[:]
        return pred, self.nodes[:]

    def forward_no_update(self, sensory):
        """不更新 self.nodes，不更新权重 — 纯推理"""
        return self.forward(sensory, update=False)

    def update_weights(self, pred, actual):
        """在线 Hebbian"""
        g = self.genome
        n = g['n_nodes']
        lr = g['params']['learning_rate']
        ri = self._region_index(actual[7])
        bias_base = ri * N_CHANNELS

        errors = [actual[j] - pred[j] for j in range(N_CHANNELS)]
        mse = sum(e ** 2 for e in errors) / N_CHANNELS

        # Update base predict_w
        for i in range(n):
            for c in range(N_CHANNELS):
                delta = lr * self.nodes[i] * errors[c]
                g['predict_w'][i][c] += delta
                g['predict_w'][i][c] = max(-3.0, min(3.0, g['predict_w'][i][c]))

        # Also update the region-specific bias (faster learning for region offset)
        for c in range(N_CHANNELS):
            delta_bias = lr * 2.0 * errors[c]  # nodes-mean ~ O(1), but bias converges fast
            g['predict_w_bias'][bias_base + c] += delta_bias
            g['predict_w_bias'][bias_base + c] = max(-5.0, min(5.0, g['predict_w_bias'][bias_base + c]))

        # Weak sensory_w update
        for j in range(N_CHANNELS):
            for i in range(n):
                delta = lr * 0.05 * actual[j] * self.nodes[i]
                g['sensory_w'][j][i] += delta
                g['sensory_w'][j][i] = max(-3.0, min(3.0, g['sensory_w'][j][i]))

        return mse

    def unroll_multi_step(self, sensory_init, proprio_val, n_steps):
        """自回归多步展开"""
        pred1, nodes = self.forward(sensory_init, update=False)
        preds = [pred1]
        nodes_cur = nodes[:]
        n = self.genome['n_nodes']
        ri = self._region_index(proprio_val)
        decay = self.genome['decay_table'][ri]
        bias_base = ri * N_CHANNELS

        for step in range(1, n_steps):
            p = [0.0] * N_CHANNELS
            for i in range(n):
                for c in range(N_CHANNELS):
                    pw = self.genome['predict_w'][i][c]
                    bias = self.genome['predict_w_bias'][bias_base + c]
                    p[c] += nodes_cur[i] * (pw + bias)
            preds.append(p)

            new_nodes = [0.0] * n
            for i in range(n):
                h = 0.0
                for j in range(N_CHANNELS):
                    h += self.genome['sensory_w'][j][i] * p[j]
                for k in range(n):
                    h += self.genome['recurrent_w'][i][k] * nodes_cur[k]
                new_nodes[i] = math.tanh(h)

            for i in range(n):
                nodes_cur[i] = (1 - decay) * new_nodes[i] + decay * nodes_cur[i]

        return preds


# ========== Intrinsic Motivation ==========

class IntrinsicMotivation:
    def __init__(self):
        self.ema_error = 0.3
        self.ema_error_fast = 0.3
        self.ema_error_slow = 0.3
        self.boredom = 0.0
        self.lr_log = []
        self.region_scores = collections.defaultdict(float)
        self.drive_energy = 0.0
        self.last_error_for_trend = 0.3
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
        if self.boredom > 0.03:
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
        self.senses_prev = None
        self.state = [0.0] * N_CHANNELS
        self.alive = True
        self.age = 0
        self.energy = 8.0
        self.lineage_id = lineage_id or self._gen_id()
        self.parents = []
        self.behavior_trace = []
        self.total_information = 0.0
        self.region_counts = collections.Counter()
        self._inbreeding = 1.0
        self._parent_max_fit = 0.0

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
                'learning_rate': random.uniform(0.05, 0.4),
            },
            # v17b: 查表式区域专用参数
            'predict_w_bias': [random.uniform(-0.3, 0.3) for _ in range(N_REGIONS * N_CHANNELS)],
            'decay_table': [random.uniform(0.4, 0.95) for _ in range(N_REGIONS)],
        }
        for i in range(n):
            g['recurrent_w'][i][i] = random.uniform(0.5, 1.5)
        return g

    def clone_genome(self):
        return copy.deepcopy(self.genome)


# ========== Evolution Engine ==========

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
    # bias
    for k in range(len(g1['predict_w_bias'])):
        if k < len(g2['predict_w_bias']):
            dist += abs(g1['predict_w_bias'][k] - g2['predict_w_bias'][k]) * 0.5
    norm = N_CHANNELS * n + N_CHANNELS * n + len(g1['predict_w_bias']) * 0.5
    return dist / norm if norm > 0 else 0.5


def crossover_genomes(g1, g2, mode='row_mix'):
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

    # sensory_w
    for j in range(N_CHANNELS):
        for i in range(child_n):
            if mode == 'row_mix':
                src = g1 if random.random() < 0.5 else g2
            else:
                src = g1 if random.random() < 0.5 else g2
            si = min(i, src['n_nodes'] - 1)
            child['sensory_w'][j][i] = src['sensory_w'][j][si]

    # predict_w
    for i in range(child_n):
        for c in range(N_CHANNELS):
            if mode == 'row_mix':
                src = g1 if random.random() < 0.5 else g2
            else:
                src = g1 if random.random() < 0.5 else g2
            si = min(i, src['n_nodes'] - 1)
            child['predict_w'][i][c] = src['predict_w'][si][c]

    # recurrent_w
    for i in range(child_n):
        for k in range(child_n):
            if mode == 'row_mix':
                src = g1 if random.random() < 0.5 else g2
            else:
                src = g1 if random.random() < 0.5 else g2
            si = min(i, src['n_nodes'] - 1)
            sk = min(k, src['n_nodes'] - 1)
            child['recurrent_w'][i][k] = src['recurrent_w'][si][sk]

    # bias & decay: 随机选父本整体
    src = g1 if random.random() < 0.5 else g2
    child['predict_w_bias'] = src['predict_w_bias'][:]
    child['decay_table'] = src['decay_table'][:]

    # params: 随机选一个父本
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
            d = genome_distance(p1.genome, p2.genome)
            dists.append(d)
    idx2 = max(range(len(survivors)), key=lambda j: dists[j] * sf[j])
    return survivors[idx1], survivors[idx2]


def mutate_genome(g, rate=0.15):
    n = g['n_nodes']

    for j in range(N_CHANNELS):
        for i in range(len(g['sensory_w'][j])):
            if random.random() < rate:
                g['sensory_w'][j][i] += random.gauss(0, 0.1)
                g['sensory_w'][j][i] = max(-3.0, min(3.0, g['sensory_w'][j][i]))
    for i in range(len(g['predict_w'])):
        for j in range(N_CHANNELS):
            if random.random() < rate:
                g['predict_w'][i][j] += random.gauss(0, 0.1)
                g['predict_w'][i][j] = max(-3.0, min(3.0, g['predict_w'][i][j]))
    for i in range(len(g['recurrent_w'])):
        for k in range(len(g['recurrent_w'][i])):
            if random.random() < rate:
                g['recurrent_w'][i][k] += random.gauss(0, 0.05)
                g['recurrent_w'][i][k] = max(-2.0, min(2.0, g['recurrent_w'][i][k]))

    # params
    if random.random() < 0.3:
        g['params']['wm_decay'] += random.gauss(0, 0.05)
        g['params']['wm_decay'] = max(0.1, min(0.99, g['params']['wm_decay']))
    if random.random() < 0.3:
        g['params']['learning_rate'] += random.gauss(0, 0.03)
        g['params']['learning_rate'] = max(0.01, min(1.0, g['params']['learning_rate']))

    # v17b: bias + decay 突变
    for i in range(len(g['predict_w_bias'])):
        if random.random() < rate:
            g['predict_w_bias'][i] += random.gauss(0, 0.15)
            g['predict_w_bias'][i] = max(-5.0, min(5.0, g['predict_w_bias'][i]))
    for i in range(len(g['decay_table'])):
        if random.random() < rate * 0.5:
            g['decay_table'][i] += random.gauss(0, 0.05)
            g['decay_table'][i] = max(0.1, min(0.99, g['decay_table'][i]))

    # 偶尔加节点
    if random.random() < 0.05 and n < MAX_NODES:
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
        self._hybrid_max_delta = []

    def compute_fitness(self, org, ticks=500):
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

            lr_mult = org.im.update(mse, self.pool.regions[current_region]['sigma'],
                                    current_region, current_region)
            org.wm.update_weights(pred, sensory)

            total_mse += mse
            region_ticks[current_region] += 1

            self.pool.step_global()

            bias = org.im.get_action_bias()
            if bias and random.random() < 0.3:
                action = max(bias, key=bias.get)
            else:
                action = random.choices([0, 1, 2], weights=[0.85, 0.1, 0.05])[0]

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
        inbreeding = getattr(org, '_inbreeding', 1.0)
        fitness *= inbreeding

        return {
            'fitness': fitness,
            'avg_mse': avg_mse,
            'total_info': total_info,
            'diversity': diversity,
            'region_ticks': dict(region_ticks),
            'region_switches': region_switch_count,
        }

    def evolve(self):
        for gen in range(self.n_gen):
            self.generation = gen
            gen_results = []

            for org in self.population:
                result = self.compute_fitness(org, ticks=300)
                org._last_result = result
                gen_results.append(result)

            sorted_pairs = sorted(zip(self.population, gen_results),
                                  key=lambda x: x[1]['fitness'], reverse=True)

            avg_fit = sum(r['fitness'] for r in gen_results) / len(gen_results)
            avg_mse = sum(r['avg_mse'] for r in gen_results) / len(gen_results)
            avg_info = sum(r['total_info'] for r in gen_results) / len(gen_results)
            avg_boredom = sum(org.im.boredom for org in self.population) / len(self.population)
            diversity = sum(r['diversity'] for r in gen_results) / len(gen_results)
            best_fit = sorted_pairs[0][1]['fitness']
            best_mse = sorted_pairs[0][1]['avg_mse']

            hybrid_deltas = [getattr(org, '_parent_max_fit', 0) for org in self.population if getattr(org, '_parent_max_fit', 0) > 0]
            avg_delta = 0
            if hybrid_deltas:
                parent_fits = hybrid_deltas
                child_fits = [org._last_result['fitness'] for org in self.population if getattr(org, '_parent_max_fit', 0) > 0]
                if parent_fits and child_fits:
                    delta = (sum(child_fits) / len(child_fits)) - (sum(parent_fits) / len(parent_fits))
                    avg_delta = delta
                    self._hybrid_max_delta.append(delta)

            div_mean = 0
            traces = [org.im.region_history[-30:] for org in self.population if org.im.region_history]
            if len(traces) >= 2:
                dists = []
                for i in range(len(traces)):
                    for j in range(i + 1, min(i + 5, len(traces))):
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

            avg_drive = sum(org.im.drive_energy for org in self.population) / len(self.population)
            total_switches = sum(r['region_switches'] for r in gen_results)

            if gen % 10 == 0 or gen == self.n_gen - 1:
                print(f"Gen {gen:4d} | fit={best_fit:.4f}/{avg_fit:.4f} | mse={best_mse:.4f}/{avg_mse:.4f} | "
                      f"info={avg_info:.0f} | bored={avg_boredom:.4f} | div={div_mean:.3f} | "
                      f"r_div={region_diversity} | switches={total_switches} | hybrid={avg_delta:+.4f}")

            self.history.append({
                'gen': gen,
                'best_fitness': best_fit,
                'avg_fitness': avg_fit,
                'best_mse': best_mse,
                'avg_mse': avg_mse,
                'avg_info': avg_info,
                'avg_drive': avg_drive,
                'avg_boredom': avg_boredom,
                'div_mean': div_mean,
                'region_diversity': region_diversity,
                'avg_hybrid_delta': avg_delta,
                'total_switches': total_switches,
                'alive': len(self.population),
            })

            if best_fit >= 0.99:
                print(f"Converged at gen {gen}")
                break

            survivors = [p[0] for p in sorted_pairs[:max(3, self.n // 3)]]
            survivor_fits = [p[1]['fitness'] for p in sorted_pairs[:max(3, self.n // 3)]]
            fit_shifted = [f - min(survivor_fits) + 0.01 for f in survivor_fits]
            total_f = sum(fit_shifted)
            sf = [f / total_f for f in fit_shifted]

            n_hybrid = max(1, int(self.n * 0.7))
            n_mutant = self.n - n_hybrid

            new_pop = []
            hybrid_count = 0
            mutant_count = 0
            attempts = 0

            while len(new_pop) < self.n and attempts < self.n * 30:
                attempts += 1
                if not survivors or sum(sf) <= 0:
                    new_pop.append(Organism())
                    continue

                if hybrid_count < n_hybrid:
                    p1, p2 = select_complementary_pair(survivors, sf)
                    mode = random.choices(['row_mix', 'uniform'], weights=[0.5, 0.5])[0]
                    child_g = crossover_genomes(p1.clone_genome(), p2.clone_genome(), mode=mode)
                    child_g = mutate_genome(child_g)
                    child = Organism(genome=child_g,
                                     lineage_id=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8])
                    child.parents = [p1.lineage_id, p2.lineage_id]
                    child._inbreeding = 1.0
                    child._parent_max_fit = max(p1._last_result['fitness'], p2._last_result['fitness'])
                    hybrid_count += 1
                    new_pop.append(child)
                elif mutant_count < n_mutant:
                    parent = random.choices(survivors, weights=sf, k=1)[0]
                    child_g = mutate_genome(parent.clone_genome())
                    child = Organism(genome=child_g,
                                     lineage_id=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8])
                    child.parents = [parent.lineage_id]
                    child._inbreeding = 1.0
                    mutant_count += 1
                    new_pop.append(child)

            self.population = new_pop

        # Final evaluation
        for org in self.population:
            result = self.compute_fitness(org, ticks=300)
            org._last_result = result

        return self.population, self.history


# ========== Tests ==========

def regional_precision_test(org, n_warmup=50, n_test=80, n_trials=3):
    """先 warmup 让节点收敛，再测量预测精度（update=False 测纯推理）"""
    results = {}
    for region_id in range(N_REGIONS):
        mses = []
        for trial in range(n_trials):
            pool = StableRegionPool(seed=42 + trial)
            pool.reset_states()
            org.wm.nodes = [0.0] * org.genome['n_nodes']

            # warmup (收敛节点)
            for _ in range(n_warmup):
                s = pool.get_sensory(region_id)
                org.wm.forward(s, update=True)
                pool.step_global()

            # test (no weight update)
            for _ in range(n_test):
                s = pool.get_sensory(region_id)
                pred, _ = org.wm.forward(s, update=False)
                mse = sum((s[j] - pred[j]) ** 2 for j in range(N_CHANNELS)) / N_CHANNELS
                mses.append(mse)
                pool.step_global()

        results[region_id] = round(sum(mses) / len(mses), 4)
    return results


def multi_step_test(org, steps=[1, 3, 5, 10, 20], n_trials=6):
    results = {}
    for n_step in steps:
        mses = []
        for trial in range(n_trials):
            region_id = trial % N_REGIONS
            pool = StableRegionPool(seed=42 + trial)
            pool.reset_states()
            org.wm.nodes = [0.0] * org.genome['n_nodes']

            # warmup (same region)
            for _ in range(50):
                s = pool.get_sensory(region_id)
                org.wm.forward(s, update=True)
                pool.step_global()

            # unroll
            s = pool.get_sensory(region_id)
            preds = org.wm.unroll_multi_step(s, pool.regions[region_id]['proprio'], n_step)

            # actual
            for _ in range(n_step):
                pool.step_global()
            actual = pool.get_sensory(region_id)

            predicted_at_n = preds[-1]
            mse = sum((actual[j] - predicted_at_n[j]) ** 2 for j in range(N_CHANNELS)) / N_CHANNELS
            mses.append(mse)

        results[n_step] = round(sum(mses) / len(mses), 5)
    return results


# ========== Main ==========

def run_experiment():
    print("=" * 70)
    print("yinyang_evo_v17c — Long-run evolution (500 gen)")
    print("=" * 70)

    engine = EvolutionaryEngine(n_organisms=40, n_generations=500, seed=7)
    pop, history = engine.evolve()

    best = max(pop, key=lambda o: getattr(o, '_last_result', {}).get('fitness', 0))
    best_fit = getattr(best, '_last_result', {}).get('fitness', 0)

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best fitness: {best_fit:.4f}")
    print(f"Best avg_mse: {getattr(best, '_last_result', {}).get('avg_mse', 'N/A')}")
    print(f"Best diversity: {getattr(best, '_last_result', {}).get('diversity', 'N/A')}")
    print(f"Genome n_nodes: {best.genome['n_nodes']}")
    print(f"learning_rate: {best.genome['params']['learning_rate']:.4f}")
    print(f"decay_table: {[f'{x:.3f}' for x in best.genome['decay_table']]}")

    # Test A: Regional precision
    print("\n--- Test A: Regional Precision (warmup+test) ---")
    regional = regional_precision_test(best, n_warmup=60, n_test=50)
    for rid, mse in sorted(regional.items()):
        bar = '#' * max(1, int(mse * 10))
        print(f"  Region {rid} (σ={engine.pool.regions[rid]['sigma']:.2f}): MSE={mse:.4f} |{bar}")

    avg_r = sum(regional.values()) / len(regional)
    min_r = min(regional.values())
    max_r = max(regional.values())
    below_01 = sum(1 for m in regional.values() if m < 0.01)
    below_05 = sum(1 for m in regional.values() if m < 0.05)
    print(f"  AVG={avg_r:.4f}  MIN={min_r:.4f}  MAX={max_r:.4f}  SPREAD={max_r - min_r:.4f}")
    print(f"  Regions with MSE<0.01: {below_01}/9   MSE<0.05: {below_05}/9")

    # Test B: Multi-step prediction
    print("\n--- Test B: Multi-step Prediction ---")
    ms = multi_step_test(best, steps=[1, 3, 5, 10, 20])
    for ns, mse in sorted(ms.items()):
        bar = '#' * max(1, int(mse * 5))
        print(f"  N={ns:3d}: MSE={mse:.4f} |{bar}")

    # Test C: Context modulation check
    print("\n--- Test C: Region-specific prediction shift ---")
    for r_idx in [0, 4, 8]:
        pool = StableRegionPool(seed=42)
        pool.reset_states()
        best.wm.nodes = [0.0] * best.genome['n_nodes']
        for _ in range(80):
            s = pool.get_sensory(r_idx)
            pred, _ = best.wm.forward(s, update=True)
            pool.step_global()
        print(f"  Region {r_idx} (σ={engine.pool.regions[r_idx]['sigma']:.2f}): pred_ch0={pred[0]:+.4f}")

    # Test D: predict_w_bias pattern
    print("\n--- Test D: Learned bias pattern per region ---")
    for ri in range(N_REGIONS):
        biases = best.genome['predict_w_bias'][ri * N_CHANNELS:(ri + 1) * N_CHANNELS]
        print(f"  Region {ri} (σ={engine.pool.regions[ri]['sigma']:.2f}): bias={[f'{b:+.2f}' for b in biases[:4]]}...")

    print()
    print("Evolution history (last 10):")
    for h in history[-10:]:
        print(f"  Gen {h['gen']:3d}: fit={h['best_fitness']:.4f} mse={h['best_mse']:.4f} "
              f"div={h['region_diversity']} switches={h['total_switches']}")

    return engine, best


if __name__ == '__main__':
    run_experiment()
