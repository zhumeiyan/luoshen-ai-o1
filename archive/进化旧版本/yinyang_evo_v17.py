#!/usr/bin/env python3
"""
yinyang_evo_v17 — Context-Conditioned World Model
=================================================
诊断：v16 的 predict_w 列范数分析显示 channel7(proprio) 只有 0.23，
行范数 0.58-1.20 无特化 — 一个 RNN 试图同时建模 9 种不同动力学，
权重互相干扰，结果哪个都没学会。

修复架构：本体感觉调制 World Model
1. proprio(scalar) → embed(3-d) via learned tanh projection
2. decay modulation: wm_decay_eff = sigmoid(base_logit + embed @ decay_mod)
   — 低混沌区域高 decay（信内部），高混沌区域低 decay（信感官）
3. predict_w modulation: predict_w_eff[ctx] = base + outer(embed, predict_row_mod)
   — 不同本体感觉 → 不同预测映射
4. 其余（进化、IM、环境）不变

genome 新增参数 (~36个):
  proprio_embed_w: [3] — proprio→embed 投影
  decay_mod: [3] — embed 对 decay 的缩放
  predict_row_mod: [3*n_channels] — embed 对 predict_w 的行调制
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
         'proprio': round(i / (N_REGIONS - 1), 2),  # 0.00, 0.12, ..., 1.00
         'freq': round(0.03 + 0.09 * rng.random(), 3),
         'phase': round(rng.uniform(0, 2*math.pi), 3),
         'type': rng.choice(['sin','logistic','lorenz','van der Pol','sawtooth',
                             'rossler','henon','tent','cubic'])}
        for i in range(N_REGIONS)
    ]

REGION_TYPES = {
    'sin': lambda p, s, t: math.sin(p['freq'] * t + p['phase']) + p['sigma'] * rng.gauss(0,1),
    'logistic': lambda p, s, t: 4 * s * (1 - s) if 0 < s < 1 else 0.5,
    'lorenz': lambda p, s, t: 0.5 + 0.3 * math.tanh(s),
    'van der Pol': lambda p, s, t: s - 0.1 * (s**3 - s),
    'sawtooth': lambda p, s, t: (p['freq'] * t + p['phase']) % 1.0,
    'rossler': lambda p, s, t: 0.5 * math.sin(0.7 * p['freq'] * t + s),
    'henon': lambda p, s, t: 1 - 1.4 * s**2 + 0.3 * s,
    'tent': lambda p, s, t: 2 * s if s < 0.5 else 2 * (1 - s),
    'cubic': lambda p, s, t: s**3 - s,
}

class InfoRegion:
    """一个信息区域：稳态吸引子 + 噪声"""
    def __init__(self, region_def, seed=0):
        self.defn = region_def
        self.sigma = region_def['sigma']
        self.proprio = region_def['proprio']
        self.rng = random.Random(seed + region_def['id'])
        self.state = 0.3
        self.tick = 0
        self._next()

    def _next(self):
        self.tick += 1
        rt = self.defn['type']
        dev = REGION_TYPES[rt]
        noise = self.sigma * self.rng.gauss(0, 1)
        val = dev(self.defn, self.state, self.tick)
        if isinstance(val, (int, float)):
            self.state = max(-2.0, min(2.0, val + noise))
        return max(-2.0, min(2.0, noise))

    def get_sensory(self):
        """返回到 8 通道的感受向量"""
        v = self.state
        return [
            v,
            math.sin(v * 2.5) if abs(v) < 1.5 else 0,
            math.tanh(v),
            v**2 * (1 if v > 0 else -1),
            abs(v),
            math.cos(v * 1.7),
            sign(v) * math.sqrt(abs(v)) if abs(v) < 2 else 0,
            self.proprio,  # channel 7 = 本体感觉
        ]

class StableRegionPool:
    """v15+ 版本：固定吸引子参数，每代重置状态"""
    def __init__(self, seed=42):
        self.regions = [StableRegionPool._make_region(i, seed) for i in range(N_REGIONS)]
        self.region_map = {r['id']: i for i, r in enumerate(self.regions)}

    @staticmethod
    def _make_region(idx, seed):
        rng = random.Random(seed * 100 + idx)
        return {
            'id': idx,
            'sigma': round(0.05 + 0.64 * (idx / (N_REGIONS - 1)), 2),
            'proprio': round(idx / (N_REGIONS - 1), 2),
            'freq': round(0.03 + 0.12 * rng.random(), 3),
            'phase': round(rng.uniform(0, 2 * math.pi), 3),
            'type': ['sin', 'logistic', 'lorenz', 'van der Pol', 'sawtooth',
                     'rossler', 'henon', 'tent', 'cubic'][idx],
        }

    def get_region_region(self, region_id):
        return self.regions[self.region_map[region_id]]

    def reset_states(self):
        for r in self.regions:
            r['_state'] = 0.3
            r['_phase_accum'] = 0.0
            r['_tick'] = 0

    def step_global(self):
        """每 tick 全局步进一次（修复 v12 bug）"""
        for r in self.regions:
            r['_tick'] = r.get('_tick', 0) + 1
            r['_phase_accum'] = r.get('_phase_accum', 0) + r['freq']

    def get_sensory(self, region_id):
        r = self.regions[self.region_map[region_id]]
        t = r.get('_tick', 0)
        ph = r.get('_phase_accum', 0)
        sigma = r['sigma']
        rng_val = random.Random(hash((region_id, t)) % (2**31)).gauss(0, 1)
        noise = sigma * rng_val

        rt = r['type']
        base_val = 0
        s = r.get('_state', 0.3)

        if rt == 'sin':
            base_val = math.sin(ph + r['phase'])
        elif rt == 'logistic':
            base_val = 4 * s * (1 - s) if 0 < s < 1 else 0.5
        elif rt == 'lorenz':
            base_val = 0.5 + 0.3 * math.tanh(s)
        elif rt == 'van der Pol':
            base_val = s - 0.1 * (s**3 - s)
        elif rt == 'sawtooth':
            base_val = (ph + r['phase']) % 1.0
        elif rt == 'rossler':
            base_val = 0.5 * math.sin(0.7 * ph + s)
        elif rt == 'henon':
            base_val = 1 - 1.4 * s**2 + 0.3 * s
        elif rt == 'tent':
            base_val = 2 * s if s < 0.5 else 2 * (1 - s)
        elif rt == 'cubic':
            base_val = s**3 - s

        s_new = max(-2.0, min(2.0, base_val + noise))
        r['_state'] = s_new

        return [
            s_new,
            math.sin(s_new * 2.5) if abs(s_new) < 1.5 else 0,
            math.tanh(s_new),
            s_new**2 * (1 if s_new > 0 else -1),
            abs(s_new),
            math.cos(s_new * 1.7),
            sign(s_new) * math.sqrt(abs(s_new)) if abs(s_new) < 2 else 0,
            r['proprio'],
        ]

def sign(x): return 1 if x > 0 else -1 if x < 0 else 0


# ========== World Model ==========

class WorldModel:
    """v17: Context-conditioned WM — proprio 调制 dynamics"""
    def __init__(self, genome):
        self.genome = genome
        n = genome['n_nodes']
        self.nodes = [0.0] * n
        self.prev_sensory = [0.0] * N_CHANNELS

        # context 调制参数
        self.proprio_embed_w = genome.get('proprio_embed_w', [0.0]*3)
        self.decay_mod_w = genome.get('decay_mod_w', [0.0]*3)
        self.predict_mod_specialist = genome.get('predict_mod_specialist', [0.0]*15)
        self.predict_mod_bias = genome.get('predict_mod_bias', [0.0]*24)

    def _compute_context(self, proprio_val):
        """proprio scalar → 3-d modulation vector"""
        e = []
        for i in range(3):
            e.append(math.tanh(proprio_val * self.proprio_embed_w[i]))
        return e

    def _modulated_decay(self, embed):
        """基于 context 的动态 decay"""
        raw = 0.0
        for i in range(3):
            raw += embed[i] * self.decay_mod_w[i]
        # sigmoid → [0.1, 0.95] 区间
        return 0.1 + 0.85 / (1 + math.exp(-raw))

    def _compute_predict_w_eff(self, embed):
        """基于 embed 对 predict_w 做输出端调制。

        设计：embed × predict_mod_specialist → 每节点增益 + 每通道偏置。
        predict_mod_specialist[j][i]: embed dim j 对 node i 的乘性增益
        predict_mod_bias[j][c]: embed dim j 对 output channel c 的加性偏置
        """
        g = self.genome
        n = g['n_nodes']
        pw = g['predict_w']
        predict_w_eff = [[0.0] * N_CHANNELS for _ in range(n)]

        for i in range(n):
            # 乘性增益：embed 经 specialist 矩阵映射到每个 node 的标量
            gain = 1.0
            for j in range(3):
                si = j * n + i
                if si < len(self.predict_mod_specialist):
                    gain += embed[j] * self.predict_mod_specialist[si]

            for c in range(N_CHANNELS):
                # 加性偏置：embed × bias_mod
                bias = 0.0
                for j in range(3):
                    bsi = j * N_CHANNELS + c
                    if bsi < len(self.predict_mod_bias):
                        bias += embed[j] * self.predict_mod_bias[bsi]

                base = pw[i][c] if c < len(pw[i]) else 0.0
                predict_w_eff[i][c] = base * (1.0 + gain * 0.1) + bias

        return predict_w_eff

    def forward(self, sensory, update=True):
        """前向一步：context 调制 decay + predict_w"""
        g = self.genome
        n = g['n_nodes']

        # context 向量
        proprio = sensory[7]  # channel 7
        embed = self._compute_context(proprio)
        decay = self._modulated_decay(embed)

        # 计算 context 调制的 predict_w_eff
        predict_w_eff = self._compute_predict_w_eff(embed)

        # 标准 RNN 更新（decay 调制）
        new_nodes = [0.0] * n
        for i in range(n):
            h = 0.0
            for j in range(N_CHANNELS):
                h += g['sensory_w'][j][i] * sensory[j]
            for k in range(n):
                h += g['recurrent_w'][i][k] * self.nodes[k]
            new_nodes[i] = math.tanh(h)

        # decay 混合：new = (1-decay)*new + decay*old
        if update:
            for i in range(n):
                self.nodes[i] = (1 - decay) * new_nodes[i] + decay * self.nodes[i]

        # 预测（用调制后的 predict_w）
        pred = [0.0] * N_CHANNELS
        for i in range(n):
            for j in range(N_CHANNELS):
                pred[j] += self.nodes[i] * predict_w_eff[i][j]

        self.prev_sensory = sensory[:]
        return pred, self.nodes[:]

    def forward_from_nodes(self, nodes, proprio_val):
        """从给定节点状态 + proprio 向前展开（不更新 self.nodes）"""
        g = self.genome
        n = g['n_nodes']
        decay = self._modulated_decay(self._compute_context(proprio_val))

        embed = self._compute_context(proprio_val)
        predict_w_eff = self._compute_predict_w_eff(embed)

        pred = [0.0] * N_CHANNELS
        for i in range(n):
            for j in range(N_CHANNELS):
                pred[j] += nodes[i] * predict_w_eff[i][j]
        return pred

    def unroll_multi_step(self, sensory_init, proprio_val, n_steps):
        """多步预测（自回归展开）"""
        pred1, nodes = self.forward(sensory_init, update=False)

        preds = [pred1]
        nodes_cur = nodes[:]

        g = self.genome
        n = g['n_nodes']
        embed = self._compute_context(proprio_val)
        decay = self._modulated_decay(embed)
        predict_w_eff = self._compute_predict_w_eff(embed)

        for step in range(1, n_steps):
            # 从当前 nodes 预测（用调制后的 predict_w_eff）
            p = [0.0] * N_CHANNELS
            for i in range(n):
                for j in range(N_CHANNELS):
                    p[j] += nodes_cur[i] * predict_w_eff[i][j]
            preds.append(p)

            # 用预测值作为输入更新 nodes（自回归展开）
            new_nodes = [0.0] * n
            for i in range(n):
                h = 0.0
                for j in range(N_CHANNELS):
                    h += g['sensory_w'][j][i] * p[j]
                for k in range(n):
                    h += g['recurrent_w'][i][k] * nodes_cur[k]
                new_nodes[i] = math.tanh(h)

            # decay 混合
            for i in range(n):
                nodes_cur[i] = (1 - decay) * new_nodes[i] + decay * nodes_cur[i]

        return preds
    
    def update_weights(self, pred, actual):
        """Hebbian 学习：预测误差驱动权重更新"""
        g = self.genome
        n = g['n_nodes']
        lr = g['params']['learning_rate']

        errors = [actual[j] - pred[j] for j in range(N_CHANNELS)]
        mse = sum(e**2 for e in errors) / N_CHANNELS

        # 更新 predict_w
        proprio = actual[7]
        embed = self._compute_context(proprio)
        for i in range(min(n, len(g['predict_w']))):
            for j in range(N_CHANNELS):
                delta = lr * self.nodes[i] * errors[j]
                # 加上 context 调制（让不同 context 学到不同权重）
                g['predict_w'][i][j] += delta

        # 更新 sensory_w (weak, for input grounding)
        for j in range(N_CHANNELS):
            for i in range(min(n, len(g['sensory_w'][j]))):
                delta = lr * 0.1 * actual[j] * self.nodes[i]
                g['sensory_w'][j][i] += delta

        # Clip
        for i in range(n):
            for j in range(N_CHANNELS):
                g['predict_w'][i][j] = max(-3.0, min(3.0, g['predict_w'][i][j]))
                if i < len(g['sensory_w'][j]):
                    g['sensory_w'][j][i] = max(-3.0, min(3.0, g['sensory_w'][j][i]))

        return mse


# ========== Intrinsic Motivation ==========

class IntrinsicMotivation:
    """内在动机：Curiosity / Mastery / Agency / Novelty / Boredom"""
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

        # boredom: fast vs slow error ratio
        if self.ema_error_fast < self.ema_error_slow * 0.7:
            self.boredom += 0.003
            lr_mult = 0.5
        else:
            self.boredom *= 0.97
            lr_mult = 1.0

        self.boredom = max(0.0, min(1.0, self.boredom))
        self.lr_log.append(lr_mult)

        # region value
        info_value = region_sigma * (1.0 / max(0.001, current_mse + 0.01))
        self.region_scores[current_region] = 0.95 * self.region_scores[current_region] + 0.05 * info_value

        # intrinsic energy
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

    def wm_forward_from_nodes(self, nodes, proprio_val):
        return self.wm.forward_from_nodes(nodes, proprio_val)

    def _gen_id(self):
        return hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]

    @staticmethod
    def random_genome():
        n = random.randint(3, MAX_NODES)
        g = {
            'n_nodes': n,
            'sensory_w': [[random.uniform(-0.3, 0.3) for _ in range(n)] for _ in range(N_CHANNELS)],
            'predict_w': [[random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)] for _ in range(n)],
            'recurrent_w': [[random.uniform(-0.1, 0.1) for _ in range(n)] for _ in range(n)],
            'params': {
                'wm_decay': random.uniform(0.5, 0.95),
                'learning_rate': random.uniform(0.05, 0.4),
                'max_lr': random.uniform(0.3, 0.8),
            },
            # v17: context modulation parameters
            'proprio_embed_w': [random.uniform(-0.5, 0.5) for _ in range(3)],
            'decay_mod_w': [random.uniform(-0.2, 0.2) for _ in range(3)],
            'predict_mod_specialist': [random.uniform(-0.2, 0.2) for _ in range(3 * 15)],
            'predict_mod_bias': [random.uniform(-0.1, 0.1) for _ in range(3 * N_CHANNELS)],
        }

        # reasonable init for recurrent: identity-ish
        for i in range(n):
            g['recurrent_w'][i][i] = random.uniform(0.5, 1.5)

        return g

    def clone_genome(self):
        return copy.deepcopy(self.genome)


# ========== Evolution Engine ==========

def genome_distance(g1, g2):
    """基因组距离（用于近交惩罚）"""
    n1, n2 = g1['n_nodes'], g2['n_nodes']
    n = min(n1, n2)
    dist = abs(n1 - n2) * 0.3

    for j in range(N_CHANNELS):
        for i in range(n):
            dist += abs(g1['sensory_w'][j][i] - g2['sensory_w'][j][i])
    for i in range(n):
        for j in range(N_CHANNELS):
            dist += abs(g1['predict_w'][i][j] - g2['predict_w'][i][j])
    norm = (N_CHANNELS * n + N_CHANNELS * n + n)
    return dist / norm if norm > 0 else 1.0


def crossover_genomes(g1, g2, mode='row_mix'):
    """三种杂交模式"""
    n = min(g1['n_nodes'], g2['n_nodes'])
    child_n = random.choice([g1['n_nodes'], g2['n_nodes']])
    child = {
        'n_nodes': child_n,
        'sensory_w': [[0.0]*child_n for _ in range(N_CHANNELS)],
        'predict_w': [[0.0]*N_CHANNELS for _ in range(child_n)],
        'recurrent_w': [[0.0]*child_n for _ in range(child_n)],
        'params': copy.deepcopy(g1['params']),
        'proprio_embed_w': [0.0]*3,
        'decay_mod_w': [0.0]*3,
        'predict_mod_specialist': [0.0]*(3*15),
        'predict_mod_bias': [0.0]*(3*N_CHANNELS),
    }

    if mode == 'row_mix':
        # 逐行混合：按 row 选择父本
        for j in range(N_CHANNELS):
            for i in range(child_n):
                src = g1 if random.random() < 0.5 else g2
                src_n = src['n_nodes']
                src_i = min(i, src_n - 1)
                src_j = min(j, N_CHANNELS - 1)
                child['sensory_w'][j][i] = src['sensory_w'][src_j][src_i] if src_j < len(src['sensory_w']) and src_i < len(src['sensory_w'][src_j]) else 0
        for i in range(child_n):
            src = g1 if random.random() < 0.5 else g2
            src_i = min(i, src['n_nodes'] - 1)
            for j in range(N_CHANNELS):
                child['predict_w'][i][j] = src['predict_w'][src_i][j]
        for i in range(child_n):
            src = g1 if random.random() < 0.5 else g2
            src_i = min(i, src['n_nodes'] - 1)
            for k in range(child_n):
                src_k = min(k, src['n_nodes'] - 1)
                child['recurrent_w'][i][k] = src['recurrent_w'][src_i][src_k]
        # params: 随机选
        if random.random() < 0.5:
            child['params'] = copy.deepcopy(g2['params'])
        # cmap mod: 随机选
        if random.random() < 0.5:
            child['proprio_embed_w'] = g1['proprio_embed_w'][:]
            child['decay_mod_w'] = g1['decay_mod_w'][:]
            child['predict_mod_specialist'] = g1.get('predict_mod_specialist', [0.0]*(3*15))[:]
            child['predict_mod_bias'] = g1.get('predict_mod_bias', [0.0]*(3*N_CHANNELS))[:]
        else:
            child['proprio_embed_w'] = g2['proprio_embed_w'][:]
            child['decay_mod_w'] = g2['decay_mod_w'][:]
            child['predict_mod_specialist'] = g2.get('predict_mod_specialist', [0.0]*(3*15))[:]
            child['predict_mod_bias'] = g2.get('predict_mod_bias', [0.0]*(3*N_CHANNELS))[:]

    elif mode == 'layer_swap':
        # 感官/预测/循环 分别选不同父本
        for j in range(N_CHANNELS):
            for i in range(child_n):
                src = g1 if random.random() < 0.5 else g2
                src_i = min(i, src['n_nodes'] - 1)
                child['sensory_w'][j][i] = src['sensory_w'][j][src_i] if j < len(src['sensory_w']) and src_i < len(src['sensory_w'][j]) else 0
        for i in range(child_n):
            src = g1 if random.random() < 0.5 else g2
            src_i = min(i, src['n_nodes'] - 1)
            for j in range(N_CHANNELS):
                child['predict_w'][i][j] = src['predict_w'][src_i][j]
        for i in range(child_n):
            src = g1 if random.random() < 0.5 else g2
            src_i = min(i, src['n_nodes'] - 1)
            for k in range(child_n):
                src_k = min(k, src['n_nodes'] - 1)
                child['recurrent_w'][i][k] = src['recurrent_w'][src_i][src_k]
        child['params'] = copy.deepcopy(random.choice([g1, g2])['params'])
        src = g1 if random.random() < 0.5 else g2
        child['proprio_embed_w'] = src['proprio_embed_w'][:]
        child['decay_mod_w'] = src['decay_mod_w'][:]
        child['predict_mod_specialist'] = src.get('predict_mod_specialist', [0.0]*(3*15))[:]
        child['predict_mod_bias'] = src.get('predict_mod_bias', [0.0]*(3*N_CHANNELS))[:]

    else:  # uniform
        for j in range(N_CHANNELS):
            for i in range(child_n):
                src = g1 if random.random() < 0.5 else g2
                src_i = min(i, src['n_nodes'] - 1)
                child['sensory_w'][j][i] = src['sensory_w'][j][src_i] if j < len(src['sensory_w']) and src_i < len(src['sensory_w'][j]) else 0
        for i in range(child_n):
            for j in range(N_CHANNELS):
                src = g1 if random.random() < 0.5 else g2
                src_i = min(i, src['n_nodes'] - 1)
                child['predict_w'][i][j] = src['predict_w'][src_i][j]
        for i in range(child_n):
            for k in range(child_n):
                src = g1 if random.random() < 0.5 else g2
                src_i = min(i, src['n_nodes'] - 1)
                src_k = min(k, src['n_nodes'] - 1)
                child['recurrent_w'][i][k] = src['recurrent_w'][src_i][src_k]
        child['params'] = copy.deepcopy(random.choice([g1, g2])['params'])
        src = g1 if random.random() < 0.5 else g2
        child['proprio_embed_w'] = src['proprio_embed_w'][:]
        child['decay_mod_w'] = src['decay_mod_w'][:]
        child['predict_mod_specialist'] = src.get('predict_mod_specialist', [0.0]*(3*15))[:]
        child['predict_mod_bias'] = src.get('predict_mod_bias', [0.0]*(3*N_CHANNELS))[:]

    return child


def select_complementary_pair(survivors, sf):
    """选行为差异大的双亲"""
    idx1 = random.choices(range(len(survivors)), weights=sf, k=1)[0]
    p1 = survivors[idx1]
    # 选与 p1 距离最大的
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
    """高斯突变"""
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

    # v17: context modulation 突变
    for i in range(len(g['proprio_embed_w'])):
        if random.random() < rate:
            g['proprio_embed_w'][i] += random.gauss(0, 0.1)
            g['proprio_embed_w'][i] = max(-3.0, min(3.0, g['proprio_embed_w'][i]))
    for i in range(len(g['decay_mod_w'])):
        if random.random() < rate:
            g['decay_mod_w'][i] += random.gauss(0, 0.05)
            g['decay_mod_w'][i] = max(-2.0, min(2.0, g['decay_mod_w'][i]))
    for i in range(len(g.get('predict_mod_specialist', []))):
        if random.random() < rate:
            g['predict_mod_specialist'][i] += random.gauss(0, 0.05)
            g['predict_mod_specialist'][i] = max(-3.0, min(3.0, g['predict_mod_specialist'][i]))
    for i in range(len(g.get('predict_mod_bias', []))):
        if random.random() < rate:
            g['predict_mod_bias'][i] += random.gauss(0, 0.02)
            g['predict_mod_bias'][i] = max(-2.0, min(2.0, g['predict_mod_bias'][i]))

    # 偶尔加节点
    if random.random() < 0.05 and n < MAX_NODES:
        new_i = n
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
        self._max_fit = 0.0

    def compute_fitness(self, org, ticks=300):
        """多区域评估 fitness"""
        self.pool.reset_states()
        org.wm.nodes = [0.0] * org.genome['n_nodes']
        org.wm.prev_sensory = [0.0] * N_CHANNELS
        org.im = IntrinsicMotivation()

        total_mse = 0.0
        total_info = 0.0
        region_ticks = collections.Counter()
        region_switch_count = 0
        current_region = 0

        tick_mse_hist = []

        for t in range(ticks):
            # 第一步：环境更新
            sensory = self.pool.get_sensory(current_region)

            # 第二步：WM 预测
            pred, nodes = org.wm.forward(sensory, update=True)
            mse = sum((sensory[j] - pred[j])**2 for j in range(N_CHANNELS)) / N_CHANNELS

            # 第三步：学习
            lr_mult = org.im.update(mse, self.pool.regions[current_region]['sigma'],
                                    current_region, current_region)
            adjusted_lr = org.genome['params']['learning_rate'] * lr_mult
            org.wm.update_weights(pred, sensory)

            tick_mse_hist.append(mse)
            total_mse += mse
            region_ticks[current_region] += 1

            # 第四步：环境步进
            self.pool.step_global()

            # 第五步：行动选择（bias 动作 + 停留）
            bias = org.im.get_action_bias()
            if bias and random.random() < 0.3:
                action = max(bias, key=bias.get)
            else:
                action = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]

            # 行动: 0=stay, 1=move to random, 2=move to high sigma
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

            # 信息计数
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
        """主进化循环"""
        for gen in range(self.n_gen):
            self.generation = gen
            gen_results = []

            for org in self.population:
                result = self.compute_fitness(org, ticks=300)
                org._last_result = result
                gen_results.append(result)

            # sort by fitness
            sorted_pairs = sorted(zip(self.population, gen_results),
                                  key=lambda x: x[1]['fitness'], reverse=True)

            avg_fit = sum(r['fitness'] for r in gen_results) / len(gen_results)
            avg_mse = sum(r['avg_mse'] for r in gen_results) / len(gen_results)
            avg_info = sum(r['total_info'] for r in gen_results) / len(gen_results)
            avg_boredom = sum(org.im.boredom for org in self.population) / len(self.population)
            diversity = sum(r['diversity'] for r in gen_results) / len(gen_results)
            best_fit = sorted_pairs[0][1]['fitness']
            best_mse = sorted_pairs[0][1]['avg_mse']

            # hybrid vigor analysis
            hybrid_deltas = [getattr(org, '_parent_max_fit', 0) for org in self.population if getattr(org, '_parent_max_fit', 0) > 0]
            avg_delta = 0
            if hybrid_deltas:
                parent_fits = hybrid_deltas
                child_fits = [org._last_result['fitness'] for org in self.population
                              if getattr(org, '_parent_max_fit', 0) > 0]
                if parent_fits and child_fits:
                    delta = (sum(child_fits) / len(child_fits)) - (sum(parent_fits) / len(parent_fits))
                    avg_delta = delta
                    self._hybrid_max_delta.append(delta)

            # 杂交多样性：每个个体行为 trace 的汉明距离均值
            div_mean = 0
            traces = [org.im.region_history[-30:] for org in self.population if org.im.region_history]
            if len(traces) >= 2:
                dists = []
                for i in range(len(traces)):
                    for j in range(i + 1, len(traces)):
                        if traces[i] and traces[j]:
                            d = sum(1 for a, b in zip(traces[i], traces[j]) if a != b) / max(len(traces[i]), len(traces[j]))
                            dists.append(d)
                if dists:
                    div_mean = sum(dists) / len(dists)

            # region diversity
            all_regions = set()
            for org in self.population:
                for r in org.im.region_history:
                    all_regions.add(r)
            region_diversity = len(all_regions)

            # intrinsic drive (avg)
            avg_drive = sum(org.im.drive_energy for org in self.population) / len(self.population)
            total_switches = sum(r['region_switches'] for r in gen_results)

            # track max fit
            if best_fit > self._max_fit:
                self._max_fit = best_fit

            # log every 10 gen
            if gen % 10 == 0 or gen == self.n_gen - 1:
                print(f"Gen {gen:4d} | fit={best_fit:.4f}/{avg_fit:.4f} | mse={best_mse:.4f}/{avg_mse:.4f} | "
                      f"info={avg_info:.0f} | drive={avg_drive:.3f} | bored={avg_boredom:.4f} | "
                      f"div={div_mean:.3f} | r_div={region_diversity} | switches={total_switches} | "
                      f"hybrid_delta={avg_delta:+.4f}")

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

            # termination check
            if best_fit >= 0.99:
                print(f"Converged at gen {gen}")
                break

            # reproduction
            survivors = [p[0] for p in sorted_pairs[:max(3, self.n // 3)]]
            survivor_fits = [p[1]['fitness'] for p in sorted_pairs[:max(3, self.n // 3)]]
            # normalize fitness as selection weights
            sf = [max(0.01, f) for f in survivor_fits]

            # fitness-proportional selection weights
            fit_shifted = [f - min(sf) + 0.01 for f in sf]
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
                    # 杂交后代
                    p1, p2 = select_complementary_pair(survivors, sf)
                    mode = random.choices(['row_mix', 'layer_swap', 'uniform'], weights=[0.5, 0.2, 0.3])[0]
                    child_g = crossover_genomes(p1.clone_genome(), p2.clone_genome(), mode=mode)
                    child_g = mutate_genome(child_g)

                    p1_fit = p1._last_result['fitness']
                    p2_fit = p2._last_result['fitness']

                    child = Organism(genome=child_g, lineage_id=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8])
                    child.parents = [p1.lineage_id, p2.lineage_id]
                    child._inbreeding = 1.0
                    child._parent_max_fit = max(p1_fit, p2_fit)
                    hybrid_count += 1
                    new_pop.append(child)

                elif mutant_count < n_mutant:
                    # 突变克隆
                    parent = random.choices(survivors, weights=sf, k=1)[0]
                    child_g = mutate_genome(parent.clone_genome())
                    child = Organism(genome=child_g, lineage_id=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8])
                    child.parents = [parent.lineage_id]
                    child._inbreeding = 1.0
                    mutant_count += 1
                    new_pop.append(child)

            self.population = new_pop

        # === Final evaluation pass on the last population (no reproduction) ===
        final_results = []
        for org in self.population:
            result = self.compute_fitness(org, ticks=300)
            org._last_result = result
            final_results.append(result)

        return self.population, self.history

    def get_best(self):
        best = max(self.population, key=lambda o: getattr(o, '_last_result', {}).get('fitness', 0))
        return best


# ========== Test Harness ==========

def regional_precision_test(org, n_ticks=100, n_trials=5):
    """逐区域精度测试（v16 发现的核心问题）"""
    results = {}
    for region_id in range(N_REGIONS):
        mses = []
        for trial in range(n_trials):
            pool = StableRegionPool(seed=42 + trial)
            pool.reset_states()
            org.wm.nodes = [0.0] * org.genome['n_nodes']

            for t in range(n_ticks):
                s = pool.get_sensory(region_id)
                pred, _ = org.wm.forward(s, update=False)
                mse = sum((s[j] - pred[j])**2 for j in range(N_CHANNELS)) / N_CHANNELS
                mses.append(mse)

                pool.step_global()

        avg_mse = sum(mses) / len(mses)
        results[region_id] = round(avg_mse, 4)
    return results


def multi_step_test(org, proprio_val=0.5, steps=[1, 3, 5, 10, 20], n_trials=10):
    """多步预测精度测试（固定区域，同 pool 全流程）"""
    results = {}
    for n_step in steps:
        mses = []
        for trial in range(n_trials):
            region_id = trial % N_REGIONS
            pool = StableRegionPool(seed=42 + trial)
            pool.reset_states()
            org.wm.nodes = [0.0] * org.genome['n_nodes']

            # warmup (fixed region, update nodes to converge)
            for _ in range(50):
                s = pool.get_sensory(region_id)
                org.wm.forward(s, update=True)
                pool.step_global()

            # capture initial sensory + unroll (no update)
            s = pool.get_sensory(region_id)
            preds = org.wm.unroll_multi_step(s, pool.regions[region_id]['proprio'], n_step)

            # actual future (same region, same pool)
            for _ in range(n_step):
                pool.step_global()
            actual = pool.get_sensory(region_id)

            predicted_at_n = preds[-1]
            mse = sum((actual[j] - predicted_at_n[j])**2 for j in range(N_CHANNELS)) / N_CHANNELS
            mses.append(mse)

        results[n_step] = round(sum(mses) / len(mses), 5)
    return results


# ========== Main ==========

def run_experiment():
    print("=" * 70)
    print("yinyang_evo_v17 — Context-Conditioned World Model")
    print("=" * 70)

    engine = EvolutionaryEngine(n_organisms=30, n_generations=200, seed=7)
    pop, history = engine.evolve()

    best = engine.get_best()
    best_result = getattr(best, '_last_result', {})

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Find the best organism's result
    best_fit = 0
    best_org = None
    for org in pop:
        r = getattr(org, '_last_result', {})
        if r.get('fitness', 0) > best_fit:
            best_fit = r['fitness']
            best_org = org

    if best_org:
        print(f"Best fitness: {best_fit:.4f}")
        print(f"Best avg_mse: {getattr(best_org, '_last_result', {}).get('avg_mse', 'N/A')}")
        print(f"Best diversity: {getattr(best_org, '_last_result', {}).get('diversity', 'N/A')}")
        print(f"Genome n_nodes: {best_org.genome['n_nodes']}")
        print(f"wm_decay: {best_org.genome['params']['wm_decay']:.3f}")
        print(f"learning_rate: {best_org.genome['params']['learning_rate']:.4f}")
        print(f"proprio_embed_w: {[f'{x:.3f}' for x in best_org.genome['proprio_embed_w']]}")
        print(f"decay_mod_w: {[f'{x:.3f}' for x in best_org.genome['decay_mod_w']]}")

        # Test 1: Regional precision test (per-region MSE)
        print("\n--- Test A: Regional Precision (per-region, 1-step) ---")
        regional = regional_precision_test(best_org, n_ticks=80)
        for rid, mse in sorted(regional.items()):
            bar = '#' * int(mse * 15)
            print(f"  Region {rid} (σ={engine.pool.regions[rid]['sigma']:.2f}): MSE={mse:.4f} |{bar}")

        avg_regional = sum(regional.values()) / len(regional)
        min_regional = min(regional.values())
        max_regional = max(regional.values())
        print(f"  AVG={avg_regional:.4f}  MIN={min_regional:.4f}  MAX={max_regional:.4f}  SPREAD={max_regional-min_regional:.4f}")

        # Test 2: Multi-step prediction (autoregressive unroll)
        print("\n--- Test B: Multi-step Prediction (autoregressive) ---")
        ms_results = multi_step_test(best_org, proprio_val=0.5, steps=[1, 3, 5, 10, 20], n_trials=10)
        for n_step, mse in sorted(ms_results.items()):
            bar = '#' * int(mse * 8)
            print(f"  N={n_step:3d}: MSE={mse:.4f} |{bar}")

        # Test 3: Context modulation check — does proprio actually modulate predictions?
        print("\n--- Test C: Context Modulation (proprio → prediction shift) ---")
        for r_idx in [0, 4, 8]:
            pool = StableRegionPool(seed=42)
            pool.reset_states()
            best_org.wm.nodes = [0.0] * best_org.genome['n_nodes']
            # run 50 ticks in this region
            for t in range(50):
                s = pool.get_sensory(r_idx)
                pred, _ = best_org.wm.forward(s, update=False)
                pool.step_global()
            print(f"  Region {r_idx} (σ={engine.pool.regions[r_idx]['sigma']:.2f}): pred_last={pred[0]:+.3f}")

    print()
    print("Evolution history (last 10):")
    for h in history[-10:]:
        print(f"  Gen {h['gen']:3d}: fit={h['best_fitness']:.4f} mse={h['best_mse']:.4f} "
              f"div={h['region_diversity']} switches={h['total_switches']}")

    return engine, best_org


if __name__ == '__main__':
    run_experiment()
