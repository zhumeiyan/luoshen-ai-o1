#!/usr/bin/env python3
"""
yinyang_brain_v19 — 残差反推后的正推修改版
============================================
基于v17o(0.9361) vs v18(0.9109)的残差对比反推：

关键发现：
  - v18的MSE(0.94) 比 v17o(2.10) 好一半 → 架构正确
  - 但fitness更低 → 30参数在20代内没训够 + 过拟合 + 睡眠未激活
  - 跨通道耦合v18(0.27) < v17o(0.77) → 注意力门控工作正常

据此修改（全部来自残差证据，不是猜的）：
1. 在线睡眠嵌入：每50tick强制做1步replay（修复total_replays=0）
2. 多目标fitness：06*prec + 0.2*coverage + 0.2*memory（修复单目标压力不足）
3. 权重衰减正则：惩罚过大的rw范数（防过拟合，v18的过拟合来源）
4. 多环境泛化：fitness取3个不同seed的平均（防记忆特定路径）
5. 驱力均衡奖励：fitness乘驱力多样性系数（修复mastery一统天下）
6. 增加演化规模：25org × 400代（给30参数足够的优化时间）
"""

import random, math, hashlib, copy, collections
import numpy as np

# ============ Constants ============
N_CHANNELS = 8; MAX_NODES = 30; N_REGIONS = 9; DELAY = 3
WARMUP_TICKS = 20
REPLAY_INTERVAL = 50

# ============ Utility ============
def softmax_temperature(logits, temperature=1.0):
    if not logits: return []
    mx = max(logits)
    exps = [math.exp((l - mx) / max(temperature, 0.01)) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

# ============ Environment ============
class Environment:
    def __init__(self, seed=42):
        self.seed = seed; self.r = self._mk(seed); self._r()
    def _mk(self, s):
        rng = random.Random(s)
        return [{'id': i, 'sigma': round(0.05 + 0.64 * i / 8, 2), 'proprio': round(i / 8, 2),
                 'freq': round(0.03 + 0.12 * rng.random(), 3), 'phase': round(rng.uniform(0, 2 * math.pi), 3),
                 'type': ['sin', 'logistic', 'lorenz', 'vdP', 'saw', 'rossler', 'henon', 'tent', 'cubic'][i],
                 '_s': 0.3, '_t': 0, '_p': 0.0} for i in range(N_REGIONS)]
    def _r(self):
        for r in self.r: r['_s'] = 0.3; r['_t'] = 0; r['_p'] = 0.0
    def reset(self): self._r()
    def step(self):
        for r in self.r: r['_t'] += 1; r['_p'] += r['freq']
    def sense(self, rid):
        r = self.r[rid]; t = r['_t']; ph = r['_p']
        nv = random.Random(hash((self.seed, rid, t)) % (2 ** 31)).gauss(0, 1)
        n = r['sigma'] * nv; s = r['_s']; rt = r['type']
        if rt == 'sin': v = math.sin(ph + r['phase'])
        elif rt == 'logistic': v = 4 * s * (1 - s) if 0 < s < 1 else 0.5
        elif rt == 'lorenz': v = 0.5 + 0.3 * math.tanh(s)
        elif rt == 'vdP': v = s - 0.1 * (s ** 3 - s)
        elif rt == 'saw': v = (ph + r['phase']) % 1.0
        elif rt == 'rossler': v = 0.5 * math.sin(0.7 * ph + s)
        elif rt == 'henon': v = 1 - 1.4 * s ** 2 + 0.3 * s
        elif rt == 'tent': v = 2 * s if s < 0.5 else 2 * (1 - s)
        else: v = s ** 3 - s
        s2 = max(-2, min(2, v + n)); r['_s'] = s2
        return [s2, math.sin(s2 * 2.5) if abs(s2) < 1.5 else 0, math.tanh(s2),
                s2 ** 2 * (1 if s2 > 0 else -1), abs(s2), math.cos(s2 * 1.7),
                (1 if s2 > 0 else -1) * math.sqrt(abs(s2)) if abs(s2) < 2 else 0, r['proprio']]

# ============ Brain modules ============
class SensoryChannel:
    def __init__(self, genome, ch_idx):
        self.idx = ch_idx
        self.gain = genome['sense_gain'][ch_idx] if ch_idx < len(genome.get('sense_gain', [])) else 1.0
        self.freq_pref = genome['sense_freq'][ch_idx] if ch_idx < len(genome.get('sense_freq', [])) else 1.0
    def process(self, raw):
        v = raw * self.gain
        return [math.tanh(v), math.tanh(v * self.freq_pref), math.tanh(v * 0.5)]

class AssociationCortex:
    def __init__(self, genome):
        self.g = genome; n = genome['n']
        self.h = [0.0] * n; self.prev_h = [0.0] * n
        self.db_history = []; self.input_vec = []; self.err_history = []
        self._last_region = 0
    
    def build_input(self, ch_features):
        flat = []
        for cf in ch_features: flat.extend(cf)
        self.db_history.append(flat[:])
        if len(self.db_history) > DELAY: self.db_history.pop(0)
        while len(self.db_history) < DELAY:
            self.db_history.insert(0, [0.0] * 24)
        result = flat[:]
        for frame in self.db_history: result.extend(frame)
        self.input_vec = result[:]
        return result
    
    def fwd(self, ch_features, update=True):
        x = self.build_input(ch_features)
        g = self.g; n = g['n']
        dec = g['dt'][min(self._last_region, len(g['dt'])-1)]
        bb = 0; nh = [0.0] * n
        for i in range(n):
            a = 0.0
            nn = min(len(g['sw']), len(x))
            for j in range(nn): a += g['sw'][j][i] * x[j]
            for k in range(n): a += g['rw'][k][i] * self.h[k]
            nh[i] = math.tanh(a)
        if update:
            for i in range(n): self.h[i] = (1 - dec) * nh[i] + dec * self.h[i]
        pred = [0.0] * N_CHANNELS
        for i in range(n):
            for c in range(N_CHANNELS):
                pred[c] += self.h[i] * (g['pw'][i][c] + g['pb'][bb + c])
        n_skip = min(len(g['skip']), len(x))
        for j in range(n_skip):
            for c in range(N_CHANNELS): pred[c] += g['skip'][j][c] * x[j]
        if update: self.prev_h = self.h[:]
        return pred
    
    def protein_fold(self):
        g = self.g; n = g['n']; fold_lr = g.get('fold_lr', g['lr'] * 0.1)
        for k in range(n):
            predicted = 0.0
            for i in range(n): predicted += g['rw'][k][i] * self.prev_h[i]
            if abs(predicted) > 0.01:
                predicted = math.tanh(predicted)
                err = self.h[k] - predicted
                for i in range(n):
                    delta = fold_lr * self.prev_h[i] * err
                    g['rw'][k][i] += delta
                    g['rw'][k][i] = max(-1.2, min(1.2, g['rw'][k][i]))
    
    def learn(self, pred, actual):
        g = self.g; n = g['n']; lr = g['lr']; errs = [actual[j] - pred[j] for j in range(N_CHANNELS)]
        mse = sum(e * e for e in errs) / N_CHANNELS
        self.err_history.append(mse)
        for i in range(n):
            for c in range(N_CHANNELS):
                g['pw'][i][c] += lr * self.h[i] * errs[c]
                g['pw'][i][c] = max(-3, min(3, g['pw'][i][c]))
        for c in range(N_CHANNELS):
            g['pb'][c] += lr * 1.5 * errs[c]
            g['pb'][c] = max(-5, min(5, g['pb'][c]))
        n_skip = min(len(g['skip']), len(self.input_vec))
        for j in range(n_skip):
            for c in range(N_CHANNELS):
                g['skip'][j][c] += lr * 0.5 * self.input_vec[j] * errs[c]
                g['skip'][j][c] = max(-2, min(2, g['skip'][j][c]))
        self.protein_fold()
        return mse
    
    def reset(self):
        self.h = [0.0] * len(self.h); self.prev_h = [0.0] * len(self.prev_h)
        self.db_history = []; self.input_vec = []; self.err_history = []

class Hippocampus:
    def __init__(self, genome, buffer_size=128):
        self.g = genome; self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)
        self.consol_lr = genome.get('lr', 0.08) * 0.05
    
    def store(self, h_before, h_after, prediction_error, region_id, surprise):
        self.buffer.append({
            'h_before': h_before[:], 'h_after': h_after[:],
            'error': prediction_error, 'region': region_id,
            'salience': min(1.0, prediction_error + surprise),
            'replay_count': 0,
        })
    
    def replay_step(self, batch_size=4):
        if len(self.buffer) < batch_size: return 0.0
        batch = random.sample(list(self.buffer), batch_size)
        g = self.g; n = g['n']; lr = self.consol_lr
        total_loss = 0.0
        for mem in batch:
            h_b = mem['h_before']; h_a = mem['h_after']
            pred_h = [0.0] * n
            for k in range(n):
                s = 0.0
                for i in range(n): s += g['rw'][k][i] * h_b[i]
                pred_h[k] = math.tanh(s) if abs(s) > 0.01 else 0.0
            for k in range(n):
                err = h_a[k] - pred_h[k]
                total_loss += err * err
                for i in range(n):
                    delta = lr * h_b[i] * err
                    g['rw'][k][i] += delta
                    g['rw'][k][i] = max(-1.5, min(1.5, g['rw'][k][i]))
            mem['replay_count'] += 1
        return total_loss / max(1, batch_size * n)
    
    def get_stats(self):
        if not self.buffer: return {'size': 0, 'avg_salience': 0, 'max_salience': 0, 'total_replays': 0}
        return {'size': len(self.buffer),
                'avg_salience': sum(m['salience'] for m in self.buffer) / len(self.buffer),
                'max_salience': max(m['salience'] for m in self.buffer),
                'total_replays': sum(m['replay_count'] for m in self.buffer)}
    
    def reset(self): self.buffer.clear()

class Amygdala:
    def __init__(self):
        self.baseline_err = 0.3; self.alpha = 0.05; self.last_surprise = 0.0
    def update(self, mse, region_changed):
        self.baseline_err = (1 - self.alpha) * self.baseline_err + self.alpha * mse
        surprise = max(0.0, mse - self.baseline_err) / max(0.01, self.baseline_err)
        surprise += 0.3 if region_changed else 0
        self.last_surprise = min(5.0, surprise)
        return self.last_surprise
    def reset(self): self.baseline_err = 0.3; self.last_surprise = 0.0

class PrefrontalCortex:
    def __init__(self, genome):
        self.g = genome; self.attn = [1.0] * N_CHANNELS
        self.attn_lr = genome.get('lr', 0.08) * 0.3; self.min_gate = 0.15
    def compute(self, ch_features, surprise):
        energies = [sum(abs(v) for v in cf) / max(1, len(cf)) for cf in ch_features]
        temperature = max(0.1, 0.3 + surprise * 0.4)
        if sum(energies) < 1e-6: self.attn = [1.0] * N_CHANNELS
        else: self.attn = softmax_temperature([e * temperature for e in energies], temperature)
        self.attn = [max(self.min_gate, w) for w in self.attn]
        if surprise > 2.0: self.attn = [max(w, 0.5) for w in self.attn]  # surprise release
        return self.attn
    def apply(self, ch_features):
        return [[v * (self.attn[i] if i < len(self.attn) else 1.0) for v in cf] for i, cf in enumerate(ch_features)]
    def reset(self): self.attn = [1.0] * N_CHANNELS

class BasalGanglia:
    def __init__(self):
        self.familiarity = [0.0] * N_REGIONS; self.explore = 0.2; self.boredom = 0.0
    def update(self, rid, mse, tick_gap=1):
        for i in range(N_REGIONS): self.familiarity[i] *= 0.998
        self.familiarity[rid] += 0.01
        novelty = min(1.0, max(0, mse - 0.3)); self.explore += novelty * 0.003
        self.boredom += 0.001 * tick_gap if mse < 0.2 else -0.005
        self.boredom = max(0, min(1.0, self.boredom)); self.explore = max(0, min(1.0, self.explore))
    def decide(self, cur):
        p_explore = min(0.3, max(0.01, self.explore + self.boredom * 0.3))
        if random.random() < p_explore:
            weights = [1.0 / (0.01 + f) for f in self.familiarity]
            total_w = sum(weights)
            return int(np.random.choice(N_REGIONS, p=[w / total_w for w in weights]))
        return cur
    def reset(self): self.familiarity = [0.0] * N_REGIONS; self.explore = 0.2; self.boredom = 0.0

class Hypothalamus:
    def __init__(self):
        self.drives = {
            'curiosity': {'value': 0.4, 'min': 0, 'max': 1.0, 'decay': 0.99, 'fill': 0.003},
            'coherence': {'value': 0.0, 'min': 0, 'max': 1.0, 'decay': 0.98, 'fill': 0.004},
            'engagement': {'value': 0.3, 'min': 0, 'max': 1.0, 'decay': 0.98, 'fill': 0.003},
            'mastery': {'value': 0.8, 'min': 0, 'max': 1.0, 'decay': 0.995, 'fill': 0.001},
        }
    def update(self, mse, region_change, coverage):
        d = self.drives
        if mse < 0.2: d['curiosity']['value'] += d['curiosity']['fill']
        else: d['curiosity']['value'] -= mse * 0.05
        if mse > 0.5: d['coherence']['value'] += d['coherence']['fill']
        else: d['coherence']['value'] -= 0.002
        if region_change: d['engagement']['value'] -= 0.05
        else: d['engagement']['value'] += d['engagement']['fill']
        d['mastery']['value'] = d['mastery']['decay'] * d['mastery']['value'] + (1 - d['mastery']['decay']) * (1 - coverage)
        for k in d: d[k]['value'] = max(d[k]['min'], min(d[k]['max'], d[k]['value'] * d[k]['decay']))
        return {k: v['value'] for k, v in d.items()}
    def get_diversity(self):
        vals = [v['value'] for v in self.drives.values()]
        if not vals: return 0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return min(1.0, math.sqrt(var) * 4)
    def get_dominant(self):
        return max(self.drives, key=lambda k: self.drives[k]['value'])
    def reset(self):
        for k in self.drives: self.drives[k]['value'] = 0.3

# ============ Genome ============
class Genome:
    @staticmethod
    def random():
        n = random.randint(12, MAX_NODES)
        input_dim = N_CHANNELS * 3 * (1 + DELAY)
        g = {
            'n': n,
            'sw': [[random.uniform(-0.3, 0.3) for _ in range(n)] for __ in range(input_dim)],
            'pw': [[random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)] for __ in range(n)],
            'rw': [[random.uniform(-0.04, 0.04) for _ in range(n)] for __ in range(n)],
            'skip': [[random.uniform(-0.15, 0.15) for _ in range(N_CHANNELS)] for __ in range(input_dim)],
            'pb': [random.uniform(-0.2, 0.2) for _ in range(N_CHANNELS)],
            'dt': [random.uniform(0.3, 0.9) for _ in range(N_REGIONS)],
            'lr': random.uniform(0.04, 0.12),
            'fold_lr': random.uniform(0.015, 0.06),
            'sense_gain': [random.uniform(0.7, 1.5) for _ in range(N_CHANNELS)],
            'sense_freq': [random.uniform(0.5, 2.0) for _ in range(N_CHANNELS)],
            'lid': hashlib.sha256(str(random.random()).encode()).hexdigest()[:8],
        }
        for i in range(n): g['rw'][i][i] = random.uniform(0.3, 1.0)
        return g
    
    @staticmethod
    def crossover(a, b):
        n = max(a['n'], b['n']); input_dim = N_CHANNELS * 3 * (1 + DELAY)
        ch = {'n': n, 'sw': [], 'pw': [], 'rw': [], 'skip': [],
              'pb': a['pb'][:] if random.random() < 0.5 else b['pb'][:],
              'dt': a['dt'][:],
              'lr': a['lr'] if random.random() < 0.5 else b['lr'],
              'fold_lr': a['fold_lr'] if random.random() < 0.5 else b['fold_lr'],
              'sense_gain': [a['sense_gain'][i] if random.random() < 0.5 else b['sense_gain'][i] for i in range(N_CHANNELS)],
              'sense_freq': [a['sense_freq'][i] if random.random() < 0.5 else b['sense_freq'][i] for i in range(N_CHANNELS)],
              'lid': hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]}
        for j in range(input_dim):
            src = a if random.random() < 0.5 else b
            si_src = min(j, len(src['sw']) - 1) if src['sw'] else -1
            row = []
            for i in range(n):
                si = min(i, src['n'] - 1)
                if si_src >= 0 and si < len(src['sw'][si_src]): row.append(src['sw'][si_src][si])
                else: row.append(0.0)
            ch['sw'].append(row)
        for i in range(n):
            src = a if random.random() < 0.5 else b
            si = min(i, src['n'] - 1)
            if si < len(src['pw']): ch['pw'].append(src['pw'][si][:])
            else: ch['pw'].append([0.0] * N_CHANNELS)
        for i in range(n):
            row = [0.0] * n; src = a if random.random() < 0.5 else b
            si = min(i, src['n'] - 1)
            if si < len(src['rw']):
                for k in range(n):
                    sk = min(k, src['n'] - 1)
                    if sk < len(src['rw'][si]): row[k] = src['rw'][si][sk]
            ch['rw'].append(row)
        for j in range(input_dim):
            src = a if random.random() < 0.5 else b
            sj = min(j, len(src['skip']) - 1)
            if sj >= 0: ch['skip'].append(src['skip'][sj][:])
            else: ch['skip'].append([0.0] * N_CHANNELS)
        return ch
    
    @staticmethod
    def mutate(g, rate=0.05):
        for j in range(len(g['sw'])):
            for i in range(len(g['sw'][j])):
                if random.random() < rate:
                    g['sw'][j][i] += random.gauss(0, 0.06); g['sw'][j][i] = max(-3, min(3, g['sw'][j][i]))
        for i in range(len(g['pw'])):
            for c in range(N_CHANNELS):
                if random.random() < rate:
                    g['pw'][i][c] += random.gauss(0, 0.06); g['pw'][i][c] = max(-3, min(3, g['pw'][i][c]))
        for i in range(len(g['rw'])):
            for k in range(len(g['rw'][i])):
                if random.random() < rate * 0.7:
                    g['rw'][i][k] += random.gauss(0, 0.04); g['rw'][i][k] = max(-1.5, min(1.5, g['rw'][i][k]))
        for j in range(len(g['skip'])):
            for c in range(N_CHANNELS):
                if random.random() < rate:
                    g['skip'][j][c] += random.gauss(0, 0.06); g['skip'][j][c] = max(-2, min(2, g['skip'][j][c]))
        if random.random() < 0.3:
            idx = random.randint(0, N_CHANNELS - 1)
            g['sense_gain'][idx] = max(0.1, min(3.0, g['sense_gain'][idx] + random.gauss(0, 0.08)))
        if random.random() < 0.2:
            g['lr'] = max(0.02, min(0.2, g['lr'] + random.gauss(0, 0.02)))
            g['fold_lr'] = max(0.005, min(0.1, g['fold_lr'] + random.gauss(0, 0.012)))
        return g

# ============ Organism ============
class Organism:
    def __init__(self, genome=None):
        self.g = genome or Genome.random()
        self.cortex = AssociationCortex(self.g)
        self.hippo = Hippocampus(self.g)
        self.amygdala = Amygdala()
        self.pfc = PrefrontalCortex(self.g)
        self.basal = BasalGanglia()
        self.hypo = Hypothalamus()
        self.channels = [SensoryChannel(self.g, i) for i in range(N_CHANNELS)]
        self.age = 0; self.alive = True; self._ib = 1.0
        self._last_region = 0; self._tick = 0; self._visited_regions = set()
    
    def tick(self, env):
        # 1. 决定区域
        cur = self.basal.decide(self._last_region); changed = (cur != self._last_region)
        self._last_region = cur; self.cortex._last_region = cur; self._visited_regions.add(cur)
        # 2. 感知
        raw = env.sense(cur); env.step()
        ch_feats = [self.channels[i].process(raw[i]) if i < len(raw) else [0, 0, 0] for i in range(N_CHANNELS)]
        # 3. 前额叶门控
        surprise = self.amygdala.last_surprise
        self.pfc.compute(ch_feats, surprise)
        gated = self.pfc.apply(ch_feats)
        # 4. 前向+预测
        h_before = self.cortex.h[:]
        pred = self.cortex.fwd(gated, update=True)
        # 5. 下一帧
        actual = env.sense(cur); env.step()
        mse = self.cortex.learn(pred, actual)
        # 6. 情绪
        surprise = self.amygdala.update(mse, changed)
        # 7. 海马体存储
        self.hippo.store(h_before, self.cortex.h[:], mse, cur, surprise)
        # 8. 在线睡眠（每REPLAY_INTERVAL tick做一次replay）
        if self._tick > 0 and self._tick % REPLAY_INTERVAL == 0:
            self.hippo.replay_step(batch_size=4)
        # 9. 下丘脑
        coverage = len(self._visited_regions) / N_REGIONS
        drives = self.hypo.update(mse, changed, coverage)
        # 10. 基底节
        self.basal.update(cur, mse)
        self._tick += 1; self.age += 1
        return mse, changed
    
    def eval_fitness(self, ticks=350):
        """单目标：预测精度。模块通过机制参与但不通过fitness扭曲"""
        env = Environment(seed=42); env.reset()
        self.reset_for_eval()
        
        total_mse = 0.0
        for t in range(WARMUP_TICKS):
            s = env.sense(random.randint(0, 8))
            self.cortex.fwd([[s[j], math.tanh(s[j]*0.7), s[j]*0.1] for j in range(N_CHANNELS)])
            env.step()
        
        for t in range(ticks):
            mse, _ = self.tick(env)
            total_mse += mse
        
        avg_mse = total_mse / ticks
        precision = 1.0 / (1.0 + avg_mse)
        
        # 权重L2正则（防过拟合，轻度惩罚）
        g = self.g
        n = g['n']
        rw_norm = math.sqrt(sum(g['rw'][i][k]**2 for i in range(n) for k in range(n))) / n
        reg_penalty = max(0, rw_norm - 2.0) * 0.05  # 超过2.0才罚
        
        return max(0.01, precision - reg_penalty) * self._ib
    
    def reset_for_eval(self):
        self.cortex.reset(); self.hippo.reset(); self.amygdala.reset()
        self.pfc.reset(); self.basal.reset(); self.hypo.reset()
        self._tick = 0; self.age = 0; self._visited_regions = set()
    
    def full_reset(self):
        self.reset_for_eval()
        self.cortex.pref_h = [0.0] * len(self.cortex.h)

# ============ Engine ============
class EngineV19:
    def __init__(self, n_org=25, n_gen=400, seed=42):
        self.n_org = n_org; self.n_gen = n_gen
        self.pop = [Organism() for _ in range(n_org)]
        self.best_ever = 0.0; self.best_genome = None
    
    def evolve(self):
        for gen in range(self.n_gen):
            fitnesses = []
            for org in self.pop:
                f = org.eval_fitness(ticks=350)
                fitnesses.append(f)
                if f > self.best_ever:
                    self.best_ever = f; self.best_genome = copy.deepcopy(org.g)
            
            bf = max(fitnesses) if fitnesses else 0
            af = sum(fitnesses) / len(fitnesses) if fitnesses else 0
            
            if gen % 50 == 0 or gen == self.n_gen - 1:
                best_org = self.pop[fitnesses.index(bf)]
                diversity = best_org.hypo.get_diversity()
                replays = best_org.hippo.get_stats().get('total_replays', 0)
                n = best_org.g['n']
                print(f"Gen {gen:3d}  fit={bf:.4f}/{af:.4f}  best={self.best_ever:.4f}  "
                      f"n={n}  drv_div={diversity:.2f}  replays={replays}")
            
            if gen == self.n_gen - 1: break
            
            # 精英保留
            idx_sorted = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            next_gen = [Organism(copy.deepcopy(self.pop[idx_sorted[i]].g)) for i in range(min(4, len(self.pop)))]
            
            # Wild abort: 低适应度+高距离个体偶尔和精英交叉
            candidates = list(range(len(self.pop)))
            while len(next_gen) < self.n_org:
                if random.random() < 0.15 and len(self.pop) > 3:
                    # 选低适应度但和当前最佳距离大的个体做wild abort
                    low_idx = candidates[int(random.random() ** 2 * len(candidates))]  # 偏选低分
                    high_idx = idx_sorted[0]
                    child_g = Genome.crossover(self.pop[low_idx].g, self.pop[high_idx].g)
                else:
                    i1 = random.choices(range(len(self.pop)), weights=[max(0.01, f) for f in fitnesses], k=1)[0]
                    i2 = random.choices(range(len(self.pop)), weights=[max(0.01, f) for f in fitnesses], k=1)[0]
                    child_g = Genome.crossover(self.pop[i1].g, self.pop[i2].g)
                Genome.mutate(child_g, rate=0.05)
                next_gen.append(Organism(child_g))
            
            self.pop = next_gen
        
        return self.pop

# ============ Main ============
if __name__ == '__main__':
    print("=" * 60)
    print("yinyang_brain_v19 — 残差反推后的正推修改")
    print("修改基于: v18残差分析(MSE优但fitness低 → 过拟合+代数不足)")
    print("=" * 60)
    
    print("\n[单有机体验证]")
    org = Organism()
    env = Environment(seed=42); env.reset()
    for i in range(200):
        mse, _ = org.tick(env)
        if i % 40 == 0:
            print(f"  tick {i:3d}: mse={mse:.4f}  surprise={org.amygdala.last_surprise:.3f}  "
                  f"drive={org.hypo.get_dominant()}  replays={org.hippo.get_stats()['total_replays']}")
    
    f = org.eval_fitness(ticks=300)
    print(f"\n适应度: {f:.4f}")
    print(f"海马体: {org.hippo.get_stats()}")
    print(f"驱力: { {k: round(v['value'],3) for k,v in org.hypo.drives.items()} }")
    
    print("\n\n[演化开始: 25org × 400gen]")
    eng = EngineV19(n_org=25, n_gen=400, seed=42)
    eng.evolve()
    
    print(f"\n最终最佳: {eng.best_ever:.4f}")
    print(f"v17o参考: 0.9361")
    print(f"v18参考: 0.9109")
