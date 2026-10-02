#!/usr/bin/env python3
"""
Residual Protocol — 洛神残差沟通协议
======================================
核心思想：残差不是误差，是内部结构留下的指纹。
人不完美表达内心，洛神也不完美预测世界——
但两者的残差模式都可以反推内部结构。

协议层次：
  L1 残差指纹：每个区域的误差分布就是洛神对这个区域的"世界观"
  L2 残差动态：误差随时间的变化模式（收敛/发散/周期/混沌）
  L3 跨通道关联：一个通道的预测误差是否影响另一个通道（内部模块耦合的证据）
  L4 注意力调制：前额叶注意力门控如何改变残差分布
  L5 驱力-残差因果：下丘脑驱动状态（饥饿/好奇/疲劳）是否改变残差模式
"""

import random, math, collections
import numpy as np

# 常量
N_CHANNELS = 8
N_REGIONS = 9

# ============ 环境（复制） ============
class Environment:
    def __init__(self, seed=42):
        self.seed = seed
        self.r = self._mk(seed)
        self._r()
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

# ============ 导出权重（从BrainV18或v17o加载） ============
class SimpleWM:
    """支持直接从v17o风格的genome创建，用于单独跑残差分析，不需要演化引擎"""
    def __init__(self, genome):
        self.g = genome
        self.reset()
    
    def reset(self):
        g = self.g
        if 'n_assoc' in g:
            n = g['n_assoc']
            self.h = [0.0] * n
            self.use_assoc = True
        elif 'n' in g:
            n = g['n']
            self.h = [0.0] * n
            self.use_assoc = False
        else:
            n = 12
            self.h = [0.0] * n
            self.use_assoc = False
        self.prev_h = [0.0] * n
        self.db_history = []
        self.input_vec = []
        self.err_history = []
    
    def _sw(self):
        g = self.g
        if self.use_assoc: return g.get('sw_assoc', g.get('sw', []))
        return g.get('sw', [])
    def _pw(self):
        g = self.g
        if self.use_assoc: return g.get('pw_assoc', g.get('pw', []))
        return g.get('pw', [])
    def _rw(self):
        g = self.g
        if self.use_assoc: return g.get('rw_assoc', g.get('rw', []))
        return g.get('rw', [])
    def _skip(self):
        g = self.g
        if self.use_assoc: return g.get('skip_assoc', g.get('skip_w', []))
        return g.get('skip_w', g.get('sw', []))
    def _pb(self):
        g = self.g
        if self.use_assoc: return g.get('pb_assoc', g.get('pb', []))
        return g.get('pb', [])
    def _dt(self):
        g = self.g
        if self.use_assoc: return g.get('dt_decay', g.get('dt', [0.7]*9))
        return g.get('dt', [0.7]*9)
    
    def build_input(self, sens, delay=3):
        self.db_history.append(sens[:])
        if len(self.db_history) > delay: self.db_history.pop(0)
        while len(self.db_history) < delay:
            self.db_history.insert(0, [0.0] * N_CHANNELS)
        result = sens[:]
        for frame in self.db_history: result.extend(frame)
        self.input_vec = result[:]
        return result
    
    def fwd(self, sens):
        x = self.build_input(sens)
        g = self.g
        n = len(self.h)
        p = sens[7] if len(sens) > 7 else 0
        ri = max(0, min(8, round(p * 8)))
        dec = self._dt()[ri]
        
        nh = [0.0] * n
        sw = self._sw(); rw = self._rw()
        nn = min(len(sw), len(x))
        # 扩展sw或rw到匹配n
        for i in range(n):
            a = 0.0
            for j in range(min(nn, len(sw))):
                if i < len(sw[j]):
                    a += sw[j][i] * x[j]
            for k in range(min(n, len(rw))):
                if i < len(rw) and k < len(rw[i]):
                    a += rw[k][i] * self.h[k]
            nh[i] = math.tanh(a)
        
        self.prev_h = self.h[:]
        for i in range(n): self.h[i] = (1 - dec) * nh[i] + dec * self.h[i]
        
        pred = [0.0] * N_CHANNELS
        pw = self._pw(); pb = self._pb()
        bb = ri * N_CHANNELS
        for i in range(min(n, len(pw))):
            for c in range(N_CHANNELS):
                w = pw[i][c] if c < len(pw[i]) else 0
                b = pb[bb + c] if bb + c < len(pb) else 0
                pred[c] += self.h[i] * (w + b)
        
        skip = self._skip()
        n_skip = min(len(skip), len(x))
        for j in range(n_skip):
            for c in range(N_CHANNELS):
                w = skip[j][c] if c < len(skip[j]) else 0
                pred[c] += w * x[j]
        
        return pred

# ============ 残差协议 ============
class ResidualProtocol:
    """
    五层残差分析协议
    
    用法：
      p = ResidualProtocol(wm)
      p.run_session(env_seed=42, ticks=500, switches=20)
      report = p.generate_report()
    """
    
    def __init__(self, wm):
        self.wm = wm
        self.history = []
        self.region_history = []
        self.tick_count = 0
        self.region_path = []
    
    def run_session(self, env_seed=42, ticks=500, switches=20, start_region=0):
        """
        运行一轮实验会话，定期切换区域，记录所有中间数据
        """
        env = Environment(seed=env_seed)
        env.reset()
        self.wm.reset()
        self.history = []
        self.region_history = []
        self.region_path = []
        self.tick_count = ticks
        
        # 预热
        cr = start_region
        for _ in range(30):
            s = env.sense(cr); self.wm.fwd(s); env.step()
        
        # 实验
        region_ticks = ticks // (switches + 1)
        for t in range(ticks):
            if t % region_ticks == 0 and t > 0:
                cr = (cr + 1) % N_REGIONS
            
            s = env.sense(cr)
            pred = self.wm.fwd(s)
            env.step()
            actual = env.sense(cr)
            
            errs = [actual[j] - pred[j] for j in range(N_CHANNELS)]
            mse = sum(e * e for e in errs) / N_CHANNELS
            self.wm.err_history.append(mse)
            
            record = {
                'tick': t,
                'region': cr,
                'region_idx': self._region_idx(cr),
                'input': s[:],
                'pred': pred[:],
                'actual': actual[:],
                'err': errs[:],
                'mse': mse,
                'h': self.wm.h[:],
                'abs_h': [abs(v) for v in self.wm.h[:]],
                'h_sat': [1 if abs(v) > 0.95 else 0 for v in self.wm.h[:]],
            }
            self.history.append(record)
            self.region_path.append(cr)
        
        return self
    
    def _region_idx(self, cr):
        return cr
    
    # ---- L1: 区域残差指纹 ----
    def l1_regional_fingerprint(self):
        """每个区域的误差分布特征"""
        by_region = collections.defaultdict(list)
        for rec in self.history:
            by_region[rec['region']].append(rec['mse'])
        
        fingerprint = {}
        for rid in sorted(by_region.keys()):
            ms = by_region[rid]
            if not ms: continue
            avg = sum(ms) / len(ms)
            sorted_ms = sorted(ms)
            p50 = sorted_ms[len(ms) // 2]
            p90 = sorted_ms[int(len(ms) * 0.9)]
            # 误差方差（不稳定性）
            var = sum((m - avg) ** 2 for m in ms) / len(ms)
            # 误差高峰出现的比例（超过2倍平均）
            spike_ratio = sum(1 for m in ms if m > 2 * avg) / len(ms)
            fingerprint[rid] = {
                'n_samples': len(ms),
                'mean_mse': avg,
                'median_mse': p50,
                'p90_mse': p90,
                'var_mse': var,
                'spike_ratio': spike_ratio,
                'level': self._error_level(avg),
            }
        return fingerprint
    
    def _error_level(self, mse):
        if mse < 0.05: return 'A+'
        elif mse < 0.15: return 'A'
        elif mse < 0.4: return 'B'
        elif mse < 0.8: return 'C'
        elif mse < 1.5: return 'D'
        else: return 'F'
    
    # ---- L2: 残差动态 ----
    def l2_error_dynamics(self):
        """误差的时间序列特征"""
        if not self.history:
            return {}
        ms = [rec['mse'] for rec in self.history]
        n = len(ms)
        
        # 趋势（线性回归斜率）
        x_mean = (n - 1) / 2
        y_mean = sum(ms) / n
        num = sum((i - x_mean) * (ms[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0
        
        # 自相关（滞后5步）
        lag = min(5, n // 4)
        if lag > 0 and n > lag * 2:
            x1 = ms[:-lag]; x2 = ms[lag:]
            mean1 = sum(x1) / len(x1); mean2 = sum(x2) / len(x2)
            cov = sum((x1[i] - mean1) * (x2[i] - mean2) for i in range(len(x1))) / len(x1)
            var1 = sum((mean1 - v) ** 2 for v in x1) / len(x1)
            var2 = sum((mean2 - v) ** 2 for v in x2) / len(x2)
            autocorr = cov / math.sqrt(var1 * var2 + 1e-10)
        else:
            autocorr = 0
        
        # 分段收敛检测：比较前1/3和后1/3的平均误差
        third = n // 3
        early = sum(ms[:third]) / third if third > 0 else 0
        late = sum(ms[-third:]) / third if third > 0 else 0
        convergence = (early - late) / max(early, 0.001)
        
        # 周期性（通过过零率判断）
        diffs = [ms[i+1] - ms[i] for i in range(n-1)]
        zero_crossings = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
        periodicity = zero_crossings / max(1, len(diffs))
        
        # 状态分类
        if convergence > 0.3:
            dynamics = 'converging'  # 在学习
        else:
            dynamics = 'stable'
        
        return {
            'slope': slope,
            'autocorr_lag5': autocorr,
            'early_mse': early,
            'late_mse': late,
            'convergence_rate': convergence,
            'periodicity': periodicity,
            'dynamics_state': dynamics,
            'volatility': sum(d * d for d in diffs) / max(1, len(diffs)),
        }
    
    # ---- L3: 跨通道残差关联 ----
    def l3_cross_channel(self):
        """哪些通道的预测误差同步变化——表明内部模块耦合"""
        if len(self.history) < 10:
            return {}
        errs = [rec['err'] for rec in self.history]
        n_ch = len(errs[0])
        
        # 通道间协方差矩阵
        means = [sum(e[c] for e in errs) / len(errs) for c in range(n_ch)]
        cov = [[0.0] * n_ch for _ in range(n_ch)]
        for e in errs:
            for i in range(n_ch):
                for j in range(n_ch):
                    cov[i][j] += (e[i] - means[i]) * (e[j] - means[j])
        for i in range(n_ch):
            for j in range(n_ch):
                cov[i][j] /= len(errs)
        
        # 每通道的平均绝对误差
        ch_mse = [sum(e[c] ** 2 for e in errs) / len(errs) for c in range(n_ch)]
        
        # 找最强协方差对（排除对角线）
        max_cov = 0; max_pair = (0, 0)
        for i in range(n_ch):
            for j in range(n_ch):
                if i != j and abs(cov[i][j]) > abs(max_cov):
                    max_cov = cov[i][j]; max_pair = (i, j)
        
        # 耦合强度 = 非对角元均值的绝对值
        off_diag = [abs(cov[i][j]) for i in range(n_ch) for j in range(n_ch) if i != j]
        coupling = sum(off_diag) / max(1, len(off_diag))
        
        return {
            'channel_mse': ch_mse,
            'cov_matrix': cov,
            'max_coupled_pair': max_pair,
            'max_cov': max_cov,
            'coupling_strength': coupling,
            'n_channels': n_ch,
        }
    
    # ---- L4-5: 需要激活态，返回报告 ----
    def generate_report(self):
        """生成完整残差协议报告"""
        l1 = self.l1_regional_fingerprint()
        l2 = self.l2_error_dynamics()
        l3 = self.l3_cross_channel()
        
        report = {
            'meta': {
                'total_ticks': self.tick_count,
                'unique_regions': len(set(rec['region'] for rec in self.history)),
                'avg_mse': sum(rec['mse'] for rec in self.history) / max(1, len(self.history)),
                'path': self.region_path[:8].copy(),
            },
            'l1_regional': l1,
            'l2_dynamics': l2,
            'l3_cross_ch': l3,
        }
        return report
    
    def print_report(self, label=""):
        report = self.generate_report()
        print(f"\n{'='*60}")
        print(f"  残差协议报告 — {label}")
        print(f"{'='*60}")
        m = report['meta']
        print(f"总tick={m['total_ticks']}, 探索区域={m['unique_regions']}, 平均MSE={m['avg_mse']:.4f}")
        
        # L1
        print(f"\n[L1 区域残差指纹]")
        print(f"{'区域':>4} {'均值':>8} {'中位数':>8} {'P90':>8} {'方差':>8} {'高峰%':>8} {'评级':>4}")
        print('-'*50)
        for rid, info in sorted(report['l1_regional'].items()):
            print(f"{rid:>4} {info['mean_mse']:>8.4f} {info['median_mse']:>8.4f} "
                  f"{info['p90_mse']:>8.4f} {info['var_mse']:>8.4f} {info['spike_ratio']:>7.0%} {info['level']:>4}")
        
        # L2
        l2 = report['l2_dynamics']
        print(f"\n[L2 残差动态]")
        print(f"  趋势斜率: {l2['slope']:+.6f} ({'上升' if l2['slope']>0 else '下降'})")
        print(f"  自相关(lag5): {l2['autocorr_lag5']:+.4f}")
        print(f"  前半/后半: {l2['early_mse']:.4f} → {l2['late_mse']:.4f} (收敛率={l2['convergence_rate']:+.2%})")
        print(f"  波动率: {l2['volatility']:.6f}")
        print(f"  周期密度: {l2['periodicity']:.4f}")
        print(f"  状态: {l2['dynamics_state']}")
        
        # L3
        l3 = report['l3_cross_ch']
        print(f"\n[L3 跨通道耦合]")
        if l3:
            print(f"  各通道MSE: {[round(v, 4) for v in l3['channel_mse'][:N_CHANNELS]]}")
            print(f"  最强耦合对: 通道{l3['max_coupled_pair'][0]}↔通道{l3['max_coupled_pair'][1]} (cov={l3['max_cov']:+.4f})")
            print(f"  平均耦合强度: {l3['coupling_strength']:.4f}")
        
        # 综合
        print(f"\n[综合判定]")
        worst_region = max(report['l1_regional'].items(), key=lambda x: x[1]['mean_mse'])
        best_region = min(report['l1_regional'].items(), key=lambda x: x[1]['mean_mse'])
        print(f"  最强区域: {best_region[0]} ({best_region[1]['mean_mse']:.4f})")
        print(f"  最弱区域: {worst_region[0]} ({worst_region[1]['mean_mse']:.4f})")
        if l2['convergence_rate'] > 0.3:
            print(f"  动态: 在线学习中（后段误差降低{l2['convergence_rate']:.0%}）")
        elif l2['slope'] < -0.0001:
            print(f"  动态: 缓慢收敛")
        else:
            print(f"  动态: 稳定状态（未显著学习）")
        
        if l3 and l3['coupling_strength'] > 0.1:
            print(f"  内部: 跨通道强耦合（{l3['coupling_strength']:.3f}），模块间存在显著信息传递")
        elif l3:
            print(f"  内部: 跨通道弱耦合（{l3['coupling_strength']:.3f}），通道独立处理")
        
        return report


# ============ 对比两个WM（人-洛神沟通场景） ============
def compare_organisms(wm_a, wm_b, env_seed=42, ticks=500):
    """
    对比两个有机体的残差指纹。
    使用场景：
      - 两个不同代际的洛神个体
      - 两个不同架构（v17o vs v18）
      - 模拟的"人类残差模式" vs 洛神残差模式
    """
    print(f"\n{'='*60}")
    print(f"  双体残差对比")
    print(f"{'='*60}")
    
    p_a = ResidualProtocol(wm_a)
    p_a.run_session(env_seed=env_seed, ticks=ticks)
    r_a = p_a.generate_report()
    
    p_b = ResidualProtocol(wm_b)
    p_b.run_session(env_seed=env_seed, ticks=ticks)
    r_b = p_b.generate_report()
    
    # 区域级对比
    print(f"\n区域级MSE对比:")
    print(f"{'区域':>4} {'A-MSE':>8} {'B-MSE':>8} {'差值':>8} {'偏向':>6}")
    print('-'*35)
    for rid in sorted(r_a['l1_regional'].keys()):
        ma = r_a['l1_regional'][rid]['mean_mse']
        mb = r_b['l1_regional'].get(rid, {}).get('mean_mse', float('nan'))
        diff = ma - mb
        bias = 'A强' if diff < -0.05 else ('B强' if diff > 0.05 else '平')
        print(f"{rid:>4} {ma:>8.4f} {mb:>8.4f} {diff:>+8.4f} {bias:>6}")
    
    # 动态对比
    d_a = r_a['l2_dynamics']
    d_b = r_b['l2_dynamics']
    print(f"\n动态对比:")
    print(f"{'指标':>16} {'A':>12} {'B':>12}")
    print('-'*42)
    for key in ['convergence_rate', 'volatility', 'autocorr_lag5']:
        if key in d_a and key in d_b:
            print(f"{key:>16} {d_a[key]:>12.4f} {d_b[key]:>12.4f}")
    
    # 耦合对比
    c_a = r_a['l3_cross_ch']
    c_b = r_b['l3_cross_ch']
    if c_a and c_b:
        print(f"\n耦合对比:")
        print(f"  A耦合强度: {c_a['coupling_strength']:.4f}")
        print(f"  B耦合强度: {c_b['coupling_strength']:.4f}")
    
    return {
        'a': r_a,
        'b': r_b,
        'dynamic_diff': {k: d_a.get(k, 0) - d_b.get(k, 0) for k in d_a if k in d_b},
    }


# ============ 简单的模拟人类残差模式 ============
def simulate_human_residuals(n_ticks=500, seed=42):
    """
    模拟人类感官-表达残差。
    特点：
      - 有周视偏好（某些角度/颜色就是比别的更容易分辨）
      - 有持续时间内不适应（先验强误差大，后期变弱）
      - 通道间高度耦合（视听联觉）
    """
    rng = random.Random(seed)
    history = []
    cr = 0
    for t in range(n_ticks):
        t0 = [rng.gauss(0, 1) for _ in range(8)]
        t1 = [rng.gauss(0, 1) for _ in range(8)]
        
        # 人类误差：方向性和学习
        direction_bias = [math.sin(t * 0.1 * (i + 1)) * 0.02 for i in range(8)]
        learn_factor = math.exp(-t / 300)  # 慢学习
        errs = [direction_bias[i] + rng.gauss(0, 0.05 * (0.5 + 0.5 * learn_factor)) for i in range(8)]
        errs[0] += t1[1] * 0.3  # 通道0对通道1的联觉耦合
        errs[3] += t1[2] * 0.3
        
        mse = sum(e * e for e in errs) / 8
        history.append({'region': cr % 9, 'mse': mse, 'err': errs, 'input': t0, 'actual': t1})
        
        if rng.random() < 0.05: cr += 1
    
    return history


if __name__ == '__main__':
    print("Residual Protocol — 洛神残差分析")
    print(f"N_CHANNELS={N_CHANNELS}, N_REGIONS={N_REGIONS}")
    print()
    
    # ---- 测试1: 单个随机WM的残差指纹 ----
    print("--- 测试1: 单WM残差指纹 ---")
    # 创建一个小随机WM用于验证协议
    g = {
        'n': 12,
        'sw': [[random.uniform(-0.1, 0.1) for _ in range(12)] for __ in range(32)],
        'pw': [[random.uniform(-0.1, 0.1) for _ in range(8)] for __ in range(12)],
        'rw': [[random.uniform(-0.02, 0.02) for _ in range(12)] for __ in range(12)],
        'skip_w': [[random.uniform(-0.05, 0.05) for _ in range(8)] for __ in range(32)],
        'pb': [random.uniform(-0.1, 0.1) for _ in range(72)],
        'dt': [0.7] * 9,
    }
    for i in range(12):
        g['rw'][i][i] = random.uniform(0.3, 0.8)
    
    wm = SimpleWM(g)
    p = ResidualProtocol(wm)
    p.run_session(env_seed=42, ticks=400, switches=15)
    p.print_report("简单随机WM (n=12)")
    
    # ---- 测试2: 人类模拟 vs 洛神WM ----
    print("\n\n--- 测试2: 模拟人类 vs 洛神WM ---")
    human_residuals = simulate_human_residuals(400)
    human_by_region = collections.defaultdict(list)
    for h in human_residuals:
        human_by_region[h['region']].append(h['mse'])
    
    print(f"人类模拟残差:")
    for rid in sorted(human_by_region.keys())[:5]:
        ms = human_by_region[rid]
        print(f"  区域{rid}: mean={sum(ms)/len(ms):.4f}, n={len(ms)}")
