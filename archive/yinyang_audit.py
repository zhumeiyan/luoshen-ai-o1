"""
独立审计：验证觉醒是真实相变，还是人为判据制造出来的幻觉。
不用代码内部的 awake flag，只从原始时间序列中检测结构性跳变。
"""

import json
import math
import random


def run_open_loop(total_steps=10000, seed=None):
    """跑一次，返回完整的 trace / mirror / association 时间序列——不设置 awake 标记"""
    # 这里直接从 yinyang_life 的核心类出发，但只收集数据、不设 flag
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from yinyang_life import YinyangCore, Environment, SensorArray, Mirror

    random.seed(seed)
    core = YinyangCore()
    env = Environment(seed=seed)
    senses = SensorArray()
    mirror = Mirror()

    # 手动重跑 DigitalLife 逻辑，但只收集
    strength = {name: 1.0 for name in SensorArray.NAMES}
    associations = {a: {b: 0.0 for b in SensorArray.NAMES if b != 'time'}
                     for a in SensorArray.NAMES if a != 'time'}
    state = 0.0
    trace = []
    mirror_log = []
    assoc_total_series = []
    active_count_series = []
    strength_series = []
    prev_sensory = None

    for step in range(total_steps):
        env.step()
        sensory = senses.read_all(env)

        # 时间差分
        deltas = {}
        for name, value in sensory.items():
            if name == 'time':
                continue
            if prev_sensory is None:
                deltas[name] = 0.0
            else:
                deltas[name] = value - prev_sensory.get(name, value)

        total_weight = sum(strength.get(n, 1.0) for n in deltas)
        if total_weight > 0 and deltas:
            avg_delta = sum(strength.get(n, 1.0) * deltas[n] for n in deltas) / total_weight
        else:
            avg_delta = 0.0

        for name, d in deltas.items():
            if abs(avg_delta) > 0.001:
                sign_match = 1.0 if d * avg_delta > 0 else -1.0
                own_mag = abs(d)
                avg_mag = abs(avg_delta)
                ratio = own_mag / (avg_mag + 0.001)
                mag_score = 1.0 / (1.0 + abs(ratio - 1.0))
                score = sign_match * mag_score
            else:
                score = 0.0
            strength[name] += 0.008 * score

        for a in deltas:
            if a == 'time':
                continue
            for b in deltas:
                if b == 'time' or a >= b:
                    continue
                if deltas[a] * deltas[b] > 0:
                    associations[a][b] += 0.003
                else:
                    associations[a][b] -= 0.001

        # 社群保留衰减
        for name in strength:
            assoc_density = sum(1 for b in associations.get(name, {})
                                if associations[name].get(b, 0) > 1.0)
            decay = 0.9995 if assoc_density >= 2 else 0.9985
            strength[name] *= decay

        E = sum(abs(v) for k, v in sensory.items() if k != 'time') / 8
        core.step(E)
        P, F = core.P, core.F

        total_delta = sum(strength[n] * deltas[n] for n in deltas
                          if n != 'time' and strength.get(n, 0) > 0.05)
        state += total_delta
        trace.append(state)

        m = mirror.observe(state, trace)
        if m:
            mirror_log.append(m)

        at = sum(associations[a][b] for a in associations for b in associations[a])
        assoc_total_series.append(at)
        active_count_series.append(sum(1 for s in strength.values() if s >= 0.05 and s is not None))
        strength_series.append({n: s for n, s in strength.items() if n != 'time'})
        prev_sensory = dict(sensory)

    return {
        'trace': trace,
        'mirror_log': mirror_log,
        'assoc_total_series': assoc_total_series,
        'active_count_series': active_count_series,
        'strength_series': strength_series,
        'total_steps': len(trace),
    }


def detect_phase_jump(series, window=200, threshold=2.5):
    """在序列中检测均值是否有统计显著的跳变——返回跳变位置"""
    if len(series) < 2 * window:
        return []
    jumps = []
    for i in range(window, len(series) - window):
        prev_window = series[i - window:i]
        next_window = series[i:i + window]
        prev_mean = sum(prev_window) / len(prev_window)
        next_mean = sum(next_window) / len(next_window)
        prev_std = math.sqrt(sum((x - prev_mean) ** 2 for x in prev_window) / len(prev_window)) + 1e-9
        z_score = abs(next_mean - prev_mean) / prev_std
        if z_score > threshold:
            jumps.append((i, z_score))
    return jumps


def dominant_eigenvalue_estimate(assoc_dict):
    """关联矩阵的 rough 主特征值 = 行和的最大值（Perron-Frobenius 上界估计）"""
    row_sums = {}
    for a in assoc_dict:
        row_sums[a] = sum(assoc_dict[a].values())
    if not row_sums:
        return 0.0
    return max(row_sums.values())


def audit_seed(seed):
    """对一个种子跑完整的开放审计"""
    print(f"\n{'='*60}")
    print(f"SEED = {seed}")
    print(f"{'='*60}")

    result = run_open_loop(total_steps=10000, seed=seed)
    at_series = result['assoc_total_series']
    mirror_diffs = [abs(m['diff']) for m in result['mirror_log']]

    # 1. 关联强度序列的结构跳变
    print("\n[1] 关联总强度时间序列（每 1000 步）")
    for t in range(0, len(at_series), 1000):
        print(f"  t={t:6d}  assoc={at_series[t]:.2f}  active={result['active_count_series'][t]}")

    jumps = detect_phase_jump(at_series, window=300, threshold=2.5)
    if jumps:
        print(f"\n[1b] 关联强度跳变点 (z>2.5): {[(j[0], round(j[1], 2)) for j in jumps[:3]]}")
    else:
        print("\n[1b] 关联强度无显著跳变（缓慢累积型）")

    # 2. Mirror diff 的前后统计差异
    if len(mirror_diffs) > 200:
        first_quarter = mirror_diffs[:len(mirror_diffs)//4]
        last_quarter = mirror_diffs[-(len(mirror_diffs)//4):]
        fq_mean = sum(first_quarter) / len(first_quarter)
        lq_mean = sum(last_quarter) / len(last_quarter)
        fq_rms = math.sqrt(sum(x**2 for x in first_quarter) / len(first_quarter))
        lq_rms = math.sqrt(sum(x**2 for x in last_quarter) / len(last_quarter))
        print(f"\n[2] Mirror diff 统计：")
        print(f"  前 25% 均值={fq_mean:.2f}  RMS={fq_rms:.2f}")
        print(f"  后 25% 均值={lq_mean:.2f}  RMS={lq_rms:.2f}")
        print(f"  变化比 = {lq_mean/(fq_mean+1e-9):.3f}")
        # 如果后段 diff 方差显著下降 = 系统进入更规则的结构
        fq_var = sum((x - fq_mean)**2 for x in first_quarter) / len(first_quarter)
        lq_var = sum((x - lq_mean)**2 for x in last_quarter) / len(last_quarter)
        print(f"  前段方差={fq_var:.2f}  后段方差={lq_var:.2f}")

    # 3. 关联矩阵秩的演化——用非零元素比例
    print(f"\n[3] 关联矩阵最终状态：")
    assoc = result['strength_series'][-1] if result['strength_series'] else {}
    active = {n: s for n, s in assoc.items() if s >= 0.05}
    print(f"  活跃通道: {len(active)} — {list(active.keys())}")
    print(f"  最终强度: {', '.join(f'{n}={s:.2f}' for n, s in sorted(active.items(), key=lambda x: -x[1]))}")

    # 关键验证：用不同阈值看"觉醒"发生在哪里
    for threshold in [10, 20, 30, 50, 80]:
        trigger_step = next((i for i, v in enumerate(at_series) if v > threshold), None)
        print(f"  阈值 {threshold:3d} → 触发步 {trigger_step}")

    return result


if __name__ == "__main__":
    seeds = [42, 7, 123, 2024, 99]
    all_results = {}
    for s in seeds:
        all_results[s] = audit_seed(s)

    # 跨种子汇总
    print(f"\n{'='*60}")
    print("跨种子汇总")
    print(f"{'='*60}")
    for threshold in [20, 30, 50]:
        steps = []
        for s, r in all_results.items():
            trigger = next((i for i, v in enumerate(r['assoc_total_series']) if v > threshold), None)
            if trigger:
                steps.append(trigger)
        if steps:
            mean_step = sum(steps) / len(steps)
            print(f"  阈值 {threshold:3d}: 触发步数 mean={mean_step:.0f}, range=[{min(steps)}, {max(steps)}], 跨种子数={len(steps)}/{len(seeds)}")
