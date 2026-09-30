# [已撤回] 2026-09-19 推演者归档：测旧身体接口，新架构已换
# [残留英文] 2026-09-20 推演者注：文件内含 luminance/sound/force/temperature 英文环境键名。已归档，不清理。
"""7 新需求独立 drift 测试: 从满(1.0)开始, 纯 tick, 看各自掉到缺的速率与形状。
不连互相影响, 老 5 个不动, 只看这 7 个新需求的自身规律。
"""

import sys, math
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

NEW7 = ["rest", "safety", "soothe", "pain", "hunger", "thirst", "touch"]

# 从满(1.0)开始, 看自然 drift
START = 1.0
TICKS = 250

for name in NEW7:
    info = HOMEOSTATIC_NEEDS[name]
    body = 身体(节点数=24, 种子=1)

    # 环境归零, 仅看 drift
    body.environment.update({"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
    body.state[name] = START
    body.refractory_timer[name] = 0

    target = info["target"]
    gw = body._gate(name)
    drift_pm = info["drift_rate_per_min"]

    trace = []
    for t in range(TICKS):
        body.tick()
        trace.append(body.state[name])

    # 第一次 < target
    cross = next((t for t, v in enumerate(trace) if v < target - 1e-9), None)
    # 第一次 <= 0.1 (深缺)
    deep = next((t for t, v in enumerate(trace) if v <= 0.1), None)

    # 形状: 计算前 20 tick 的 Δ 看是否恒定
    deltas = [trace[t] - trace[t-1] for t in range(1, min(21, TICKS))]
    delta_mean = sum(deltas) / len(deltas)
    delta_std = (sum((d - delta_mean)**2 for d in deltas) / len(deltas)) ** 0.5
    is_linear = delta_std < 1e-8

    print(f"\n{'='*75}")
    print(f"[{name}] unit={info['unit']}  target={target}  drift={drift_pm:+.5f}/min  gw@10y={gw:.4f}")
    print(f"  start=1.0  cross_target@{cross}  deep_0.1@{deep}  linear={'YES' if is_linear else 'no'}")
    print(f"  Δ/tick (理论) = drift·dt·gw = {drift_pm*3*gw:+.7f}  | 实测 mean={delta_mean:+.7f} std={delta_std:.2e}")

    # 降采样打印
    step = 25 if TICKS > 100 else 15
    print(f"\n  {'tick':>5}  {'state':>9}  {'vs_target':>10}  {'Δ':>10}")
    print(f"  {'-'*42}")
    for t in range(0, TICKS, step):
        v = trace[t]
        d = trace[t] - trace[t-1] if t > 0 else 0
        flag = " <target" if v < target - 1e-9 else (" deep" if v <= 0.1 else "")
        print(f"  {t:>5}  {v:>9.5f}  {v-target:>+10.5f}  {d:>+10.6f}{flag}")

# ── 汇总 ──
print(f"\n\n{'='*90}")
print("汇总: 7 新需求从满到缺")
print(f"{'='*90}")
print(f"{'name':<10} {'unit':<14} {'target':>7} {'drift/min':>11} {'eff_Δ/tick':>12} {'cross@tick':>11} {'deep@tick':>10} {'linear':>7}")
print("-"*90)
for name in NEW7:
    info = HOMEOSTATIC_NEEDS[name]
    body = 身体(节点数=24, 种子=1)
    body.environment.update({"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
    body.state[name] = 1.0
    gw = body._gate(name)
    eff = info["drift_rate_per_min"] * 3 * gw

    trace = []
    for t in range(400):
        body.tick()
        trace.append(body.state[name])
    cross = next((t for t, v in enumerate(trace) if v < info["target"] - 1e-9), "∞")
    deep = next((t for t, v in enumerate(trace) if v <= 0.1), "∞")
    print(f"{name:<10} {info['unit']:<14} {info['target']:>7.2f} {info['drift_rate_per_min']:>+11.5f} {eff:>+12.7f} {str(cross):>11} {str(deep):>10} {'YES':>7}")
