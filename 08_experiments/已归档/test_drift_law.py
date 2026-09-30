# [已撤回] 2026-09-19 推演者归档：测旧身体接口，新架构已换
# [残留英文] 2026-09-20 推演者注：文件内含 luminance/sound/force/temperature 英文环境键名。已归档，不清理。
"""规律测试: 5 个生理需求, 从满开始纯 tick drift, 看掉落速率与形状。
不改环境, 不推 select/serve, 只看 drift 本身 + thermal 物理模型。
age=10 全开, 用 3 min/tick。
"""

import sys, math
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

NEEDS = ["thermal", "glucose", "social", "exploration", "sexual"]

# 起点: 热/血糖用 target (刚好在平衡点), index-type 用 1.0 (真正满)
START = {"thermal": 310.15, "glucose": 5.0, "social": 1.0, "exploration": 1.0, "sexual": 1.0}

TICKS = 120  # 360 min = 6h

# 快降需求(social/exploration)多报, 慢降少报
EVERY = {"thermal": 5, "glucose": 5, "social": 10, "exploration": 10, "sexual": 15}

results = {}

for name in NEEDS:
    info = HOMEOSTATIC_NEEDS[name]
    body = 身体(节点数=24, 种子=1)

    # 环境: 中性偏凉, 不振 any need
    body.environment.update({"光": 0, "声": 0, "力": 0, "温": 293})
    body.state[name] = START[name]

    # 实际 ref 不动 — body.tick() 只跑 drift + thermal 物理

    target = info["target"]
    drift_pm = info["drift_rate_per_min"]

    trace = []
    for t in range(TICKS):
        body.tick()
        trace.append(body.state[name])

    # 找第一次落到 target 以下/以上的 tick
    # thermal 降温(低于 target), 别的触 0
    if name == "thermal":
        cross = next((t for t, v in enumerate(trace) if v < target), None)
        floor = 293  # ambient
    elif name == "glucose":
        cross = next((t for t, v in enumerate(trace) if v < target), None)
        floor = 0.0
    else:
        cross = next((t for t, v in enumerate(trace) if v < target), None)
        floor = 0.0

    # 落到 floor
    reach_floor = next((t for t, v in enumerate(trace) if v <= floor + 1e-9), None)

    results[name] = dict(
        target=target, drift_pm=drift_pm, start=START[name],
        cross_target_tick=cross, reach_floor_tick=reach_floor,
        trace=trace, every=EVERY[name],
    )

    print(f"\n{'='*70}")
    print(f"[{name}] unit={info['unit']}  target={target}  drift={drift_pm:+.4f}/min")
    print(f"  start={START[name]}  ->  cross_target@{cross}  reach_floor@{reach_floor}")
    print(f"  ref = {body.refractory_timer[name]:.1f}  (应=0, 未被触发)")
    eh = EVERY[name]
    print(f"\n  {'tick':>5}  {'state':>10}  {'Δ/tick':>10}  below_target")
    print(f"  {'-'*50}")
    for t in range(0, TICKS, eh):
        v = trace[t]
        d = trace[t] - trace[t-1] if t > 0 else 0
        flag = " <<<" if (name == "thermal" and v < target) or (name != "thermal" and v < target - 1e-9) else ""
        print(f"  {t:>5}  {v:>10.4f}  {d:>+10.6f}{flag}")

# ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("汇总: 5 需求从满到缺的速率")
print("=" * 80)
print(f"{'need':<14} {'unit':<10} {'start':>8} {'target':>8} {'drift/min':>10} {'gw@10y':>8} {'tick_sz':>8} {'Δ/tick':>10} {'cross@tick':>11} {'零点@tick':>11}")
print("-" * 100)
for name in NEEDS:
    r = results[name]
    body = 身体(节点数=24, 种子=1)
    gw = body._gate(name)
    eff_drift = r["drift_pm"] * 3 * gw
    print(f"{name:<14} {HOMEOSTATIC_NEEDS[name]['unit']:<10} {r['start']:>8.2f} {r['target']:>8.2f} {r['drift_pm']:>+10.5f} {gw:>8.4f} {'3min':>8} {eff_drift:>+10.6f} {str(r['cross_target_tick']):>11} {str(r['reach_floor_tick']):>11}")
