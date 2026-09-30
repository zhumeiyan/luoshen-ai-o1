# [已撤回] 2026-09-19 推演者归档：测旧身体接口，新架构已换
# [残留英文] 2026-09-20 推演者注：文件内含 luminance/sound/force/temperature 英文环境键名。已归档，不清理。
"""12-需求互相压/松对照测试。

单独隔离每个压/松关系, 对照组(无跨需求) vs 实验组(固定源需求缺),
看目标需求的 drift 速率变化。

每个场景固定 4 个对照:
  BL   = 基线 (无压迫, 12 需求各自跑)
  PG   = 压：源需求低
  RL   = 松：安抚/被抱高
  PG+RL= 同时

速度比 = 实验组 drift / 基线 drift。> 1 = 掉得更快; < 1 = 掉得更慢。
"""

import sys
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

TICKS = 80
DT = 3

def fresh_pin(src_name, src_val, extra=None):
    """构造 body, 把 src_name 钉在 src_val, 12 需求其他放到 1.0。extra 是额外要钉的 {name: val}."""
    body = 身体(节点数=24, 种子=1)
    body.environment.update({"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
    for n in HOMEOSTATIC_NEEDS:
        body.state[n] = 1.0
    body.state[src_name] = src_val
    if extra:
        for k, v in extra.items():
            body.state[k] = v
    return body

def run(body, ticks=TICKS, pin=None):
    """纯 tick, 如果 pin 则在每一 tick 开头重写被钉的值。返回 {name: trace(list)}。"""
    traces = {n: [] for n in HOMEOSTATIC_NEEDS}
    for t in range(ticks):
        if pin:
            for k, v in pin.items():
                body.state[k] = v
        body.tick()
        for n in HOMEOSTATIC_NEEDS:
            traces[n].append(body.state[name_of(n, body)])
    return traces

def name_of(n, body):
    return n

def avg_drift(trace):
    if len(trace) < 2:
        return 0.0
    return (trace[-1] - trace[0]) / (len(trace) - 1)

def drift_of(body, target, ticks=TICKS, pin_src=None, pin_val=None, extra_pin=None):
    """跑一段, 返回 target 的平均 drift/tick。"""
    b = fresh_pin(pin_src, pin_val, extra_pin) if pin_src else 身体(节点数=24, 种子=1)
    if not pin_src:
        b.environment.update({"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
        for n in HOMEOSTATIC_NEEDS:
            b.state[n] = 1.0
    pin_dict = {}
    if pin_src:
        pin_dict[pin_src] = pin_val
    if extra_pin:
        pin_dict.update(extra_pin)

    trace = []
    for t in range(ticks):
        for k, v in pin_dict.items():
            b.state[k] = v
        b.tick()
        trace.append(b.state[target])
    return trace

# 基线: 无任何压, 各自从 1.0 纯 drift
print("=" * 95)
print("基线 (BL): 12 需求全在 1.0, 无跨需求压")
print("=" * 95)
bl = {}
for n in HOMEOSTATIC_NEEDS:
    if n in ("thermal", "glucose"):
        continue  # 环境类, 不在这测
    body = 身体(节点数=24, 种子=1)
    body.environment.update({"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
    for k in HOMEOSTATIC_NEEDS:
        body.state[k] = 1.0
    trace = []
    for t in range(TICKS):
        body.tick()
        trace.append(body.state[n])
    bl[n] = avg_drift(trace)
    print(f"  {n:<14} Δ/tick = {bl[n]:+.7f}")

#  松: soothe=1.0 → safety/pain/rest 掉得慢
print(f"\n{'='*95}")
print("松 (RL): soothe 钉在 1.0, 看 safety/pain/rest 减速")
print("=" * 95)
for tgt in ("safety", "pain", "rest"):
    tr = drift_of(None, tgt, pin_src="soothe", pin_val=1.0)
    rl = avg_drift(tr)
    print(f"  {tgt:<14} Δ/tick = {rl:+.7f}  (BL={bl[tgt]:+.7f})  松比 = {rl/bl[tgt]:.3f}")

# 松: touch=1.0 (连着安抚)
print(f"\n{'='*95}")
print("松 (touch=1.0, 被抱连着安抚)")
print("=" * 95)
for tgt in ("safety", "pain", "rest"):
    tr = drift_of(None, tgt, pin_src="touch", pin_val=1.0)
    rl = avg_drift(tr)
    print(f"  {tgt:<14} Δ/tick = {rl:+.7f}  (BL={bl[tgt]:+.7f})  松比 = {rl/bl[tgt]:.3f}")

# 压: safety=0.0 (极怕)
print(f"\n{'='*95}")
print("压: safety=0.0 (极怕)  → social, exploration")
print("=" * 95)
for tgt in ("social", "exploration"):
    tr = drift_of(None, tgt, pin_src="safety", pin_val=0.0)
    pg = avg_drift(tr)
    print(f"  {tgt:<14} Δ/tick = {pg:+.7f}  (BL={bl[tgt]:+.7f})  压比 = {pg/bl[tgt]:.3f}")

# 压: pain=0.0 (极疼)
print(f"\n{'='*95}")
print("压: pain=0.0 (极疼)  → social, exploration, touch")
print("=" * 95)
for tgt in ("social", "exploration", "touch"):
    tr = drift_of(None, tgt, pin_src="pain", pin_val=0.0)
    pg = avg_drift(tr)
    print(f"  {tgt:<14} Δ/tick = {pg:+.7f}  (BL={bl[tgt]:+.7f})  压比 = {pg/bl[tgt]:.3f}")

# 压: rest=0.0 (极困)
print(f"\n{'='*95}")
print("压: rest=0.0 (极困)  → exploration")
print("=" * 95)
tr = drift_of(None, "exploration", pin_src="rest", pin_val=0.0)
pg = avg_drift(tr)
print(f"  exploration    Δ/tick = {pg:+.7f}  (BL={bl['exploration']:+.7f})  压比 = {pg/bl['exploration']:.3f}")

# 压: hunger=0.0 (极饿) — 慢速压
print(f"\n{'='*95}")
print("压: hunger=0.0 (极饿, 慢压)  → social, exploration")
print("=" * 95)
for tgt in ("social", "exploration"):
    tr = drift_of(None, tgt, pin_src="hunger", pin_val=0.0)
    pg = avg_drift(tr)
    print(f"  {tgt:<14} Δ/tick = {pg:+.7f}  (BL={bl[tgt]:+.7f})  压比 = {pg/bl[tgt]:.3f}")

# 压: thirst=0.0 (极渴) — 快速压
print(f"\n{'='*95}")
print("压: thirst=0.0 (极渴, 急压)  → social, exploration")
print("=" * 95)
for tgt in ("social", "exploration"):
    tr = drift_of(None, tgt, pin_src="thirst", pin_val=0.0)
    pg = avg_drift(tr)
    print(f"  {tgt:<14} Δ/tick = {pg:+.7f}  (BL={bl[tgt]:+.7f})  压比 = {pg/bl[tgt]:.3f}")

# 综合表
print(f"\n\n{'='*95}")
print("速度比对照 (基线 drift = 1.0)")
print("=" * 95)
print(f"{'场景':<30} {'目标':<14} {'drift实验':>12} {'drift基线':>12} {'比(↑快↓慢)':>12}")
print("-"*95)

def row(scene, tgt, tr):
    g = avg_drift(tr)
    b = bl[tgt]
    print(f"{scene:<30} {tgt:<14} {g:>+12.7f} {b:>+12.7f} {g/b:>12.3f}")

r=drift_of(None, "social", pin_src="safety", pin_val=0.0); row("怕→压social",  "social", r)
r=drift_of(None, "exploration", pin_src="safety", pin_val=0.0); row("怕→压explore",  "exploration", r)
r=drift_of(None, "social", pin_src="pain", pin_val=0.0); row("疼→压social",  "social", r)
r=drift_of(None, "exploration", pin_src="pain", pin_val=0.0); row("疼→压explore", "exploration", r)
r=drift_of(None, "touch", pin_src="pain", pin_val=0.0); row("疼→压touch",   "touch", r)
r=drift_of(None, "exploration", pin_src="rest", pin_val=0.0); row("困→压explore", "exploration", r)
r=drift_of(None, "social", pin_src="hunger", pin_val=0.0); row("饿→压social",  "social", r)
r=drift_of(None, "social", pin_src="thirst", pin_val=0.0); row("渴→压social",  "social", r)
r=drift_of(None, "safety", pin_src="soothe", pin_val=1.0); row("安抚松safety", "safety", r)
r=drift_of(None, "pain", pin_src="soothe", pin_val=1.0); row("安抚松pain",   "pain", r)
r=drift_of(None, "rest", pin_src="soothe", pin_val=1.0); row("安抚松rest",   "rest", r)
r=drift_of(None, "safety", pin_src="touch", pin_val=1.0); row("被抱松safety", "safety", r)
