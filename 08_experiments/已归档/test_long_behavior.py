# [已撤回] 2026-09-19 推演者归档：测旧身体接口，新架构已换
# [残留英文] 2026-09-20 推演者注：文件内含 luminance/sound/force/temperature 英文环境键名。已归档，不清理。
"""长跑行为测试: 12 需求 + 血糖/热 一起。
四项修改后重跑:
  改1: thermal drift 方向下降 (-0.002/min)
  改2: thermal dead_band 0.5K (死区内不响)
  改3: glucose 糖异生地板 3.3 (低于此不再下掉)
  改4: pain/safety 改为事件驱动 (不再匀速下漂, 只有 trigger_event 才掉)

分四段跑:
  A. pure drift 60 tick
  B. 带 select/serve 400 tick
  C. trigger pain -> social/探索 被压
  D. trigger safety -> social/探索 被压 + 恢复
"""

import sys
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

PSY = ["social", "exploration", "sexual", "rest", "safety", "soothe", "pain", "hunger", "thirst", "touch"]
BIO = ["thermal", "glucose"]
ALL = BIO + PSY

def body_start_at_one(env=None):
    b = 身体(节点数=24, 种子=1)   # 1 拍 = 4 分钟
    if env:
        b.environment.update(env)
    for n in ALL:
        b.state[n] = 1.0
    b.state["glucose"] = 5.0
    b.state["thermal"] = 310.15
    return b

# ════ 段 A ════
print("=" * 110)
print("段 A: 纯 drift, 12 需求从 1.0 开始, 60 tick")
print("=" * 110)
b = body_start_at_one(env={"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
trA = {n: [] for n in ALL}
for t in range(60):
    for n in ALL:
        trA[n].append(b.state[n])
    b.tick()

def curvature(trace):
    d1 = [trace[i+1] - trace[i] for i in range(len(trace)-1)]
    if len(d1) < 3: return 0.0
    d2 = [d1[i+1] - d1[i] for i in range(len(d1)-1)]
    mean_d2 = sum(d2) / len(d2)
    return (sum((x - mean_d2)**2 for x in d2) / len(d2)) ** 0.5

print(f"\n{'need':<14} {'t=0':>9} {'t=59':>9} {'head_delta':>12} {'tail_delta':>12} {'curv':>10} v")
print("-" * 75)
for n in ALL:
    head = (trA[n][10] - trA[n][0]) / 10
    tail = (trA[n][59] - trA[n][50]) / 9
    c = curvature(trA[n])
    diff = abs(head - tail)
    v = "弯" if diff > 1e-5 else "~直"
    print(f"{n:<14} {trA[n][0]:>9.4f} {trA[n][59]:>9.4f} {head:>+12.7f} {tail:>+12.7f} {c:>10.2e} {v}")

# ════ 段 B ════
print(f"\n{'='*110}")
print("段 B: 带 select/serve, 400 tick")
print("=" * 110)
b = body_start_at_one(env={"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
trB = {n: [] for n in ALL}
serve_ticks = {n: [] for n in ALL}
sel_info = []
for t in range(400):
    snap = {n: b.state[n] for n in ALL}
    for n in ALL:
        trB[n].append(snap[n])
    chosen, dv = b.select()
    sel_info.append((t, chosen, dv))
    ok = b.serve(chosen) if dv > 0 else False
    if ok:
        serve_ticks[chosen].append(t)
    b.tick()

print("\n[第一次低于 target: tick]")
print(f"{'need':<14} {'target':>8} {'first<tick':>10}")
print("-" * 35)
for n in ALL:
    tgt = HOMEOSTATIC_NEEDS[n]["target"]
    first = next((t for t, v in enumerate(trB[n]) if v < tgt - 1e-9), None)
    print(f"{n:<14} {tgt:>8.2f} {first if first is not None else '未达':>10}")

print("\n[serve 间隔 (tick 距离) 与 总次数]")
print(f"{'need':<14} {'count':>6} {'gap_mean':>12} {'gap_median':>12}")
print("-" * 50)
for n in ALL:
    ts = serve_ticks[n]
    if not ts:
        print(f"{n:<14} {0:>6} -")
        continue
    if len(ts) == 1:
        print(f"{n:<14} {len(ts):>6} - (once)")
        continue
    gaps = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
    print(f"{n:<14} {len(ts):>6} {sum(gaps)/len(gaps):>12.1f} {sorted(gaps)[len(gaps)//2]:>12}")

# ════ 段 C ════
print(f"\n{'='*110}")
print("段 C: trigger pain severity=0.9 @ t=50, 看 social/exploration 被压")
print("=" * 110)
b = body_start_at_one(env={"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
trC = {n: [] for n in ALL}
modC_social = []
modC_explore = []
selC = []
for t in range(200):
    snap = {n: b.state[n] for n in ALL}
    for n in ALL:
        trC[n].append(snap[n])
    modC_social.append(b._drift_mod("social"))
    modC_explore.append(b._drift_mod("exploration"))
    if t == 50:
        b.trigger_event("pain", 0.9)
        print(f"  @ t=50 trigger_event(pain, 0.9) -> pain state={b.state['pain']}")
    chosen, dv = b.select()
    selC.append(chosen)
    b.serve(chosen) if dv > 0 else None
    b.tick()

print(f"\nsocial drift_mod mean (t=0-49, before trigger):   {sum(modC_social[0:50])/50:.4f}")
print(f"social drift_mod mean (t=51-100, after trigger):  {sum(modC_social[51:101])/50:.4f}")
print(f"explore drift_mod mean (t=0-49, before):          {sum(modC_explore[0:50])/50:.4f}")
print(f"explore drift_mod mean (t=51-100, after):         {sum(modC_explore[51:101])/50:.4f}")

def safe_avg_gap(ticks):
    if len(ticks) <= 1: return "N/A"
    return f"{(ticks[-1]-ticks[0])/(len(ticks)-1):.1f}"

print(f"\nsocial selected: total={selC.count('social')}, after_t50 avg_gap={safe_avg_gap([i for i,c in enumerate(selC[51:]) if c=='social'])}")
print(f"exploration selected: total={selC.count('exploration')}, after_t50 avg_gap={safe_avg_gap([i for i,c in enumerate(selC[51:]) if c=='exploration'])}")

print(f"\n[tick 45-65 逐字: pain 触发前后]")
print(f"{'t':>3} {'soc':>7} {'exp':>7} {'pain':>7} {'mod_s':>9} {'mod_e':>9} {'chosen':>10}")
print("-" * 65)
for t in range(45, 65):
    print(f"{t:>3} {trC['social'][t]:>7.3f} {trC['exploration'][t]:>7.3f} {trC['pain'][t]:>7.3f} "
          f"{modC_social[t]:>9.4f} {modC_explore[t]:>9.4f} {(selC[t] or ''):>10}")

# ════ 段 D ════
print(f"\n{'='*110}")
print("段 D: trigger safety severity=0.8 @ t=30, 看 social/exploration 被压+恢复")
print("=" * 110)
b = body_start_at_one(env={"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
trD = {n: [] for n in ALL}
selD = []
safety_served_at = []
for t in range(150):
    snap = {n: b.state[n] for n in ALL}
    for n in ALL:
        trD[n].append(snap[n])
    if t == 30:
        b.trigger_event("safety", 0.8)
        print(f"  @ t=30 trigger_event(safety, 0.8) -> safety state={b.state['safety']}")
    chosen, dv = b.select()
    selD.append(chosen)
    ok = b.serve(chosen) if dv > 0 else False
    if ok and chosen == "safety":
        safety_served_at.append(t)
    b.tick()

print(f"\nsafety served {len(safety_served_at)} times: {safety_served_at[:10]}")

print(f"\n[tick 25-50 逐字: safety 触发前后]")
print(f"{'t':>3} {'soc':>7} {'exp':>7} {'safe':>7} {'mod_s':>9} {'chosen':>10}")
print("-" * 60)
# 重跑一遍带 mod 计算
b2 = body_start_at_one(env={"luminance": 0, "sound": 0, "force": 0, "temperature": 293})
selD2 = []
for t in range(50):
    if t == 30:
        b2.trigger_event("safety", 0.8)
    if 25 <= t < 50:
        mod_soc = b2._drift_mod("social")
        chosen, _ = b2.select()
        selD2.append(chosen)
        print(f"{t:>3} {b2.state['social']:>7.3f} {b2.state['exploration']:>7.3f} {b2.state['safety']:>7.3f} "
              f"{mod_soc:>9.4f} {(chosen or ''):>10}")
    else:
        chosen, _ = b2.select()
        selD2.append(chosen)
    b2.serve(chosen) if chosen else None
    b2.tick()

# 血糖地板验证
print(f"\n{'='*110}")
print("附: glucose 糖异生地板验证 (从 5.0 纯 drift, 看是否停在 3.3)")
print("=" * 110)
b = 身体(节点数=24, 种子=1)
b.state['glucose'] = 5.0
for t in range(400):
    b._drift('glucose', 4)
    if t % 40 == 0 or t in (180, 199, 200, 250):
        print(f"  t={t:>3}: glucose={b.state['glucose']:.4f}")

# thermal dead band 验证
print(f"\n{'='*110}")
print("附: thermal dead band 验证 (400 tick, 看 thermal 是否每 3 tick 响)")
print("=" * 110)
b = 身体(节点数=24, 种子=1)   # 1 拍 = 4 分钟
thermal = []
sel_count = 0
for t in range(400):
    thermal.append(b.state['thermal'])
    chosen, d = b.select()
    if chosen == "thermal":
        sel_count += 1
    b.serve(chosen) if chosen else None
    b.tick()
print(f"thermal selected {sel_count} times / 400 ticks = 每 {400/sel_count:.1f} tick 响一次" if sel_count > 0 else "thermal 从未被选中")
print(f"thermal range: min={min(thermal):.4f}, max={max(thermal):.4f}, final={thermal[-1]:.4f}")
