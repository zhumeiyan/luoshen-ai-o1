# [已撤回] 2026-09-19 推演者归档：测旧身体接口，新架构已换
"""长跑测试: 12 需求 + 热/血糖 一起, 跨压制结节全开。
跑 300 tick (900 min = 15h), 每 tick 记录:
  · 12 需求 state
  · 每个需求的 drift_mod
  · select 选了谁, serve 是否成功
  · Δ 的变化 (判断弯不弯)
"""

import sys, math
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

ALL = ["thermal", "glucose", "social", "exploration", "sexual",
       "rest", "safety", "soothe", "pain", "hunger", "thirst", "touch"]

def long_run(ticks=300, dt=3, env=None, label="run"):
    body = 身体(节点数=24, 种子=1)
    if env:
        body.environment.update(env)

    # 记录
    trace = {n: [] for n in ALL}
    mods  = {n: [] for n in ALL if n not in ("thermal", "glucose")}
    selects = []
    serves  = []
    deltas  = {n: [] for n in ALL}

    for t in range(ticks):
        # 记录 tick 前的 state
        snap = {n: body.state[n] for n in ALL}
        for n in ALL:
            trace[n].append(snap[n])

        # 记录 mod
        for n in mods:
            mods[n].append(body._drift_mod(n))

        # select + serve
        chosen, dv = body.select()
        selects.append(chosen)
        ok = body.serve(chosen) if dv > 0 else False
        serves.append((chosen, ok))

        body.tick()

        # Δ (tick 前后)
        for n in ALL:
            deltas[n].append(body.state[n] - snap[n])

    return trace, mods, selects, serves, deltas, body

# ── 场景 1: 正常环境 (293K 冷室, 无刺激) ──
print("=" * 100)
print("场景 A: 正常凉室 (293K, sound=0, light=0, force=0)")
print("=" * 100)

tr, md, sel, srv, dl, body = long_run(300, env={"光": 0, "声": 0, "力": 0, "温": 293})

# ── 想知道各级 "原来匀速的, 现在弯不弯" ──
# 连续 delta 的标准差: 纯直线则相邻 Δ 差=0, 弯则有变化。
# 检测: 看 Δ[t+1] - Δ[t] 在整段中有多次变号
print("\n[弯度检测: 相邻 Δ 差的标准差 (大=弯, ~0=直)]")
print(f"{'need':<14} {'Δ均值':>12} {'Δ-bend(std)':>14} {'判断':>8}")
print("-"*55)
for n in ALL:
    ds = dl[n][1:]  # skip 1st (serve/rest)
    if len(ds) < 3:
        continue
    d_mean = sum(ds) / len(ds)
    # 二阶差分: 弯度
    d2 = [ds[i+1] - ds[i] for i in range(len(ds)-1)]
    d2_std = (sum((x - sum(d2)/len(d2))**2 for x in d2) / len(d2)) ** 0.5
    verdict = "弯" if d2_std > 1e-6 else "直"
    print(f"{n:<14} {d_mean:>+12.7f} {d2_std:>14.2e} {verdict:>8}")

# ── 第一响/触发顺序: tick 第一次 < target 的时刻 ──
print("\n[第一次低于 target 的时刻]")
print(f"{'need':<14} {'target':>8} {'首次<tick':>10} {'当时state':>12}")
print("-"*48)
for n in ALL:
    tgt = HOMEOSTATIC_NEEDS[n]["target"]
    first = next((t for t, v in enumerate(tr[n]) if v < tgt - 1e-9), None)
    if first is not None:
        print(f"{n:<14} {tgt:>8.2f} {first:>10} {tr[n][first]:>12.4f}")
    else:
        print(f"{n:<14} {tgt:>8.2f} {'未达':>10} {tr[n][-1]:>12.4f}")

# ── 谁被选了多少次? ──
from collections import Counter
sc = Counter(sel)
print("\n[select 分布 (被看上的次数)]")
print(f"{'need':<14} {'次数':>6} {'占比':>8}")
print("-"*30)
for n in ALL:
    cnt = sc.get(n, 0)
    print(f"{n:<14} {cnt:>6} {cnt/len(sel)*100:>7.1f}%")

# ── serve 成功 vs 被 ref 挡 ──
served_ok = Counter()
served_fail = Counter()
for name, ok in srv:
    if ok:
        served_ok[name] += 1
    else:
        served_fail[name] += 1

print("\n[serve 结果 (成功 vs 被 ref 挡)]")
print(f"{'need':<14} {'成功':>6} {'被挡':>6}")
print("-"*30)
for n in ALL:
    print(f"{n:<14} {served_ok.get(n,0):>6} {served_fail.get(n,0):>6}")

# ── 社交/探索 被压后多久回来? ──
# 找 safety=0 被人为逼低的区间 → 看 social 弯度
# 不人为逼, 自然跑: 看 social 在 safety 自然低的时段 Δ 变化
print("\n[social 在 safety 低时的弯度]")
safety = tr["safety"]
social = tr["social"]
so_far_compressed = 0
for t in range(len(safety)):
    if safety[t] < 0.85:  # 怕
        so_far_compressed += 1
print(f"  safety<0.85 的 tick: {so_far_compressed}/{len(safety)}")
if so_far_compressed > 5:
    # 在这些 tick 看 social Δ
    bad_deltas = [dl["social"][t] for t in range(1, len(safety)) if safety[t] < 0.85]
    good_deltas = [dl["social"][t] for t in range(1, len(safety)) if safety[t] >= 0.85]
    if bad_deltas and good_deltas:
        print(f"  social Δ: 怕低时 mean={sum(bad_deltas)/len(bad_deltas):+.7f} | 安全时 mean={sum(good_deltas)/len(good_deltas):+.7f}")
        print(f"  怕压加速比: {(sum(bad_deltas)/len(bad_deltas))/(sum(good_deltas)/len(good_deltas)):.3f}")

# ── mod 直方图: 统计各需求 mod 落在 [1.0, 1.2, 1.5, 2.0+] 区间多久 ──
print("\n[drift_mod 分布: 压多久?]")
print(f"{'need':<14} {'=1.00':>7} {'1.0-1.2':>8} {'1.2-1.5':>8} {'1.5-2.0':>8} {'2.0+':>7}  {'max':>7}")
print("-"*65)
for n in md:
    vs = md[n]
    total = len(vs)
    b1 = sum(1 for v in vs if abs(v - 1.0) < 0.001)
    b2 = sum(1 for v in vs if 1.0 <= v < 1.2)
    b3 = sum(1 for v in vs if 1.2 <= v < 1.5)
    b4 = sum(1 for v in vs if 1.5 <= v < 2.0)
    b5 = sum(1 for v in vs if v >= 2.0)
    print(f"{n:<14} {b1/total*100:>6.1f}% {b2/total*100:>7.1f}% {b3/total*100:>7.1f}% {b4/total*100:>7.1f}% {b5/total*100:>6.1f}%  {max(vs):>7.3f}")


# ── 场景 B: 极端热 + 低血糖同时 ──
print("\n\n" + "=" * 100)
print("场景 B: 热室 (thermal=313K) + 同时低血糖 (pinned glucose=2.5)")
print("=" * 100)

# 强制 thermal 高 / glucose 低 的环境
body2 = 身体(节点数=24, 种子=1)
body2.environment.update({"光": 0, "声": 0, "力": 0, "温": 31})
tr2, md2, sel2, srv2, dl2, _ = long_run(300, env={"光": 0, "声": 0, "力": 0, "温": 313})
# 另外在 tick 里钉 glucose 低
bodyB = 身体(节点数=24, 种子=1)
bodyB.environment.update({"光": 0, "声": 0, "力": 0, "温": 313})
trB = {n: [] for n in ALL}
mdB = {n: [] for n in ALL if n not in ("thermal","glucose")}
selB, srvB, dlB = [], [], {n: [] for n in ALL}
for t in range(300):
    bodyB.state["glucose"] = 2.5  # 持续低血糖
    snap = {n: bodyB.state[n] for n in ALL}
    for n in ALL:
        trB[n].append(snap[n])
    for n in mdB:
        mdB[n].append(bodyB._drift_mod(n))
    c, dv = bodyB.select()
    selB.append(c)
    ok = bodyB.serve(c) if dv > 0 else False
    srvB.append((c, ok))
    bodyB.tick()
    for n in ALL:
        dlB[n].append(bodyB.state[n] - snap[n])

print("\n[低血糖+热 时的 drift_mod 分布]")
print(f"{'need':<14} {'mean':>7} {'max':>7} {'min':>7} {'压>1.5':>8}")
print("-"*50)
for n in mdB:
    vs = mdB[n]
    press = sum(1 for v in vs if v > 1.5)
    print(f"{n:<14} {sum(vs)/len(vs):>7.3f} {max(vs):>7.3f} {min(vs):>7.3f} {press/len(vs)*100:>7.1f}%")

print("\n[低血糖+热 时 select 分布]")
scB = Counter(selB)
for n in ALL:
    cnt = scB.get(n, 0)
    print(f"  {n:<14} {cnt:>4} ({cnt/len(selB)*100:>5.1f}%)")

print("\n[低血糖+热 时首次 < target]")
for n in ALL:
    tgt = HOMEOSTATIC_NEEDS[n]["target"]
    first = next((t for t, v in enumerate(trB[n]) if v < tgt - 1e-9), None)
    if first is not None:
        print(f"  {n:<14} tick {first:>4}  当时 state={trB[n][first]:.4f}")
    else:
        print(f"  {n:<14} 未达 target  state_end={trB[n][-1]:.4f}")

# ── 逐 tick 前 30 个 (看细节)
print("\n\n[前 50 tick 逐次明细 (normal room)]")
print(f"{'t':>3}  {'soc':>6}  {'exp':>6}  {'rest':>6}  {'safe':>6}  {'hunger':>7}  {'thirst':>7}  {'gluc':>6}  {'therm':>6}  {'chosen':>8}")
print("-"*85)
for t in range(50):
    print(f"{t:>3}  {tr['social'][t]:>6.3f}  {tr['exploration'][t]:>6.3f}  {tr['rest'][t]:>6.3f}  "
          f"{tr['safety'][t]:>6.3f}  {tr['hunger'][t]:>7.3f}  {tr['thirst'][t]:>7.3f}  "
          f"{tr['glucose'][t]:>6.3f}  {tr['thermal'][t]:>6.3f}  {str(sel[t]):>8}")
