# [已撤回] 2026-09-19 推演者归档：测旧身体接口，新架构已换
"""补测: 压边修正 + 松边修正 + 困→社交新边
协议: 单源、单目标、其余在 target、1 拍
压源: 怕/疼/困/饿/渴 (五个真压源)
松源: 安抚/被抱 (仅两个真松源)
不含: 社交/探索 作为源 (它们是被压目标, 不是源)
"""

import sys
sys.path.insert(0, '.')
from 身体稳态 import 身体

ALL = ["thermal","glucose","social","exploration","sexual","rest","safety","soothe","pain","hunger","thirst","touch"]

def body_at_target():
    """所有需求在 target, thermal/glucose 也在 target"""
    b = 身体(节点数=24, 种子=1)
    # __init__ 已把 state[n] 设为 target, 这里只是显式再保一遍
    for n in ALL:
        b.state[n] = HOMEOSTATIC_NEEDS[n]["target"]
    return b

# ============================================================
print("=" * 70)
print("一. 压边真源测试 (怕/疼/困/饿/渴)")
print("协议: 单源=0, 单目标测 drift_mod, 其余在 target, 不打 tick")
print("=" * 70)

COMPRESSION_EDGES = [
    # (源, =0, [目标列表], 设计说明)
    ("safety",  "怕=0",  ["social", "exploration"], "怕压社交/探索"),
    ("pain",    "疼=0",  ["social", "exploration", "touch", "sexual"], "疼压社交/探索/被抱/性"),
    ("rest",    "困=0",  ["social", "exploration", "sexual"], "困压社交/探索/性"),
    ("thirst",  "渴=0",  ["social", "exploration"], "渴压社交/探索"),
    ("hunger",  "饿=0",  ["social", "exploration"], "饿压社交/探索"),
]

print(f"\n{'压源':<10} {'目标':<12} {'压前dmod':>10} {'压后dmod':>10} {'压制量':>10} {'说明'}")
print("-" * 75)
for src, label, targets, desc in COMPRESSION_EDGES:
    b = body_at_target()
    pre_all = {t: b._drift_mod(t) for t in targets}
    b.state[src] = 0.0
    post_all = {t: b._drift_mod(t) for t in targets}
    for t in targets:
        damp = pre_all[t] - post_all[t]  # 正数=被压
        print(f"{label:<10} {t:<12} {pre_all[t]:>10.4f} {post_all[t]:>10.4f} {damp:>+10.4f}  {desc}")

# 困→社交 新边独立标定
print("\n[困→社交 新增边 独立标定]")
b = body_at_target()
pre_s = b._drift_mod('social')
b.state['rest'] = 0.0
post_s = b._drift_mod('social')
print(f"  rest=0: social dmod {pre_s:.4f} → {post_s:.4f}  压制量={pre_s-post_s:+.4f}")
b2 = body_at_target()
b2.state['rest'] = 0.5  # 半压
post_s2 = b2._drift_mod('social')
print(f"  rest=0.5: social dmod {pre_s:.4f} → {post_s2:.4f}  压制量={pre_s-post_s2:+.4f}")
b3 = body_at_target()
b3.state['rest'] = 0.8  # 微压
post_s3 = b3._drift_mod('social')
print(f"  rest=0.8: social dmod {pre_s:.4f} → {post_s3:.4f}  压制量={pre_s-post_s3:+.4f}")

# ============================================================
print("\n" + "=" * 70)
print("二. 松边真源测试 (安抚/被抱 仅两个)")
print("协议: 单源=1.0, 单目标测 drift_mod, 其余在 target")
print("=" * 70)

RELEASE_EDGES = [
    ("soothe", "安抚=1.0", ["safety", "pain", "rest"], "安抚松怕/疼/困"),
    ("touch",  "被抱=1.0", ["safety", "pain", "rest"], "被抱松怕/疼/困"),
]

print(f"\n{'松源':<10} {'目标':<12} {'松前dmod':>10} {'松后dmod':>10} {'松量':>10}")
print("-" * 60)
for src, label, targets, desc in RELEASE_EDGES:
    b = body_at_target()
    pre_all = {t: b._drift_mod(t) for t in targets}
    b.state[src] = 1.0
    post_all = {t: b._drift_mod(t) for t in targets}
    for t in targets:
        damp = pre_all[t] - post_all[t]  # 正数=被松(掉得更慢)
        print(f"{label:<10} {t:<12} {pre_all[t]:>10.4f} {post_all[t]:>10.4f} {damp:>+10.4f}  {desc}")

# 补点: 0.8/0.9/1.0 强度梯度
print("\n[松 补点: 安抚/被抱 三强度]")
for sev in [0.8, 0.9, 1.0]:
    print(f"\n-- 强度 {sev} --")
    b_s = body_at_target(); b_s.state['soothe'] = sev
    b_t = body_at_target(); b_t.state['touch'] = sev
    for t in ['safety', 'pain', 'rest']:
        pre = body_at_target()._drift_mod(t)
        ds = b_s._drift_mod(t) - pre
        dt = b_t._drift_mod(t) - pre
        print(f"  {t:<8}: soothe松{ds:+.4f}  /  touch松{dt:+.4f}")

# ============================================================
print("\n" + "=" * 70)
print("三. 第 13 组 (安抚=1.0 + 被抱=1.0 同时)")
print("=" * 70)

b13 = body_at_target()
pre13 = {t: b13._drift_mod(t) for t in ['safety','pain','rest']}
b13.state['soothe'] = 1.0
b13.state['touch'] = 1.0
post13 = {t: b13._drift_mod(t) for t in ['safety','pain','rest']}
for t in ['safety','pain','rest']:
    print(f"  {t}: {pre13[t]:.4f} → {post13[t]:.4f}  松量={pre13[t]-post13[t]:+.4f}")

print(f"\n叠加验证: safety damp(安抚)+damp(被抱) 应等于 damp(同时)")
b_s = body_at_target(); b_s.state['soothe'] = 1.0
b_t = body_at_target(); b_t.state['touch'] = 1.0
safety_damp_soothe = 1.0 - b_s._drift_mod('safety')
safety_damp_touch  = 1.0 - b_t._drift_mod('safety')
safety_damp_both   = pre13['safety'] - post13['safety']
print(f"  safety: 安抚松{safety_damp_soothe:.4f} + 被抱松{safety_damp_touch:.4f} = {safety_damp_soothe+safety_damp_touch:.4f}")
print(f"  safety: 同时松{safety_damp_both:.4f}")
print(f"  差 (应≈0, 由 cap min 0.90 截断所致): {(safety_damp_soothe+safety_damp_touch)-safety_damp_both:+.4f}")

# ============================================================
print("\n" + "=" * 70)
print("四. 删除验证: 社交/探索 不是压源 (应当无效果)")
print("=" * 70)

for src, label in [("social", "社交=0"), ("exploration", "探索=0")]:
    b = body_at_target()
    b.state[src] = 0.0
    print(f"\n{label}:")
    for t in ALL:
        if t == src:
            continue
        d = b._drift_mod(t)
        if abs(d - 1.0) > 1e-6:
            print(f"  {t}: dmod={d:.4f} (非1.0!)" )
    print(f"  (所有目标 dmod 应均为 1.0 — 社交/探索不是压源)")
