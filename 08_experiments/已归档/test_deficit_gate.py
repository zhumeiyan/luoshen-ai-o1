# [已撤回] 2026-09-19 推演者归档：测旧身体接口，新架构已换
"""缺口测试: 低 heat 起点 + 加密 deficit 采样, 完整暴露 _sat 曲线。
"""

import sys, math
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

T   = HOMEOSTATIC_NEEDS["social"]["target"]            # 0.8
REC = HOMEOSTATIC_NEEDS["social"]["recovery_per_contact"]  # 0.35
RMAX = 0.7
K    = 0.03

sat = lambda x, K: x / (K + x) if x else 0.0

def run(pre_state, start_heat=0.0):
    body = 身体(节点数=24, 种子=1)
    body.state["social"]       = pre_state
    body.refractory_timer["social"] = 0
    body._social_heat          = start_heat

    deficit_raw = T - pre_state
    deficit_pos = max(0.0, deficit_raw)
    gate        = sat(deficit_pos, K)
    rech_theory = RMAX * gate

    s0 = body.state["social"]
    h0 = body._social_heat
    body.serve("social")
    s1 = body.state["social"]
    h1 = body._social_heat

    return dict(
        pre_state=s0, target=T, deficit_raw=deficit_raw, deficit_pos=deficit_pos,
        gate=gate, rech_theory=rech_theory,
        heat_pre=h0, heat_post=h1, rech_actual=h1 - h0,
        state_pre=s0, state_post=s1, state_delta=s1 - s0,
    )

# 采样: 覆盖 _sat 曲线 0 → 1 全程
#   deficit: 0, 0.005, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7
#   pre_state = T - deficit  (T=0.8)
samples = [0.0, 0.005, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7]
pre_states = [round(T - d, 6) for d in samples]

print(f"target={T}  recovery/contact={REC}  RMAX={RMAX}  K(半饱和)={K}  start_heat=0.0\n")
print(f"{'deficit':>8} {'pre_st':>8} {'gate':>8} {'rech_th':>9} {'rech_act':>9} {'heat_pre':>9} {'heat_post':>10} {'state_pre':>10} {'state_post':>11} {'Δstate':>8}")

rows = []
for ps in pre_states:
    r = run(ps, start_heat=0.0)
    rows.append(r)
    print(f"{r['deficit_pos']:>8.4f} {r['pre_state']:>8.2f} {r['gate']:>8.4f} {r['rech_theory']:>9.6f} {r['rech_actual']:>9.6f} {r['heat_pre']:>9.4f} {r['heat_post']:>10.4f} {r['state_pre']:>10.4f} {r['state_post']:>11.4f} {r['state_delta']:>+8.4f}")

# ── _sat 理论值核对表 ──
print(f"\n{'='*72}")
print("_sat(dep, 0.03) · 0.7 理论对照 (无封顶):\n")
print(f"{'deficit':>10} {'gate的理论':>12} {'recharge的理论':>16} {'说明':>20}")
print("-"*60)
for d in [0.0, 0.005, 0.01, 0.015, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    g = sat(d, K)
    rech = RMAX * g
    note = ""
    if d == 0:     note = "def=0 → 不充"
    elif d == 0.03: note = "def=K → 半饱和 (gate=0.5)"
    elif d == 0.3: note = "K*10 -> gate~0.91"
    elif d >= 0.7: note = "deep deficit -> near max"
    print(f"{d:>10.3f} {g:>12.6f} {rech:>16.6f} {note:>20}")
