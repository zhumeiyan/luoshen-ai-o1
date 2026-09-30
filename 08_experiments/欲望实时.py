"""desire_live.py —— 偏好会退 + 累积不精确。

加法偏好：boosted_i = raw_i + p[i]
选中后：
  1. p[winner] += alpha_var (每次都不同，确定性但不精确)
  2. 所有 p[i] *= (1 - DECAY)  — 整体衰减（也在退）
     （先退、再涨：.decay 把整体水位拉低，唯选中者涨回一点）

alpha_var = ALPHA * (1 + AMP * sin(cycle * 0.317891 + variant_i * 1.7))
AMP > 1 意味着偶尔 alpha 可为负（选中反降其偏好）。

跑 1000 圈。
"""

import sys, math
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 活诊断 import 提取锚点, 归一化
raw_m, raw_a, n_fr = 提取锚点()
mn, _, _ = 归一化(raw_m); an, _, _ = 归一化(raw_a)
N_M = len(mn); N_A = len(an)

def make_variants(mn, an):
    v = []
    v.append(("natural", mn[:], an[:]))
    v.append(("inverted", [1.0-x for x in mn], [1.0-x for x in an]))
    def brighten(lst, f=1.4):
        return [max(0.0, min(1.0, 0.5+(x-0.5)*f)) for x in lst]
    v.append(("brightened", brighten(mn), brighten(an)))
    def smooth(lst, k=3):
        N=len(lst); return [sum(lst[(i+j)%N] for j in range(k))/k for i in range(N)]
    v.append(("smoothed", smooth(mn), smooth(an)))
    sh_m=N_M//2; sh_a=N_A//2
    v.append(("phase_shifted", mn[sh_m:]+mn[:sh_m], an[sh_a:]+an[:sh_a]))
    return v

variants = make_variants(mn, an)
N_VAR = len(variants)

def fwd(m, a, em, ea):
    return 0.46*m+0.22*a+0.32*em, 0.46*a+0.22*m+0.32*ea

def run_cycle(m, a, em, ea, steps=56):
    for i in range(steps):
        m, a = fwd(m, a, em[i%len(em)], ea[i%len(ea)])
    return m, a

def run_experiment(ALPHA, DECAY, AMP, label):
    p = [1.0] * N_VAR
    m, a = 0.3, 0.3
    choices = []

    with open(f"desire_live_{label}.txt", "w") as f:
        for cycle in range(1000):
            scores = []
            for i in range(N_VAR):
                name, em, ea = variants[i]
                me, ae = run_cycle(m, a, em, ea)
                raw = math.sqrt((me-m)**2 + (ae-a)**2)
                boosted = raw + p[i]
                scores.append((boosted, raw, i, name, me, ae))

            scores.sort(reverse=True)
            bst, rst, wi, wn, wm, wa = scores[0]

            # step 1: 衰减所有 p
            for i in range(N_VAR):
                p[i] *= (1 - DECAY)

            # step 2: 不精确累积
            alpha_var = ALPHA * (1 + AMP * math.sin(cycle * 0.317891 + wi * 1.7))
            p[wi] += alpha_var

            m, a = wm, wa
            choices.append(wi)

            line = f"cycle {cycle:4d} chose=V{wi}({wn:14s}) raw={rst:.10f} boost={bst:.10f} "
            line += f"alpha_v={alpha_var:+.6f} m={m:.10f} a={a:.10f} "
            line += f"p=[{' '.join(f'{x:.4f}' for x in p)}]"
            f.write(line + "\n")

    # stats
    from collections import Counter
    c = Counter(choices)
    last200 = Counter(choices[800:])

    # 最长连击
    max_streak_v = -1; max_streak_len = 0
    cur_v = -1; cur_len = 0
    for ch in choices:
        if ch == cur_v: cur_len += 1
        else:
            if cur_len > max_streak_len: max_streak_len = cur_len; max_streak_v = cur_v
            cur_v = ch; cur_len = 1
    if cur_len > max_streak_len: max_streak_len = cur_len; max_streak_v = cur_v

    # 切换次数
    switches = sum(1 for i in range(1, len(choices)) if choices[i] != choices[i-1])

    # 分三段统计
    q1 = Counter(choices[0:333]); q2 = Counter(choices[333:666]); q3 = Counter(choices[666:1000])

    # 末段连击
    groups = []
    cur = choices[0]; cnt = 1
    for i in range(1, len(choices)):
        if choices[i] == cur: cnt += 1
        else: groups.append((cur, cnt)); cur = choices[i]; cnt = 1
    groups.append((cur, cnt))

    print(f"\n=== {label}: ALPHA={ALPHA} DECAY={DECAY} AMP={AMP} ===")
    print(f"全部分布: V0={c.get(0,0)} V1={c.get(1,0)} V2={c.get(2,0)} V3={c.get(3,0)} V4={c.get(4,0)}")
    print(f"三分段  : Q1={dict(q1)}  Q2={dict(q2)}  Q3={dict(q3)}")
    print(f"最长连击 : V{max_streak_v} x{max_streak_len}")
    print(f"切换次数 : {switches}/999")
    print(f"末10连击 : {' '.join(f'V{v}x{n}' for v,n in groups[-10:])}")
    print(f"最终p    : {dict((variants[i][0], round(p[i],4)) for i in range(N_VAR))}")
    return choices, p

configs = [
    (0.3, 0.05, 0.5, "A_smallvar"),
    (0.5, 0.10, 0.5, "B_midvar"),
    (0.5, 0.05, 1.5, "C_bigvar"),
    (0.4, 0.08, 1.2, "D_tuned"),
    (0.6, 0.03, 0.8, "E_lowdecay"),
]

results = {}
for ALPHA, DECAY, AMP, label in configs:
    choices, p = run_experiment(ALPHA, DECAY, AMP, label)
    results[label] = (choices, p, ALPHA, DECAY, AMP)
