"""desire_flex.py —— 偏好会退 + 可负累积 + 部分步。

关键：alpha_var = ALPHA * (1 + AMP*sin(...))，AMP=2.0
      sin ∈ [-1, 1] → alpha_var ∈ [-ALPHA, +3*ALPHA]
      选中 V1 时，V1 的偏好可能涨也可能跌。
      锁死后如果连续选 V1 且 alpha 为负，V1 会自己掉下去——锁被自己的动作打断。

加法偏好：boosted_i = raw_i + p[i]
选中后：所有 p *= (1-DECAY)，然后 p[wi] += alpha_var。
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

def run_k_steps(m, a, em, ea, K):
    for i in range(K):
        m, a = fwd(m, a, em[i%len(em)], ea[i%len(ea)])
    return m, a

def experiment(ALPHA, DECAY, AMP, K, label):
    p = [1.0]*N_VAR
    m, a = 0.3, 0.3
    choices = []
    streak_v = None; streak_len = 0
    max_streak = (0, -1)

    with open(f"desire_flex_{label}.txt", "w") as f:
        for cycle in range(1000):
            scores = []
            for i in range(N_VAR):
                name, em, ea = variants[i]
                me, ae = run_k_steps(m, a, em, ea, K)
                raw = math.sqrt((me-m)**2 + (ae-a)**2)
                boosted = raw + p[i]
                scores.append((boosted, raw, i, name, me, ae))

            scores.sort(reverse=True)
            bst, rst, wi, wn, wm, wa = scores[0]

            for i in range(N_VAR):
                p[i] *= (1 - DECAY)
            alpha_var = ALPHA * (1 + AMP * math.sin(cycle * 0.29173 + wi * 1.9))
            p[wi] += alpha_var

            m, a = wm, wa
            choices.append(wi)

            if wi == streak_v:
                streak_len += 1
            else:
                if streak_len > max_streak[0]:
                    max_streak = (streak_len, streak_v)
                streak_v = wi; streak_len = 1

            line = f"cycle {cycle:4d} chose=V{wi}({wn:14s}) raw={rst:.10f} boost={bst:.10f} "
            line += f"alpha_v={alpha_var:+.6f} m={m:.10f} a={a:.10f} "
            line += f"p=[{' '.join(f'{x:.4f}' for x in p)}]"
            f.write(line + "\n")

    if streak_len > max_streak[0]:
        max_streak = (streak_len, streak_v)

    from collections import Counter
    c = Counter(choices)
    last200 = Counter(choices[800:])

    switches = sum(1 for i in range(1, len(choices)) if choices[i] != choices[i-1])

    # 分布末段 p vs 全段偏好强度
    # 统计 raw 平均值
    with open(f"desire_flex_{label}.txt") as f:
        lines = f.readlines()
    raw_vals = [float(l.split('raw=')[1].split()[0]) for l in lines if 'chose=' in l]
    # print分段：
    q1 = Counter(choices[0:333]); q2 = Counter(choices[333:666]); q3 = Counter(choices[666:1000])

    groups = []
    cur = choices[0]; cnt = 1
    for i in range(1, len(choices)):
        if choices[i] == cur: cnt += 1
        else: groups.append((cur, cnt)); cur = choices[i]; cnt = 1
    groups.append((cur, cnt))

    print(f"\n=== {label}: ALPHA={ALPHA} DECAY={DECAY} AMP={AMP} K={K} ===")
    print(f"全: V0={c.get(0,0)} V1={c.get(1,0)} V2={c.get(2,0)} V3={c.get(3,0)} V4={c.get(4,0)}")
    print(f"三分: Q1={dict(q1)} Q2={dict(q2)} Q3={dict(q3)}")
    print(f"最长连击: V{max_streak[1]} x{max_streak[0]}")
    print(f"切换: {switches}/999")
    print(f"末10连击: {' '.join(f'V{v}x{n}' for v,n in groups[-10:])}")
    print(f"最后8圈: {choices[-8:]}")
    print(f"最终p: {dict((variants[i][0], round(p[i],4)) for i in range(N_VAR))}")
    return choices, p

configs = [
    # ALPHA, DECAY, AMP, K, label
    (0.4, 0.05, 2.0, 55, "neg_alpha_full"),
    (0.3, 0.03, 2.0, 55, "neg_alpha_lowdec"),
    (0.5, 0.08, 2.0, 55, "neg_alpha_highamp"),
    (0.2, 0.02, 3.0, 55, "neg_alpha_bigamp"),
    (0.3, 0.05, 2.0, 55, "neg_alpha_bal"),
]

for ALPHA, DECAY, AMP, K, label in configs:
    experiment(ALPHA, DECAY, AMP, K, label)
