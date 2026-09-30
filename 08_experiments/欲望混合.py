"""desire_hybrid.py —— 偏好会退 + 累积不精确 + 部分步（不收敛到固定点）。

核心：每圈跑的步数短（如 10~20，不是 56），状态停在两固定点之间。
两个变体都有正 raw，偏好在中间真正起作用。

加法偏好：boosted_i = raw_i + p[i]
选中：
  1. 所有 p *= (1 - DECAY)      — 退
  2. p[wi] += alpha_var          — 涨（确定性但不精确，AMP*正弦无公度频率）

状态更新：跑 K 步（不跑满 56）。
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
    with open(f"desire_hybrid_{label}.txt", "w") as f:
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
            line = f"cycle {cycle:4d} chose=V{wi}({wn:14s}) raw={rst:.10f} boost={bst:.10f} m={m:.10f} a={a:.10f} p=[{' '.join(f'{x:.4f}' for x in p)}]"
            f.write(line + "\n")

    from collections import Counter
    c = Counter(choices)
    last200 = Counter(choices[800:])
    max_streak_v = -1; max_streak_len = 0
    cur_v = -1; cur_len = 0
    for ch in choices:
        if ch == cur_v: cur_len += 1
        else:
            if cur_len > max_streak_len: max_streak_len = cur_len; max_streak_v = cur_v
            cur_v = ch; cur_len = 1
    if cur_len > max_streak_len: max_streak_len = cur_len; max_streak_v = cur_v

    switches = sum(1 for i in range(1, len(choices)) if choices[i] != choices[i-1])

    groups = []
    cur = choices[0]; cnt = 1
    for i in range(1, len(choices)):
        if choices[i] == cur: cnt += 1
        else: groups.append((cur, cnt)); cur = choices[i]; cnt = 1
    groups.append((cur, cnt))

    print(f"\n=== {label}: ALPHA={ALPHA} DECAY={DECAY} AMP={AMP} K={K} ===")
    print(f"全部分布: V0={c.get(0,0)} V1={c.get(1,0)} V2={c.get(2,0)} V3={c.get(3,0)} V4={c.get(4,0)}")
    print(f"末200圈: {dict(last200)}")
    print(f"最长连击: V{max_streak_v} x{max_streak_len}")
    print(f"切换次数: {switches}/999")
    print(f"末10连击: {' '.join(f'V{v}x{n}' for v,n in groups[-10:])}")
    print(f"最后5圈: {choices[-5:]}")
    print(f"最终 p: {dict((variants[i][0], round(p[i],4)) for i in range(N_VAR))}")
    return choices, p

# 试不同步数 + 参数
configs = [
    (0.3, 0.05, 0.5, 15, "K15_smallvar"),
    (0.3, 0.05, 0.5, 30, "K30_smallvar"),
    (0.3, 0.05, 0.5, 10, "K10_smallvar"),
    (0.5, 0.08, 0.7, 15, "K15_midvar"),
    (0.4, 0.06, 1.0, 20, "K20_bigvar"),
]

for ALPHA, DECAY, AMP, K, label in configs:
    experiment(ALPHA, DECAY, AMP, K, label)
