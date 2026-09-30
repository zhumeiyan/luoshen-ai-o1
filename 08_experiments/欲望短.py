"""desire_short.py —— K=2（极短步），状态不进固定点，偏好主导选择。

K=2 时状态只走两步，离任何固定点都远。
两个 variants raw 同阶 (~0.2~0.4)，偏好在中间掰手腕。
加法偏好 + 衰减 + 不精确累积（AMP=1.2, alpha可负）。
跑 5000 圈，看偏好能不能稳定在某一端但偶尔翻盘。
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
N = len(variants)

def fwd_2(m, a, em, ea):
    """跑 2 步"""
    m1 = 0.46*m + 0.22*a + 0.32*em[0]
    a1 = 0.46*a + 0.22*m + 0.32*ea[0]
    m2 = 0.46*m1 + 0.22*a1 + 0.32*em[1 % len(em)]
    a2 = 0.46*a1 + 0.22*m1 + 0.32*ea[1 % len(ea)]
    return m2, a2

def run_experiment(ALPHA, DECAY, AMP, K, label, NCYCLES=5000):
    p = [1.0]*N
    m, a = 0.3, 0.3
    choices = []

    with open(f"desire_short_{label}.txt", "w") as f:
        for cycle in range(NCYCLES):
            scores = []
            for i in range(N):
                name, em, ea = variants[i]
                me, ae = fwd_2(m, a, em, ea)
                raw = math.sqrt((me-m)**2 + (ae-a)**2)
                boosted = raw + p[i]
                scores.append((boosted, raw, i, name, me, ae))
            scores.sort(reverse=True)
            bst, rst, wi, wn, wm, wa = scores[0]

            for i in range(N):
                p[i] *= (1 - DECAY)
            alpha_var = ALPHA * (1 + AMP * math.sin(cycle * 0.29173 + wi * 1.9))
            p[wi] += alpha_var

            m, a = wm, wa
            choices.append(wi)

            if cycle < 20 or cycle % 200 == 0:
                line = f"c{cycle:5d} V{wi}({wn[:6]}) raw={rst:.6f} boost={bst:.6f} alpha_v={alpha_var:+.4f} m={m:.8f} a={a:.8f} "
                line += f"p=[{' '.join(f'{x:.3f}' for x in p)}]"
                f.write(line + "\n")

    from collections import Counter
    c = Counter(choices)
    last1k = Counter(choices[4000:5000])

    max_streak_v = -1; max_streak_len = 0
    cur_v = -1; cur_len = 0
    for ch in choices:
        if ch == cur_v: cur_len += 1
        else:
            if cur_len > max_streak_len: max_streak_len = cur_len; max_streak_v = cur_v
            cur_v = ch; cur_len = 1
    if cur_len > max_streak_len: max_streak_len = cur_len; max_streak_v = cur_v

    switches = sum(1 for i in range(1, len(choices)) if choices[i] != choices[i-1])

    # 分段分布
    seg = []
    for s in range(0, NCYCLES, 1000):
        seg.append(dict(Counter(choices[s:s+1000])))

    groups = []
    cur = choices[0]; cnt = 1
    for i in range(1, len(choices)):
        if choices[i] == cur: cnt += 1
        else: groups.append((cur, cnt)); cur = choices[i]; cnt = 1
    groups.append((cur, cnt))

    print(f"\n=== {label}: ALPHA={ALPHA} DECAY={DECAY} AMP={AMP} K={K} ===")
    print(f"全{NCYCLES}圈: {dict(c)}")
    print(f"末1000: {dict(last1k)}")
    print(f"分段([0-999,1000-1999,...]): {seg}")
    print(f"最长连击: V{max_streak_v} x{max_streak_len}")
    print(f"切换次数: {switches}/{NCYCLES-1}")
    print(f"末10连击: {' '.join(f'V{v}x{n}' for v,n in groups[-10:])}")
    print(f"最终p: {dict((variants[i][0][:8], round(p[i],4)) for i in range(N))}")

    # 命中率和偏好差
    return choices, p

configs = [
    (0.25, 0.04, 1.2, 2, "K2_mid"),
    (0.30, 0.05, 1.5, 2, "K2_big"),
    (0.20, 0.03, 1.8, 2, "K2_neg"),
    (0.15, 0.08, 1.0, 2, "K2_fastdec"),
    (0.40, 0.02, 2.0, 2, "K2_strong"),
]

for ALPHA, DECAY, AMP, K, label in configs:
    run_experiment(ALPHA, DECAY, AMP, K, label, NCYCLES=5000)
