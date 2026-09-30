"""desire_want.py —— 偏好会退 + 累积不精确 + 无聊效应（未选中项偏好缓慢上升）。

每圈：
  1. 评分：boosted_i = raw_i + p[i]
  2. 选最大
  3. 所有 p[i] *= (1 - DECAY)        — 整体退
  4. 选中者：p[wi] += alpha_var       — 涨
  5. 未选中者：p[i] += BOREDOM         — 无聊补涨（没被选 → 越想要）

平衡：BOREDOM < alpha_var 均值 → 涨得比无聊多，总体偏好领先
      BOREDOM * N_cycles > alpha_gain → 无聊最终追上，翻转发生
      系统应在两者之间振荡：偏好领先后被打断，再建立，再打断。
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

def fwd(m, a, em, ea):
    return 0.46*m+0.22*a+0.32*em, 0.46*a+0.22*m+0.32*ea

def run_k(m, a, em, ea, K):
    for i in range(K):
        m, a = fwd(m, a, em[i%len(em)], ea[i%len(ea)])
    return m, a

def rng_seed(cycle, i):
    """确定性不精确数，频率无理数，不重复"""
    return math.sin(cycle * 0.29173 + i * 1.7) * math.cos(cycle * 0.13791 + i * 0.83)

def run_run(ALPHA, DECAY, AMP, BOREDOM, K, NCYCLES, label):
    p = [1.0]*N
    m, a = 0.3, 0.3
    choices = []

    with open(f"desire_want_{label}.txt", "w") as f:
        for cycle in range(NCYCLES):
            scores = []
            for i in range(N):
                name, em, ea = variants[i]
                me, ae = run_k(m, a, em, ea, K)
                raw = math.sqrt((me-m)**2 + (ae-a)**2)
                boosted = raw + p[i]
                scores.append((boosted, raw, i, name, me, ae))
            scores.sort(reverse=True)
            bst, rst, wi, wn, wm, wa = scores[0]

            # step 1: 退
            for i in range(N):
                p[i] *= (1 - DECAY)
            # step 2: 选中者 alpha
            alpha_var = ALPHA * (1 + AMP * rng_seed(cycle, wi))
            p[wi] += alpha_var
            # step 3: 未选中者 BOREDOM
            for i in range(N):
                if i != wi:
                    p[i] += BOREDOM

            m, a = wm, wa
            choices.append(wi)

            if cycle < 20 or cycle % 200 == 0:
                line = f"c{cycle:5d} V{wi}({wn[:6]}) raw={rst:.6f} boost={bst:.6f} alpha={alpha_var:+.4f} m={m:.8f} a={a:.8f} "
                line += f"p=[{' '.join(f'{x:.3f}' for x in p)}]"
                f.write(line + "\n")

    from collections import Counter
    c = Counter(choices)
    last1k = Counter(choices[NCYCLES-1000:])

    # 最长连击
    max_s = (0, -1)
    cur = choices[0]; cnt = 1
    for ch in choices[1:]:
        if ch == cur: cnt += 1
        else:
            if cnt > max_s[0]: max_s = (cnt, cur)
            cur = ch; cnt = 1
    if cnt > max_s[0]: max_s = (cnt, cur)

    switches = sum(1 for i in range(1, len(choices)) if choices[i] != choices[i-1])

    segs = [dict(Counter(choices[s:s+1000])) for s in range(0, NCYCLES, 1000)]

    # 找每段"主导"（最多圈数）
    leaders = []
    for seg in segs:
        if seg:
            leader = max(seg, key=seg.get)
            leaders.append((leader, seg[leader]))

    groups = []
    cur = choices[0]; cnt = 1
    for ch in choices[1:]:
        if ch == cur: cnt += 1
        else: groups.append((cur, cnt)); cur = ch; cnt = 1
    groups.append((cur, cnt))

    print(f"\n=== {label}: A={ALPHA} D={DECAY} AMP={AMP} B={BOREDOM} K={K} ===")
    print(f"全: {dict(c)}")
    print(f"末1000: {dict(last1k)}")
    print(f"分段主导: {leaders}")
    print(f"最长连击: V{max_s[1]} x{max_s[0]}")
    print(f"切换次数: {switches}/{NCYCLES-1}")
    print(f"末10连击: {' '.join(f'V{v}x{n}' for v,n in groups[-10:])}")
    print(f"最终p: {dict((variants[i][0][:8], round(p[i],4)) for i in range(N))}")
    return choices, p

configs = [
    # ALPHA, DECAY, AMP, BOREDOM, K, NCYCLES, label
    (0.15, 0.05, 1.0, 0.06, 30, 10000, "stab1"),
    (0.12, 0.04, 1.2, 0.05, 30, 10000, "stab2"),
    (0.10, 0.05, 1.5, 0.04, 30, 10000, "stab3"),
    (0.18, 0.06, 0.8, 0.07, 30, 10000, "stab4"),
    (0.20, 0.05, 1.8, 0.06, 30, 10000, "stab5"),
    (0.08, 0.03, 2.5, 0.05, 30, 10000, "stab6"),
]

for ALPHA, DECAY, AMP, BOREDOM, K, NCYCLES, label in configs:
    run_run(ALPHA, DECAY, AMP, BOREDOM, K, NCYCLES, label)
