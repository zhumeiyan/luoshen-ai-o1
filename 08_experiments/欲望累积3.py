"""desire_accum3.py —— ALPHA 超过 raw 差，看偏好能否锁定。

乘法：boosted = raw * p[i]
ALPHA = 1.0 (>> 0.832)
让 V1 在 cycle 0 先选，p1=2.0，p2=1.0。
若 p1 - p2 > (V2_raw_at_state_A) = 0.832，V1 可以赢下 V2 的固定点。
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

def run_cycle(m, a, em, ea, steps=56):
    for i in range(steps):
        m, a = fwd(m, a, em[i%len(em)], ea[i%len(ea)])
    return m, a

# 试 4 个 ALPHA 值
for ALPHA in [0.1, 0.5, 1.0, 2.0]:
    p = [1.0]*N
    m, a = 0.3, 0.3
    first_lock_cycle = None
    lock_variant = None

    for cycle in range(1000):
        scores = []
        for i in range(N):
            name, em, ea = variants[i]
            me, ae = run_cycle(m, a, em, ea)
            raw = math.sqrt((me-m)**2 + (ae-a)**2)
            boosted = raw * p[i]  # 乘法
            scores.append((boosted, raw, i, name, me, ae))
        scores.sort(reverse=True)
        bst, rst, wi, wn, wm, wa = scores[0]
        p[wi] += ALPHA
        m, a = wm, wa

        if cycle > 10 and first_lock_cycle is None:
            # 检查最近 10 圈是否都选同一个
            pass

    # 统计最后 500 圈
    # 重新跑以记录
    p = [1.0]*N
    m, a = 0.3, 0.3
    choices = []
    for cycle in range(1000):
        scores = []
        for i in range(N):
            name, em, ea = variants[i]
            me, ae = run_cycle(m, a, em, ea)
            raw = math.sqrt((me-m)**2 + (ae-a)**2)
            boosted = raw * p[i]
            scores.append((boosted, raw, i, name, me, ae))
        scores.sort(reverse=True)
        bst, rst, wi, wn, wm, wa = scores[0]
        p[wi] += ALPHA
        m, a = wm, wa
        choices.append(wi)

    from collections import Counter
    last500 = Counter(choices[500:])
    first20 = choices[:20]

    # 找锁定圈（连续 50 圈选同一个）
    lock_cycle = None
    lock_v = '?'
    for i in range(50, len(choices)):
        if len(set(choices[i-50:i])) == 1:
            lock_cycle = i - 50
            lock_v = choices[i-1]
            break

    print(f"ALPHA={ALPHA}: first_lock={lock_cycle} V{lock_v}  last500_counts={dict(last500)}  first20={first20}")
    print(f"  final p: {dict((variants[i][0], round(p[i],2)) for i in range(N))}")
    print()
