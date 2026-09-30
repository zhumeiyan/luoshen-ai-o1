"""desire_accum.py —— 偏好累积，选谁就涨谁的权重。

每个变体 i 有偏好权重 p[i]，初始 1.0。
每圈：对所有变体，跑 56 步，算 raw_strength(i) = |Δ(m,a)|。
      boosted(i) = raw_strength(i) * p[i]
      选 boosted 最大的。
      被选中的：p[i] += alpha。
    跑 1000 圈。

不共享、不重置。只涨被选中的那个。
涨的多 → 下次更容易被选 → 正反馈。
"""

import sys, math
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 活诊断 import 提取锚点, 归一化
raw_m, raw_a, n_fr = 提取锚点()
mn, _, _ = 归一化(raw_m)
an, _, _ = 归一化(raw_a)
N_M = len(mn)
N_A = len(an)

def make_variants(mn, an):
    v = []
    v.append(("natural", mn[:], an[:]))
    inv_m = [1.0 - x for x in mn]; inv_a = [1.0 - x for x in an]
    v.append(("inverted", inv_m, inv_a))
    def brighten(lst, f=1.4):
        return [max(0.0, min(1.0, 0.5 + (x-0.5)*f)) for x in lst]
    v.append(("brightened", brighten(mn), brighten(an)))
    def smooth(lst, k=3):
        N = len(lst); return [sum(lst[(i+j)%N] for j in range(k))/k for i in range(N)]
    v.append(("smoothed", smooth(mn), smooth(an)))
    sh_m = N_M // 2; sh_a = N_A // 2
    v.append(("phase_shifted", mn[sh_m:]+mn[:sh_m], an[sh_a:]+an[:sh_a]))
    return v

variants = make_variants(mn, an)
N_VAR = len(variants)

def fwd(m, a, em, ea):
    return 0.46*m + 0.22*a + 0.32*em, 0.46*a + 0.22*m + 0.32*ea

def run_cycle(m, a, em, ea, steps=56):
    for i in range(steps):
        m, a = fwd(m, a, em[i%len(em)], ea[i%len(ea)])
    return m, a

# 偏好权重
p = [1.0] * N_VAR
ALPHA = 0.1

CYCLES = 1000
m, a = 0.3, 0.3

log = []
with open("desire_accum_log.txt", "w") as f:
    for cycle in range(CYCLES):
        scores = []
        for i in range(N_VAR):
            name, em, ea = variants[i]
            me, ae = run_cycle(m, a, em, ea)
            raw = math.sqrt((me-m)**2 + (ae-a)**2)
            boosted = raw * p[i]
            scores.append((boosted, raw, i, name, me, ae))

        scores.sort(reverse=True)
        winner = scores[0]
        bst, rst, wi, wn, wm, wa = winner

        # 更新偏好
        p[wi] += ALPHA

        m, a = wm, wa
        log.append((cycle, wi, wn, rst, bst, m, a, list(p)))

        line = f"cycle {cycle:4d} chose=V{wi}({wn:14s}) raw={rst:.10f} boosted={bst:.10f} m={m:.10f} a={a:.10f} p=[{' '.join(f'{x:.4f}' for x in p)}]"
        f.write(line + "\n")

        if cycle < 20 or cycle % 100 == 0:
            print(line)

print("done")
print(f"\nfinal p: {dict((variants[i][0], round(p[i],4)) for i in range(N_VAR))}")
