"""desire_accum2.py —— 加法偏好。

boosted = raw_strength + p[i]  (加法，不是乘法)
选中：p[i] += ALPHA
ALPHA = 0.1 (和 desire_accum.py 一样，但加法)

关键差异：乘法时 raw=0 * p = 0（永远赢不了）
         加法时 raw=0 + p = p（p 够大就能赢）
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

p = [1.0]*N_VAR
ALPHA = 0.1
CYCLES = 1000
m, a = 0.3, 0.3

with open("desire_accum2_log.txt", "w") as f:
    for cycle in range(CYCLES):
        scores = []
        for i in range(N_VAR):
            name, em, ea = variants[i]
            me, ae = run_cycle(m, a, em, ea)
            raw = math.sqrt((me-m)**2 + (ae-a)**2)
            boosted = raw + p[i]
            scores.append((boosted, raw, i, name, me, ae))
        scores.sort(reverse=True)
        bst, rst, wi, wn, wm, wa = scores[0]
        p[wi] += ALPHA
        m, a = wm, wa
        line = f"cycle {cycle:4d} chose=V{wi}({wn:14s}) raw={rst:.10f} boosted={bst:.10f} m={m:.10f} a={a:.10f} p=[{' '.join(f'{x:.4f}' for x in p)}]"
        f.write(line + "\n")
        if cycle < 30 or cycle % 100 == 0:
            print(line)

from collections import Counter
print("\n=== summary ===")
lines = open("desire_accum2_log.txt").readlines()
choices = []
for line in lines:
    if 'chose=' in line:
        vi = int(line.split('chose=V')[1].split('(')[0])
        cyc = int(line.split('cycle')[1].split()[0])
        choices.append((cyc, vi))
c = Counter(v for _, v in choices)
print(f"V0:{c.get(0,0)} V1:{c.get(1,0)} V2:{c.get(2,0)} V3:{c.get(3,0)} V4:{c.get(4,0)}")
print(f"final p: {dict((variants[i][0], round(p[i],4)) for i in range(N_VAR))}")
