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

p = [1.0]*N
ALPHA = 2.0
m, a = 0.3, 0.3

print(f"additive, ALPHA={ALPHA}:")
for cycle in range(1000):
    scores = []
    for i in range(N):
        name, em, ea = variants[i]
        me, ae = run_cycle(m, a, em, ea)
        raw = math.sqrt((me-m)**2 + (ae-a)**2)
        boosted = raw + p[i]  # 加法
        scores.append((boosted, raw, i, name, me, ae))
    scores.sort(reverse=True)
    bst, rst, wi, wn, wm, wa = scores[0]
    p[wi] += ALPHA
    m, a = wm, wa

    if cycle < 15 or (wi != 1 and wi != 2):
        print(f"cycle {cycle:4d} chose=V{wi}({wn:14s}) raw={rst:.10f} boosted={bst:.10f} m={m:.10f} a={a:.10f} p=[{' '.join(f'{x:.1f}' for x in p)}]")

from collections import Counter
# Re-count
p2 = [1.0]*N
m, a = 0.3, 0.3
choices = []
for cycle in range(1000):
    scores = []
    for i in range(N):
        name, em, ea = variants[i]
        me, ae = run_cycle(m, a, em, ea)
        raw = math.sqrt((me-m)**2 + (ae-a)**2)
        boosted = raw + p2[i]
        scores.append((boosted, raw, i, name, me, ae))
    scores.sort(reverse=True)
    bst, rst, wi, wn, wm, wa = scores[0]
    p2[wi] += ALPHA
    m, a = wm, wa
    choices.append(wi)

c = Counter(choices)
print(f"\\nsummary: {dict(c)}")
# find lock
lock_cycle = None
lock_v = '?'
for i in range(20, len(choices)):
    if len(set(choices[i-20:i])) == 1:
        lock_cycle = i - 20
        lock_v = choices[i-1]
        break
print(f"lock: cycle {lock_cycle} V{lock_v}")
print(f"final p: {dict((variants[i][0], round(p2[i],1)) for i in range(N))}")
