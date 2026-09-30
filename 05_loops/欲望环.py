"""desire_loop.py —— 多输入竞争，偏好从反应强度中浮现。

N 个不同的外部输入（同一份数据的变形）。
每圈：对所有 N 个输入，跑一整圈 56 步，算出终态 (m, a)。
      选 |Δ(m,a)| 最大的那个——反应最强的。
      那个终态成为下一圈的起点。
       跑 CYCLES 圈，记录每圈的：选了谁、强度多少、终态在哪。

不指定偏好。偏好 = 哪个输入把状态推得最远。
"""

import sys, math
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 活诊断 import 提取锚点, 归一化

# —— 加载原始数据 ——
raw_m, raw_a, n_fr = 提取锚点()
mn, _, _ = 归一化(raw_m)
an, _, _ = 归一化(raw_a)
N_M = len(mn)
N_A = len(an)

print(f"loaded: mouth_norm {N_M} frames, aud_norm {N_A} frames")

# —— 构建 N 个不同的外部输入 ——
def make_variants(mn, an):
    variants = []

    # V0: natural — 原数据
    variants.append(("natural", mn[:], an[:]))

    # V1: inverted — 镜像（哪里有峰值，哪里变低谷）
    inv_m = [1.0 - v for v in mn]
    inv_a = [1.0 - v for v in an]
    variants.append(("inverted", inv_m, inv_a))

    # V2: brightened — 对比度增强（推离 0.5）
    def brighten(lst, factor=1.4):
        return [max(0.0, min(1.0, 0.5 + (v - 0.5) * factor)) for v in lst]
    variants.append(("brightened", brighten(mn), brighten(an)))

    # V3: smoothed — 平滑（时域上抹掉锋利的跳变）
    def smooth(lst, k=3):
        out = []
        N = len(lst)
        for i in range(N):
            s = sum(lst[(i + j) % N] for j in range(k)) / k
            out.append(s)
        return out
    variants.append(("smoothed", smooth(mn), smooth(an)))

    # V4: phase_shifted — 相位错半拍
    sh_m = N_M // 2
    sh_a = N_A // 2
    ps_m = mn[sh_m:] + mn[:sh_m]
    ps_a = an[sh_a:] + an[:sh_a]
    variants.append(("phase_shifted", ps_m, ps_a))

    return variants

variants = make_variants(mn, an)
N_VAR = len(variants)
for v in variants:
    print(f"  variant: {v[0]:15s}  ext_m[0]={v[1][0]:.6f}  ext_a[0]={v[2][0]:.6f}")

# —— 前向步进 ——
def fwd(m, a, ext_m, ext_a):
    mn = 0.46 * m + 0.22 * a + 0.32 * ext_m
    an = 0.46 * a + 0.22 * m + 0.32 * ext_a
    return mn, an

def run_full_cycle(m0, a0, ext_m_seq, ext_a_seq, steps=56):
    m, a = m0, a0
    for i in range(steps):
        m, a = fwd(m, a, ext_m_seq[i % len(ext_m_seq)], ext_a_seq[i % len(ext_a_seq)])
    return m, a

# —— 主循环 ——
CYCLES = 1000
m, a = 0.3, 0.3

log = []
with open("desire_log.txt", "w") as f:
    for cycle in range(CYCLES):
        # 评估所有变体
        scores = []
        for vi in range(N_VAR):
            name, em, ea = variants[vi]
            m_end, a_end = run_full_cycle(m, a, em, ea)
            strength = math.sqrt((m_end - m)**2 + (a_end - a)**2)
            scores.append((strength, vi, name, m_end, a_end))

        # 选最强的
        scores.sort(reverse=True)
        winner = scores[0]
        w_strength, w_vi, w_name, w_m, w_a = winner

        m, a = w_m, w_a
        log.append((cycle, w_vi, w_name, w_strength, m, a))

        line = f"cycle {cycle:4d}  chose=V{w_vi}({w_name:14s})  strength={w_strength:.10f}  m={m:.16f}  a={a:.16f}"
        f.write(line + "\n")

        if cycle < 10:
            print(line + "  | " + " ".join(f"{s[2]}={s[0]:.8f}" for s in scores))

print("done")

# —— 偏好统计 ——
from collections import Counter
c = Counter(s[2] for s in log)
print(f"\n共 {CYCLES} 圈，各变体被选中的圈数（原始计票）:")
for name, cnt in c.most_common():
    print(f"  {name:15s}: {cnt:4d} 圈  ({100*cnt/CYCLES:.1f}%)")

# 检查哪圈后偏好稳定：
print("\n偏好变化点（圈号 → 选谁）:")
prev = None
for cycle, vi, name, strength, m, a in log:
    if prev != vi:
        print(f"  cycle {cycle:4d}: V{vi}({name}), strength={strength:.10f}, m={m:.10f}, a={a:.10f}")
        prev = vi
