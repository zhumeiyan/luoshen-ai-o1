"""绑定测试（修正）：留出法。

Memory 积累 N 个山，测试留出的山。
预判 = 记忆中找最像的图，取其配对音。
预判 vs 测试山实际音 → γ。
γ 低 = 预判准。
随 N 增大（记忆增多），留出山的预判 γ 是否下降。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from 矩阵.跨模态 import 跨模态差分, 归一化N维, 矩阵转六维

def mountain_frame(kind):
    rows, cols = 8, 64
    m = np.zeros((rows, cols))
    if kind == "thin":
        m[0:2, :] = 50
        for r in range(2, 5):
            for c in range(cols):
                dist = abs(c - 32) / 12.0
                m[r, c] = max(0, (1 - dist)) * 6000
        m[5:8, :] = 200
    elif kind == "wide":
        m[0:2, :] = 50
        for r in range(2, 5):
            for c in range(cols):
                dist = abs(c - 32) / 28.0
                m[r, c] = max(0, (1 - dist)) * 4500
        m[5:8, :] = 200
    elif kind == "double":
        m[0:2, :] = 50
        for r in range(2, 5):
            for c in range(cols):
                d1 = abs(c - 20) / 10.0
                d2 = abs(c - 44) / 10.0
                m[r, c] = max(0, (1-d1)) * 5000 + max(0, (1-d2)) * 4000
        m[5:8, :] = 200
    elif kind == "left":
        m[0:2, :] = 50
        for r in range(2, 5):
            for c in range(cols):
                dist = abs(c - 18) / 14.0
                m[r, c] = max(0, (1 - dist)) * 5500
        m[5:8, :] = 200
    elif kind == "low":
        m[0:3, :] = 50
        for r in range(3, 5):
            for c in range(cols):
                dist = abs(c - 32) / 24.0
                m[r, c] = max(0, (1 - dist)) * 3000
        m[5:8, :] = 200
    return m.tolist()

def audio_frame(kind):
    rows, cols = 8, 64
    base = np.zeros((rows, cols))
    if kind == "thin":
        base[0:3, 6:50] = 800; base[3:5, 6:50] = 1300; base[5:8, 6:50] = 2600
    elif kind == "wide":
        base[0:3, 6:50] = 600; base[3:5, 6:50] = 1000; base[5:8, 6:50] = 2200
    elif kind == "double":
        base[0:3, 6:50] = 750; base[3:5, 6:50] = 1400; base[5:8, 6:50] = 2800
    elif kind == "left":
        base[0:3, 6:50] = 700; base[3:5, 6:50] = 1200; base[5:8, 6:50] = 2500
    elif kind == "low":
        base[0:3, 6:50] = 500; base[3:5, 6:50] = 900; base[5:8, 6:50] = 1800
    return base.tolist()

ALL = ["thin", "wide", "double", "left", "low"]

# Pre-compute
img_data = {k: mountain_frame(k) for k in ALL}
aud_data = {k: audio_frame(k) for k in ALL}
img_l2 = {k: 归一化N维(矩阵转六维(img_data[k])) for k in ALL}
aud_l2 = {k: 归一化N维(矩阵转六维(aud_data[k])) for k in ALL}

print("=" * 80)
print("【绑定测试 · 留出法】")
print("=" * 80)

# 自身配对 γ（基线）
print(f"\n── 基线：山图 vs 山音（同一山）──")
baselines = {}
for k in ALL:
    cd = 跨模态差分("baseline", img_data[k], aud_data[k], f"图{k}", f"音{k}")
    baselines[k] = cd["gamma"]
    print(f"  山({k:6s}): γ={cd['gamma']:.6f}")

# ── 留出法 ──
print(f"\n── 留出法：积累 N 个山，预判留出山 ──")
for test_kind in ALL:
    memory_kinds = [k for k in ALL if k != test_kind]
    print(f"\n  留出: 山({test_kind}), 基线 γ={baselines[test_kind]:.4f}")

    for n in range(1, len(memory_kinds) + 1):
        memory = memory_kinds[:n]

        # 在 memory 中找最像 test 图的山
        test_l = img_l2[test_kind]
        best_mem = min(memory, key=lambda k: sum((a-b)**2 for a,b in zip(test_l, img_l2[k]))**0.5)
        dist = sum((a-b)**2 for a,b in zip(test_l, img_l2[best_mem]))**0.5

        # 预判音 = 记忆中最像图的山的配对音
        pred_aud = aud_data[best_mem]

        # 实际音 = 留出山的音
        actual_aud = aud_data[test_kind]

        # γ 预判 vs 实际
        cd = 跨模态差分("binding", pred_aud, actual_aud, f"预判音_{best_mem}", f"实际音_{test_kind}")
        gamma_pred = cd["gamma"]

        print(f"    N={n}: memory={memory}, 最像=山{best_mem}(距离{dist:.4f})  γ预判={gamma_pred:.4f}")

# ── 反向：给音，预判图 ──
print(f"\n── 反向留出法：给音，预判图 ──")
for test_kind in ALL:
    memory_kinds = [k for k in ALL if k != test_kind]
    test_aud_l = aud_l2[test_kind]

    # 实际图 baseline（同一山 音→图 反转）
    cd_bl = 跨模态差分("baseline_rev", aud_data[test_kind], img_data[test_kind], f"音{test_kind}", f"图{test_kind}")
    print(f"\n  留出: 山({test_kind}), 反向基线 γ={cd_bl['gamma']:.4f}")

    for n in range(1, len(memory_kinds) + 1):
        memory = memory_kinds[:n]

        best_mem = min(memory, key=lambda k: sum((a-b)**2 for a,b in zip(test_aud_l, aud_l2[k]))**0.5)
        dist = sum((a-b)**2 for a,b in zip(test_aud_l, aud_l2[best_mem]))**0.5

        pred_img = img_data[best_mem]
        actual_img = img_data[test_kind]

        cd = 跨模态差分("binding_rev", pred_img, actual_img, f"预判图_{best_mem}", f"实际图_{test_kind}")
        gamma_pred = cd["gamma"]

        print(f"    N={n}: 预判图=山{best_mem}(距离{dist:.4f})  γ={gamma_pred:.4f}")

# ── 汇总 ──
print(f"\n── 汇总 ──")
print(f"  留出法现象：")
print(f"  当最像记忆越不像留出山，预判 γ 越高（预判越差）")
print(f"  当最像记忆越像留出山，预判 γ 越低（预判越好）")
print(f"  随 N 增大，最像距离可能下降（因为候选多了）→ γ 可能下降")
