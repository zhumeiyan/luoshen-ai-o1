"""听说闭环：发一个音 → 自己听回 → γ 对比。

发"啊" → 听回"啊" → γ 应该低（同源）
发"啊" → 听回"哦" → γ 应该高（不同源）

落盘：gray/sensory/
状态：[待观察]
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from 矩阵.发声 import 发声矩阵, 建测试帧
from 矩阵.听觉 import 听矩阵
from 矩阵.跨模态 import 跨模态差分, 归一化N维, 矩阵转六维

# ── 两种音的声学模型 ────────────────────────

def build_a_sound(rows=8, cols=64):
    """啊：开元音，低频能量集中（F1~700Hz），低矮共振峰"""
    base = np.zeros((rows, cols))
    base[0:3, 8:48] = 700   # 声带+咽腔+鼻腔，F1 ~700Hz
    base[3:5, 8:48] = 1200  # F2
    base[5:8, 8:48] = 2500  # F3
    return base.tolist()

def build_o_sound(rows=8, cols=64):
    """哦：圆唇元音，F1~500Hz，F2偏低，高频衰减"""
    base = np.zeros((rows, cols))
    base[0:3, 8:52] = 500   # F1 ~500Hz（更低）
    base[3:5, 8:52] = 800   # F2 偏低
    base[5:8, 8:36] = 1500  # F3 更弱
    return base.tolist()

# ── 实测 ───────────────────────────────────

def run():
    print("=" * 80)
    print("【听说闭环 v0.1 — 发声矩阵 vs 听觉矩阵】")
    print("=" * 80)

    # 生成"啊" 和 "哦" 的帧
    a_frame = build_a_sound()
    o_frame = build_o_sound()

    # ---- 路径 A: 发声 ----
    vocal_a = 发声矩阵.from_frame(a_frame, sub_modality="基频_Hz", source="发啊")
    vocal_o = 发声矩阵.from_frame(o_frame, sub_modality="基频_Hz", source="发哦")

    # ---- 路径 B: 听觉 ----
    aud_a = 听矩阵.from_frame(a_frame, sub_modality="频率_Hz", source="听啊")
    aud_o = 听矩阵.from_frame(o_frame, sub_modality="频率_Hz", source="听哦")

    print(f"\n── 矩阵尺寸 ──")
    print(f"发声矩阵（基频_Hz）: {len(vocal_a.data)}×{len(vocal_a.data[0])}")
    print(f"听觉矩阵（频率_Hz）: {len(aud_a.data)}×{len(aud_a.data[0])}")

    # ── 四组跨模态差分 ──
    print(f"\n── 跨模态差分（发声 vs 听觉） ──")
    tests = [
        ("发啊 vs 听啊", vocal_a.data, aud_a.data),
        ("发哦 vs 听哦", vocal_o.data, aud_o.data),
        ("发啊 vs 听哦", vocal_a.data, aud_o.data),
        ("发哦 vs 听啊", vocal_o.data, aud_a.data),
    ]

    results = []
    for name, vm, am in tests:
        cd = 跨模态差分(
            "speak_hear_loop", vm, am, name.split(" vs ")[0], name.split(" vs ")[1],
            role_a="output", role_b="input"
        )
        results.append((name, cd))
        gamma_str = f"{cd['gamma']:.6f}" if cd['gamma'] is not None else "None"
        print(f"  {name:12s}  γ={gamma_str}")

    # ── 向量 ──
    print(f"\n── 六维特征 ──")
    channels = {
        "发啊(基频)": vocal_a.data,
        "发哦(基频)": vocal_o.data,
        "听啊(频率)": aud_a.data,
        "听哦(频率)": aud_o.data,
    }
    feats = {}
    for name, mat in channels.items():
        raw = 矩阵转六维(mat)
        nf = 归一化N维(raw)
        feats[name] = (raw, nf)
        r = ", ".join(["{:.2f}".format(v) for v in raw])
        n = ", ".join(["{:.4f}".format(v) for v in nf])
        print(f"  {name}")
        print(f"    原始: [{r}]")
        print(f"    L2:   [{n}]")

    return results, feats

if __name__ == "__main__":
    results, feats = run()
