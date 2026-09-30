"""单通道区分度溯源：5 个音，内部 γ。

发声(vocal) 5 个音，内部 C(5,2)=10 对 γ
听觉(auditory) 5 个音，内部 C(5,2)=10 对 γ

落盘 gray/sensory/
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from 矩阵.发声 import 发声矩阵
from 矩阵.听觉 import 听矩阵
from 矩阵.跨模态 import 跨模态差分, 归一化N维, 矩阵转六维

# ── 5 个元音 ───────────────────────────────
VOWELS = {
    "啊": {"F1": 700,  "F2": 1200, "F3": 2500},  # a 开元前
    "哦": {"F1": 500,  "F2": 800,  "F3": 1500},  # o 圆唇后
    "咦": {"F1": 300,  "F2": 2200, "F3": 3500},  # i 闭前
    "呜": {"F1": 350,  "F2": 700,  "F3": 2500},  # u 闭后圆唇
    "诶": {"F1": 500,  "F2": 1800, "F3": 2800},  # e 中前
}

def build_vowel_frame(f1, f2, f3, rows=8, cols=64):
    """生成一帧：不同共振峰激活不同声道分区，不同时间区间"""
    base = np.zeros((rows, cols))
    # 声带/咽腔/鼻腔 = F1 区
    base[0:3, 8:48] = f1
    # 口腔/唇 = F2 区（前后位置）
    base[3:5, 8:48] = f2
    # 舌/齿/下颌 = F3 区（精细调节）
    base[5:8, 8:48] = f3
    return base.tolist()

results = {}

for channel_name, factory_class, sub_mode in [
    ("发声", 发声矩阵, "基频_Hz"),
    ("听觉", 听矩阵, "频率_Hz"),
]:
    frames = {}
    for vname, formants in VOWELS.items():
        frame = build_vowel_frame(formants["F1"], formants["F2"], formants["F3"])
        mat = factory_class.from_frame(frame, sub_modality=sub_mode, source=vname)
        frames[vname] = mat.data

    vowel_names = list(VOWELS.keys())
    n = len(vowel_names)

    print(f"\n=== {channel_name} 通道（内部 {n*(n-1)//2} 对 γ）===")

    feats_raw = {}
    feats_l2 = {}
    for vname in vowel_names:
        raw = 矩阵转六维(frames[vname])
        nf = 归一化N维(raw)
        feats_raw[vname] = raw
        feats_l2[vname] = nf

    print(f"\n六维 L2 向量:")
    for vname in vowel_names:
        nf = feats_l2[vname]
        s = ", ".join(["{:.4f}".format(v) for v in nf])
        f = feats_raw[vname]
        print(f"  {vname}: mean={f[0]:.1f}, max={f[1]:.1f}  L2=[{s}]")

    gamma_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = vowel_names[i], vowel_names[j]
            cd = 跨模态差分(
                "intra_channel", frames[vi], frames[vj], f"{channel_name}_{vi}", f"{channel_name}_{vj}"
            )
            gamma_matrix[(vi, vj)] = cd["gamma"]
            ga = "{:.6f}".format(cd['gamma']) if cd['gamma'] is not None else "None"
            print(f"  {vi} vs {vj}: γ={ga}")

    # 统计
    gammas = [v for v in gamma_matrix.values() if v is not None]
    if gammas:
        print(f"\n统计: min={min(gammas):.4f}, max={max(gammas):.4f}, mean={np.mean(gammas):.4f}, std={np.std(gammas):.4f}")
        near = [k for k,v in gamma_matrix.items() if v < 0.05]
        far = [k for k,v in gamma_matrix.items() if v > 0.10]
        if near:
            print(f"  近零(γ<0.05): {near}")
        if far:
            print(f"  高位(γ>0.10): {far}")
        if not near and not far:
            print(f"  集中区间无分化")

    results[channel_name] = {"feats_l2": feats_l2, "gamma_matrix": gamma_matrix}

# ── 跨通道对比（同一音，发 vs 听）──
print(f"\n=== 同一音 · 发 vs 听 跨通道 ===")
for vname in VOWELS.keys():
    frame = build_vowel_frame(VOWELS[vname]["F1"], VOWELS[vname]["F2"], VOWELS[vname]["F3"])
    vm = 发声矩阵.from_frame(frame, sub_modality="基频_Hz", source=f"发{vname}").data
    am = 听矩阵.from_frame(frame, sub_modality="频率_Hz", source=f"听{vname}").data
    cd = 跨模态差分("cross_channel", vm, am, f"发{vname}", f"听{vname}")
    ga = "{:.6f}".format(cd['gamma']) if cd['gamma'] is not None else "None"
    print(f"  {vname}: γ={ga}")

# 整体结论
print(f"\n=== 整体 ===")
for ch in ["发声", "听觉"]:
    gm = results[ch]["gamma_matrix"]
    gammas = [v for v in gm.values() if v is not None]
    print(f"  {ch}: γ 范围 [{min(gammas):.4f}, {max(gammas):.4f}]")
