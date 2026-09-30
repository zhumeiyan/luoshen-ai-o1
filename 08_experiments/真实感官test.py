"""真感官输入测试 v0.1 — 一张山的图 + 一段"山"的音。

不是文字码点。
图：模拟山的视觉轮廓（8×64 亮度矩阵）
音：模拟人声"山"的声压（8×64 声压矩阵）

两路同时进：
  视觉矩阵 → 六维特征 → ↘
                        跨模态差分 → γ → 对上了没有？
  听觉矩阵 → 六维特征 → ↗

标 [判读·真感官输入]。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from 矩阵.跨模态 import (
    跨模态差分, 归一化N维, 矩阵转六维,
    compute_coverage, remove_baseline
)


# ────────────────────────────────────────
# 一、生成"山的图"
# ────────────────────────────────────────

def build_mountain_image(rows=8, cols=64):
    """生成山的视觉轮廓（模拟像素亮度）。

    不是文字码点。
    是物理场景的亮度映射：天空（上）→ 山峰（中）→ 地面（下）。

    结构：
      row 0-1: 天空 → 低亮度（~50 cd/m²）
      row 2-4: 山峰轮廓 → 中间高两侧低（锥形，山顶最亮 ~5000）
      row 5-7: 地面 → 中等亮度（~200 cd/m²）

    来源：模拟数据（非真实摄像头），用于跑通链路。
    """
    img = np.zeros((rows, cols))

    # 天空
    img[0:2, :] = np.random.uniform(30, 70, (2, cols))

    # 山峰轮廓（row 2-4）: 三角形，顶点在 col=32
    for r in range(2, 5):
        height_factor = (r - 2) / 2.0  # 0, 0.5, 1
        for c in range(cols):
            # 到山顶的水平距离
            dist = abs(c - 32) / 32.0
            # 高度 = 三角形 * height_factor
            mountain = max(0, 1.0 - dist) * height_factor
            img[r, c] = mountain * 5000 + np.random.uniform(-20, 20)

    # 地面
    img[5:8, :] = np.random.uniform(150, 250, (3, cols))

    return img.tolist()


# ────────────────────────────────────────
# 二、生成"山"的音
# ────────────────────────────────────────

def build_shan_audio(rows=8, cols=64):
    """生成人声"山"的声压（模拟声压波形）。

    不是文字码点。
    是物理声波的声压映射。

    结构（时间轴从左到右）：
      起始 ~col 8:  辅音 sh → 清擦音，高频噪声
      col 8-48:     元音 a → 周期波（基频 ~250Hz 对应发声周期）
      收尾 col 48-64: 鼻音 n → 声压衰减 + 低频共振

    每一行是不同频段的声压（模拟频谱图）：
      row 0: 低频（<100Hz）
      row 1-3: 中低频（100-500Hz，元音谐波）
      row 4-6: 中高频（500-2000Hz，辅音能量）
      row 7: 高频（>2000Hz，擦音）

    来源：模拟数据（非真实麦克风），用于跑通链路。
    """
    audio = np.zeros((rows, cols))

    t = np.linspace(0, 1, cols)

    # === 频段 0: 低频（<100Hz）===
    # 元音 a 的低频能量
    env = np.zeros(cols)
    env[8:48] = 1.0  # 元音区间
    env[48:56] = np.linspace(1.0, 0.3, 8)  # 衰减
    audio[0, :] = env * np.sin(2 * np.pi * 80 * t) * 0.5 + np.random.normal(0, 0.02, cols)

    # === 频段 1-3: 中低频（100-500Hz，元音谐波）===
    for i, freq in enumerate([200, 300, 400], start=1):
        if i > 3:
            break
        env = np.zeros(cols)
        env[8:48] = 1.0
        env[48:56] = np.linspace(1.0, 0.3, 8)
        audio[i, :] = env * np.sin(2 * np.pi * freq * t) * (0.6 / (i)) + np.random.normal(0, 0.015, cols)

    # === 频段 4-6: 中高频（500-2000Hz，辅音 sh 的擦音能量）===
    # 擦音只在起始阶段（col 0-8）
    frication = np.zeros(cols)
    frication[0:8] = 1.0
    for i, freq in enumerate(np.array([600, 1000, 1500]), start=4):
        audio[i, :] = frication * np.sin(2 * np.pi * freq * t) * 0.3 + np.random.normal(0, 0.1, cols)

    # === 频段 7: 高频（>2000Hz，擦音噪声）===
    frication = np.zeros(cols)
    frication[0:8] = 1.0
    audio[7, :] = frication * np.random.normal(0, 0.4, cols)

    return audio.tolist()


# ────────────────────────────────────────
# 三、自净
# ────────────────────────────────────────

CHANNEL_CONFIG = {
    "视觉":    {"role": "evidence", "baseline": None},
    "听觉":    {"role": "evidence", "baseline": None},
    "触觉":    {"role": "sparse",   "baseline": None},
    "温度觉":  {"role": "evidence", "baseline": 293.0},
    "痛觉":    {"role": "modulation", "baseline": None},
    "发声":    {"role": "evidence", "baseline": None},
    "红外":    {"role": "evidence", "baseline": None},
    "雷达":    {"role": "sparse",   "baseline": None},
    "LiDAR":   {"role": "sparse",   "baseline": None},
}


def apply_self_cleaning(matrix, channel_name):
    config = CHANNEL_CONFIG[channel_name]
    role = config["role"]
    baseline = config["baseline"]
    m = matrix
    if baseline is not None:
        m = remove_baseline(matrix, baseline)
    coverage = compute_coverage(m)
    return m, coverage, role


# ────────────────────────────────────────
# 四、主链路
# ────────────────────────────────────────

def run_pipeline():
    print("=" * 80)
    print("【真感官输入 v0.1 — 山的图 + 山的声音】")
    print("=" * 80)

    # ── 输入 ──
    print(f"\n── 输入（模拟） ──")
    img = build_mountain_image()
    audio = build_shan_audio()
    print(f"视觉: 8×64 亮度矩阵（山的轮廓）")
    print(f"听觉: 8×64 声压矩阵（人声\"山\"）")

    # ── 输入可视化 ──
    print(f"\n── 视觉矩阵（亮度） ──")
    for r in range(8):
        vals = img[r]
        bar = "".join(["#" if v > 1000 else "+" if v > 200 else "." for v in vals[::4]])
        print(f"  row {r}: [{bar}]")

    print(f"\n── 听觉矩阵（声压） ──")
    for r in range(8):
        vals = audio[r]
        max_v = max(abs(v) for v in vals)
        if max_v < 0.01:
            bar = "." * 16
        else:
            bar = "".join(["#" if abs(v) > max_v * 0.5 else "+" if abs(v) > max_v * 0.2 else "." for v in vals[::4]])
        print(f"  row {r}: [{bar}]")

    # ── 六维特征 ──
    print(f"\n── 六维特征 ──")
    feat_names = ["mean", "max", "skew", "kurt", "p25", "p75"]

    feat_vis = 矩阵转六维(img)
    feat_aud = 矩阵转六维(audio)
    cov_vis = compute_coverage(img)
    cov_aud = compute_coverage(audio)

    print(f"视觉  : [{', '.join(f'{feat_names[i]}={feat_vis[i]:.2f}' for i in range(6))}]")
    print(f"听觉  : [{', '.join(f'{feat_names[i]}={feat_aud[i]:.4f}' for i in range(6))}]")

    # L2 归一化
    nf_vis = 归一化N维(feat_vis)
    nf_aud = 归一化N维(feat_aud)
    print(f"视觉 L2: [{', '.join(f'{v:.4f}' for v in nf_vis)}]")
    print(f"听觉 L2: [{', '.join(f'{v:.4f}' for v in nf_aud)}]")

    # ── 交叉模态差分 ──
    print(f"\n── 跨模态差分 ──")
    cd = 跨模态差分(
        "real_sensory", img, audio, "视觉", "听觉",
        coverage_a=cov_vis, coverage_b=cov_aud,
        role_a="evidence", role_b="evidence"
    )

    fa, fb = cd['特征_a'], cd['特征_b']
    nfa, nfb = cd['规一_a'], cd['规一_b']
    g_vec, h_vec = cd['g'], cd['h']
    fa_str = ', '.join(['{}={:.2f}'.format(feat_names[i], fa[i]) for i in range(6)])
    fb_str = ', '.join(['{}={:.4f}'.format(feat_names[i], fb[i]) for i in range(6)])
    print("视觉 feat  : [{}]".format(fa_str))
    print("听觉 feat  : [{}]".format(fb_str))
    print("视觉 L2    : [{}]".format(', '.join(['{:.4f}'.format(v) for v in nfa])))
    print("听觉 L2    : [{}]".format(', '.join(['{:.4f}'.format(v) for v in nfb])))
    print("g          : [{}]".format(', '.join(['{:.4f}'.format(v) for v in g_vec])))
    print("h (锚点差) : [{}]".format(', '.join(['{:.4f}'.format(v) for v in h_vec])))
    sig = cd.get('sigma')
    gam = cd.get('gamma')
    print("σ          : {:.6f}".format(sig) if sig is not None else "σ: None")
    print("γ          : {:.6f}".format(gam) if gam is not None else "γ: None")

    # ── 判定 ──
    print(f"\n── 判定 ──")
    gamma = cd["gamma"]
    if gamma is None:
        print(f"  γ = None → 无法判定")
    elif abs(gamma) > 0.8:
        print(f"  |γ| > 0.8 → 两路差异极大，可能代表不同事物")
    elif abs(gamma) > 0.3:
        print(f"  0.3 < |γ| < 0.8 → 有差异但可关联（可能同一事物不同模态）")
    elif abs(gamma) > 0.05:
        print(f"  0.05 < |γ| < 0.3 → 差异小，可能高度关联（同一事件不同模态）")
    else:
        print(f"  |γ| < 0.05 → 几乎无差异")

    print(f"\n状态：[判读·真感官输入]")
    return cd


if __name__ == "__main__":
    cd = run_pipeline()
