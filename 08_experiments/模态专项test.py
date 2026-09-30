"""模态专有特征测试。

每个模态用自己的结构特征，不套通用 6D。
测：
  1. 同一东西（山图 vs 山音），两路专有特征差多少
  2. 同一模态（视觉），不同东西（山图 vs 文字图），专有特征能否分开
  3. 对比通用 6D：专有特征的区分度有没有变好
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from 矩阵.跨模态 import 跨模态差分, 归一化N维, 矩阵转六维

# ════════════════════════════════════════════════════════════════
# 1. 模态专有特征提取器
# ════════════════════════════════════════════════════════════════

def visual_specific(matrix):
    """视觉专有特征。

    矩阵结构：8行 = 8条垂直带（上→下），64列 = 64像素（左→右）。
    空间结构是视觉的固有属性。
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # 垂直廓线：每行均值（8维）— 上亮还是下亮
    v_profile = [sum(matrix[r]) / cols for r in range(rows)]

    # 水平廓线：每列均值（8维，8列一组）— 左亮还是右亮
    group = cols // 8
    h_profile = []
    for g in range(8):
        s = 0.0
        count = 0
        for c in range(g * group, min((g + 1) * group, cols)):
            for r in range(rows):
                s += matrix[r][c]
                count += 1
        h_profile.append(s / count if count > 0 else 0.0)

    # 行质心（垂直重心）
    total = sum(sum(row) for row in matrix)
    if total < 1e-12:
        c_row = 0.0
        c_col = 0.0
        v_spread = 0.0
        h_spread = 0.0
        symmetry = 0.0
        edge_h = 0.0
        edge_v = 0.0
    else:
        c_row = sum(r * sum(matrix[r]) for r in range(rows)) / total
        c_col = sum(c * sum(matrix[r][c] for r in range(rows)) for c in range(cols)) / total

        # 行散布（偏离质心的方差）
        v_spread = sum((r - c_row) ** 2 * sum(matrix[r]) for r in range(rows)) / total
        h_spread = sum((c - c_col) ** 2 * sum(matrix[r][c] for r in range(rows)) for c in range(cols)) / total

        # 对称性：左半 vs 右半 之差的绝对值
        symmetry = 0.0
        for r in range(rows):
            for c in range(cols // 2):
                mirror_c = cols - 1 - c
                symmetry += abs(matrix[r][c] - matrix[r][mirror_c])
        symmetry /= total

        # 边缘密度：水平相邻差
        edge_h = 0.0
        for r in range(rows):
            for c in range(cols - 1):
                edge_h += abs(matrix[r][c + 1] - matrix[r][c])
        edge_h /= (rows * (cols - 1))

        # 边缘密度：垂直相邻差
        edge_v = 0.0
        for r in range(rows - 1):
            for c in range(cols):
                edge_v += abs(matrix[r + 1][c] - matrix[r][c])
        edge_v /= ((rows - 1) * cols)

    # 组合：廓线 + 几何
    feat = v_profile + h_profile + [c_row / rows, c_col / cols,
                                     v_spread / (rows ** 2), h_spread / (cols ** 2),
                                     symmetry, edge_h / (total / (rows * cols)) if total > 1e-12 else 0.0,
                                     edge_v / (total / (rows * cols)) if total > 1e-12 else 0.0]
    return feat


def auditory_specific(matrix):
    """听觉专有特征。

    矩阵结构：8行 = 8个频带（低→高），64列 = 64时间帧。
    频谱结构是听觉的固有属性。
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # 频谱廓线：每行均值（8维）— 低频强还是高频强
    s_profile = [sum(matrix[r]) / cols for r in range(rows)]

    # 时间廓线：8段（每段8帧）— 前后能量分布
    group = cols // 8
    t_profile = []
    for g in range(8):
        s = 0.0
        count = 0
        for c in range(g * group, min((g + 1) * group, cols)):
            for r in range(rows):
                s += matrix[r][c]
                count += 1
        t_profile.append(s / count if count > 0 else 0.0)

    # 频谱重心（哪个频带主导）
    total = sum(sum(row) for row in matrix)
    if total < 1e-12:
        s_centroid = 0.0
        t_centroid = 0.0
        s_spread = 0.0
        harmonic_ratio = 0.0
        spectral_flux = 0.0
        peak_band = 0.0
        onset_rate = 0.0
    else:
        s_centroid = sum(r * sum(matrix[r]) for r in range(rows)) / total
        t_centroid = sum(c * sum(matrix[r][c] for r in range(rows)) for c in range(cols)) / total

        s_spread = sum((r - s_centroid) ** 2 * sum(matrix[r]) for r in range(rows)) / total

        # 谐波比：低频能量 / 总能量
        low_energy = sum(matrix[r][c] for r in range(rows // 2) for c in range(cols))
        harmonic_ratio = low_energy / total if total > 1e-12 else 0.0

        # 频谱通量：相邻帧频谱变化
        spectral_flux = 0.0
        for c in range(cols - 1):
            for r in range(rows):
                spectral_flux += abs(matrix[r][c + 1] - matrix[r][c])
        spectral_flux /= (rows * (cols - 1))

        # 峰值频带（能量最高行的位置）
        band_energy = [sum(matrix[r]) for r in range(rows)]
        peak_band = band_energy.index(max(band_energy)) / rows

        # 上升率：前 1/4 帧内的能量变化
        q1 = cols // 4
        early = sum(matrix[r][c] for r in range(rows) for c in range(q1))
        late = sum(matrix[r][c] for r in range(rows) for c in range(cols - q1, cols))
        onset_rate = (late - early) / total if total > 1e-12 else 0.0

    feat = s_profile + t_profile + [s_centroid / rows, t_centroid / cols,
                                     s_spread / (rows ** 2), harmonic_ratio,
                                     spectral_flux / (total / (rows * cols)) if total > 1e-12 else 0.0,
                                     peak_band, onset_rate]
    return feat


def tactile_specific(matrix):
    """触觉专有特征。

    矩阵：8行 = 8个触觉分区，64列 = 64个空间位点。
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # 分区压力廓线（8维）
    zone_profile = [sum(matrix[r]) / cols for r in range(rows)]

    # 峰值压力及位置
    total = sum(sum(row) for row in matrix)
    if total < 1e-12:
        peak_zone = 0.0
        peak_loc = 0.0
        pressure_spread = 0.0
        max_pressure = 0.0
        coverage = 0.0
        gradient = 0.0
    else:
        zone_energy = [sum(matrix[r]) for r in range(rows)]
        peak_zone = zone_energy.index(max(zone_energy)) / rows
        # 找到压力最大的行和列
        max_v = -1
        mr, mc = 0, 0
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] > max_v:
                    max_v = matrix[r][c]
                    mr, mc = r, c
        peak_loc = mc / cols
        pressure_spread = sum((r - peak_zone * rows) ** 2 * zone_energy[r] for r in range(rows)) / total
        max_pressure = max_v
        coverage = sum(1 for r in range(rows) for c in range(cols) if matrix[r][c] > max_v * 0.3) / (rows * cols)
        # 压力梯度（相邻位点的变化剧烈程度）
        gradient = sum(abs(matrix[r][c] - matrix[r][c-1]) for r in range(rows) for c in range(1, cols)) / (rows * (cols - 1))

    feat = zone_profile + [peak_zone, peak_loc, pressure_spread / (rows ** 2),
                            max_pressure, coverage, gradient / (total / (rows * cols)) if total > 1e-12 else 0.0]
    return feat


def thermal_specific(matrix):
    """温度觉专有特征。

    绝对温度是温度觉的核心特征（不同于其他模态的相对值）。
    """
    flat = [v for row in matrix for v in row]
    n = len(flat)

    mean_t = sum(flat) / n
    mx = max(flat)
    mn = min(flat)
    rng = mx - mn

    # 温度梯度（相邻帧温差）
    rows = len(matrix)
    cols = len(matrix[0])
    gradient = sum(abs(matrix[r][c] - matrix[r][c-1]) for r in range(rows) for c in range(1, cols)) / (rows * (cols - 1))

    # 高温面积占比（> mean + 0.5*std）
    std_t = (sum((v - mean_t) ** 2 for v in flat) / n) ** 0.5
    warm_area = sum(1 for v in flat if v > mean_t + 0.5 * std_t) / n
    cool_area = sum(1 for v in flat if v < mean_t - 0.5 * std_t) / n

    return [mean_t, mx, mn, rng, std_t, gradient, warm_area, cool_area]


def pain_specific(matrix):
    """痛觉专有特征。

    痛觉是稀疏的，核心特征：峰值、覆盖率、累积。
    """
    rows = len(matrix)
    cols = len(matrix[0])
    flat = [v for row in matrix for v in row]
    n = len(flat)

    peak = max(flat) if flat else 0.0
    mean_p = sum(flat) / n if n > 0 else 0.0

    # 覆盖率（>10% peak 的像素占比）
    coverage = sum(1 for v in flat if v > peak * 0.1) / n if peak > 1e-12 else 0.0

    # 痛觉灶数（连续高值区域数）
    rows_data = [sum(1 for v in row if v > peak * 0.3) for row in matrix]
    active_zones = sum(1 for rcnt in rows_data if rcnt > cols * 0.1)

    # 痛觉脉冲：最大梯度（相邻帧差）
    max_gradient = 0.0
    for r in range(rows):
        for c in range(1, cols):
            diff = abs(matrix[r][c] - matrix[r][c - 1])
            if diff > max_gradient:
                max_gradient = diff

    # 总累积（用于强度估计）
    total_p = sum(flat)

    return [peak, mean_p, coverage, active_zones / rows, max_gradient, total_p]


def lidar_specific(matrix):
    """LiDAR 专有特征。

    LiDAR 是距离传感器，核心特征：距离分布、最近点、点云密度。
    """
    rows = len(matrix)
    cols = len(matrix[0])
    flat = [abs(v) for row in matrix for v in row]
    n = len(flat)

    mean_d = sum(flat) / n if n > 0 else 0.0
    min_d = min(flat) if flat else 0.0
    max_d = max(flat) if flat else 0.0

    std_d = (sum((v - mean_d) ** 2 for v in flat) / n) ** 0.5 if n > 0 else 0.0

    # 点云密度（非零占比）
    density = sum(1 for v in flat if v > 1e-10) / n

    # 最近点方位（哪一行最近）
    row_min = [min(abs(v) for v in row) for row in matrix]
    nearest_row = row_min.index(min(row_min)) / rows if row_min else 0.0

    # 距离梯度（远→近的变化）
    gradient = sum(abs(matrix[r][c] - matrix[r][c-1]) for r in range(rows) for c in range(1, cols))

    return [mean_d, min_d, max_d, std_d, density, nearest_row, gradient / (rows * cols) if rows * cols > 0 else 0.0]


def generic_specific(matrix, modality_name):
    """通用回退：用原有的 6D。"""
    return 矩阵转六维(matrix)


# ════════════════════════════════════════════════════════════════
# 2. 测试数据
# ════════════════════════════════════════════════════════════════

def mountain_frame(kind="thin"):
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
                m[r, c] = max(0, (1 - d1)) * 5000 + max(0, (1 - d2)) * 4000
        m[5:8, :] = 200
    return m.tolist()


def audio_frame(kind="thin"):
    rows, cols = 8, 64
    base = np.zeros((rows, cols))
    if kind == "thin":
        base[0:3, 6:50] = 800; base[3:5, 6:50] = 1300; base[5:8, 6:50] = 2600
    elif kind == "wide":
        base[0:3, 6:50] = 600; base[3:5, 6:50] = 1000; base[5:8, 6:50] = 2200
    elif kind == "double":
        base[0:3, 6:50] = 750; base[3:5, 6:50] = 1400; base[5:8, 6:50] = 2800
    return base.tolist()


def text_to_visual(seed_text, rows=8, cols=64):
    """文字生成视觉（与真实图像对比用）。"""
    seed = [ord(c) % 256 + 1 for c in seed_text]
    visual = []
    for r in range(rows):
        row = []
        for c in range(cols):
            val = (seed[r % len(seed)] * 100 + c * seed[0]) % 12000
            row.append(float(val))
        visual.append(row)
    return visual


def text_to_auditory(seed_text, rows=8, cols=64):
    """文字生成听觉。"""
    seed = [ord(c) % 256 + 1 for c in seed_text]
    auditory = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            auditory[r][c] = float(seed[r % len(seed)] * 5 + 80)
    return auditory


# ════════════════════════════════════════════════════════════════
# 3. 辅助计算
# ════════════════════════════════════════════════════════════════

def l2_distance(a, b):
    """两个向量（任意维）的 L2 距离，先归一化再比。"""
    na = 归一化N维(list(a))
    nb = 归一化N维(list(b))
    # 同维直接比；不同维取短长度的公共尾部比较
    dim = min(len(na), len(nb))
    return sum((na[i] - nb[i]) ** 2 for i in range(dim)) ** 0.5


def l2_distance_same_modality(a, b):
    """同模态:feature 直接比。"""
    na = 归一化N维(list(a))
    nb = 归一化N维(list(b))
    return sum((na[i] - nb[i]) ** 2 for i in range(len(na))) ** 0.5


# ════════════════════════════════════════════════════════════════
# 4. 主测试
# ════════════════════════════════════════════════════════════════

print("=" * 80)
print("【模态专有特征测试】")
print("=" * 80)

# 提取各通道专有特征
mountain_img = mountain_frame("thin")
mountain_aud = audio_frame("thin")
text_visual_img = text_to_visual("山")
text_auditory = text_to_auditory("山")
text_visual_hello = text_to_visual("你好洛神")

# 专有特征
vis_spec = visual_specific(mountain_img)
aud_spec = auditory_specific(mountain_aud)
text_vis_spec = visual_specific(text_visual_img)
text_vis_hello_spec = visual_specific(text_visual_hello)
text_aud_spec = auditory_specific(text_auditory)

# 通用 6D 特征
vis_6d = 矩阵转六维(mountain_img)
aud_6d = 矩阵转六维(mountain_aud)
text_vis_6d = 矩阵转六维(text_visual_img)

print(f"\n── 特征维度对比 ──")
print(f"  视觉专有: {len(vis_spec)}维")
print(f"  听觉专有: {len(aud_spec)}维")
print(f"  通用 6D:  6维")

print(f"\n── 原始特征值 ──")
print(f"  山图视觉专有(前8): {[f'{v:.2f}' for v in vis_spec[:8]]}")
print(f"  山图视觉 6D:       {[f'{v:.2f}' for v in vis_6d]}")
print(f"  山音听觉专有(前8): {[f'{v:.2f}' for v in aud_spec[:8]]}")
print(f"  山音听觉 6D:       {[f'{v:.2f}' for v in aud_6d]}")
print(f"  文字图视觉专有(前8): {[f'{v:.2f}' for v in text_vis_spec[:8]]}")
print(f"  文字图视觉 6D:     {[f'{v:.2f}' for v in text_vis_6d]}")

# ══════════════════════════════════════════════════════════════
# TEST A: 同一东西，两路专有特征形状差多少？
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"【TEST A】同一山：专有特征 视觉 vs 听觉 形状差")
print(f"{'=' * 80}")

# 由于视觉专有(19维)和听觉专有(18维)维度不同，用 L2 距离对各自归一化后的前 min_dim 维
# 也报归一化后的向量做 6D 截断对比
min_dim = min(len(vis_spec), len(aud_spec))
vis_spec_trunc = vis_spec[:min_dim]
aud_spec_trunc = aud_spec[:min_dim]

cross_dist_spec = l2_distance_same_modality(vis_spec_trunc, aud_spec_trunc)
print(f"  专有特征 L2 距离(前{min_dim}维截断): {cross_dist_spec:.6f}")

# 通用 6D 跨模态
cross_dist_6d = l2_distance_same_modality(vis_6d, aud_6d)
print(f"  通用 6D L2 距离:                    {cross_dist_6d:.6f}")

# 用跨模态差分算 γ（两个都需要 6D 做 跨模态差分）
# 为了公平，用专有特征的前 6 维跑 跨模态差分
cd_spec = 跨模态差分("test_a", mountain_img, mountain_aud, "山图", "山音")
print(f"\n  【原 跨模态差分 γ】{cd_spec['gamma']:.6f}  (6D 特征)")

# ══════════════════════════════════════════════════════════════
# TEST B: 同一模态，不同东西，专有特征能否分开？
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"【TEST B】视觉：山图 vs 文字图，能否分开")
print(f"{'=' * 80}")

vis_dist_spec = l2_distance_same_modality(vis_spec, text_vis_spec)
vis_dist_6d_b = l2_distance_same_modality(vis_6d, text_vis_6d)
print(f"  专有特征 L2 距离: {vis_dist_spec:.6f}")
print(f"  通用 6D L2 距离:  {vis_dist_6d_b:.6f}")

# 增加：山图 vs "你好洛神" 文字图（更远的东西）
vis_dist_hello_spec = l2_distance_same_modality(vis_spec, text_vis_hello_spec)
vis_hello_6d = 矩阵转六维(text_visual_hello)
vis_dist_hello_6d = l2_distance_same_modality(vis_6d, vis_hello_6d)
print(f"\n  山图 vs 你好洛神图：")
print(f"    专有特征 L2 距离: {vis_dist_hello_spec:.6f}")
print(f"    通用 6D L2 距离:  {vis_dist_hello_6d:.6f}")

# 归一化后的视觉专有特征向量
print(f"\n  归一化视觉专有向量对比：")
nv = 归一化N维(vis_spec)
nt = 归一化N维(text_vis_spec)
print(f"    山图:   {[f'{v:.3f}' for v in nv[:10]]}")
print(f"    文字图: {[f'{v:.3f}' for v in nt[:10]]}")

# ══════════════════════════════════════════════════════════════
# TEST C: 同一模态，不同东西（听觉）
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"【TEST C】听觉：山音 vs 文字音")
print(f"{'=' * 80}")

# 文字"山"听觉 vs 文字"你好洛神"听觉
text_aud_hello = text_to_auditory("你好洛神")
text_aud_spec_hello = auditory_specific(text_aud_hello)

aud_dist_spec = l2_distance_same_modality(aud_spec, text_aud_spec)
aud_dist_hello_spec = l2_distance_same_modality(aud_spec, text_aud_spec_hello)
print(f"  山音 vs 文字山音  专有 L2: {aud_dist_spec:.6f}")
print(f"  山音 vs 你好洛神音 专有 L2: {aud_dist_hello_spec:.6f}")

# ══════════════════════════════════════════════════════════════
# TEST D: 综合对比 — 专有 vs 通用，同模态内 vs 跨模态
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"【TEST D】专有 vs 通用 6D — 区分度对比")
print(f"{'=' * 80}")

# 山 thin vs 山 wide（视觉）
mw = mountain_frame("wide")
vis_spec_wide = visual_specific(mw)
vis_6d_wide = 矩阵转六维(mw)

dist_varia_spec = l2_distance_same_modality(vis_spec, vis_spec_wide)
dist_varia_6d = l2_distance_same_modality(vis_6d, vis_6d_wide)
print(f"\n  视觉：山thin vs 山wide")
print(f"    专有 L2: {dist_varia_spec:.6f}")
print(f"    通用 6D: {dist_varia_6d:.6f}")

# 山 thin vs 山 double（视觉）
md = mountain_frame("double")
vis_spec_double = visual_specific(md)
vis_6d_double = 矩阵转六维(md)
dist_varia_spec2 = l2_distance_same_modality(vis_spec, vis_spec_double)
dist_varia_6d2 = l2_distance_same_modality(vis_6d, vis_6d_double)
print(f"\n  视觉：山thin vs 山double")
print(f"    专有 L2: {dist_varia_spec2:.6f}")
print(f"    通用 6D: {dist_varia_6d2:.6f}")

# 山 thin vs 文字图（视觉）
print(f"\n  视觉：山thin vs 文字山图")
print(f"    专有 L2: {vis_dist_spec:.6f}")
print(f"    通用 6D: {vis_dist_6d_b:.6f}")

# ──────────────────────────────────────────────────────────────
# 专有特征里，山thin vs 山变体 vs 文字图 的 完整矩阵
# ──────────────────────────────────────────────────────────────
print(f"\n── 视觉专有特征 · 完整距离矩阵 ──")
vis_all = {
    "山thin": visual_specific(mountain_frame("thin")),
    "山wide": visual_specific(mountain_frame("wide")),
    "山double": visual_specific(mountain_frame("double")),
    "文字山": visual_specific(text_to_visual("山")),
    "文字你好": visual_specific(text_to_visual("你好洛神")),
}
print(f"{'':>10}", end="")
for k in vis_all:
    print(f"  {k:>10}", end="")
print()
for k1 in vis_all:
    print(f"{k1:>10}", end="")
    for k2 in vis_all:
        if k1 == k2:
            print(f"  {'—':>10}", end="")
        else:
            d = l2_distance_same_modality(vis_all[k1], vis_all[k2])
            print(f"  {d:>10.4f}", end="")
    print()

print(f"\n── 通用 6D · 完整距离矩阵 ──")
vis_all_6d = {
    "山thin": 矩阵转六维(mountain_frame("thin")),
    "山wide": 矩阵转六维(mountain_frame("wide")),
    "山double": 矩阵转六维(mountain_frame("double")),
    "文字山": 矩阵转六维(text_to_visual("山")),
    "文字你好": 矩阵转六维(text_to_visual("你好洛神")),
}
print(f"{'':>10}", end="")
for k in vis_all_6d:
    print(f"  {k:>10}", end="")
print()
for k1 in vis_all_6d:
    print(f"{k1:>10}", end="")
    for k2 in vis_all_6d:
        if k1 == k2:
            print(f"  {'—':>10}", end="")
        else:
            d = l2_distance_same_modality(vis_all_6d[k1], vis_all_6d[k2])
            print(f"  {d:>10.4f}", end="")
    print()

# ──────────────────────────────────────────────────────────────
# 听觉距离矩阵
# ──────────────────────────────────────────────────────────────
aud_all_spec = {
    "山thin": auditory_specific(audio_frame("thin")),
    "山wide": auditory_specific(audio_frame("wide")),
    "山double": auditory_specific(audio_frame("double")),
    "文字山": auditory_specific(text_to_auditory("山")),
    "文字你好": auditory_specific(text_to_auditory("你好洛神")),
}
aud_all_6d = {
    "山thin": 矩阵转六维(audio_frame("thin")),
    "山wide": 矩阵转六维(audio_frame("wide")),
    "山double": 矩阵转六维(audio_frame("double")),
    "文字山": 矩阵转六维(text_to_auditory("山")),
    "文字你好": 矩阵转六维(text_to_auditory("你好洛神")),
}

print(f"\n── 听觉专有特征 · 完整距离矩阵 ──")
print(f"{'':>10}", end="")
for k in aud_all_spec:
    print(f"  {k:>10}", end="")
print()
for k1 in aud_all_spec:
    print(f"{k1:>10}", end="")
    for k2 in aud_all_spec:
        if k1 == k2:
            print(f"  {'—':>10}", end="")
        else:
            d = l2_distance_same_modality(aud_all_spec[k1], aud_all_spec[k2])
            print(f"  {d:>10.4f}", end="")
    print()

print(f"\n── 听觉通用 6D · 完整距离矩阵 ──")
print(f"{'':>10}", end="")
for k in aud_all_6d:
    print(f"  {k:>10}", end="")
print()
for k1 in aud_all_6d:
    print(f"{k1:>10}", end="")
    for k2 in aud_all_6d:
        if k1 == k2:
            print(f"  {'—':>10}", end="")
        else:
            d = l2_distance_same_modality(aud_all_6d[k1], aud_all_6d[k2])
            print(f"  {d:>10.4f}", end="")
    print()

# ══════════════════════════════════════════════════════════════
# TEST E: 区分度指标（类间 / 类内 距离比）
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"【TEST E】区分度指标：类间/类内 距离比")
print(f"{'=' * 80}")

# 视觉：类内距离（山 thin vs 山 wide） vs 类间距离（山 thin vs 文字）
vis_intra_spec = l2_distance_same_modality(vis_all["山thin"], vis_all["山wide"])
vis_inter_spec = l2_distance_same_modality(vis_all["山thin"], vis_all["文字山"])
vis_intra_6d = l2_distance_same_modality(vis_all_6d["山thin"], vis_all_6d["山wide"])
vis_inter_6d = l2_distance_same_modality(vis_all_6d["山thin"], vis_all_6d["文字山"])

print(f"\n  视觉 thin vs wide (类内):")
print(f"    专有 L2: {vis_intra_spec:.6f}")
print(f"    通用 6D: {vis_intra_6d:.6f}")
print(f"  视觉 thin vs 文字山 (类间):")
print(f"    专有 L2: {vis_inter_spec:.6f}")
print(f"    通用 6D: {vis_inter_6d:.6f}")
print(f"  区分度比 (类间/类内):")
print(f"    专有: {vis_inter_spec / vis_intra_spec:.4f}" if vis_intra_spec > 1e-12 else f"    专有: INF")
print(f"    通用: {vis_inter_6d / vis_intra_6d:.4f}" if vis_intra_6d > 1e-12 else f"    通用: INF")

# 听觉
aud_intra_spec = l2_distance_same_modality(aud_all_spec["山thin"], aud_all_spec["山wide"])
aud_inter_spec = l2_distance_same_modality(aud_all_spec["山thin"], aud_all_spec["文字山"])
aud_intra_6d = l2_distance_same_modality(aud_all_6d["山thin"], aud_all_6d["山wide"])
aud_inter_6d = l2_distance_same_modality(aud_all_6d["山thin"], aud_all_6d["文字山"])

print(f"\n  听觉 thin vs wide (类内):")
print(f"    专有 L2: {aud_intra_spec:.6f}")
print(f"    通用 6D: {aud_intra_6d:.6f}")
print(f"  听觉 thin vs 文字山 (类间):")
print(f"    专有 L2: {aud_inter_spec:.6f}")
print(f"    通用 6D: {aud_inter_6d:.6f}")
print(f"  区分度比 (类间/类内):")
print(f"    专有: {aud_inter_spec / aud_intra_spec:.4f}" if aud_intra_spec > 1e-12 else f"    专有: INF")
print(f"    通用: {aud_inter_6d / aud_intra_6d:.4f}" if aud_intra_6d > 1e-12 else f"    通用: INF")

# 山的 thin vs double vs wide 三组类内取平均
vis_intra_avg_spec = np.mean([
    l2_distance_same_modality(vis_all["山thin"], vis_all["山wide"]),
    l2_distance_same_modality(vis_all["山thin"], vis_all["山double"]),
    l2_distance_same_modality(vis_all["山wide"], vis_all["山double"]),
])
vis_intra_avg_6d = np.mean([
    l2_distance_same_modality(vis_all_6d["山thin"], vis_all_6d["山wide"]),
    l2_distance_same_modality(vis_all_6d["山thin"], vis_all_6d["山double"]),
    l2_distance_same_modality(vis_all_6d["山wide"], vis_all_6d["山double"]),
])
vis_inter_avg_spec = np.mean([
    l2_distance_same_modality(vis_all["山thin"], vis_all["文字山"]),
    l2_distance_same_modality(vis_all["山thin"], vis_all["文字你好"]),
    l2_distance_same_modality(vis_all["山wide"], vis_all["文字山"]),
    l2_distance_same_modality(vis_all["山wide"], vis_all["文字你好"]),
])
vis_inter_avg_6d = np.mean([
    l2_distance_same_modality(vis_all_6d["山thin"], vis_all_6d["文字山"]),
    l2_distance_same_modality(vis_all_6d["山thin"], vis_all_6d["文字你好"]),
    l2_distance_same_modality(vis_all_6d["山wide"], vis_all_6d["文字山"]),
    l2_distance_same_modality(vis_all_6d["山wide"], vis_all_6d["文字你好"]),
])

print(f"\n  视觉 均类内 L2: 专有={vis_intra_avg_spec:.6f} 通用={vis_intra_avg_6d:.6f}")
print(f"  视觉 均类间 L2: 专有={vis_inter_avg_spec:.6f} 通用={vis_inter_avg_6d:.6f}")
print(f"  视觉 均区分度比: 专有={vis_inter_avg_spec/vis_intra_avg_spec:.4f} 通用={vis_inter_avg_6d/vis_intra_avg_6d:.4f}"
      if vis_intra_avg_spec > 1e-12 and vis_intra_avg_6d > 1e-12 else "")

# ══════════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"【总结】")
print(f"{'=' * 80}")
print(f"  1. 专有特征维度：视觉 {len(vis_spec)}D，听觉 {len(aud_spec)}D（vs 通用 6D）")
print(f"  2. 同一山（thin）：视觉专有 vs 听觉专有 L2 = {cross_dist_spec:.6f}")
print(f"  3. 同一山（thin）：视觉 6D vs 听觉 6D L2 = {cross_dist_6d:.6f}")
print(f"  4. 视觉专有 区分度比(类间/类内) = {vis_inter_spec/vis_intra_spec:.4f}" if vis_intra_spec > 1e-12 else "")
print(f"     视觉通用6D 区分度比(类间/类内) = {vis_inter_6d/vis_intra_6d:.4f}" if vis_intra_6d > 1e-12 else "")
print(f"  5. 听觉专有 区分度比(类间/类内) = {aud_inter_spec/aud_intra_spec:.4f}" if aud_intra_spec > 1e-12 else "")
print(f"     听觉通用6D 区分度比(类间/类内) = {aud_inter_6d/aud_intra_6d:.4f}" if aud_intra_6d > 1e-12 else "")
