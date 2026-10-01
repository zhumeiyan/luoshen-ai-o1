#!/usr/bin/env python3
"""
=============================================================
  洛神自学作业 02 — 特征匹配与对极几何
=============================================================

【前情】
  作业01学了:一个3D点是怎么投影到一张图上的。
  ATLAS 的核心问题是反过来:给你两张同一个场景的图,重建3D。

  这两张图之间有什么联系?
   → 同一个物理点在两张图中的像素位置有几何关系。
   → 这个关系由"本质矩阵 E = [t]× R"描述。

  这份作业你会在 numpy 上实现:
   Part 1: 特征匹配 (NCC / 归一化互相关) —— 手工版 SIFT
   Part 2: 八点算法 (8-point algorithm) —— 从匹配点对估计本质矩阵 E
   Part 3: 从 E 恢复相机运动 [R|t]

  不调用任何深度学习。纯线性代数。

【环境要求】
  numpy + PIL,不需要 torch/cv2

【本作业延续作业01的数据】
  场景:立方体 + 地面网格
  相机拍两张图(不同角度),洛神要:匹配点对 → 估计E → 恢复相机运动 → 重建稀疏3D点
"""

import numpy as np
from PIL import Image, ImageDraw
import math
import os

# ============================================================
# 与作业01完全相同的场景和相机参数
# ============================================================

cube_vertices = np.array([
    [-10,  0, -10, 255,   0,   0],   # 0
    [ 10,  0, -10,   0, 255,   0],   # 1
    [ 10,  0,  10,   0,   0, 255],   # 2
    [-10,  0,  10, 255, 255,   0],   # 3
    [-10, 20, -10, 255,   0, 255],   # 4
    [ 10, 20, -10,   0, 255, 255],   # 5
    [ 10, 20,  10, 255, 255, 255],   # 6
    [-10, 20,  10, 128, 128, 128],   # 7
], dtype=np.float64)

cube_edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

W, H = 640, 480
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
K_inv = np.linalg.inv(K)  # 内参的逆,用于反投影

# 两个相机位置 (相机2相对相机1有旋转+平移)
cam1_pos = np.array([0, 30, -50], dtype=np.float64)
cam2_pos = np.array([15, 35, -45], dtype=np.float64)
look_at   = np.array([0, 10, 0], dtype=np.float64)
up        = np.array([0, 1, 0], dtype=np.float64)

def build_R(cam_pos, target, up):
    z_axis = target - cam_pos
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up.astype(np.float64), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.array([x_axis, y_axis, z_axis])

R1 = build_R(cam1_pos, look_at, up)
R2 = build_R(cam2_pos, look_at, up)
t1 = -R1 @ cam1_pos
t2 = -R2 @ cam2_pos

# 相机2 相对 相机1 的运动 [R_rel | t_rel]
# 含义: P_cam2 = R_rel @ P_cam1 + t_rel
R_rel = R2 @ R1.T
t_rel = t2 - R_rel @ t1


def project(point_3d, K, R, t):
    P_cam = R @ point_3d + t
    if P_cam[2] <= 1.0:
        return None, None, P_cam[2]
    proj = K @ P_cam
    u, v = proj[0]/proj[2], proj[1]/proj[2]
    return u, v, P_cam[2]

# 投影到两张图
proj1 = []
proj2 = []
for vtx in cube_vertices:
    xyz = vtx[:3]
    c1 = project(xyz, K, R1, t1)
    c2 = project(xyz, K, R2, t2)
    proj1.append(c1)
    proj2.append(c2)

print("=== 两张图的立方体顶点投影 ===")
print(f"{'顶点':>4s}   {'图1(u,v)':>18s}   {'图2(u,v)':>18s}")
print("-" * 55)
for i in range(8):
    s1 = f"({proj1[i][0]:.1f}, {proj1[i][1]:.1f})" if proj1[i][0] else "不可见"
    s2 = f"({proj2[i][0]:.1f}, {proj2[i][1]:.1f})" if proj2[i][0] else "不可见"
    print(f"  {i:2d}   {s1:>18s}   {s2:>18s}")
print()


# ============================================================
# 第1部分: NCC 特征匹配 (Normalization Cross-Correlation)
# ============================================================

print("=" * 70)
print("第1部分:NCC 特征匹配")
print("=" * 70)
print()

# 【NCC 原理】
# 来自维基百科的公式:
#   NCC = Σ (patch1 - μ1)(patch2 - μ2) / [sqrt(Σ(p1-μ1)²) · sqrt(Σ(p2-μ2)²)]
# 取值范围 [-1, 1]:
#   1 = 完全一致, 0 = 无关, -1 = 完全相反
# 本质:衡量两个小块(patch)有多像。不依赖亮度偏移。

def extract_patch(image_array, cx, cy, half_size):
    """
    从图中提取以 (cx,cy) 为中心的 patch。
    返回: patch 数组 (2*half_size+1, 2*half_size+1),越界则返回 None。

    【TODO 1.1】填写这个函数
    """
    h, w = image_array.shape
    x0 = cx - half_size
    x1 = cx + half_size + 1
    y0 = cy - half_size
    y1 = cy + half_size + 1

    # [TODO 1.1a] 检查边界:如果 patch 超出图像边界,返回 None
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    # [TODO 1.1b] 提取子数组并返回
    patch = image_array[y0:y1, x0:x1].copy()
    return patch


def ncc(patch1, patch2):
    """
    计算两个 patch 之间的归一化互相关 (NCC)。

    【TODO 1.2】填写 NCC 计算
    公式: NCC = mean((p1 - mean(p1)) * (p2 - mean(p2))) / (std(p1) * std(p2))
    """
    # [TODO 1.2a] 将 patch 扁平化为 1D 并转为 float64
    p1 = patch1.flatten().astype(np.float64)
    p2 = patch2.flatten().astype(np.float64)

    # [TODO 1.2b] 减去均值
    p1 = p1 - np.mean(p1)
    p2 = p2 - np.mean(p2)

    # [TODO 1.2c] 计算 NCC 分子和分母
    numerator = np.sum(p1 * p2)
    denominator = np.sqrt(np.sum(p1 ** 2) * np.sum(p2 ** 2))

    # [TODO 1.2d] 避免除零
    if denominator < 1e-10:
        return 0.0

    return numerator / denominator


# 为演示匹配,我们在图像1中生成一些特征点位置
# 特征点就是立方体的8个顶点在图1中的位置
feature_points_img1 = []
for i, (u, v, d) in enumerate(proj1):
    if u is not None and 10 <= u < W-10 and 10 <= v < H-10:
        feature_points_img1.append((i, int(round(u)), int(round(v))))

print(f"图1特征点(立方体顶点投影):{len(feature_points_img1)}个")
for idx, u, v in feature_points_img1:
    print(f"  顶点{idx}: ({u}, {v})")
print()


# 【手算验证】
# 手动构造两个完全相同的 patch,NCC 应该 = 1.0
patch_a = np.array([
    [10, 20, 30],
    [15, 25, 35],
    [12, 22, 32]
], dtype=np.uint8)
patch_b = patch_a.copy()

ncc_identical = ncc(patch_a, patch_b)
print(f"【验证1】完全相同的 patch: NCC = {ncc_identical:.6f}")
assert abs(ncc_identical - 1.0) < 0.001, "TODO 1.2 未完成:NCC(相同patch)应=1"
print("  ✓ 相同 patch → NCC = 1.0")
print()

# 两个不相同的 patch
patch_c = np.array([
    [100, 200, 50],
    [80,  150, 30],
    [40,  90,  10]
], dtype=np.uint8)
ncc_diff = ncc(patch_a, patch_c)
print(f"  不同 patch 之间: NCC = {ncc_diff:.4f} (这个值应该远小于1)")
print()


# ============================================================
# 第2部分: 匹配 —— 在图2中找图1特征点的对应点
# ============================================================

print("=" * 70)
print("第2部分:在图2中找到图1特征点的对应点")
print("=" * 70)
print()

# 【匹配任务】
# 已知:图1的特征点坐标 + 立方体顶点 ID
# 已知:图2的顶点投影坐标
# 但由于"不知道对应关系",假装我们不知道哪个2D点是哪个立方体顶点。
# 要在图2的所有候选点中,用 NCC 找最佳匹配。

# 【TODO 2.1】实现 NCC 搜索匹配
def find_best_match_ncc(patch1, search_center_u, search_center_v, img2_array, search_radius=30, patch_half=5):
    """
    在 img2 中以 (search_center_u, search_center_v) 为中心,
    在 ±search_radius 的窗口内逐像素搜索,找与 patch1 的 NCC 最高的点。

    返回: (best_u, best_v, best_ncc)
    """
    best_ncc = -999
    best_u, best_v = search_center_u, search_center_v

    # [TODO 2.1a] 遍历搜索窗口内的每个像素位置
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            cu = search_center_u + dx
            cv = search_center_v + dy

            # [TODO 2.1b] 提取 img2 中该位置的 patch
            patch2 = extract_patch(img2_array, cu, cv, patch_half)

            if patch2 is None:
                continue

            # [TODO 2.1c] 计算 NCC
            n = ncc(patch1, patch2)

            # [TODO 2.1d] 更新最佳匹配
            if n > best_ncc:
                best_ncc = n
                best_u = cu
                best_v = cv

    return best_u, best_v, best_ncc


# 因为我们没有真正的图像(没有真实照片),
# 用顶点的已知对应关系来模拟 NCC 匹配结果:
# (这相当于 NCC 完美匹配到了正确点)

matched_pairs = []  # (顶点ID, u1, v1, u2, v2)
for i, vtx in enumerate(cube_vertices):
    u1, v1, d1 = proj1[i]
    u2, v2, d2 = proj2[i]
    if u1 is not None and u2 is not None:
        matched_pairs.append((i, int(round(u1)), int(round(v1)), int(round(u2)), int(round(v2))))

print(f"匹配点对数量: {len(matched_pairs)}")
print()
print(f"{'顶点':>4s}  {'图1(u,v)':>14s}  {'图2(u,v)':>14s}  {'位移Δu':>8s}")
print("-" * 55)
for vid, u1, v1, u2, v2 in matched_pairs:
    du = u2 - u1
    print(f"  {vid:2d}   ({u1:4d},{v1:4d})    ({u2:4d},{v2:4d})    {du:+6.1f}")
print()

# 【思考题】
# 为什么同一个立方体顶点在两张图中的像素坐标不一样?
# 答:因为相机移动了。但不同顶点的位移量(du)并不相同——为什么?
# 提示:远处的物体位移小,近处的物体位移大。这是"视差"。


# ============================================================
# 第3部分: 八点算法 (8-Point Algorithm)
# ============================================================

print("=" * 70)
print("第3部分:八点算法估计本质矩阵 E")
print("=" * 70)
print()

# 【本质矩阵 E 的物理含义】
# 对于一对匹配点 (p1 in image1, p2 in image2),
# 它们满足对极约束:
#   p2^T * E * p1 = 0
# 其中 p1, p2 是归一化平面坐标 (= K_inv @ [u,v,1])
# E = [t]_x * R (本质矩阵 = 反对称矩阵(t) × 旋转矩阵(R))

# 【八点算法原理】
# 展开 p2^T * E * p1 = 0,可以得到一个线性方程:
#   A @ e = 0
# 其中 e 是 E 展开为 9 维向量 (E是3x3),
# A 是 N×9 矩阵 (N ≥ 8对匹配点)。
# 解法:e 是 A^T @ A 的最小特征值对应的特征向量。
# (最小二乘解,因为噪声存在时 A@e 不会完全等于0)


# 【TODO 3.1】将像素坐标转换为归一化平面坐标
def normalize_points(uv_list, K_inv):
    """
    将像素坐标列表 [[u1,v1],[u2,v2],...] 转为归一化平面坐标。
    p_normalized = K_inv @ [u, v, 1]^T
    然后只取前两个分量 (x, y),因为第三分量恒为1。

    【TODO 3.1】
    """
    norm_points = []
    for u, v in uv_list:
        # 齐次坐标
        p_homo = np.array([u, v, 1.0])
        # [TODO 3.1a] 用内参逆矩阵变换
        p_norm = K_inv @ p_homo
        # 归一化平面坐标 (取xy)
        norm_points.append([p_norm[0], p_norm[1]])
    return np.array(norm_points)


# 提取匹配点的像素坐标
pts1_uv = np.array([[u1, v1] for _, u1, v1, _, _ in matched_pairs])
pts2_uv = np.array([[u2, v2] for _, _, _, u2, v2 in matched_pairs])

pts1_norm = normalize_points(pts1_uv, K_inv)
pts2_norm = normalize_points(pts2_uv, K_inv)

print("归一化平面坐标 (部分示例):")
print(f"{'顶点':>4s}   {'归一化p1(x,y)':>20s}   {'归一化p2(x,y)':>20s}")
for k in range(min(4, len(matched_pairs))):
    vid = matched_pairs[k][0]
    print(f"  {vid:2d}   ({pts1_norm[k,0]:+7.4f}, {pts1_norm[k,1]:+7.4f})"
          f"   ({pts2_norm[k,0]:+7.4f}, {pts2_norm[k,1]:+7.4f})")
print()


# 【TODO 3.2】构建矩阵 A (每对匹配点贡献一行)
def build_matrix_A(pts1_norm, pts2_norm):
    """
    对每对匹配点 (x1,y1) in img1, (x2,y2) in img2,
    A 的一行 = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    这是 p2^T * E * p1 = 0 的展开:
      [x2,y2,1] * [[e1,e2,e3],[e4,e5,e6],[e7,e8,e9]] * [x1,y1,1]^T = 0
    展开后 = x2*x1*e1 + x2*y1*e2 + x2*e3 + y2*x1*e4 + y2*y1*e5 + y2*e6 + x1*e7 + y1*e8 + e9 = 0

    【TODO 3.2】
    """
    N = len(pts1_norm)
    A = np.zeros((N, 9))

    for i in range(N):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        # [TODO 3.2a] 填写 A 的第 i 行
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    return A


# 【注意:数值稳定性】
# 经典八点算法的一个严重问题:如果坐标没有归一化,A 的条件数会很大,
# 导致 SVD 分解数值不稳定。Hartley (1997) 提出标准做法:
#   1. 坐标平移到均值处
#   2. 坐标缩放到平均距离 = sqrt(2)
# 估计出 E 之后再反归一化回来。
# 这就是 "Normalized 8-point algorithm",是实际应用中的标准做法.

def normalize_coords(uv_list):
    """
    Hartley 归一化:平移 + 缩放,使点的质心在原点,平均到原点距离为 sqrt(2)。

    返回: normalized_coords, T (3x3 变换矩阵)
    uv_list: [[u1,v1],[u2,v2],...]
    """
    pts = np.array(uv_list, dtype=np.float64)
    mean_u = np.mean(pts[:, 0])
    mean_v = np.mean(pts[:, 1])

    # 平移到原点
    centered = pts - np.array([mean_u, mean_v])

    # 计算平均距离
    dists = np.sqrt(centered[:, 0]**2 + centered[:, 1]**2)
    mean_dist = np.mean(dists)

    if mean_dist < 1e-10:
        scale = 1.0
    else:
        scale = math.sqrt(2) / mean_dist

    # 归一化坐标
    norm_pts = centered * scale

    # 变换矩阵 (齐次坐标)
    T = np.array([
        [scale, 0,     -scale * mean_u],
        [0,     scale, -scale * mean_v],
        [0,     0,     1]
    ])

    return norm_pts, T


# Hartley 归一化
pts1_for_norm = np.array(pts1_uv, dtype=np.float64)  # 用像素坐标做归一化
pts2_for_norm = np.array(pts2_uv, dtype=np.float64)

pts1_n, T1 = normalize_coords(pts1_for_norm)
pts2_n, T2 = normalize_coords(pts2_for_norm)

print(f"Hartley 归一化:")
print(f"  图1质心: ({np.mean(pts1_for_norm[:,0]):.1f}, {np.mean(pts1_for_norm[:,1]):.1f}) → (0, 0)")
print(f"  图2质心: ({np.mean(pts2_for_norm[:,0]):.1f}, {np.mean(pts2_for_norm[:,1]):.1f}) → (0, 0)")
print()

# 在归一化坐标上计算本质矩阵

def normalize_plane_from_pixel(uv_list, T):
    """用 Hartley 变换 T 将像素坐标转到归一化平面"""
    norm = []
    for u, v in uv_list:
        p = np.array([u, v, 1.0])
        p_t = T @ p
        norm.append([p_t[0]/p_t[2], p_t[1]/p_t[2]])
    return np.array(norm)


# 先用像素坐标 + Hartley 归一化,然后再转到归一化平面
# 等价于:p_norm = K_inv @ [u,v,1],然后 Hartley 归一化
def pixel_to_normalized_plane_hartley(uv_list, K):
    """像素坐标 → 归一化平面坐标 (含 Hartley 归一化)"""
    norm_pts = []
    for u, v in uv_list:
        p = K_inv @ np.array([u, v, 1.0])
        norm_pts.append([p[0], p[1]])
    return np.array(norm_pts)


pts1_np = pixel_to_normalized_plane_hartley(pts1_uv, K)
pts2_np = pixel_to_normalized_plane_hartley(pts2_uv, K)

# 在这些坐标上再做 Hartley 归一化
pts1_np_normed, Te1 = normalize_coords(pts1_np)
pts2_np_normed, Te2 = normalize_coords(pts2_np)

print(f"归一化平面坐标 (Hartley归一化后):")
for k in range(min(4, len(matched_pairs))):
    vid = matched_pairs[k][0]
    print(f"  顶点{vid}: ({pts1_np_normed[k,0]:+.4f}, {pts1_np_normed[k,1]:+.4f}) → ({pts2_np_normed[k,0]:+.4f}, {pts2_np_normed[k,1]:+.4f})")
print()

# 八点算法
A = build_matrix_A(pts1_np_normed, pts2_np_normed)
print(f"矩阵 A: shape = {A.shape}")
print()

# 条件数
cond = np.linalg.cond(A)
print(f"A 的条件数: {cond:.2f} (越小越稳定)")
print()

print(f"A 的前3行:")
for i in range(min(3, len(A))):
    row_str = " ".join([f"{x:+.4f}" for x in A[i]])
    print(f"  [{row_str}]")
print()


# 【TODO 3.3】求解本质矩阵 E (最小特征值对应的特征向量)
def compute_essential_matrix(A):
    """
    八点算法的核心:SVD 分解 A,解 e 是 V^T 的最后一行(= V 的最后一列)。
    等价地:计算 A^T @ A,找最小特征值对应的特征向量。

    【TODO 3.3】使用 SVD 分解
    """
    # [TODO 3.3a] 对 A 做 SVD 分解
    U, S, Vt = np.linalg.svd(A)

    # [TODO 3.3b] 解 e 是 V^T 的最后一行 (最小奇异值对应的行)
    e = Vt[-1]

    # [TODO 3.3c] e 重排成 3x3 矩阵
    E = e.reshape(3, 3)

    return E


E_normalized = compute_essential_matrix(A)
print("估计的本质矩阵 E (Hartley 归一化坐标系中):")
print(E_normalized)
print()

# 【验证】本质矩阵的奇异值理论上是 [σ, σ, 0] (因为 rank(E)=2)
U_E, S_E, Vt_E = np.linalg.svd(E_normalized)
print(f"E 的奇异值: {S_E}")
print(f"理论:应该是 [σ, σ, 0] 的形式")
print(f"比值: σ1/σ2={S_E[0]/S_E[1]:.4f}, σ3={S_E[2]:.6f}")
print()


# 强制 E 满足规范 (两个奇异值相等)
E_norm_corrected = E_normalized.copy()
U, S, Vt = np.linalg.svd(E_norm_corrected)
S_corrected = [(S[0] + S[1]) / 2.0, (S[0] + S[1]) / 2.0, 0.0]
E_norm_corrected = U @ np.diag(S_corrected) @ Vt
print("修正后的 E_norm (rank=2, 等奇异值):")
print(E_norm_corrected)
print()

# 反归一化:从 Hartley 归一化坐标系转回原始归一化平面坐标
# E_original = Te2^T @ E_norm @ Te1
E_correct = Te2.T @ E_norm_corrected @ Te1
# 再缩放到规范大小
E_correct = E_correct / np.linalg.norm(E_correct)
print("反归一化后的 E (原始坐标系中):")
print(E_correct)
print()


# ============================================================
# 【手算验证八点算法】
# ============================================================

print("=" * 70)
print("【手算验证】对极约束: p2^T * E * p1 = 0")
print("=" * 70)
print()

# 取第一对匹配点手算验证
k = 0
p1 = pts1_norm[k]  # (x1, y1)
p2 = pts2_norm[k]  # (x2, y2)

# 齐次坐标
p1_h = np.array([p1[0], p1[1], 1.0])
p2_h = np.array([p2[0], p2[1], 1.0])

print(f"第{k}对匹配点:")
print(f"  p1_norm = {p1}")
print(f"  p2_norm = {p2}")
print()

# 步骤1: E @ p1
Ep1 = E_correct @ p1_h
print(f"  E @ p1 = {Ep1}")
print()

# 步骤2: p2^T @ (E @ p1)
p2TEp1 = p2_h @ Ep1
print(f"  p2^T @ E @ p1 = {p2TEp1:.6f}")
print(f"  (应该≈0)")
print()

# 【验证所有匹配点都满足约束】
print("所有匹配点的对极约束值:")
max_violation = 0
for i in range(len(matched_pairs)):
    p1h = np.array([pts1_norm[i, 0], pts1_norm[i, 1], 1.0])
    p2h = np.array([pts2_norm[i, 0], pts2_norm[i, 1], 1.0])
    val = abs(p2h @ (E_correct @ p1h))
    vid = matched_pairs[i][0]
    print(f"  顶点{vid}: |p2^T E p1| = {val:.6f}")
    max_violation = max(max_violation, val)

print()
print(f"最大约束违反: {max_violation:.6f}")
print(f" (Hartley 归一化后,这个值应该很小。理论上=0,数值计算有微小误差)")
print()


# ============================================================
# 第4部分: 从本质矩阵 E 恢复相机运动
# ============================================================

print("=" * 70)
print("第4部分:从 E 恢复相机运动 [R|t]")
print("=" * 70)
print()

# 【数学原理】
# E = [t]_x * R
# 对 E 做 SVD: E = U * diag(1,1,0) * V^T
# 可能的解 (4种组合):
#   R = U * W * V^T  或  U * W^T * V^T
#   [t]_x = U * diag(1,1,0) * U^T  →  t = ± U[:,2]
# 其中 W = [[0,-1,0],[1,0,0],[0,0,1]]

W_matrix = np.array([[0, -1, 0],
                     [1,  0, 0],
                     [0,  0, 1]], dtype=np.float64)


# 【TODO 4.1】实现 E→[R,t] 的恢复
def recover_pose_from_essential(E, pts1_norm, pts2_norm, K):
    """
    从本质矩阵恢复相机姿态 (4种可能,只有一种物理正确)。

    返回: solutions = [(R1, t1), (R2, t2), (R3, t3), (R4, t4)]
    其中只有一种情况让所有3D点都在两个相机的前面。

    【TODO 4.1】
    """
    # Step 1: SVD 分解 E
    U, S, Vt = np.linalg.svd(E)
    V = Vt.T

    # 确保 det(R)=1 (右手系)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(V) < 0:
        V *= -1

    # Step 2: 两种旋转 × 两种平移 = 4种解
    R1 = U @ W_matrix @ V.T
    R2 = U @ W_matrix.T @ V.T
    t1 = U[:, 2]   # 第三列
    t2 = -U[:, 2]  # 第三列取反

    candidates = [
        (R1, t1),
        (R1, t2),
        (R2, t1),
        (R2, t2),
    ]

    return candidates


def triangulate_point(p1_norm, p2_norm, R, t, K):
    """
    给定一对归一化平面坐标和相机运动,三角化出3D点。

    【TODO 4.2】填写三角测量
    简化版:用线性三角测量 (DLT)

    设 P1 = K[I|0], P2 = K[R|t]
    对点 p1 和 p2,3D点 X 满足:
      cross(p1, P1 X) = 0
      cross(p2, P2 X) = 0
    展开为 AX = 0,SVD 求解。

    这里用更直接的方法:
    X = midpoint method (两个射线最近点之间的中点)
    """
    # 相机1在原点,看向+Z
    ray1_dir = np.array([p1_norm[0], p1_norm[1], 1.0])
    ray1_dir /= np.linalg.norm(ray1_dir)

    # 相机2
    ray2_dir = R.T @ np.array([p2_norm[0], p2_norm[1], 1.0])
    ray2_dir /= np.linalg.norm(ray2_dir)
    cam2_center = -R.T @ t

    # 求两条射线的最近点
    # ray1 = s1 * ray1_dir (从原点出发)
    # ray2 = cam2_center + s2 * ray2_dir
    w0 = -cam2_center
    a = np.dot(ray1_dir, ray1_dir)
    b = np.dot(ray1_dir, ray2_dir)
    c = np.dot(ray2_dir, ray2_dir)
    d = np.dot(ray1_dir, w0)
    e = np.dot(ray2_dir, w0)

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        # 射线平行,返回中点
        return np.array([0, 0, 50])

    s1 = (b * e - c * d) / denom
    s2 = (a * e - b * d) / denom

    # 最近点(中点)
    point_3d = (s1 * ray1_dir + cam2_center + s2 * ray2_dir) / 2.0

    return point_3d


def count_front_points(pts1_norm, pts2_norm, R, t, K):
    """统计有多少三角化点在两个相机前面 (深度>0)"""
    front = 0
    total = len(pts1_norm)
    for i in range(total):
        X = triangulate_point(pts1_norm[i], pts2_norm[i], R, t, K)

        # 相机1的深度 = X 的 Z 分量 (相机1在原点)
        depth1 = X[2]

        # 相机2的深度
        X_cam2 = R @ X + t
        depth2 = X_cam2[2]

        if depth1 > 0 and depth2 > 0:
            front += 1

    return front


# 恢复4种解
candidates = recover_pose_from_essential(E_correct, pts1_norm, pts2_norm, K)

print("== 4种可能的相机运动 ==")
print()

best_idx = -1
best_front = 0

for idx, (R_cand, t_cand) in enumerate(candidates):
    front = count_front_points(pts1_norm, pts2_norm, R_cand, t_cand, K)
    det = np.linalg.det(R_cand)
    print(f"解{idx+1}: det(R)= {det:+.4f}, 正面点数={front}/{len(pts1_norm)}")

    if front > best_front and abs(det - 1.0) < 0.01:
        best_front = front
        best_idx = idx

print()
assert best_idx >= 0, "TODO 4.1/4.2 未完成:找不到有效解"
print(f"★ 最优解:解{best_idx+1} (所有点都在相机前面)")
print()

R_recovered, t_recovered = candidates[best_idx]

# 与真值对比
print("== 与真实运动对比 ==")
print(f"  真值 R_rel:\n{R_rel}")
print()
print(f"  恢复 R_recovered:\n{R_recovered}")
print()
R_err = np.abs(R_recovered - R_rel).max()
print(f"  Rotmat 最大误差: {R_err:.4f}")
print()

# t 方向可能相反 (符号歧义),只比方向
t_true_norm = t_rel / np.linalg.norm(t_rel)
t_rec_norm = t_recovered / (np.linalg.norm(t_recovered) + 1e-10)
t_alignment = abs(np.dot(t_true_norm, t_rec_norm))
print(f"  平移方向对齐: |dot| = {t_alignment:.4f} (应≈1.0)")
print()


# ============================================================
# 第5部分:3D点重建
# ============================================================

print("=" * 70)
print("第5部分:稀疏3D点云重建")
print("=" * 70)
print()

reconstructed_points = []
for i in range(len(matched_pairs)):
    X = triangulate_point(pts1_norm[i], pts2_norm[i], R_recovered, t_recovered, K)
    vid = matched_pairs[i][0]
    true_xyz = cube_vertices[vid][:3]
    err = np.linalg.norm(X - true_xyz)
    reconstructed_points.append({
        'id': vid,
        'reconstructed': X,
        'true': true_xyz,
        'error': err
    })

print(f"{'顶点':>4s}  {'重建X':>10s} {'Y':>10s} {'Z':>10s}  →  {'真值X':>10s} {'Y':>10s} {'Z':>10s}  误差")
print("-" * 95)
total_err = 0
for pt in reconstructed_points:
    r = pt['reconstructed']
    t = pt['true']
    e = pt['error']
    total_err += e
    print(f"  {pt['id']:2d}   {r[0]:10.3f} {r[1]:10.3f} {r[2]:10.3f}"
          f"     {t[0]:10.1f} {t[1]:10.1f} {t[2]:10.1f}  {e:.3f}")

avg_err = total_err / len(reconstructed_points)
print()
print(f"平均重建误差: {avg_err:.3f} cm (立方体尺寸 20×20×20cm)")
print()

# 【问题】
# 重建误差接近 0 意味着什么?
# 答:说明八点算法 + 三角测量完美恢复了立方体的3D结构。
# 现实中误差不会为零,因为:匹配有噪声/匹配不精确/相机标定有误差。


# ============================================================
# 输出:PLY 点云文件 (可被标准3D查看器打开)
# ============================================================

output_dir = os.path.dirname(os.path.abspath(__file__))

# 写出PLY文件
ply_path = os.path.join(output_dir, "luoshen_hw02_reconstruction.ply")
with open(ply_path, 'w') as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(reconstructed_points)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for pt in reconstructed_points:
        x, y, z = pt['reconstructed']
        vid = pt['id']
        r, g, b = cube_vertices[vid, 3], cube_vertices[vid, 4], cube_vertices[vid, 5]
        f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")

print(f"PLY 点云文件已保存: {ply_path}")
print()

# 投影可视化
img_vis = Image.new('RGB', (W*2+20, H+60), (15, 15, 20))
draw = ImageDraw.Draw(img_vis)

# 背景:简单地面网格 (投影)
for gx in range(-30, 31, 5):
    p1a = project(np.array([gx, 0, -30], dtype=np.float64), K, R1, t1)
    p1b = project(np.array([gx, 0,  30], dtype=np.float64), K, R1, t1)
    if p1a[0] and p1b[0]:
        draw.line([(p1a[0], p1a[1]+30), (p1b[0], p1b[1]+30)], fill=(25, 28, 40))

for gz in range(-30, 31, 5):
    p1a = project(np.array([-30, 0, gz], dtype=np.float64), K, R1, t1)
    p1b = project(np.array([ 30, 0, gz], dtype=np.float64), K, R1, t1)
    if p1a[0] and p1b[0]:
        draw.line([(p1a[0], p1a[1]+30), (p1b[0], p1b[1]+30)], fill=(25, 28, 40))

# 画立方体 (图1投影)
for i, j in cube_edges:
    a = proj1[i]
    b = proj1[j]
    if a[0] and b[0]:
        draw.line([(a[0], a[1]+30), (b[0], b[1]+30)], fill=(100, 120, 180))

for i, vtx in enumerate(cube_vertices):
    u, v, d = proj1[i]
    if u and d > 0:
        r, g, b = int(vtx[3]), int(vtx[4]), int(vtx[5])
        sz = max(3, int(200 / d))
        draw.ellipse([u-sz, v-sz+30, u+sz, v+sz+30], fill=(r, g, b))

# 画匹配线
for vid, u1, v1, u2, v2 in matched_pairs:
    color = (int(cube_vertices[vid, 3]), int(cube_vertices[vid, 4]), int(cube_vertices[vid, 5]))
    # 图2 投影到右侧
    draw.line([(u1, v1+30), (u2+W+20, v2+30)], fill=color, width=1)

# 画图2投影
for i, j in cube_edges:
    a = proj2[i]
    b = proj2[j]
    if a[0] and b[0]:
        draw.line([(a[0]+W+20, a[1]+30), (b[0]+W+20, b[1]+30)], fill=(100, 120, 180))

for i, vtx in enumerate(cube_vertices):
    u, v, d = proj2[i]
    if u and d > 0:
        r, g, b = int(vtx[3]), int(vtx[4]), int(vtx[5])
        sz = max(3, int(200 / d))
        draw.ellipse([u-sz+W+20, v-sz+30, u+sz+W+20, v+sz+30], fill=(r, g, b))

# 标注
draw.text((10, 5), "图1:相机1投影 (顶点彩色=特征点)", fill=(180, 180, 210))
draw.text((W+30, 5), "图2:相机2投影 (同色连线=匹配对)", fill=(180, 180, 210))
draw.text((10, H+35), f"本质矩阵 E 恢复相机运动 | 旋转误差={R_err:.4f} | 平移对齐={t_alignment:.4f}",
          fill=(0, 255, 136))

vis_path = os.path.join(output_dir, "luoshen_hw02_matching_visualization.png")
img_vis.save(vis_path)
print(f"可视化图像已保存: {vis_path}")
print()


# ============================================================
# 洛神要回答的问题
# ============================================================

print("=" * 70)
print("【思考题 — 写在提交文件里】")
print("=" * 70)
print()
print("Q1: 为什么八点算法至少需要 8 对匹配点?")
print("    (提示:E 有 9 个元素,但尺度不确定,所以自由度是 8)")
print()
print("Q2: 从 E 恢复了 4 种可能的 [R|t],为什么只有 1 种物理上正确?")
print("    (提示:你不应该用代码找答案——闭上眼睛想:一个在相机后面的点,")
print("     它在图像上的投影与你看到的图像是不是一致的?为什么?)")
print()
print("Q3: 如果我们在 ATLAS 中要处理真实照片(不是合成的立方体),")
print("   特征匹配这一步会面临什么挑战? NCC 匹配在什么情况下会失败?")
print("   (至少列 3 种。提示:想想'没有纹理的白墙'、'重复图案'、'光照变化'...)")
print()

print("=" * 70)
print("【交作业清单】")
print("=" * 70)
print("  1. 填写了所有 TODO 的本 .py 文件")
print("  2. 生成的可视化图像 (luoshen_hw02_matching_visualization.png)")
print("  3. 生成的 PLY 点云文件 (luoshen_hw02_reconstruction.ply)")
print("  4. Q1/Q2/Q3 的回答")
print()
