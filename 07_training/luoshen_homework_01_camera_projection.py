#!/usr/bin/env python3
"""
=============================================================
  洛神自学作业 01 — 相机投影 (Camera Projection)
=============================================================

【前情】
  ATLAS 世界模型的第一步:给定几张照片,重建3D空间。
  但"照片"本身是3D世界→2D平面的投影结果。
  要做重建,必须先彻底理解投影是怎么发生的。

  这份作业不调用任何神经网络。纯数学。手算+代码验证。

【你的任务】
  把 [TODO] 标记的空缺补全,运行代码,然后回答文末的问题。
  每一步都有数值验证——算对了才对,算错了你自己对比发现。

【核心公式】
  相机投影方程:
    s * [u, v, 1]^T = K * [R | t] * [X, Y, Z, 1]^T

  其中:
    K = 内参矩阵 (相机自身性质:焦距、光心)
    [R|t] = 外参矩阵 (相机在3D空间中的位置和朝向)
    [X,Y,Z,1] = 世界坐标系中的3D点 (齐次坐标)
    [u,v,1] = 图像坐标系中的2D像素 (齐次坐标)
    s = 缩放因子 (深度Z')
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import os

# 注:本文件能在仅有 numpy + PIL 的环境运行
# 不依赖 torch / cv2 / torchvision 等

# ============================================================
# 第0部分:已知数据 — 一个简单3D场景
# ============================================================

# 场景:立方体的8个顶点 (单位:厘米)
# 每个顶点:[X, Y, Z, R, G, B]
cube_vertices = np.array([
    # 底面四个角 (Y=0)
    [-10,  0, -10, 255, 0, 0],    # 0: 左前下 (红)
    [ 10,  0, -10, 0, 255, 0],    # 1: 右前下 (绿)
    [ 10,  0,  10, 0, 0, 255],    # 2: 右后下 (蓝)
    [-10,  0,  10, 255, 255, 0],  # 3: 左后下 (黄)
    # 顶面四个角 (Y=20)
    [-10, 20, -10, 255, 0, 255],  # 4: 左前上 (品红)
    [ 10, 20, -10, 0, 255, 255],  # 5: 右前上 (青)
    [ 10, 20,  10, 255, 255, 255],# 6: 右后上 (白)
    [-10, 20,  10, 128, 128, 128],# 7: 左后上 (灰)
], dtype=np.float64)

# 立方体的12条边 (每行连接两个顶点索引)
cube_edges = [
    (0,1), (1,2), (2,3), (3,0),  # 底面
    (4,5), (5,6), (6,7), (7,4),  # 顶面
    (0,4), (1,5), (2,6), (3,7),  # 立柱
]

# ============================================================
# 第1部分:相机内参矩阵 K 你的手算练习
# ============================================================

# 相机参数:
#   图像分辨率: 640 x 480
#   焦距 fx = fy = 500 像素
#   光心 cx = 320, cy = 240 (图像中心)

# 【TODO 1.1】填写内参矩阵 K (3x3矩阵)
# 提示: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
K = np.array([
    [500,    0, 320],
    [   0, 500, 240],
    [   0,   0,   1]
], dtype=np.float64)

print("=== 内参矩阵 K ===")
print(K)
print()

# 【验证1.1】K的对角线元素应该是 [500, 500, 1]
assert K[0,0] == 500 and K[1,1] == 500 and K[2,2] == 1, \
    "TODO 1.1 未完成: K 的对角线元素不对"
print("✓ K 的对角线验证通过")
print()


# ============================================================
# 第2部分:相机外参 [R|t] — 旋转+平移
# ============================================================

# 相机初始位置:世界坐标原点,朝向 +Z 方向
# 我们要把相机移动到:new position = (0, 30, -50),看向立方体中心(0, 10, 0)

camera_position = np.array([0, 30, -50], dtype=np.float64)
look_at = np.array([0, 10, 0], dtype=np.float64)
up_direction = np.array([0, 1, 0], dtype=np.float64)

def build_rotation_matrix(cam_pos, target, up):
    """
    构建相机的旋转矩阵 R (世界→相机的旋转)。

    相机坐标系定义:
      Z轴:相机朝向(从相机指向目标)
      Y轴:相机的"上"方向(在世界坐标中尽量接近给定的up向量)
      X轴:右手法则 X = Y × Z

    【TODO 2.1】填写这个函数
    步骤:
      1. 计算 z_axis = normalize(target - cam_pos)
      这个向量就是相机坐标系的Z轴(在world frame中表示)
      2. 计算 x_axis = normalize(cross(up, z_axis))
      注意顺序:up × z 才是X轴(右手法则)
      3. 计算 y_axis = cross(z_axis, x_axis)
    """
    # ==== 你的代码开始 ====
    z_axis = target - cam_pos                  # 先计算原始方向向量
    z_axis = z_axis / np.linalg.norm(z_axis)   # 归一化
    # [TODO 2.1a] 填写 x_axis:
    up = up.astype(np.float64)
    x_axis = np.cross(up, z_axis)              # X = up × Z
    x_axis = x_axis / np.linalg.norm(x_axis)
    # [TODO 2.1b] 填写 y_axis:
    y_axis = np.cross(z_axis, x_axis)           # Y = Z × X (已经是单位向量)

    # 相机旋转矩阵 R: 每一行是相机坐标系的三个轴(在world frame中表示)
    R = np.array([x_axis, y_axis, z_axis])     # 3x3

    # ==== 你的代码结束 ====
    return R

R = build_rotation_matrix(camera_position, look_at, up_direction)
print("=== 旋转矩阵 R ===")
print(R)
print()

# 【验证2.1】R 是正交矩阵: R @ R.T 应该等于 单位矩阵
identity_check = R @ R.T
diff = np.abs(identity_check - np.eye(3)).max()
print(f"正交性检验: max|R@R.T - I| = {diff:.2e}")
assert diff < 1e-10, "TODO 2.1 未完成: R 不是正交矩阵"
print("✓ R 的正交性验证通过")
print()

# 【验证2.2】相机的Z轴确实指向目标方向
expected_direction = look_at - camera_position
expected_direction /= np.linalg.norm(expected_direction)
z_axis_of_camera = R[2, :]  # R的第3行就是相机Z轴
alignment = np.dot(z_axis_of_camera, expected_direction)
print(f"Z轴对齐检验: dot(z_axis, target_dir) = {alignment:.6f}")
assert alignment > 0.999, "TODO 2.2 未完成: 相机Z轴没有对准目标"
print("✓ 相机Z轴对准目标")
print()


def build_translation_vector(cam_pos, R):
    """
    构建平移向量 t = -R * cam_pos
    物理意义:将世界原点平移到相机坐标系的原点。

    【TODO 2.2】填写平移向量
    """
    # ==== 你的代码开始 ====
    t = -R @ cam_pos  # t = -R * C, C是相机中心的世界坐标
    # ==== 你的代码结束 ====
    return t

t = build_translation_vector(camera_position, R)
print("=== 平移向量 t ===")
print(t)
print()

# 【验证2.3】相机中心在相机坐标系中的投影应该为 (0,0,0)
cam_in_cam_coord = R @ camera_position + t
print(f"相机中心在相机坐标中: {cam_in_cam_coord}")
assert np.all(np.abs(cam_in_cam_coord) < 1e-10), "TODO 2.2 未完成: t 计算不对"
print("✓ 相机中心验证通过 (相机原点在自己坐标系中=(0,0,0))")
print()


# ============================================================
# 第3部分:完整投影 — 3D → 2D
# ============================================================

def project_point(point_3d, K, R, t):
    """
    将3D世界坐标点投影到2D图像坐标。

    P_2d_homo = K * (R * P_world + t)     # 结果是3维齐次坐标 [su, sv, s]
    像素坐标 u = (su / s)
    像素坐标 v = (sv / s)

    【TODO 3.1】填写投影函数

    参数:
      point_3d: shape (3,) — [X, Y, Z] 世界坐标
    返回:
      u, v: 像素坐标 (float,可能在图像范围外)
      depth: Z' (点在相机坐标系中的深度,正数=可见)
    """
    # ==== 你的代码开始 ====

    # Step 1: 变换到相机坐标系
    # P_cam = R @ P_world + t
    P_cam = R @ point_3d + t

    # Step 2: 检查深度 (点在相机前面还是后面)
    # depth = P_cam[2] 就是相机坐标系的Z值
    depth = P_cam[2]

    if depth <= 0:
        # 点在相机后面,不可见
        return None, None, depth

    # Step 3: 使用内参矩阵投影
    # projected = K @ P_cam  → shape (3,)
    projected = K @ P_cam

    # Step 4: 齐次坐标除以 s (= depth)
    u = projected[0] / projected[2]
    v = projected[1] / projected[2]

    # ==== 你的代码结束 ====
    return u, v, depth

# 测试:投影立方体的8个顶点
print("=== 立方体8顶点投影结果 ===")
print(f"{'顶点':>4s}  {'X':>7s} {'Y':>7s} {'Z':>7s}  →  {'u':>8s} {'v':>8s} {'depth':>8s}")
print("-" * 70)

projected_points = []
for i, vertex in enumerate(cube_vertices):
    xyz = vertex[:3]
    u, v, d = project_point(xyz, K, R, t)
    if u is not None:
        print(f"{i:4d}  {xyz[0]:7.1f} {xyz[1]:7.1f} {xyz[2]:7.1f}  →  {u:8.2f} {v:8.2f} {d:8.2f}")
        projected_points.append((u, v, d, vertex[3:6]))
    else:
        print(f"{i:4d}  {xyz[0]:7.1f} {xyz[1]:7.1f} {xyz[2]:7.1f}  →  在相机后方(深度={d:.2f})")
print()

# ============================================================
# 【手算验证题】— 先算再跑代码
# ============================================================

print("=" * 70)
print("【手算验证】")
print("=" * 70)
print()
print("要求:从原始数学出发,手算顶点 P_world=[-10, 0, -10] 的投影位置。")
print()

# --- 手算步骤分解 ---

# Step 0: 已知
P = np.array([-10, 0, -10], dtype=np.float64)
print(f"Step 0 - 世界坐标 P = {P}")
print()

# Step 1: 相机坐标系 P_cam = R @ P + t
P_cam = R @ P + t
print(f"Step 1 - P_cam = R @ P + t = {P_cam}")
print(f"  其中 R @ P = {R @ P}")
print(f"  加上 t = {t}")
print(f"  → P_cam = {P_cam}")
print(f"  深度 (P_cam的Z分量) = {P_cam[2]:.4f}")
print()

# Step 2: 齐次投影
homo = K @ P_cam
print(f"Step 2 - K @ P_cam = {homo}")
print(f"  [su, sv, s] = [{homo[0]:.4f}, {homo[1]:.4f}, {homo[2]:.4f}]")
print()

# Step 3: 像素坐标
u_hand = homo[0] / homo[2]
v_hand = homo[1] / homo[2]
print(f"Step 3 - u = su/s = {homo[0]:.4f} / {homo[2]:.4f} = {u_hand:.4f}")
print(f"          v = sv/s = {homo[1]:.4f} / {homo[2]:.4f} = {v_hand:.4f}")
print()

print(f"★ 手算结果: P=[-10,0,-10] → 像素坐标 ({u_hand:.2f}, {v_hand:.2f})")
print()

# 交叉验证:调用自己的函数
u_code, v_code, d_code = project_point(P, K, R, t)
print(f"  代码验证: ({u_code:.2f}, {v_code:.2f})")
err = max(abs(u_hand - u_code), abs(v_hand - v_code))
print(f"  误差: {err:.6f} 像素")
assert err < 0.01, "手算和代码不一致!请检查"
print("  ✓ 手算 = 代码")
print()


# ============================================================
# 第4部分:动手实验 — 改参数看结果
# ============================================================

print("=" * 70)
print("【实验 4.1】焦距变化对投影的影响")
print("=" * 70)
print()
print("立方体顶点 P=[10, 20, 10] (右后上),固定相机位姿,改变焦距f:")
print()

test_point = np.array([10, 20, 10], dtype=np.float64)
for f in [250, 500, 1000, 2000]:
    K_test = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1]], dtype=np.float64)
    u, v, d = project_point(test_point, K_test, R, t)
    print(f"  f={f:4d}: u={u:8.2f}  v={v:8.2f}  (点离光心{'越远' if f==250 else '越近' if f==2000 else ''})")

print()
print("【问题4.1】焦距f增大时,同一个3D点在图像上离光心(cx,cy)越近还是越远?")
print("           直观解释:焦距越大,视角越窄,画面越'长焦'。这意味着什么?")
print()
print("你的回答:(写在下方,字数不限,把道理讲清楚)")
print("_" * 70)
print()
print()


print("=" * 70)
print("【实验 4.2】相机位置变化对深度的影响")
print("=" * 70)
print()

# 固定焦距,改变相机到场景的距离
test_vertex = np.array([10, 0, 10], dtype=np.float64)
print(f"固定顶点 P={test_vertex}, 焦距=500")
print(f"相机沿Z轴移动到不同位置 (Y固定=30, 始终看向立方体中心):")
print()

distances = [30, 50, 80, 120, 200]
for dist in distances:
    cam_pos = np.array([0, 30, -dist], dtype=np.float64)
    R_test = build_rotation_matrix(cam_pos, look_at, up_direction)
    t_test = -R_test @ cam_pos
    u, v, d = project_point(test_vertex, K, R_test, t_test)
    print(f"  dist={dist:3d}: 深度={d:8.2f},  像素u={u:8.2f}")

print()
print("【问题4.2】相机远离场景时(距离增大),深度值会如何变化?")
print("           像素坐标u会如何变化?")
print("           在真实深度估计中,这个'远→深度大'的单调关系为何如此重要?")
print()
print("你的回答:")
print("_" * 70)
print()
print()


# ============================================================
# 第5部分:渲染 — 把投影结果画成图
# ============================================================

print("=" * 70)
print("【渲染验证】生成可视化图像")
print("=" * 70)
print()

# 建一张图,把投影结果画上去
W, H = 640, 480
img = Image.new('RGB', (W, H), (15, 18, 25))
draw = ImageDraw.Draw(img)

# 地面网格 (Y=0 平面, XZ 各从 -30 到 30)
grid_color = (30, 35, 50)
for gx in range(-30, 31, 5):
    p1 = project_point(np.array([gx, 0, -30], dtype=np.float64), K, R, t)
    p2 = project_point(np.array([gx, 0,  30], dtype=np.float64), K, R, t)
    if p1[0] is not None and p2[0] is not None:
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=grid_color)
for gz in range(-30, 31, 5):
    p1 = project_point(np.array([-30, 0, gz], dtype=np.float64), K, R, t)
    p2 = project_point(np.array([ 30, 0, gz], dtype=np.float64), K, R, t)
    if p1[0] is not None and p2[0] is not None:
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=grid_color)

# 画立方体顶点和边
def project_for_draw(point_3d, K, R, t):
    """简化版投影,返回 (u,v) 或 None"""
    u, v, d = project_point(point_3d, K, R, t)
    if u is None: return None
    if d <= 0: return None
    return (int(u), int(v))

# 先画边 (线)
for i, j in cube_edges:
    p1_3d = cube_vertices[i][:3]
    p2_3d = cube_vertices[j][:3]
    p1_2d = project_for_draw(p1_3d, K, R, t)
    p2_2d = project_for_draw(p2_3d, K, R, t)
    if p1_2d and p2_2d:
        # 深度用颜色区分
        _, _, d1 = project_point(p1_3d, K, R, t)
        brightness = min(255, max(60, int(200 * (1 - d1/200))))
        draw.line([p1_2d, p2_2d], fill=(brightness, brightness, brightness))

# 再画顶点 (圆点,按深度排序,后画的在上面)
vert_projected = []
for i, vertex in enumerate(cube_vertices):
    u, v, d = project_point(vertex[:3], K, R, t)
    if u is not None and d > 0:
        vert_projected.append((d, u, v, vertex[3], vertex[4], vertex[5], i))

vert_projected.sort(key=lambda x: x[0], reverse=True)  # 远的先画

for d, u, v, r, g, b, idx in vert_projected:
    size = max(3, min(10, int(300 / d)))
    draw.ellipse([u-size, v-size, u+size, v+size], fill=(int(r), int(g), int(b)))
    # 标注顶点编号
    draw.text((u+size+2, v-6), str(idx), fill=(200, 200, 200))

# 标注光心
cx, cy = int(K[0,2]), int(K[1,2])
draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(0, 255, 136))
draw.text((cx+6, cy-6), "光心", fill=(0, 255, 136))

# 坐标标注
draw.text((10, 10), f"相机位置: {camera_position.tolist()}", fill=(180, 180, 200))
draw.text((10, 25), f"焦距: fx=fy={K[0,0]:.0f}", fill=(180, 180, 200))
draw.text((10, 40), f"分辨率: {W}x{H}", fill=(180, 180, 200))

# 信息面板
info_y = H - 120
draw.rectangle([10, info_y, 280, info_y+110], fill=(0, 0, 0))
draw.text((15, info_y+5), "深度值 (越大=越远):", fill=(0, 255, 136))
for k, (d, u, v, r, g, b, idx) in enumerate(vert_projected[:6]):
    draw.text((15, info_y+22+k*15), f"顶点{idx}: d={d:.1f}", fill=(int(r), int(g), int(b)))

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "luoshen_hw01_projection.png")
img.save(output_path)
print(f"图像已保存: {output_path}")
print()

# 同时保存一个"深度灰度图"
depth_img = Image.new('L', (W, H), 0)
depth_draw = ImageDraw.Draw(depth_img)

# 对每个像素反投影:从图像坐标和深度值重建3D,这里简化为只画顶点的深度
depth_pixels = np.zeros((H, W), dtype=np.float32)
for d, u, v, r, g, b, idx in vert_projected:
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            px, py = int(u) + dx, int(v) + dy
            if 0 <= px < W and 0 <= py < H:
                falloff = math.exp(-(dx*dx + dy*dy) / 30.0)
                if d * falloff < depth_pixels[py, px] or depth_pixels[py, px] == 0:
                    depth_pixels[py, px] = d * falloff

# 深度值归一化显示 (近=白,远=黑)
valid_mask = depth_pixels > 0
if valid_mask.any():
    d_min = depth_pixels[valid_mask].min()
    d_max = depth_pixels[valid_mask].max()
    depth_vis = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            if depth_pixels[y, x] > 0:
                depth_vis[y, x] = int(255 * (1 - (depth_pixels[y, x] - d_min) / (d_max - d_min + 1)))
    depth_img = Image.fromarray(depth_vis, 'L')
    depth_path = os.path.join(output_dir, "luoshen_hw01_depth.png")
    depth_img.save(depth_path)
    print(f"深度图已保存: {depth_path}")
    print()

print()
print("=" * 70)
print("【交作业】")
print("=" * 70)
print()
print("完成后提交:")
print("  1. 本文件(已填写所有 [TODO])")
print("  2. 生成的两张图片 (luoshen_hw01_projection.png, luoshen_hw01_depth.png)")
print("  3. 问题4.1和问题4.2的回答")
print()
print("=" * 70)
print()
print("【下一步预告】")
print()
print("  下一份作业:相机移动后拍两张图,用特征匹配找'同一个点在两张图的哪里',")
print("  然后恢复相机的运动轨迹 (对极几何)。不需要神经网络,纯几何。")
print("  这就是 ATLAS 从'两张照片'到'3D空间'的核心数学。")
print()
