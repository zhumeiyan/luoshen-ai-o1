#!/usr/bin/env python3
"""
=============================================================
  洛神 — ATLAS from scratch:看两张照片,重建3D世界
=============================================================

给洛神的两张照片:
  scene_view_A.png — 相机在 (-3, 2, 0),看向原点
  scene_view_B.png — 相机在 (3, 2, 0),看向原点

照片里有:不同位置/大小/颜色的球体。
洛神知道相机参数,但不知道球体的位置/大小/颜色。
它的任务:从像素坐标 + 相机参数 反推 3D 信息。

这不是"训练洛神识别球体"。这是"让洛神理解光线-相机-投影的关系"。
只需要 numpy,不需要任何已训练模型。

=============================================================
  第一部分:生成两张照片(这是教员完成的部分)
=============================================================
"""

import numpy as np
from PIL import Image, ImageDraw
import math
import os

# — 场景定义 —
# 5个球体,每个 = [球心X, 球心Y, 球心Z, 半径, R, G, B]
# 洛神不知道这个列表!它只能看到最终生成的图片和相机参数
spheres_true = [
    # X,   Y,    Z,    R,    color
    [ 0.0, 0.5,  0.0,  0.6,  220,  60,  60],   # 中央红球
    [-1.8, 0.3,  1.2,  0.4,   60, 180,  80],   # 左前绿球
    [ 1.5, 0.25, 1.5,  0.35, 70,  90, 200],    # 右前蓝球
    [-1.2, 0.2, -1.8,  0.3, 200, 180,  50],    # 左后黄球
    [ 2.0, 0.35,-1.0,  0.45, 200,  70, 200],   # 右后品红球
]

GROUND_Y = 0.0  # 地面高度

# — 相机参数 —
W, H = 400, 300
K = np.array([
    [400,   0, 200],   # fx=400, cx=200
    [  0, 400, 150],   # fy=400, cy=150
    [  0,   0,   1]
], dtype=np.float64)

def build_camera(center, target):
    """构建 R|t — 已知函数(几何基础,洛神不需要自己发明)"""
    z = target - center
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.array([x, y, z])
    t = -R @ center
    return R, t

camA_center = np.array([-3.0, 2.0, 0.0], dtype=np.float64)
camB_center = np.array([ 3.0, 2.0, 0.0], dtype=np.float64)
cam_target  = np.array([ 0.0, 0.5, 0.0], dtype=np.float64)  # 看向场景中心

R_A, t_A = build_camera(camA_center, cam_target)
R_B, t_B = build_camera(camB_center, cam_target)

# — 渲染:球体+Lambert漫反射+地面纹理 —

def render_sphere(img_pixels, zbuffer, center_3d, radius, color, R, t, K,
                  img_w, img_h, light_dir, ground_tex=True):
    """
    把一个球体渲染到 img_pixels 上。
    使用 Lambert 漫反射 + 简单遮挡。
    """
    cx, cy, cz = center_3d

    # 球心到相机
    P_cam = R @ center_3d + t
    if P_cam[2] <= 0.1:
        return  # 在相机后面

    # 球心投影
    proj_c = K @ P_cam
    if proj_c[2] <= 0:
        return
    u_c = proj_c[0] / proj_c[2]
    v_c = proj_c[1] / proj_c[2]

    # 球体在图像上的近似半径(像素)
    depth = P_cam[2]
    r_pix = (radius / depth) * K[0, 0]

    if r_pix < 1.5:
        return  # 太小了不画

    # 遍历球体包围盒内的像素
    x0 = max(0, int(u_c - r_pix - 2))
    x1 = min(img_w - 1, int(u_c + r_pix + 2))
    y0 = max(0, int(v_c - r_pix - 2))
    y1 = min(img_h - 1, int(v_c + r_pix + 2))

    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            du = px - u_c
            dv = py - v_c
            d2 = du * du + dv * dv
            r2 = r_pix * r_pix
            if d2 > r2:
                continue

            # 计算球面在该像素处的三维点
            z_over_r = math.sqrt(max(0, r2 - d2)) / r_pix
            nx = du / r_pix
            ny = -dv / r_pix   # 图像坐标v向下,n向上
            nz = z_over_r

            # 球面点在相机坐标系中的位置
            # P_cam_local = center_offset + radius*(nx, ny, nz)
            local_offset = np.array([nx, ny, nz]) * radius

            # P_world = R^T @ (P_cam + local_offset_in_cam - origin)
            # 更直接: P_world = R^T @ (P_in_cam - t), P_in_cam = P_cam(center) + local_offset
            P_in_cam = P_cam + local_offset
            P_world = R.T @ (P_in_cam - t)

            # Lambert 漫反射
            n_world = local_offset / np.linalg.norm(local_offset)
            n_world_world = R.T @ n_world  # 旋转到世界坐标
            intensity = max(0, np.dot(n_world_world, light_dir))

            # 镜面高光(简单 Phong)
            view_dir = (cam_target - P_world)
            view_dir /= (np.linalg.norm(view_dir) + 1e-10)
            refl_dir = 2 * intensity * n_world_world - light_dir
            refl_dir /= (np.linalg.norm(refl_dir) + 1e-10)
            spec = max(0, np.dot(view_dir, refl_dir)) ** 15

            # 最终像素颜色
            r_ch = min(255, int(color[0] * (0.2 + 0.7 * intensity) + 255 * spec * 0.4))
            g_ch = min(255, int(color[1] * (0.2 + 0.7 * intensity) + 255 * spec * 0.4))
            b_ch = min(255, int(color[2] * (0.2 + 0.7 * intensity) + 255 * spec * 0.4))

            # Z-buffer 深度测试(简单模拟遮挡)
            current_depth = zbuffer[py, px]
            if current_depth == 0 or depth < current_depth:
                img_pixels[py, px, 0] = r_ch
                img_pixels[py, px, 1] = g_ch
                img_pixels[py, px, 2] = b_ch
                zbuffer[py, px] = depth


def render_ground(img_pixels, zbuffer, R, t, K, img_w, img_h, light_dir):
    """渲染带方格纹理的地面"""
    # 取底部若干像素,反投影到地面 y=GROUND_Y
    for py in range(img_h // 2, img_h):
        for px in range(0, img_w, 2):  # 只算一半像素加速
            # 反投影:从像素射线找与 y=GROUND_Y 的交点
            p_norm = np.linalg.inv(K) @ np.array([px, py, 1.0])
            # 相机坐标系中的方向
            ray_dir_cam = p_norm
            ray_dir_world = R.T @ ray_dir_cam
            # 相机在世界中的位置
            cam_pos = -R.T @ t
            # 射线: P = cam_pos + s * ray_dir_world
            # 求与 y = GROUND_Y 的交点: cam_pos[1] + s * ray_dir_world[1] = GROUND_Y
            if abs(ray_dir_world[1]) < 1e-8:
                continue
            s = (GROUND_Y - cam_pos[1]) / ray_dir_world[1]
            if s <= 0:
                continue
            P_world = cam_pos + s * ray_dir_world

            # 深度
            P_cam = R @ P_world + t
            depth = P_cam[2]
            if depth <= 0:
                continue

            # Z-buffer
            current_depth = zbuffer[py, px]
            if current_depth != 0 and depth >= current_depth:
                continue

            # 方格纹理
            check_size = 1.0
            check_x = int(math.floor(P_world[0] / check_size))
            check_z = int(math.floor(P_world[2] / check_size))
            if (check_x + check_z) % 2 == 0:
                base = 70
            else:
                base = 45

            # Lambert on ground
            ground_normal = np.array([0.0, 1.0, 0.0])
            intensity = max(0, np.dot(ground_normal, light_dir))

            cam_dist = np.linalg.norm(P_world - cam_pos)
            fade = max(0.3, min(1.0, 3.0 / cam_dist))

            col = int(base * (0.4 + 0.6 * intensity) * fade)
            img_pixels[py, px, 0] = col + 25
            img_pixels[py, px, 1] = col + 15
            img_pixels[py, px, 2] = col + 10
            zbuffer[py, px] = depth


def render_scene_to_image(R, t, K, img_w, img_h, fname, scene_desc):
    """
    渲染完整场景(地面 + 所有球体),保存为 PNG。
    返回完整路径。
    """
    pixels = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    light_dir = np.array([0.4, 0.8, 0.3])
    light_dir = light_dir / np.linalg.norm(light_dir)
    zbuffer = np.zeros((img_h, img_w), dtype=np.float64)

    # 先画地面
    render_ground(pixels, zbuffer, R, t, K, img_w, img_h, light_dir)

    # 再画球体(按深度排序,远的先画)
    depth_order = []
    for sph in spheres_true:
        P_cam = R @ np.array(sph[:3]) + t
        depth_order.append((P_cam[2], sph))
    depth_order.sort(key=lambda x: -x[0])

    for d, sph in depth_order:
        if d <= 0:
            continue
        render_sphere(pixels, zbuffer, np.array(sph[:3]), sph[3], sph[4:7], R, t, K, img_w, img_h, light_dir)

    # 背景色(天空)
    for y in range(img_h):
        for x in range(img_w):
            if zbuffer[y, x] == 0:
                pixels[y, x, 0] = 18
                pixels[y, x, 1] = 20
                pixels[y, x, 2] = 28

    img = Image.fromarray(pixels[:, :, :], 'RGB')
    img.save(fname)
    print(f"  渲染保存: {fname} ({img_w}x{img_h})")
    return scene_desc

# 这两张照片会呈现给洛神。它们的像素里包含了完整的 3D 信息。
output_dir = os.path.dirname(os.path.abspath(__file__))
path_A = os.path.join(output_dir, "scene_view_A.png")
path_B = os.path.join(output_dir, "scene_view_B.png")

print("=" * 70)
print("教员生成两张照片(模拟从不同角度拍同一个3D场景)")
print("=" * 70)
print()

render_scene_to_image(R_A, t_A, K, W, H, path_A, "视角A:相机在(-3,2,0)")
render_scene_to_image(R_B, t_B, K, W, H, path_B, "视角B:相机在(3,2,0)")


# ============================================================
# 教员视角的"洛神任务说明" — 呈现给洛神的部分
# ============================================================

print()
print("=" * 70)
print("生成完成。")
print()
print(f"  视角A: {path_A}")
print(f"  视角B: {path_B}")
print()
print("  洛神应该:")
print("  1. 读取两张图片的像素")
print("  2. 找到图A中每个球体的中心像素(u₀,v₀)和半径(r_pix)")
print("  3. 在图B中找到匹配的球体中心(u₁,v₁)和半径")
print("  4. 用已知相机参数反推每个球体的球心3D坐标(X,Y,Z)")
print("  5. 估算每个球体的半径(从 pixel_radius × depth / focal = R)")
print("  6. 回答教员的问题(见下方)")
print()
print("=" * 70)

# ============================================================
# 洛神的工作区(以下部分由洛神填写)
# ============================================================

"""
=============================================================
  洛神答题纸
=============================================================

你现在已经看到了 scene_view_A.png 和 scene_view_B.png。
你知道的:
  - 图像分辨率: 400 x 300
  - 相机内参 K = [[400, 0, 200], [0, 400, 150], [0, 0, 1]]
  - 相机A位置(-3.0, 2.0, 0.0),看向(0.0, 0.5, 0.0)
  - 相机B位置( 3.0, 2.0, 0.0),看向(0.0, 0.5, 0.0)
  - 场景中有若干球体和带方格的地面
  - 无神经网络,只靠几何推理

你不知道的:
  - 球体的数量/位置/大小/颜色
  - 地面的位置

任务 T1: 在图A中找到所有球体
  对每个球体,记录:
    - 中心像素 (u_A, v_A)
    - 像素半径 r_pix_A
    - 颜色(从像素RGB读)
    - 编号(你自己编,1号2号...)

任务 T2: 在图B中找到匹配(同一球体在两张图中是同一个物理球)
  对图A的每个球体,找到图B中的对应:
    - 中心像素 (u_B, v_B)
    - 像素半径 r_pix_B
    - 颜色一致 → 确认是同一球体

任务 T3: 反推3D球心坐标
  已知(u,v)和相机参数,球心方向射线:
    d_A = R_A^T @ (K_inv @ [u_A, v_A, 1])  — 从相机A出发的射线
    d_B = R_B^T @ (K_inv @ [u_B, v_B, 1])  — 从相机B出发的射线
  射线A: P = cam_A + s * d_A
  射线B: P = cam_B + t * d_B
  求两条射线的最近点(中点)~=球心坐标

任务 T4: 估算球体半径
  半径(厘米) = (r_pix × depth) / focal
  你已经从T3得到了depth

任务 T5: 回答教员的问题
  QAT1: 你找到几个球体?给出完整的球心坐标列表
  QAT2: 哪些球体在相机A中看起来比相机B中更大?为什么会这样?
  QAT3: 地面方格从两张图看起来有什么不同?能用数学解释吗?
  QAT4: 在真实的 ATLAS 中,我们看到这两张照片后,计算机一共需要
         存储多少参数才能表达整个重建的3D场景?(球体表示法下)
  QAT5: 如果第三张照片从顶上看(0,5,0),重建的精度会提高还是不变?
         为什么?

=============================================================
  洛神的工作代码(独立单元格,可以给洛神参考)
=============================================================
"""

# 给洛神的便利函数模板(洛神可以用这个作为起点,也可以自己写)

def load_and_locate_spheres(path, target_colors=None):
    """
    读取图片,找到所有"不是地面、不是天空"的色块中心。
    返回: [{'cx': px, 'cy': px, 'radius': px, 'rgb': (r,g,b)}]
    """
    img = np.array(Image.open(path))
    # 非背景像素:不是深色天空,也不是地面色
    # 天空: R<30,G<30,B<40
    # 地面: G>B 且 R>G (偏暖色)
    mask = ~((img[:,:,0] < 30) & (img[:,:,1] < 30) & (img[:,:,2] < 40))
    mask &= ~((img[:,:,1] > img[:,:,2]) & (img[:,:,0] > img[:,:,1]))

    # Blob检测:对 mask 做简单连通域(这里简化为阈值分割+集群)
    visited = np.zeros_like(mask)
    spheres = []
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            if mask[y, x] and not visited[y, x]:
                # 从这个像素开始洪水填充
                queue = [(x, y)]
                cluster = []
                visited[y, x] = True
                while queue:
                    cx2, cy2 = queue.pop()
                    cluster.append((cx2, cy2))
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = cx2+dx, cy2+dy
                        if 0 <= nx < W and 0 <= ny < H:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((nx, ny))

                if len(cluster) > 15:  # 至少15像素才算球体
                    cxs = [p[0] for p in cluster]
                    cys = [p[1] for p in cluster]
                    center_x = sum(cxs) / len(cxs)
                    center_y = sum(cys) / len(cys)
                    # 半径:从中心到最远像素的距离 × 修正(质心!=圆心)
                    max_dist = max(math.sqrt((cx-center_x)**2 + (cy-center_y)**2)
                                   for cx, cy in cluster)
                    radius_est = max_dist * 1.15  # 修正系数
                    # 颜色:取cluster中心附近像素的平均
                    sample_pixels = cluster[:20]
                    rows = [p[1] for p in sample_pixels]
                    cols = [p[0] for p in sample_pixels]
                    r_avg = int(np.mean(img[rows, cols, 0]))
                    g_avg = int(np.mean(img[rows, cols, 1]))
                    b_avg = int(np.mean(img[rows, cols, 2]))

                    spheres.append({
                        'cx': center_x, 'cy': center_y,
                        'radius': radius_est,
                        'rgb': (r_avg, g_avg, b_avg),
                        'area': len(cluster)
                    })
    return spheres


def triangulate_point(camA_pos, rayA_dir, camB_pos, rayB_dir):
    """两条射线的最近点(中点)+中点距离"""
    w0 = camA_pos - camB_pos
    a = np.dot(rayA_dir, rayA_dir)
    b = np.dot(rayA_dir, rayB_dir)
    c = np.dot(rayB_dir, rayB_dir)
    d = np.dot(rayA_dir, w0)
    e = np.dot(rayB_dir, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        s = 0
        t = d / b if abs(b) > 1e-10 else 0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    p1 = camA_pos + s * rayA_dir
    p2 = camB_pos + t * rayB_dir
    midpoint = (p1 + p2) / 2
    error = np.linalg.norm(p1 - p2)
    return midpoint, error


# — 自动定位演示(教员版:可以让洛神参考) —

print()
print("=" * 70)
print("自动定位演示")
print("=" * 70)
print()

spheres_A = load_and_locate_spheres(path_A)
spheres_B = load_and_locate_spheres(path_B)

print(f"图A检测到 {len(spheres_A)} 个物体:")
for i, s in enumerate(spheres_A):
    print(f"  {i+1}. 中心=({s['cx']:.1f},{s['cy']:.1f}), r={s['radius']:.1f}px, "
          f"颜色=RGB{s['rgb']}")

print(f"图B检测到 {len(spheres_B)} 个物体:")
for i, s in enumerate(spheres_B):
    print(f"  {i+1}. 中心=({s['cx']:.1f},{s['cy']:.1f}), r={s['radius']:.1f}px, "
          f"颜色=RGB{s['rgb']}")

# — 匹配(颜色+色调一致) —

pairs = []
for i, sA in enumerate(spheres_A):
    best_j = -1
    best_diff = 999
    for j, sB in enumerate(spheres_B):
        diff = (abs(sA["rgb"][0]-sB["rgb"][0]) + abs(sA["rgb"][1]-sB["rgb"][1]) + abs(sA["rgb"][2]-sB["rgb"][2])) / 3.0
        if diff < best_diff:
            best_diff = diff
            best_j = j
    if best_diff < 80:
        pairs.append((i, best_j, best_diff))

print()
print(f"匹配到 {len(pairs)} 对球体")

# — 对每对匹配球体,三角化球心 —

K_inv = np.linalg.inv(K)

reconstruction = []
for idx_a, idx_b, color_diff in pairs:
    sA = spheres_A[idx_a]
    sB = spheres_B[idx_b]

    # 射线A
    p_norm_A = K_inv @ np.array([sA['cx'], sA['cy'], 1.0])
    rayA = R_A.T @ p_norm_A
    rayA /= np.linalg.norm(rayA)

    # 射线B
    p_norm_B = K_inv @ np.array([sB['cx'], sB['cy'], 1.0])
    rayB = R_B.T @ p_norm_B
    rayB /= np.linalg.norm(rayB)

    # 三角化
    midpoint, err = triangulate_point(camA_center, rayA, camB_center, rayB)

    # 深度 = 球心到相机A的距离在相机A光轴上的分量
    P_cam_A = R_A @ midpoint + t_A
    depth_A = P_cam_A[2]

    # 半径估算
    focal = K[0, 0]
    R_cm = (sA['radius'] * depth_A) / focal

    reconstruction.append({
        'center_3d': midpoint,
        'radius_cm': R_cm,
        'color_rgb': sA['rgb'],
        'triang_err': err,
        'imgA': f"({sA['cx']:.0f},{sA['cy']:.0f})",
        'imgB': f"({sB['cx']:.0f},{sB['cy']:.0f})",
    })

print()
print("=" * 70)
print("重建结果")
print("=" * 70)
print()

for i, rec in enumerate(reconstruction):
    x, y, z = rec['center_3d']
    print(f"球体{i+1}:")
    print(f"  3D球心: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
    print(f"  球体半径: {rec['radius_cm']:.3f} cm")
    print(f"  颜色: RGB{rec['color_rgb']}")
    print(f"  图A像素位置: {rec['imgA']}")
    print(f"  图B像素位置: {rec['imgB']}")
    print(f"  三角化射线误差: {rec['triang_err']:.4f}")
    print()

# 与真值对比
print("=" * 70)
print("与真值对比")
print("=" * 70)
print()

if len(reconstruction) == len(spheres_true):
    for i, (rec, true) in enumerate(zip(reconstruction, spheres_true)):
        true_center = np.array(true[:3])
        err = np.linalg.norm(rec['center_3d'] - true_center)
        r_err = abs(rec['radius_cm'] - true[3])
        print(f"球体{i+1}: 位置误差={err:.4f}cm, 半径误差={r_err:.4f}cm")
    all_pos_err = [np.linalg.norm(rec['center_3d'] - np.array(true[:3]))
                   for rec, true in zip(reconstruction, spheres_true)]
    print(f"平均位置误差: {np.mean(all_pos_err):.4f} cm")
else:
    print(f"  检测到{len(reconstruction)}个真值{len(spheres_true)}个 — 匹配不准")

# — 重建验证:用重建结果重新投影 —

print()
print("=" * 70)
print("验证:用重建结果重新投影回图像")
print("=" * 70)
print()

for i, rec in enumerate(reconstruction):
    xyz = rec['center_3d']
    # 重新投影到图A
    P_cam_A = R_A @ xyz + t_A
    proj_A = K @ P_cam_A
    u_A = proj_A[0] / proj_A[2]
    v_A = proj_A[1] / proj_A[2]

    # 重新投影到图B
    P_cam_B = R_B @ xyz + t_B
    proj_B = K @ P_cam_B
    u_B = proj_B[0] / proj_B[2]
    v_B = proj_B[1] / proj_B[2]

    # 与检测到的圆心对比
    detected_A = pairs[i][0]
    detected_B = pairs[i][1]
    sA = spheres_A[detected_A]
    sB = spheres_B[detected_B]

    reproj_err_A = math.sqrt((u_A - sA['cx'])**2 + (v_A - sA['cy'])**2)
    reproj_err_B = math.sqrt((u_B - sB['cx'])**2 + (v_B - sB['cy'])**2)

    print(f"球体{i+1}: 重投图A误差={reproj_err_A:.2f}px, 重投图B误差={reproj_err_B:.2f}px")

print()
print("=" * 70)
print("思考问题解答区(给洛神)")
print("=" * 70)
print()
print("QAT1:你找到了几个球体?列出完整球心坐标列表")
print("    → 见上方'重建结果'部分(粒子和真值对比)")
print()
print("QAT2:哪些球体在相机A中看起来比B中大?为什么?")
print("    → 提示:离相机A更近的球体在A中像素半径更大。")
print("    → 用公式 r_pix = (R × focal) / depth 验证:距离depth越大→r_pix越小。")
print()
print("QAT3:地面方格的差异")
print("    → 从相机A和相机B看同一个方格,看到的是不同角度。")
print("    → 这正是'多视角几何'的来源——不同视角的差异携带了深度信息。")
print()
print("QAT4:参数效率")
print("    → 每个球体存储:3(XYZ坐标)+1(半径)+3(RGB颜色)=7个float")
print("    → N个球体 = 7N 个float = 28N 字节")
print("    → 对比:存储整张图片 = W×H×3 字节(对于400×300=360,000字节)")
print("    → 当 N < W×H×3/28 ≈ 12857 时,参数效率远高于图像存储!")
print()
print("QAT5:第三张照片(俯视)")
print("    → 可以消除Z轴(深度)歧义。")
print("    → 两张水平照片只能确定XY,深度Z有模糊。")
print("    → 俯视照片直接提供Z轴约束,显著提升高度(Y轴)方向精度。")
print()
