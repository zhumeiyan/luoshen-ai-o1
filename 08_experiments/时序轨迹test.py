"""专有特征 + 时间流测试。

同一个东西连续变形，每帧出专有特征向量，
看变化过程是否形成独特轨迹。
不是比单帧像不像，是比变化过程本身。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from 矩阵.跨模态 import 归一化N维
from modality_specific_test import visual_specific, auditory_specific

# ════════════════════════════════════════════════════════════════
# 1. 变形生成器
# ════════════════════════════════════════════════════════════════

def interp(a, b, t):
    """线性插值，t=0 全 a，t=1 全 b。"""
    rows = len(a)
    cols = len(a[0])
    return [[a[r][c] * (1 - t) + b[r][c] * t for c in range(cols)] for r in range(rows)]


def visual_morph(kind_a, kind_b, n_frames):
    """在两幅视觉之间变形（n 帧，含首尾）。"""
    a = mountain_shape(kind_a)
    b = mountain_shape(kind_b)
    return [interp(a, b, t / max(1, n_frames - 1)) for t in range(n_frames)]


def auditory_morph(kind_a, kind_b, n_frames):
    """在两段音频之间变形。"""
    a = audio_shape(kind_a)
    b = audio_shape(kind_b)
    return [interp(a, b, t / max(1, n_frames - 1)) for t in range(n_frames)]


def mountain_shape(kind):
    """生成视觉山的 8×64 帧。"""
    rows, cols = 8, 64
    m = np.zeros((rows, cols)).tolist()
    if kind == "thin":
        for c in range(cols):
            for r in range(2, 5):
                dist = abs(c - 32) / 12.0
                m[r][c] = max(0, (1 - dist)) * 6000
            m[5][c] = 200; m[6][c] = 200; m[7][c] = 200
    elif kind == "wide":
        for c in range(cols):
            for r in range(2, 5):
                dist = abs(c - 32) / 28.0
                m[r][c] = max(0, (1 - dist)) * 4500
            m[5][c] = 200; m[6][c] = 200; m[7][c] = 200
    elif kind == "double":
        for c in range(cols):
            for r in range(2, 5):
                d1 = abs(c - 20) / 10.0
                d2 = abs(c - 44) / 10.0
                m[r][c] = max(0, (1 - d1)) * 5000 + max(0, (1 - d2)) * 4000
            m[5][c] = 200; m[6][c] = 200; m[7][c] = 200
    elif kind == "left":
        for c in range(cols):
            for r in range(2, 5):
                dist = abs(c - 18) / 14.0
                m[r][c] = max(0, (1 - dist)) * 5500
            m[5][c] = 200; m[6][c] = 200; m[7][c] = 200
    return m


def audio_shape(kind):
    """生成 8×64 音频帧。"""
    rows, cols = 8, 64
    m = np.zeros((rows, cols)).tolist()
    if kind == "thin":
        for c in range(6, 50):
            for r in range(3): m[r][c] = 800
            for r in range(3, 5): m[r][c] = 1300
            for r in range(5, 8): m[r][c] = 2600
    elif kind == "wide":
        for c in range(6, 50):
            for r in range(3): m[r][c] = 600
            for r in range(3, 5): m[r][c] = 1000
            for r in range(5, 8): m[r][c] = 2200
    elif kind == "double":
        for c in range(6, 50):
            for r in range(3): m[r][c] = 750
            for r in range(3, 5): m[r][c] = 1400
            for r in range(5, 8): m[r][c] = 2800
    elif kind == "low":
        for c in range(6, 50):
            for r in range(3): m[r][c] = 500
            for r in range(3, 5): m[r][c] = 900
            for r in range(5, 8): m[r][c] = 1800
    return m


# ════════════════════════════════════════════════════════════════
# 2. 时间流分析
# ════════════════════════════════════════════════════════════════

def trajectory(features):
    """给定一系列特征向量，计算轨迹统计。

    返回：
      - frame_deltas: 相邻帧的 L2 距离列表（帧间差异）
      - path_length: 帧间距离之和（轨迹总长）
      - displacement: 首尾帧的 L2 距离（净位移）
      - direction_change: 相邻帧间方向变化角度（反映轨迹弯曲程度）
    """
    n = len(features)
    if n < 2:
        return {"frame_deltas": [], "path_length": 0.0, "displacement": 0.0}

    # L2 归一化
    norms = [归一化N维(f) for f in features]
    dim = len(norms[0])

    # 帧间 delta
    deltas = []
    for i in range(1, n):
        d = sum((norms[i][j] - norms[i-1][j]) ** 2 for j in range(dim)) ** 0.5
        deltas.append(d)

    path_length = sum(deltas)
    displacement = sum((norms[-1][j] - norms[0][j]) ** 2 for j in range(dim)) ** 0.5

    # 方向变化（连续帧向量的夹角变化）
    angles = []
    for i in range(1, n - 1):
        v1 = [norms[i][j] - norms[i-1][j] for j in range(dim)]
        v2 = [norms[i+1][j] - norms[i][j] for j in range(dim)]
        dot = sum(v1[j] * v2[j] for j in range(dim))
        n1 = sum(v1[j] ** 2 for j in range(dim)) ** 0.5
        n2 = sum(v2[j] ** 2 for j in range(dim)) ** 0.5
        if n1 < 1e-12 or n2 < 1e-12:
            angles.append(0.0)
        else:
            cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
            angles.append(np.arccos(cosang))

    return {
        "frame_deltas": deltas,
        "path_length": path_length,
        "displacement": displacement,
        "direction_changes": angles,
        "mean_delta": np.mean(deltas) if deltas else 0.0,
        "std_delta": np.std(deltas) if deltas else 0.0,
    }


def cross_trajectory_dist(traj_a, traj_b, mode="endpoint"):
    """两条轨迹间的距离。

    mode:
      "endpoint": 末态特征之间的距离
      "mean_feature": 所有帧特征均值的距离
      "path_shape": 帧间 delta 序列的 L2 距离（比较变化模式）
    """
    n = min(len(traj_a["raw"]), len(traj_b["raw"]))
    if n == 0:
        return 0.0

    na = [归一化N维(v) for v in traj_a["raw"]]
    nb = [归一化N维(v) for v in traj_b["raw"]]

    if mode == "endpoint":
        return sum((na[-1][j] - nb[-1][j]) ** 2 for j in range(len(na[-1]))) ** 0.5
    elif mode == "mean_feature":
        dim = len(na[0])
        mean_a = [sum(na[f][j] for f in range(n)) / n for j in range(dim)]
        mean_b = [sum(nb[f][j] for f in range(n)) / n for j in range(dim)]
        return sum((mean_a[j] - mean_b[j]) ** 2 for j in range(dim)) ** 0.5
    elif mode == "path_shape":
        # 帧间 delta 序列对比
        da = traj_a["frame_deltas"]
        db = traj_b["frame_deltas"]
        m = min(len(da), len(db))
        if m == 0:
            return 0.0
        return sum((da[i] - db[i]) ** 2 for i in range(m)) ** 0.5


# ════════════════════════════════════════════════════════════════
# 3. 主测试
# ════════════════════════════════════════════════════════════════

print("=" * 80)
print("【专有特征 + 时间流测试】")
print("=" * 80)

N_FRAMES = 12  # 每个变形序列的帧数

# ── A. 视觉轨迹 ──
print(f"\n{'=' * 80}")
print("【A. 视觉轨迹】每个变形序列 N_FRAMES 帧")
print(f"{'=' * 80}")

vis_trajectories = {}

morphs = [
    ("thin → wide", "thin", "wide"),
    ("wide → double", "wide", "double"),
    ("thin → double", "thin", "double"),
    ("thin → left", "thin", "left"),
    ("double → left", "double", "left"),
]

for name, a, b in morphs:
    frames = visual_morph(a, b, N_FRAMES)
    features = [visual_specific(f) for f in frames]
    traj = trajectory(features)
    traj["raw"] = features
    traj["name"] = name
    vis_trajectories[name] = traj

    print(f"\n  【{name}】({a}→{b})")
    print(f"    帧间 delta: {[f'{d:.4f}' for d in traj['frame_deltas']]}")
    print(f"    均值: {traj['mean_delta']:.6f}, 标准差: {traj['std_delta']:.6f}")
    print(f"    路径总长: {traj['path_length']:.6f}")
    print(f"    位移(首尾): {traj['displacement']:.6f}")
    print(f"    方向变化(角度和): {sum(traj['direction_changes']):.4f}")

# ── B. 帧间差异汇总 ──
print(f"\n{'=' * 80}")
print("【B. 视觉帧间 delta 汇总】")
print(f"{'=' * 80}")

for name, traj in vis_trajectories.items():
    print(f"  {name:>20}: mean_delta={traj['mean_delta']:.6f}  path_len={traj['path_length']:.6f}  displacement={traj['displacement']:.6f}")

# ── C. 不同轨迹间的距离 ──
print(f"\n{'=' * 80}")
print("【C. 不同轨迹之间的距离】")
print(f"{'=' * 80}")

modes = ["endpoint", "mean_feature", "path_shape"]
mode_names = {"endpoint": "末态距离", "mean_feature": "均值特征距离", "path_shape": "变化模式距离"}

for mode in modes:
    print(f"\n  ── {mode_names[mode]} ──")
    names = list(vis_trajectories.keys())
    # 表头
    print(f"  {'':>20}", end="")
    for n2 in names:
        print(f"  {n2:>14}", end="")
    print()
    for n1 in names:
        print(f"  {n1:>20}", end="")
        for n2 in names:
            if n1 == n2:
                print(f"  {'—':>14}", end="")
            else:
                d = cross_trajectory_dist(vis_trajectories[n1], vis_trajectories[n2], mode)
                print(f"  {d:>14.4f}", end="")
        print()

# ── D. 静态 vs 动态 ──
print(f"\n{'=' * 80}")
print("【D. 静态（不动） vs 动态（在变）】")
print(f"{'=' * 80}")

# 静态 thin（12 帧全是 thin）
static_frames = [mountain_shape("thin")] * N_FRAMES
static_feats = [visual_specific(f) for f in static_frames]
static_traj = trajectory(static_feats)
static_traj["raw"] = static_feats

# 动态 thin→wide（同样的帧数）
dynamic_traj = vis_trajectories["thin → wide"]

print(f"\n  静态 thin × {N_FRAMES}帧:")
print(f"    mean_delta: {static_traj['mean_delta']:.8f}")
print(f"    path_len:   {static_traj['path_length']:.8f}")
print(f"    displacement: {static_traj['displacement']:.8f}")

print(f"\n  动态 thin→wide:")
print(f"    mean_delta: {dynamic_traj['mean_delta']:.8f}")
print(f"    path_len:   {dynamic_traj['path_length']:.8f}")
print(f"    displacement: {dynamic_traj['displacement']:.8f}")

print(f"\n  动态/静态 ratio:")
print(f"    mean_delta: {dynamic_traj['mean_delta'] / max(static_traj['mean_delta'], 1e-12):.1f}x")
print(f"    path_len:   {dynamic_traj['path_length'] / max(static_traj['path_length'], 1e-12):.1f}x")

# ── E. 同一个起点、不同方向 ──
print(f"\n{'=' * 80}")
print("【E. 同起点(thin)、不同方向的轨迹】")
print(f"{'=' * 80}")

same_start = ["thin → wide", "thin → double", "thin → left"]
print(f"\n  起点: thin（帧0 特征相同）")
print(f"  终点:")
for name in same_start:
    traj = vis_trajectories[name]
    end_pt = 归一化N维(traj["raw"][-1])
    print(f"    {name}: 末态特征前5维={[f'{v:.3f}' for v in end_pt[:5]]}")

print(f"\n  帧0 → 帧{ N_FRAMES-1 } 的方向变化:")
# thin→wide 方向
tw = vis_trajectories["thin → wide"]
td = vis_trajectories["thin → double"]
tl = vis_trajectories["thin → left"]

na_tw = [归一化N维(v) for v in tw["raw"]]
na_td = [归一化N维(v) for v in td["raw"]]
na_tl = [归一化N维(v) for v in tl["raw"]]

# 初始步方向（帧0→帧1）
dim = len(na_tw[0])
dir_tw = [na_tw[1][j] - na_tw[0][j] for j in range(dim)]
dir_td = [na_td[1][j] - na_td[0][j] for j in range(dim)]
dir_tl = [na_tl[1][j] - na_tl[0][j] for j in range(dim)]

# 方向夹角
def angle(v1, v2):
    dot = sum(v1[j] * v2[j] for j in range(dim))
    n1 = sum(v1[j] ** 2 for j in range(dim)) ** 0.5
    n2 = sum(v2[j] ** 2 for j in range(dim)) ** 0.5
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return np.arccos(max(-1.0, min(1.0, dot / (n1 * n2))))

print(f"    thin→wide  vs thin→double 初始步夹角: {angle(dir_tw, dir_td):.4f} rad ({np.degrees(angle(dir_tw, dir_td)):.1f}°)")
print(f"    thin→wide  vs thin→left   初始步夹角: {angle(dir_tw, dir_tl):.4f} rad ({np.degrees(angle(dir_tw, dir_tl)):.1f}°)")
print(f"    thin→double vs thin→left  初始步夹角: {angle(dir_td, dir_tl):.4f} rad ({np.degrees(angle(dir_td, dir_tl)):.1f}°)")

# ── F. 听觉轨迹 ──
print(f"\n{'=' * 80}")
print("【F. 听觉轨迹】")
print(f"{'=' * 80}")

aud_trajectories = {}
aud_morphs = [
    ("thin → wide", "thin", "wide"),
    ("thin → double", "thin", "double"),
    ("wide → double", "wide", "double"),
    ("thin → low", "thin", "low"),
]

for name, a, b in aud_morphs:
    frames = auditory_morph(a, b, N_FRAMES)
    features = [auditory_specific(f) for f in frames]
    traj = trajectory(features)
    traj["raw"] = features
    traj["name"] = name
    aud_trajectories[name] = traj

    print(f"\n  【{name}】({a}→{b})")
    print(f"    帧间 delta: {[f'{d:.4f}' for d in traj['frame_deltas']]}")
    print(f"    均值: {traj['mean_delta']:.6f}, 标准差: {traj['std_delta']:.6f}")
    print(f"    路径总长: {traj['path_length']:.6f}")
    print(f"    位移(首尾): {traj['displacement']:.6f}")

# ── G. 听觉轨迹间距离 ──
print(f"\n{'=' * 80}")
print("【G. 听觉轨迹间距离】")
print(f"{'=' * 80}")

for mode in modes:
    print(f"\n  ── {mode_names[mode]} ──")
    names = list(aud_trajectories.keys())
    print(f"  {'':>20}", end="")
    for n2 in names:
        print(f"  {n2:>14}", end="")
    print()
    for n1 in names:
        print(f"  {n1:>20}", end="")
        for n2 in names:
            if n1 == n2:
                print(f"  {'—':>14}", end="")
            else:
                d = cross_trajectory_dist(aud_trajectories[n1], aud_trajectories[n2], mode)
                print(f"  {d:>14.4f}", end="")
        print()

# ── H. 视觉-听觉同变形，轨迹形状差 ──
print(f"\n{'=' * 80}")
print("【H. 同一变形（thin→wide），视觉轨迹 vs 听觉轨迹】")
print(f"{'=' * 80}")

vis_traj_tw = vis_trajectories["thin → wide"]
aud_traj_tw = aud_trajectories["thin → wide"]

print(f"\n  视觉 thin→wide:")
print(f"    mean_delta: {vis_traj_tw['mean_delta']:.6f}, path: {vis_traj_tw['path_length']:.6f}")
print(f"    帧间delta: {[f'{d:.4f}' for d in vis_traj_tw['frame_deltas']]}")

print(f"\n  听觉 thin→wide:")
print(f"    mean_delta: {aud_traj_tw['mean_delta']:.6f}, path: {aud_traj_tw['path_length']:.6f}")
print(f"    帧间delta: {[f'{d:.4f}' for d in aud_traj_tw['frame_deltas']]}")

# 形状比较：帧间 delta 序列
delta_shape_dist = sum(
    (vis_traj_tw["frame_deltas"][i] - aud_traj_tw["frame_deltas"][i]) ** 2
    for i in range(min(len(vis_traj_tw["frame_deltas"]), len(aud_traj_tw["frame_deltas"])))
) ** 0.5
print(f"\n  视觉 vs 听觉 帧间delta序列距离: {delta_shape_dist:.6f}")

# ── I. 轨迹弯曲度：方向变化 ──
print(f"\n{'=' * 80}")
print("【I. 轨迹弯曲度（方向变化）】")
print(f"{'=' * 80}")

print(f"\n  视觉轨迹：")
for name, traj in vis_trajectories.items():
    dc_sum = sum(traj["direction_changes"])
    print(f"    {name:>20}: 方向变化和={dc_sum:.4f} rad, 每步均值={dc_sum/max(1,len(traj['direction_changes'])):.4f}")

print(f"\n  听觉轨迹：")
for name, traj in aud_trajectories.items():
    dc_sum = sum(traj["direction_changes"])
    print(f"    {name:>20}: 方向变化和={dc_sum:.4f} rad, 每步均值={dc_sum/max(1,len(traj['direction_changes'])):.4f}")

# ── J. 三帧快速概览：thin→wide 特征向量沿时间变化 ──
print(f"\n{'=' * 80}")
print("【J. 具体特征随时间变化（thin→wide）】")
print(f"{'=' * 80}")

v_feats = vis_trajectories["thin → wide"]["raw"]
print(f"\n  视觉 thin→wide — 各帧前5维特征：")
for i, f in enumerate(v_feats):
    nf = 归一化N维(f)
    print(f"    帧{i:2d}: {[f'{v:.3f}' for v in nf[:5]]}")

a_feats = aud_trajectories["thin → wide"]["raw"]
print(f"\n  听觉 thin→wide — 各帧前5维特征：")
for i, f in enumerate(a_feats):
    nf = 归一化N维(f)
    print(f"    帧{i:2d}: {[f'{v:.3f}' for v in nf[:5]]}")

# ── K. 总结 ──
print(f"\n{'=' * 80}")
print("【总结】")
print(f"{'=' * 80}")
print(f"  1. 静态 mean_delta: {static_traj['mean_delta']:.10f}")
print(f"  2. 动态 mean_delta: {dynamic_traj['mean_delta']:.6f}")
print(f"  3. 动态/静态 ratio: {dynamic_traj['mean_delta'] / max(static_traj['mean_delta'], 1e-12):.0f}x")
print(f"  4. 同起点不同方向的初始步夹角(视): wide={np.degrees(angle(dir_tw, dir_td)):.1f}° left={np.degrees(angle(dir_tw, dir_tl)):.1f}°")
print(f"  5. 视觉轨迹 path 范围: {min(t['path_length'] for t in vis_trajectories.values()):.4f} ~ {max(t['path_length'] for t in vis_trajectories.values()):.4f}")
print(f"  6. 听觉轨迹 path 范围: {min(t['path_length'] for t in aud_trajectories.values()):.4f} ~ {max(t['path_length'] for t in aud_trajectories.values()):.4f}")
print(f"  7. 视觉 vs 听觉同变形帧间delta距离: {delta_shape_dist:.4f}")
