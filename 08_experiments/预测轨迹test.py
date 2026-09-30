"""轨迹预判测试。

从一条轨迹的前半段，预判后半段怎么走。
不是复述过去，是从走了一半的路推接下来往哪走。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from 矩阵.跨模态 import 归一化N维
from modality_specific_test import visual_specific, auditory_specific

# ════════════════════════════════════════════════════════════════
# 1. 生成和之前一样的轨迹
# ════════════════════════════════════════════════════════════════

def interp(a, b, t):
    rows = len(a); cols = len(a[0])
    return [[a[r][c] * (1 - t) + b[r][c] * t for c in range(cols)] for r in range(rows)]

def mountain_shape(kind):
    rows, cols = 8, 64
    m = [[0.0]*cols for _ in range(rows)]
    if kind == "thin":
        for c in range(cols):
            for r in range(2,5):
                d = abs(c - 32)/12.0
                m[r][c] = max(0,(1-d))*6000
            for r in range(5,8): m[r][c] = 200
    elif kind == "wide":
        for c in range(cols):
            for r in range(2,5):
                d = abs(c - 32)/28.0
                m[r][c] = max(0,(1-d))*4500
            for r in range(5,8): m[r][c] = 200
    elif kind == "double":
        for c in range(cols):
            for r in range(2,5):
                d1 = abs(c-20)/10.0; d2 = abs(c-44)/10.0
                m[r][c] = max(0,(1-d1))*5000 + max(0,(1-d2))*4000
            for r in range(5,8): m[r][c] = 200
    elif kind == "left":
        for c in range(cols):
            for r in range(2,5):
                d = abs(c-18)/14.0
                m[r][c] = max(0,(1-d))*5500
            for r in range(5,8): m[r][c] = 200
    elif kind == "low":
        for c in range(cols):
            for r in range(2,5):
                d = abs(c-32)/24.0
                m[r][c] = max(0,(1-d))*3000
            for r in range(5,8): m[r][c] = 200
    return m

def audio_shape(kind):
    rows, cols = 8, 64
    m = [[0.0]*cols for _ in range(rows)]
    if kind == "thin":
        for c in range(6,50):
            for r in range(3): m[r][c]=800
            for r in range(3,5): m[r][c]=1300
            for r in range(5,8): m[r][c]=2600
    elif kind == "wide":
        for c in range(6,50):
            for r in range(3): m[r][c]=600
            for r in range(3,5): m[r][c]=1000
            for r in range(5,8): m[r][c]=2200
    elif kind == "double":
        for c in range(6,50):
            for r in range(3): m[r][c]=750
            for r in range(3,5): m[r][c]=1400
            for r in range(5,8): m[r][c]=2800
    elif kind == "low":
        for c in range(6,50):
            for r in range(3): m[r][c]=500
            for r in range(3,5): m[r][c]=900
            for r in range(5,8): m[r][c]=1800
    return m

def visual_morph(kind_a, kind_b, n):
    a = mountain_shape(kind_a); b = mountain_shape(kind_b)
    return [interp(a, b, t/max(1,n-1)) for t in range(n)]

def auditory_morph(kind_a, kind_b, n):
    a = audio_shape(kind_a); b = audio_shape(kind_b)
    return [interp(a, b, t/max(1,n-1)) for t in range(n)]

# ════════════════════════════════════════════════════════════════
# 2. 预判机制
# ════════════════════════════════════════════════════════════════

def extract_trajectory(frames, extractor):
    """给定帧序列，返回专有特征向量序列。"""
    return [extractor(f) for f in frames]

def predict_linear(prefix_feats, n_ahead):
    """线性外推：用 prefix 最后一步的方向，按此方向继续走 n_ahead 步。

    返回预测的后续特征向量序列（含当前最后帧作为起点）。
    """
    if len(prefix_feats) < 2:
        return []
    dim = len(prefix_feats[0])
    # 当前最后一步的方向
    last_step = [prefix_feats[-1][j] - prefix_feats[-2][j] for j in range(dim)]
    predictions = []
    current = list(prefix_feats[-1])
    for _ in range(n_ahead):
        current = [current[j] + last_step[j] for j in range(dim)]
        predictions.append(list(current))
    return predictions

def predict_accelerated(prefix_feats, n_ahead):
    """加速感知外推：用 prefix 最后两步的方向变化（加速度），预判接下来的步。

    如果有 n>=3 帧：加速度 = (最后一步) - (倒数第二步)
    下一步 = 最后一步 + 加速度 → 位置 += 下一步
    """
    if len(prefix_feats) < 3:
        return predict_linear(prefix_feats, n_ahead)
    dim = len(prefix_feats[0])
    s1 = [prefix_feats[-1][j] - prefix_feats[-2][j] for j in range(dim)]  # 最后一步
    s0 = [prefix_feats[-2][j] - prefix_feats[-3][j] for j in range(dim)]  # 倒数第二步
    accel = [s1[j] - s0[j] for j in range(dim)]
    predictions = []
    current = list(prefix_feats[-1])
    step = list(s1)
    for _ in range(n_ahead):
        step = [step[j] + accel[j] for j in range(dim)]
        current = [current[j] + step[j] for j in range(dim)]
        predictions.append(list(current))
    return predictions

def predict_from_library(prefix_feats, library):
    """库匹配：在已知轨迹库中，找前缀最相似的，用其后半段作为预判。

    library: list of {name, raw: [特征0, 特征1, ...], trajectory: {...}}
    """
    best_name = None
    best_dist = float('inf')
    best_raw = None

    prefix_len = len(prefix_feats)
    nprefix = [归一化N维(f) for f in prefix_feats]

    for entry in library:
        lib_feats = entry["raw"]
        if len(lib_feats) < prefix_len:
            continue
        lib_norm = [归一化N维(f) for f in lib_feats[:prefix_len]]
        dist = sum(
            sum((nprefix[i][j] - lib_norm[i][j])**2 for j in range(len(nprefix[0])))
            for i in range(prefix_len)
        ) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = entry["name"]
            best_raw = lib_feats

    if best_raw is None:
        return None, None, float('inf')

    # 返回该轨迹的后半段（已知轨迹中 prefix 之后的部分）
    remainder = best_raw[prefix_len:]
    return remainder, best_name, best_dist


# ════════════════════════════════════════════════════════════════
# 3. 主测试
# ════════════════════════════════════════════════════════════════

print("=" * 80)
print("【轨迹预判测试】")
print("=" * 80)

N = 12  # 总帧数

# 生成轨迹库（视觉 + 听觉）
trajectories = {}

vis_morphs = [
    ("thin→wide", "thin", "wide"),
    ("wide→double", "wide", "double"),
    ("thin→double", "thin", "double"),
    ("thin→left", "thin", "left"),
    ("double→left", "double", "left"),
]

for name, a, b in vis_morphs:
    frames = visual_morph(a, b, N)
    raw = extract_trajectory(frames, visual_specific)
    trajectories[name] = {"name": name, "modality": "visual", "raw": raw}

aud_morphs = [
    ("thin→wide", "thin", "wide"),
    ("thin→double", "thin", "double"),
    ("wide→double", "wide", "double"),
    ("thin→low", "thin", "low"),
]

for name, a, b in aud_morphs:
    frames = auditory_morph(a, b, N)
    raw = extract_trajectory(frames, auditory_specific)
    trajectories["aud_" + name] = {"name": name, "modality": "auditory", "raw": raw}

# ══════════════════════════════════════════════════════════════
# TEST 1: 线性外推，看到第 1 帧（索引0）→ 预判剩余
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 1】视觉 thin→double：看前缀 → 预判后半")
print(f"{'=' * 70}")

target_name = "thin→double"
target = trajectories[target_name]
feats = target["raw"]

print(f"\n轨迹共 {len(feats)} 帧（0..{len(feats)-1}）")
print(f"实际后几帧delta: ", end="")
for i in range(2, len(feats)):
    d = sum((归一化N维(feats[i])[j] - 归一化N维(feats[i-1])[j])**2 for j in range(len(feats[0])))**0.5
    print(f"[{i-1}→{i}]={d:.4f} ", end="")
print()

# 试不同前缀长度
for prefix_len in [2, 3, 4, 5, 6]:
    prefix = feats[:prefix_len]
    actual_remainder = feats[prefix_len:]
    n_ahead = len(actual_remainder)

    # 线性外推
    pred_linear = predict_linear(prefix, n_ahead)
    # 加速外推
    pred_accel = predict_accelerated(prefix, n_ahead)

    # 比较：每个预判帧 vs 实际帧 的特征 L2 距离
    errs_linear = []
    errs_accel = []
    for i, actual in enumerate(actual_remainder):
        if i < len(pred_linear):
            na = 归一化N维(actual)
            npl = 归一化N维(pred_linear[i])
            npla = 归一化N维(pred_accel[i])
            dim = len(na)
            el = sum((na[j]-npl[j])**2 for j in range(dim))**0.5
            ea = sum((na[j]-npla[j])**2 for j in range(dim))**0.5
            errs_linear.append(el)
            errs_accel.append(ea)
        else:
            errs_linear.append(float('nan'))
            errs_accel.append(float('nan'))

    mean_l = np.mean([e for e in errs_linear if not np.isnan(e)])
    mean_a = np.mean([e for e in errs_accel if not np.isnan(e)])

    print(f"\n  前缀={prefix_len}帧 → 预判剩余{n_ahead}帧:")
    print(f"    线性误差: {[f'{e:.4f}' if not np.isnan(e) else '—' for e in errs_linear]}")
    print(f"    加速误差: {[f'{e:.4f}' if not np.isnan(e) else '—' for e in errs_accel]}")
    print(f"    线性均值: {mean_l:.4f}  加速均值: {mean_a:.4f}")

# ══════════════════════════════════════════════════════════════
# TEST 2: 误差随预判距离的变化
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 2】误差随预判步数的变化")
print(f"{'=' * 70}")

print(f"\n使用 thin→double，前缀长度=3帧")
prefix_len = 3
prefix = feats[:prefix_len]
actual_all = feats
pred_linear_all = predict_linear(prefix, len(actual_all) - prefix_len)
pred_accel_all = predict_accelerated(prefix, len(actual_all) - prefix_len)

print(f"\n  {'预判步':>6}  {'线性误差':>10}  {'加速误差':>10}  {'备注':>15}")
for step in range(1, len(actual_all) - prefix_len + 1):
    actual = actual_all[prefix_len + step - 1]
    na = 归一化N维(actual)
    el = sum((na[j]-归一化N维(pred_linear_all[step-1])[j])**2 for j in range(len(na)))**0.5
    ea = sum((na[j]-归一化N维(pred_accel_all[step-1])[j])**2 for j in range(len(na)))**0.5
    note = ""
    if step <= 3:
        note = "近"
    elif step <= 6:
        note = "中"
    else:
        note = "远"
    print(f"  {step:>6}  {el:>10.4f}  {ea:>10.4f}  {note:>15}")

# ══════════════════════════════════════════════════════════════
# TEST 3: 同起点分岔，看前缀预判路往哪走
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 3】同起点(thin)分岔：thin→double vs thin→left")
print(f"{'=' * 70}")

td = trajectories["thin→double"]
tl = trajectories["thin→left"]

print(f"\n实际: thin→double 末帧 delta序列尾部: ", end="")
feats_td = td["raw"]
for i in range(max(1, len(feats_td)-4), len(feats_td)):
    d = sum((归一化N维(feats_td[i])[j]-归一化N维(feats_td[i-1])[j])**2 for j in range(len(feats_td[0])))**0.5
    print(f"{d:.4f} ", end="")
print()

print(f"实际: thin→left   末帧 delta序列尾部: ", end="")
feats_tl = tl["raw"]
for i in range(max(1, len(feats_tl)-4), len(feats_tl)):
    d = sum((归一化N维(feats_tl[i])[j]-归一化N维(feats_tl[i-1])[j])**2 for j in range(len(feats_tl[0])))**0.5
    print(f"{d:.4f} ", end="")
print()

# 从 thin→double 的前缀预判后4帧
for prefix_len in [2, 3]:
    prefix = feats_td[:prefix_len]
    pred_td = predict_linear(prefix, 4)

    # 计算预判的末帧 delta
    print(f"\n  前缀{prefix_len}帧 → 预判后4帧:")
    print(f"    预判 delta: ", end="")
    for i in range(1, len(pred_td)):
        d = sum((归一化N维(pred_td[i])[j]-归一化N维(pred_td[i-1])[j])**2 for j in range(len(pred_td[0])))**0.5
        print(f"{d:.4f} ", end="")
    print()

    # 实际 thin→double 后4帧的 delta
    print(f"    实际thin→double: ", end="")
    for i in range(prefix_len, min(prefix_len+4, len(feats_td))):
        d = sum((归一化N维(feats_td[i])[j]-归一化N维(feats_td[i-1])[j])**2 for j in range(len(feats_td[0])))**0.5
        print(f"{d:.4f} ", end="")
    print()

    # 实际 thin→left 后4帧的 delta
    print(f"    实际thin→left:   ", end="")
    for i in range(prefix_len, min(prefix_len+4, len(feats_tl))):
        d = sum((归一化N维(feats_tl[i])[j]-归一化N维(feats_tl[i-1])[j])**2 for j in range(len(feats_tl[0])))**0.5
        print(f"{d:.4f} ", end="")
    print()

    # 预判 vs 两条实际路的距离
    if len(pred_td) >= 4:
        pred_final = 归一化N维(pred_td[3])  # 预判第4帧
        actual_td_final = 归一化N维(feats_td[min(prefix_len+3, len(feats_td)-1)])
        actual_tl_final = 归一化N维(feats_tl[min(prefix_len+3, len(feats_tl)-1)])
        dim = len(pred_final)
        dist_to_td = sum((pred_final[j]-actual_td_final[j])**2 for j in range(dim))**0.5
        dist_to_tl = sum((pred_final[j]-actual_tl_final[j])**2 for j in range(dim))**0.5
        print(f"    预判末帧距thin→double: {dist_to_td:.4f}  距thin→left: {dist_to_tl:.4f}  {'→预测到double✓' if dist_to_td < dist_to_tl else '→预测到left✗'}")

# ══════════════════════════════════════════════════════════════
# TEST 4: 库匹配预判
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 4】库匹配预判（前缀匹配轨迹库）")
print(f"{'=' * 70}")

library = [v for v in trajectories.values() if v["modality"] == "visual"]

# 用 thin→left 的前缀，看库中匹配到谁
target_tl = trajectories["thin→left"]
for prefix_len in [2, 3, 4]:
    prefix = target_tl["raw"][:prefix_len]
    remainder, matched_name, match_dist = predict_from_library(prefix, library)
    actual_remainder = target_tl["raw"][prefix_len:]

    if remainder:
        # 计算预判误差
        errs = []
        for i in range(min(len(remainder), len(actual_remainder))):
            na = 归一化N维(actual_remainder[i])
            nr = 归一化N维(remainder[i])
            dim = len(na)
            errs.append(sum((na[j]-nr[j])**2 for j in range(dim))**0.5)
        mean_err = np.mean(errs) if errs else float('nan')
        print(f"\n  看thin->left前缀{prefix_len}帧 -> 库匹配到【{matched_name}】(距离={match_dist:.4f})")
        print(f"    预判误差: {[f'{e:.4f}' for e in errs[:6]]}  均值={mean_err:.4f}")
    else:
        print(f"\n  看thin->left前缀{prefix_len}帧 -> 无匹配")

# ══════════════════════════════════════════════════════════════
# TEST 5: 视觉 vs 听觉预判对比
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 5】视觉 vs 听觉预判精度对比（thin→wide）")
print(f"{'=' * 70}")

vis_td = trajectories["thin→wide"]
aud_td = trajectories["aud_thin→wide"]

for modality, traj in [("视觉", vis_td), ("听觉", aud_td)]:
    fts = traj["raw"]
    print(f"\n  {modality} thin→wide:")
    for prefix_len in [2, 3, 4]:
        prefix = fts[:prefix_len]
        actual_remainder = fts[prefix_len:]
        pred = predict_linear(prefix, len(actual_remainder))
        errs = []
        for i, actual in enumerate(actual_remainder):
            na = 归一化N维(actual)
            npl = 归一化N维(pred[i])
            dim = len(na)
            errs.append(sum((na[j]-npl[j])**2 for j in range(dim))**0.5)
        mean_err = np.mean([e for e in errs if not np.isnan(e)]) if errs else float('nan')
        print(f"    前缀{prefix_len}帧 → 误差: {[f'{e:.4f}' for e in errs[:5]]}... 均值={mean_err:.4f}")

# ══════════════════════════════════════════════════════════════
# TEST 6: 算术——特征变化是否真的非线性
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 6】特征空间非线性验证")
print(f"{'=' * 70}")

fts = feats_td  # thin→double
norms = [归一化N维(f) for f in fts]
# 在特征空间中，相邻步的步长变化
steps = []
for i in range(1, len(norms)):
    step = [norms[i][j] - norms[i-1][j] for j in range(len(norms[0]))]
    steps.append(step)

# 步长的模长变化
step_lengths = [sum(s[j]**2 for j in range(len(s)))**0.5 for s in steps]
print(f"\n  thin→double 特征空间步长序列:")
print(f"  {[f'{l:.4f}' for l in step_lengths]}")
print(f"  步长标准差: {np.std(step_lengths):.6f}")
print(f"  步长变异系数: {np.std(step_lengths)/np.mean(step_lengths):.4f}")

# 如果线性：步长应该恒等
# 如果非线性：步长会变化
print(f"\n  结论: 步长 {'变化大 → 高度非线性' if np.std(step_lengths)/np.mean(step_lengths) > 0.1 else '变化小 → 近似线性'}")

# ══════════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【总结】")
print(f"{'=' * 70}")
print(f"  TEST结果见于上方输出")
print(f"  关键指标:")
print(f"    - 线性外推误差随预判距离增加的变化（TEST 2）")
print(f"    - 同起点分岔识别率（TEST 3）")
print(f"    - 库匹配准确率（TEST 4）")
print(f"    - 特征空间线性度（TEST 6）")
