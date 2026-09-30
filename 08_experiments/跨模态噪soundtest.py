"""跨模态抗噪测试。

一路被噪声糊了，另一路还清楚，能不能帮它拉回来。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from 矩阵.跨模态 import 归一化N维
from modality_specific_test import visual_specific, auditory_specific

# ════════════════════════════════════════════════════════════════
# 轨迹生成
# ════════════════════════════════════════════════════════════════

def interp(a, b, t):
    rows = len(a); cols = len(a[0])
    return [[a[r][c]*(1-t)+b[r][c]*t for c in range(cols)] for r in range(rows)]

def mountain_shape(kind):
    rows, cols = 8, 64
    m = [[0.0]*cols for _ in range(rows)]
    if kind == "thin":
        for c in range(cols):
            for r in range(2,5):
                d = abs(c-32)/12.0; m[r][c]=max(0,(1-d))*6000
            for r in range(5,8): m[r][c]=200
    elif kind == "double":
        for c in range(cols):
            for r in range(2,5):
                d1=abs(c-20)/10.0; d2=abs(c-44)/10.0
                m[r][c]=max(0,(1-d1))*5000+max(0,(1-d2))*4000
            for r in range(5,8): m[r][c]=200
    elif kind == "wide":
        for c in range(cols):
            for r in range(2,5):
                d = abs(c-32)/28.0; m[r][c]=max(0,(1-d))*4500
            for r in range(5,8): m[r][c]=200
    elif kind == "left":
        for c in range(cols):
            for r in range(2,5):
                d = abs(c-18)/14.0; m[r][c]=max(0,(1-d))*5500
            for r in range(5,8): m[r][c]=200
    elif kind == "low":
        for c in range(cols):
            for r in range(2,5):
                d = abs(c-32)/24.0; m[r][c]=max(0,(1-d))*3000
            for r in range(5,8): m[r][c]=200
    return m

def audio_shape(kind):
    rows, cols = 8, 64
    m = [[0.0]*cols for _ in range(rows)]
    if kind == "thin":
        for c in range(6,50):
            for r in range(3): m[r][c]=800
            for r in range(3,5): m[r][c]=1300
            for r in range(5,8): m[r][c]=2600
    elif kind == "double":
        for c in range(6,50):
            for r in range(3): m[r][c]=750
            for r in range(3,5): m[r][c]=1400
            for r in range(5,8): m[r][c]=2800
    elif kind == "wide":
        for c in range(6,50):
            for r in range(3): m[r][c]=600
            for r in range(3,5): m[r][c]=1000
            for r in range(5,8): m[r][c]=2200
    elif kind == "low":
        for c in range(6,50):
            for r in range(3): m[r][c]=500
            for r in range(3,5): m[r][c]=900
            for r in range(5,8): m[r][c]=1800
    return m

signal_amp_vis = 6000
signal_amp_aud = 2800

def add_noise_frames(frames, level, seed=42):
    rng = np.random.RandomState(seed)
    noisy = []
    amp = signal_amp_vis if len(frames[0][0]) == 64 else signal_amp_aud
    # Both visual and auditory have cols=64. Need to differentiate by signal amplitude.
    # Actually, signal amplitudes differ. Let's use a fixed value per call.
    for f in frames:
        noise = rng.normal(0, level * amp, (8, 64))
        noisy.append([[f[r][c] + noise[r][c] for c in range(64)] for r in range(8)])
    return noisy

def add_noise_vis(frames, level, seed=42):
    return add_noise_frames_visaud(frames, level, seed, signal_amp_vis)

def add_noise_aud(frames, level, seed=42):
    return add_noise_frames_visaud(frames, level, seed, signal_amp_aud)

def add_noise_frames_visaud(frames, level, seed, amp):
    rng = np.random.RandomState(seed)
    noisy = []
    for f in frames:
        noise = rng.normal(0, level * amp, (8, 64))
        noisy.append([[f[r][c] + noise[r][c] for c in range(64)] for r in range(8)])
    return noisy

# ════════════════════════════════════════════════════════════════
# 预判函数
# ════════════════════════════════════════════════════════════════

def predict_linear(prefix_feats, n_ahead):
    if len(prefix_feats) < 2:
        return []
    dim = len(prefix_feats[0])
    last_step = [prefix_feats[-1][j] - prefix_feats[-2][j] for j in range(dim)]
    predictions = []
    current = list(prefix_feats[-1])
    for _ in range(n_ahead):
        current = [current[j] + last_step[j] for j in range(dim)]
        predictions.append(list(current))
    return predictions

def distance(a, b):
    if not a or not b:
        return float('inf')
    dim = min(len(a), len(b))
    return sum((归一化N维(a)[j] - 归一化N维(b)[j])**2 for j in range(dim))**0.5

def prediction_error(pred, actual):
    errs = []
    for i in range(min(len(pred), len(actual))):
        na = 归一化N维(actual[i])
        np_ = 归一化N维(pred[i])
        dim = len(na)
        errs.append(sum((na[j]-np_[j])**2 for j in range(dim))**0.5)
    return np.mean(errs) if errs else float('inf')

# ════════════════════════════════════════════════════════════════
# 生成干净轨迹
# ════════════════════════════════════════════════════════════════

N = 12  # 帧数

clean_vis_trajs = {}
clean_aud_trajs = {}

morphs = [
    ("thin→double", "thin", "double"),
    ("thin→wide", "thin", "wide"),
    ("wide→double", "wide", "double"),
    ("thin→left", "thin", "left"),
]

for name, a, b in morphs:
    vis_frames = [interp(mountain_shape(a), mountain_shape(b), t/max(1,N-1)) for t in range(N)]
    aud_frames = [interp(audio_shape(a), audio_shape(b), t/max(1,N-1)) for t in range(N)]
    clean_vis_trajs[name] = [visual_specific(f) for f in vis_frames]
    clean_aud_trajs[name] = [auditory_specific(f) for f in aud_frames]

# 构建双模态库
library = []
for name, a, b in morphs:
    library.append({
        "name": name,
        "vis_raw": clean_vis_trajs[name],
        "aud_raw": clean_aud_trajs[name],
    })

# ════════════════════════════════════════════════════════════════
# 跨模态匹配预判
# ════════════════════════════════════════════════════════════════

def cross_modal_match(vis_prefix, aud_prefix, library, vis_weight=1.0, aud_weight=1.0):
    """用双模态前缀在库中匹配。"""
    best_name = None
    best_dist = float('inf')
    best_entry = None
    prefix_len = min(len(vis_prefix), len(aud_prefix))

    for entry in library:
        if len(entry["vis_raw"]) < prefix_len or len(entry["aud_raw"]) < prefix_len:
            continue
        d = 0.0
        for i in range(prefix_len):
            d += vis_weight * distance(vis_prefix[i], entry["vis_raw"][i])
            d += aud_weight * distance(aud_prefix[i], entry["aud_raw"][i])
        if d < best_dist:
            best_dist = d
            best_name = entry["name"]
            best_entry = entry

    return best_entry, best_name, best_dist

# ════════════════════════════════════════════════════════════════
# 主测试
# ════════════════════════════════════════════════════════════════

print("=" * 80)
print("【跨模态抗噪测试】")
print("=" * 80)

# 测试目标：thin→double
target = "thin→double"
test_cases = [
    ("两路都干净", 0.0, 0.0),
    ("图5%噪声+音干净", 0.05, 0.0),
    ("图10%噪声+音干净", 0.10, 0.0),
    ("图干净+音5%噪声", 0.0, 0.05),
    ("图干净+音10%噪声", 0.0, 0.10),
    ("两路都5%噪声", 0.05, 0.05),
    ("两路都10%噪声", 0.10, 0.10),
]

N_TRIALS = 3

results = []

for case_name, vis_noise, aud_noise in test_cases:
    trial_results = {"name": case_name}

    for trial in range(N_TRIALS):
        seed = 42 + trial

        # 生成含噪声版本
        # 需要重新生成帧，加不同噪声
        vis_frames = [interp(mountain_shape("thin"), mountain_shape("double"), t/max(1,N-1)) for t in range(N)]
        aud_frames = [interp(audio_shape("thin"), audio_shape("double"), t/max(1,N-1)) for t in range(N)]

        if vis_noise > 0:
            vis_frames = add_noise_frames_visaud(vis_frames, vis_noise, seed, signal_amp_vis)
        if aud_noise > 0:
            aud_frames = add_noise_frames_visaud(aud_frames, aud_noise, seed+100, signal_amp_aud)

        vis_feats = [visual_specific(f) for f in vis_frames]
        aud_feats = [auditory_specific(f) for f in aud_frames]

        # ────────────────────────────────────────────────────
        # 策略1: 单模态线性外推（无帮助）
        # ────────────────────────────────────────────────────
        vis_prefix = vis_feats[:2]
        aud_prefix = aud_feats[:2]
        actual_vis_rem = vis_feats[2:]
        actual_aud_rem = aud_feats[2:]

        pred_vis_alone = predict_linear(vis_prefix, len(actual_vis_rem))
        pred_aud_alone = predict_linear(aud_prefix, len(actual_aud_rem))
        err_vis_alone = prediction_error(pred_vis_alone, actual_vis_rem)
        err_aud_alone = prediction_error(pred_aud_alone, actual_aud_rem)

        # ────────────────────────────────────────────────────
        # 策略2: 单模态库匹配（仅用库，不跨模态）
        # ────────────────────────────────────────────────────
        # 视觉库匹配
        best_vis_entry = None
        best_vis_dist = float('inf')
        for entry in library:
            if len(entry["vis_raw"]) < 2:
                continue
            d = sum(distance(vis_prefix[i], entry["vis_raw"][i]) for i in range(2))
            if d < best_vis_dist:
                best_vis_dist = d
                best_vis_entry = entry

        if best_vis_entry:
            pred_vis_lib = best_vis_entry["vis_raw"][2:]
            err_vis_lib = prediction_error(pred_vis_lib, actual_vis_rem)
        else:
            err_vis_lib = float('inf')

        # 听觉库匹配
        best_aud_entry = None
        best_aud_dist = float('inf')
        for entry in library:
            if len(entry["aud_raw"]) < 2:
                continue
            d = sum(distance(aud_prefix[i], entry["aud_raw"][i]) for i in range(2))
            if d < best_aud_dist:
                best_aud_dist = d
                best_aud_entry = entry

        if best_aud_entry:
            pred_aud_lib = best_aud_entry["aud_raw"][2:]
            err_aud_lib = prediction_error(pred_aud_lib, actual_aud_rem)
        else:
            err_aud_lib = float('inf')

        # ────────────────────────────────────────────────────
        # 策略3: 跨模态库匹配（双模态加权，更侧重干净的那路）
        # ────────────────────────────────────────────────────
        # 计算每路的噪声权重：噪声越小权重越大
        # 噪声=0时权重=1.0；噪声=0.1时权重=0.3（降权）
        vw = max(0.1, 1.0 - vis_noise * 5)
        aw = max(0.1, 1.0 - aud_noise * 5)

        cross_entry, cross_name, cross_dist = cross_modal_match(
            vis_prefix, aud_prefix, library, vis_weight=vw, aud_weight=aw
        )

        if cross_entry:
            pred_vis_cross = cross_entry["vis_raw"][2:]
            pred_aud_cross = cross_entry["aud_raw"][2:]
            err_vis_cross = prediction_error(pred_vis_cross, actual_vis_rem)
            err_aud_cross = prediction_error(pred_aud_cross, actual_aud_rem)
        else:
            err_vis_cross = float('inf')
            err_aud_cross = float('inf')

        # ────────────────────────────────────────────────────
        # 汇总
        # ────────────────────────────────────────────────────
        trial_results[f"trial_{trial}"] = {
            "err_vis_alone": err_vis_alone,
            "err_aud_alone": err_aud_alone,
            "err_vis_lib": err_vis_lib,
            "err_aud_lib": err_aud_lib,
            "err_vis_cross": err_vis_cross,
            "err_aud_cross": err_aud_cross,
            "cross_matched": cross_name,
        }

    # 试验平均
    avg = {}
    for key in ["err_vis_alone", "err_aud_alone", "err_vis_lib", "err_aud_lib", "err_vis_cross", "err_aud_cross"]:
        vals = [trial_results[f"trial_{trial}"][key] for trial in range(N_TRIALS) if key in trial_results[f"trial_{trial}"]]
        avg[key] = np.mean(vals) if vals else float('inf')
    trial_results["avg"] = avg
    results.append((case_name, vis_noise, aud_noise, trial_results))

# ════════════════════════════════════════════════════════════════
# 报告
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("【结果】")
print(f"{'=' * 80}")

print(f"\n{'条件':>20}     视觉外推 视觉库匹配 视觉跨模 | 听觉外推 听觉库匹配 听觉跨模 | 跨模匹配到")
print(f"{'-'*20}     {'-'*9} {'-'*9} {'-'*9} | {'-'*9} {'-'*9} {'-'*9} |")
for case_name, vn, audn, tr in results:
    a = tr["avg"]
    matched = tr["trial_0"].get("cross_matched", "?")
    print(f"{case_name:>20}   {a['err_vis_alone']:>9.4f} {a['err_vis_lib']:>9.4f} {a['err_vis_cross']:>9.4f} | "
          f"{a['err_aud_alone']:>9.4f} {a['err_aud_lib']:>9.4f} {a['err_aud_cross']:>9.4f} | {matched}")

# ──────────────────────────────────────────────────────────────
# 帮助分析
# ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("【帮助分析】")
print(f"{'=' * 80}")

print("\n视觉 thin→double，一路一路看：")
for case_name, vn, audn, tr in results:
    a = tr["avg"]
    vis_help_from_aud = a["err_vis_alone"] - a["err_vis_cross"]  # 正值 = 跨模比单模好
    aud_help_from_vis = a["err_aud_alone"] - a["err_aud_cross"]
    print(f"  {case_name:>20}: 视觉获帮助={vis_help_from_aud:>+.4f}, 听觉获帮助={aud_help_from_vis:>+.4f}")

# 基数归一
print("\n归一化改善率 (跨模误差 - 单模外推误差) / 单模外推误差：")
for case_name, vn, audn, tr in results:
    a = tr["avg"]
    vis_improve = (a["err_vis_cross"] - a["err_vis_alone"]) / max(a["err_vis_alone"], 0.001)
    aud_improve = (a["err_aud_cross"] - a["err_aud_alone"]) / max(a["err_aud_alone"], 0.001)
    print(f"  {case_name:>20}: 视觉={vis_improve:>+6.1%}, 听觉={aud_improve:>+6.1%}")

# ══════════════════════════════════════════════════════════════
# 新东西：交叉救援成功率
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("【交叉救援成功率】跨模匹配是否优于单模匹配")
print(f"{'=' * 80}")
for case_name, vn, audn, tr in results:
    a = tr["avg"]
    vis_cross_wins = a["err_vis_cross"] < a["err_vis_lib"]
    aud_cross_wins = a["err_aud_cross"] < a["err_aud_lib"]
    print(f"  {case_name:>20}: 跨模>{'是' if vis_cross_wins else '否'}  听跨模>{'是' if aud_cross_wins else '否'}")
