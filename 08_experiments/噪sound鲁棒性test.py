"""噪声下的轨迹预判鲁棒性。

同一条 thin→double 轨迹，往每帧输入里加噪声，
看预撑在乱里还能不能撑住。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from 矩阵.跨模态 import 归一化N维
from modality_specific_test import visual_specific

# ════════════════════════════════════════════════════════════════
# 轨迹生成（thin→double）
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
    return m

signal_amplitude = 6000  # thin→double 中的峰值

def clean_trajectory(n=12):
    return visual_morph("thin", "double", n)

def visual_morph(kind_a, kind_b, n):
    a = mountain_shape(kind_a); b = mountain_shape(kind_b)
    return [interp(a, b, t/max(1,n-1)) for t in range(n)]

def add_noise(frames, level, seed=42):
    """往每帧加高斯噪声，level 是 signal_amplitude 的比例。"""
    rng = np.random.RandomState(seed)
    noisy = []
    for f in frames:
        noise = rng.normal(0, level * signal_amplitude, (8, 64))
        noisy_frame = [[f[r][c] + noise[r][c] for c in range(64)] for r in range(8)]
        noisy.append(noisy_frame)
    return noisy

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

# ════════════════════════════════════════════════════════════════
# 主测试
# ════════════════════════════════════════════════════════════════

N_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
N_TRIALS = 5  # 每个噪声档跑多次取平均
N_FRAMES = 12

print("=" * 80)
print("【噪声下的轨迹预判鲁棒性】")
print("=" * 80)

# 无噪声的基准
clean_frames = clean_trajectory(N_FRAMES)
clean_feats = [visual_specific(f) for f in clean_frames]

print("\n── 基准：无噪声预判 thin→double ──")
prefix = clean_feats[:2]
actual = clean_feats[2:]
pred = predict_linear(prefix, len(actual))
errs = []
for i, a in enumerate(actual):
    na = 归一化N维(a); np_ = 归一化N维(pred[i]); dim = len(na)
    errs.append(sum((na[j]-np_[j])**2 for j in range(dim))**0.5)
print(f"  误差: {[f'{e:.4f}' for e in errs]}")
print(f"  均值: {np.mean(errs):.6f}")

# ══════════════════════════════════════════════════════════════
# 主扫描：不同噪声水平
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【噪声扫描】每档 5 次试验取平均")
print(f"{'=' * 70}")

results = {}

for level in N_LEVELS:
    trial_errs = []
    trial_end_errs = []
    for trial in range(N_TRIALS):
        seed = 42 + trial
        noisy_frames = add_noise(clean_frames, level, seed)
        noisy_feats = [visual_specific(f) for f in noisy_frames]

        # 用 noisy 前缀 → 预判 → 对比 noisy 实际
        prefix = noisy_feats[:2]
        actual_rem = noisy_feats[2:]
        pred = predict_linear(prefix, len(actual_rem))

        errs = []
        for i, a in enumerate(actual_rem):
            if i < len(pred):
                na = 归一化N维(a); np_ = 归一化N维(pred[i]); dim = len(na)
                errs.append(sum((na[j]-np_[j])**2 for j in range(dim))**0.5)
            else:
                errs.append(float('nan'))

        end_err = ((归一化N维(noisy_feats[-1])[j] - 归一化N维(pred[-1])[j])**2 for j in range(len(na)))
        end_err = sum(end_err)**0.5

        trial_errs.append(np.mean([e for e in errs if not np.isnan(e)]))
        trial_end_errs.append(end_err)

    results[level] = {
        "mean": np.mean(trial_errs),
        "std": np.std(trial_errs),
        "end_mean": np.mean(trial_end_errs),
        "end_std": np.std(trial_end_errs),
    }

print(f"\n  {'噪声档次':>8} | {'均值误差':>10} ± {'标准差':>8} | {'末帧误差':>10} ± {'标准差':>8} | {'结论':>15}")
print(f"  {'-'*8}-+-{'-'*20}-+-{'-'*20}-+-{'-'*15}")
for level in N_LEVELS:
    r = results[level]
    if r["mean"] < 0.001:
        status = "完美"
    elif r["mean"] < 0.01:
        status = "轻微退化"
    elif r["mean"] < 0.05:
        status = "可用"
    elif r["mean"] < 0.20:
        status = "显著退化"
    else:
        status = "失效"
    print(f"  {level*100:>7.1f}% | {r['mean']:>10.4f}   {r['std']:>8.4f} | {r['end_mean']:>10.4f}   {r['end_std']:>8.4f} | {status}")

# ══════════════════════════════════════════════════════════════
# TEST 2: 误差随步数在不同噪声下的变化
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 2】不同噪声下，误差随预判步数的变化（单次试验）")
print(f"{'=' * 70}")

key_levels = [0.0, 0.01, 0.05, 0.20, 0.50]

for level in key_levels:
    noisy_frames = add_noise(clean_frames, level, seed=42)
    noisy_feats = [visual_specific(f) for f in noisy_frames]

    prefix = noisy_feats[:2]
    actual_rem = noisy_feats[2:]
    pred = predict_linear(prefix, len(actual_rem))

    print(f"\n  噪声 {level*100:.1f}%:")
    print(f"    步    误差    备注")
    for i, a in enumerate(actual_rem):
        na = 归一化N维(a); np_ = 归一化N维(pred[i]); dim = len(na)
        err = sum((na[j]-np_[j])**2 for j in range(dim))**0.5
        note = ""
        if i == 0:
            note = "即第2→3帧"
        elif i == 4:
            note = "中段"
        elif i == len(actual_rem)-1:
            note = "末帧(帧11)"
        print(f"    {i+1:>3}  {err:.4f}  {note}")

# ══════════════════════════════════════════════════════════════
# TEST 3: 同一噪声下，前缀变长能否补偿
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 3】长前缀能否在噪声下补偿（噪声=10%）")
print(f"{'=' * 70}")

level = 0.10
noisy_frames = add_noise(clean_frames, level, seed=42)
noisy_feats = [visual_specific(f) for f in noisy_frames]

print(f"\n  前缀长 | 预判均值误差 | 末帧误差 | 备注")
for prefix_len in [2, 3, 4, 5, 6, 8]:
    prefix = noisy_feats[:prefix_len]
    actual_rem = noisy_feats[prefix_len:]
    if len(actual_rem) == 0:
        continue
    pred = predict_linear(prefix, len(actual_rem))
    errs = []
    for i, a in enumerate(actual_rem):
        if i < len(pred):
            na = 归一化N维(a); np_ = 归一化N维(pred[i]); dim = len(na)
            errs.append(sum((na[j]-np_[j])**2 for j in range(dim))**0.5)
    mean_err = np.mean([e for e in errs if not np.isnan(e)])
    end_err = errs[-1] if errs else float('nan')
    # 计算前缀自身方向噪声
    actual_step_0to1 = sum((归一化N维(noisy_feats[1])[j] - 归一化N维(noisy_feats[0])[j])**2
                           for j in range(len(noisy_feats[0])))**0.5
    actual_step_last = sum((归一化N维(noisy_feats[prefix_len-1])[j] - 归一化N维(noisy_feats[prefix_len-2])[j])**2
                           for j in range(len(noisy_feats[0])))**0.5 if prefix_len >= 2 else 0
    print(f"  {prefix_len:>6} | {mean_err:>11.4f} | {end_err:>8.4f} | 首步={actual_step_0to1:.4f} 末步={actual_step_last:.4f}")

# ══════════════════════════════════════════════════════════════
# TEST 4: 噪声对首步方向的影响有多大
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 4】噪声对首步方向向量的影响")
print(f"{'=' * 70}")

clean_dir = [归一化N维(clean_feats[1])[j] - 归一化N维(clean_feats[0])[j] for j in range(len(clean_feats[0]))]

for level in [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]:
    noisy_frames = add_noise(clean_frames, level, seed=42)
    noisy_feats = [visual_specific(f) for f in noisy_frames]
    noisy_dir = [归一化N维(noisy_feats[1])[j] - 归一化N维(noisy_feats[0])[j] for j in range(len(noisy_feats[0]))]

    # 方向偏差角（弧度）
    dot = sum(clean_dir[j]*noisy_dir[j] for j in range(len(clean_dir)))
    nc = sum(clean_dir[j]**2 for j in range(len(clean_dir)))**0.5
    nn = sum(noisy_dir[j]**2 for j in range(len(noisy_dir)))**0.5
    angle = np.arccos(max(-1.0, min(1.0, dot/(nc*nn)))) if nc > 1e-12 and nn > 1e-12 else 0.0

    # 首步实际步长
    clean_step = sum(clean_dir[j]**2 for j in range(len(clean_dir)))**0.5
    noisy_step = sum(noisy_dir[j]**2 for j in range(len(noisy_dir)))**0.5

    print(f"  噪声 {level*100:>5.1f}%: 方向偏差 {np.degrees(angle):>6.1f}°, "
          f"步长变化 {noisy_step/clean_step:.3f}x  (clean_step={clean_step:.4f})")

# ══════════════════════════════════════════════════════════════
# TEST 5: 模式匹配在噪声下能走多远
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 5】模式匹配 vs 线性外推 在噪声=5%")
print(f"{'=' * 70}")

level = 0.05

# 模式匹配：库匹配（相似前缀 → 取其后半段）
def match_in_library(prefix_feats, library):
    best_name = None; best_dist = float('inf'); best_raw = None
    prefix_len = len(prefix_feats)
    nprefix = [归一化N维(f) for f in prefix_feats]
    for entry in library:
        lib = entry["raw"]
        if len(lib) < prefix_len: continue
        lib_n = [归一化N维(f) for f in lib[:prefix_len]]
        dim = len(nprefix[0])
        dist = sum(sum((nprefix[i][j]-lib_n[i][j])**2 for j in range(dim)) for i in range(prefix_len))**0.5
        if dist < best_dist:
            best_dist = dist; best_name = entry["name"]; best_raw = lib
    return best_raw[prefix_len:] if best_raw else None, best_name, best_dist

# 干净库（不含噪声版本）
clean_library = [
    {"name": "thin→wide", "raw": [visual_specific(f) for f in visual_morph("thin", "wide", N_FRAMES)]},
    {"name": "wide→double", "raw": [visual_specific(f) for f in visual_morph("wide", "double", N_FRAMES)]},
    {"name": "thin→double", "raw": [visual_specific(f) for f in visual_morph("thin", "double", N_FRAMES)]},
    {"name": "thin→left", "raw": [visual_specific(f) for f in visual_morph("thin", "left", N_FRAMES)]},
    {"name": "double→left", "raw": [visual_specific(f) for f in visual_morph("double", "left", N_FRAMES)]},
]

# 跑 5 次试验
for trial in range(5):
    noisy_frames = add_noise(clean_frames, level, seed=42+trial)
    noisy_feats = [visual_specific(f) for f in noisy_frames]

    prefix = noisy_feats[:3]
    actual_rem = noisy_feats[3:]

    # 线性外推
    pred_linear = predict_linear(prefix, len(actual_rem))
    err_linear = np.mean([sum((归一化N维(actual_rem[i])[j]-归一化N维(pred_linear[i])[j])**2
                               for j in range(len(actual_rem[0])))**0.5
                          for i in range(min(len(actual_rem), len(pred_linear)))])

    # 模式匹配
    matched_rem, matched_name, match_dist = match_in_library(prefix, clean_library)
    if matched_rem:
        err_match = np.mean([sum((归一化N维(actual_rem[i])[j]-归一化N维(matched_rem[i])[j])**2
                                  for j in range(len(actual_rem[0])))**0.5
                             for i in range(min(len(actual_rem), len(matched_rem)))])
    else:
        err_match = float('nan')

    print(f"  试验{trial}: 线性误差={err_linear:.4f}  匹配误差={err_match:.4f}  匹配到={matched_name}  匹配距离={match_dist:.4f}")

# ══════════════════════════════════════════════════════════════
# TEST 6: 信息衰减 — 哪一帧的定位信息最重要
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("【TEST 6】不同帧的定位信息（哪帧决定去向）")
print(f"{'=' * 70}")

# 假设只有部分帧可见，其余帧全黑（零输入）
# 看能否从稀疏帧中恢复轨迹方向

full_feats = clean_feats
target_end = 归一化N维(full_feats[-1])

# 只用第0帧+最后帧（不含中间） → 近似线性步长
step_from_ends = [(归一化N维(full_feats[-1])[j] - 归一化N维(full_feats[0])[j]) / (len(full_feats)-1)
                  for j in range(len(full_feats[0]))]
recovered_feats_jumps = []
current = list(归一化N维(full_feats[0]))
for i in range(len(full_feats)-1):
    current = [current[j] + step_from_ends[j] for j in range(len(current))]
    recovered_feats_jumps.append(list(current))
# 最后一个就是 step_from_ends 延伸
rec_err = sum((归一化N维(full_feats[-1])[j] - recovered_feats_jumps[-1][j])**2 for j in range(len(full_feats[0])))**0.5
print(f"\n  只看首帧+末帧（折线近似）: 末帧误差 = {rec_err:.6f}")
print(f"    -> 如果系统只需判断[大致去向]，稀疏采样足够")

# 只用第0帧和第1帧（线性外推）→ 这是 TEST 1 的结果
print(f"  只看前2帧（线性外推）: 均值误差 = {np.mean(errs):.6f}")
print(f"  → 两种策略差异大：稀疏端点 vs 前2帧")

# 只用奇数帧（0,2,4,6,8,10）
odd_feats = [归一化N维(full_feats[i]) for i in range(0, len(full_feats), 2)]
odd_step = [(odd_feats[1][j]-odd_feats[0][j]) for j in range(len(odd_feats[0]))]
recovered_odd = []
current = list(odd_feats[0])
for i in range(len(full_feats)-1):
    current = [current[j] + odd_step[j] for j in range(len(current))]
    recovered_odd.append(list(current))
odd_err = sum((归一化N维(full_feats[-1])[j]-recovered_odd[-1][j])**2 for j in range(len(full_feats[0])))**0.5
print(f"  只用奇数帧方向 → 末帧误差 = {odd_err:.6f}")

print(f"\n{'=' * 70}")
print("【总结】")
print(f"{'=' * 70}")
print(f"  基准无噪声:   {results[0.0]['mean']:.6f}")
print(f"  仅噪声 1%:    {results[0.01]['mean']:.4f}")
print(f"  仅噪声 5%:    {results[0.05]['mean']:.4f}")
print(f"  仅噪声 10%:   {results[0.10]['mean']:.4f}")
print(f"  仅噪声 20%:   {results[0.20]['mean']:.4f}")
print(f"  仅噪声 50%:   {results[0.50]['mean']:.4f}")
