"""对照实验：基线 & 稀疏性对跨模态差分的影响。

测算AI预测：基线未除→平坦通道互相似，稀疏→余弦趋0。

三组（全部合成，不用真设备）：
  组1 对数间隔：值在对数尺度分布，无基线偏移，无稀疏
  组2 不同基线：组1 + 强基线偏移（模拟环境底噪）
  组3 不同稀疏度：组1 + 随机置零（模拟点云稀疏性）

每组造 9 个通道（模拟 9 类传感器），跑 C(9,2)=36 对 γ。
对照三组 γ 的分布差异，验证预测方向。

纪律：
  - 纯数值，不用 LLM
  - 只验证"预测方向对不对"
  - 不升 [事实]
  - [判读·对照实验]
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from 矩阵.视觉 import 默认行, 默认列
from 矩阵.跨模态 import 矩阵转二维, 归一化二维, 跨模态差分

np.random.seed(42)

N_CHANNELS = 9
ROWS, COLS = 默认行, 默认列
PIXELS = ROWS * COLS


def build_channel_group(group_id):
    """生成 N_CHANNELS 个通道矩阵（list of 2D lists）。"""
    channels = []

    if group_id == 1:
        # 组1：对数间隔，无基线，无稀疏
        # 每个通道的"对数范围"不同（模拟不同动态范围的设备）
        for i in range(N_CHANNELS):
            lo = 0.1 + i * 0.05
            hi = lo + 1.5
            arr = np.logspace(lo, hi, num=PIXELS).reshape(ROWS, COLS)
            channels.append(arr.tolist())

    elif group_id == 2:
        # 组2：不同基线（对数分布 + 大偏移）
        # 相同对数分布形态，但叠加不同基线偏移
        base = np.logspace(0.5, 2.0, num=PIXELS)  # 相同形态
        for i in range(N_CHANNELS):
            baseline = (i + 1) * 100.0  # 不同偏移（模拟环境底噪）
            arr = (base + baseline).reshape(ROWS, COLS)
            channels.append(arr.tolist())

    elif group_id == 3:
        # 组3：不同稀疏度（对数分布 + 随机置零）
        # 相同分布形态 + 不同置零比例（模拟点云稀疏性）
        base = np.logspace(0.5, 2.0, num=PIXELS)
        for i in range(N_CHANNELS):
            sparsity = 0.5 + i * 0.05  # 50% → 90% 置零
            mask = np.random.random(PIXELS) < sparsity
            arr = base.copy()
            arr[mask] = 0.0
            # 至少保留 1 个非零，避免全零
            if arr.sum() == 0:
                arr[0] = base[0]
            channels.append(arr.reshape(ROWS, COLS).tolist())

    return channels


def pairs_from_channels(channels):
    """跑 C(n,2) 对跨模态差分，返回 γ 列表。"""
    n = len(channels)
    gammas = {}
    for i in range(n):
        for j in range(i + 1, n):
            r = 跨模态差分("exp", channels[i], channels[j],
                                 f"ch{i}", f"ch{j}")
            gammas[f"ch{i}-ch{j}"] = r["gamma"]
    return gammas


def stats(gammas):
    """γ 列表的统计。"""
    vals = list(gammas.values())
    if not vals:
        return {}
    return {
        "mean":   np.mean(vals),
        "std":    np.std(vals),
        "min":    np.min(vals),
        "max":    np.max(vals),
        "near_zero_frac": sum(1 for v in vals if abs(v) < 0.01) / len(vals),
        "clustered_frac": sum(1 for v in vals if 0.3 < abs(v) < 0.5) / len(vals),
    }


def run_group(group_id):
    """跑一组实验。"""
    channels = build_channel_group(group_id)
    gammas = pairs_from_channels(channels)
    return gammas, stats(gammas)


def print_group_result(group_id, label, gammas, st):
    print(f"\n{'='*80}")
    print(f"【组{group_id}】{label}")
    print(f"{'='*80}")
    print(f"  统计：mean={st['mean']:.6f}  std={st['std']:.6f}  "
          f"min={st['min']:.6f}  max={st['max']:.6f}")
    print(f"  |γ|<0.01 占比（近零）：{st['near_zero_frac']:.2%}")
    print(f"  0.3<|γ|<0.5 占比（撞簇）：{st['clustered_frac']:.2%}")
    print()
    print("  36对 γ 值：")
    for k, v in sorted(gammas.items(), key=lambda x: abs(x[1])):
        flag = ""
        if abs(v) < 0.01:
            flag = " ← 近零"
        elif 0.3 < abs(v) < 0.5:
            flag = " ← 撞簇"
        print(f"    {k:8s} γ={v:+.6f}{flag}")


def print_comparison(s1, s2, s3):
    """三组对照。"""
    print(f"\n{'='*80}")
    print("【三组对照】")
    print(f"{'='*80}")
    print(f"  {'指标':<20s} {'组1(对数)':>12s} {'组2(基线)':>12s} {'组3(稀疏)':>12s}")
    print(f"  {'─'*56}")
    for k in ["mean", "std", "min", "max", "near_zero_frac", "clustered_frac"]:
        v1 = s1.get(k, float('nan'))
        v2 = s2.get(k, float('nan'))
        v3 = s3.get(k, float('nan'))
        print(f"  {k:<20s} {v1:>12.6f} {v2:>12.6f} {v3:>12.6f}")
    print()
    print("  预测验证：")
    print(f"    组1→组2 簇增大 {'✓' if s2['clustered_frac'] > s1['clustered_frac'] else '✗'}"
          f"  ({s1['clustered_frac']:.0%} → {s2['clustered_frac']:.0%})")
    print(f"    组3 近零占比 {'✓' if s3['near_zero_frac'] > 0.3 else '✗'}"
          f"  ({s3['near_zero_frac']:.0%})")
    print(f"    基线使γ更聚 {'✓' if s2['std'] < s1['std'] else '✗'}"
          f"  (std: {s1['std']:.4f} → {s2['std']:.4f})")


if __name__ == "__main__":
    g1, s1 = run_group(1)
    g2, s2 = run_group(2)
    g3, s3 = run_group(3)

    print_group_result(1, "对数间隔分布（无基线，无稀疏）", g1, s1)
    print_group_result(2, "不同基线偏移（同形态 + 基线偏移）", g2, s2)
    print_group_result(3, "不同稀疏度（同形态 + 随机置零）", g3, s3)

    print_comparison(s1, s2, s3)

    print(f"\n状态：[判读·对照实验]")
