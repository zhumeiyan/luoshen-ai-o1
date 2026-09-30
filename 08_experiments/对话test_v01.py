"""对话测试 v0.1 — 整条链路：文本 → 全矩阵 → 跨模态差分 → 融合。

最小一组：一句话（文字）走完整链路。
标 [判读·对话测试]。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import hashlib
import numpy as np

from 矩阵.跨模态 import (
    跨模态差分, 归一化N维, 矩阵转六维,
    compute_coverage, remove_baseline
)
from 矩阵.视觉 import 视觉矩阵
from 矩阵.听觉 import 听矩阵
from 矩阵.触觉 import 触矩阵
from 矩阵.温度觉 import 温矩阵
from 矩阵.痛觉 import 痛矩阵
from 矩阵.发声 import 发声矩阵
from 矩阵.红外 import 红外矩阵
from 矩阵.雷达 import 雷达矩阵
from 矩阵.激光雷达 import 激光雷达矩阵


# ── 1. 文本 → 种子序列 ──
def text_to_seed(text):
    """文本 → Unicode 码点序列（归一化到 0..256）。"""
    return [ord(c) % 256 + 1 for c in text]


# ── 2. 种子 → 各矩阵 ──
def seed_to_matrix(name, seed):
    """不同通道用不同变换，把种子映射到矩阵。"""
    n = 64 * 8
    arr = np.zeros(n)

    if name == "视觉":
        # 文本空间分布：字的位置 → 亮度
        for i, v in enumerate(seed):
            idx = i % n
            arr[idx] = v * 1000  # 高动态
        # 加一些随机扰动
        for i in range(n):
            if arr[i] == 0:
                arr[i] = np.random.uniform(10, 500)

    elif name == "听觉":
        # 文本 → 频率特征：字符序 → 基频
        for i, v in enumerate(seed):
            idx = i % n
            arr[idx] = 80 + v * 5  # 基频范围

    elif name == "触觉":
        # 文本 → 压力：稀疏（只在有字的位置有压力）
        for i, v in enumerate(seed):
            idx = i % n
            arr[idx] = v * 4  # kPa 压力
        # 其余为 0（稀疏）

    elif name == "温度觉":
        # 文本 → 温度：字符热值 → 温度偏移 + 环境基线
        baseline = 293.0  # 300K 环境基线
        for i in range(n):
            arr[i] = baseline + np.random.uniform(-5, 15)

    elif name == "痛觉":
        # 文本 → 痛觉调制路：字符强度 → 痛觉强度
        for i, v in enumerate(seed):
            idx = i % n
            arr[idx] = v / 25.6  # 0-10 强度

    elif name == "发声":
        # 文本 → 基频 + 声压：模拟语音
        for i, v in enumerate(seed):
            idx = i % n
            arr[idx] = 200 + v * 3  # Hz

    elif name == "红外":
        # 文本 → 热像：字符热特征
        for i, v in enumerate(seed):
            idx = i % n
            arr[idx] = 200 + v * 2  # 温度分量

    elif name == "雷达":
        # 文本 → 速度/频移（稀疏）
        for i, v in enumerate(seed):
            idx = i % n
            if v > 100:  # 只有"强"字符产生速度
                arr[idx] = (v - 100) / 156 * 200 - 100  # -100..100 m/s

    elif name == "LiDAR":
        # 文本 → 距离/ToF（稀疏）
        for i, v in enumerate(seed):
            idx = i % n
            if v % 3 == 0:  # 1/3 覆盖率
                arr[idx] = 0.3 + (v / 255) * 100  # 距离

    return arr.reshape(8, 64).tolist()


def generate_all_matrices(text):
    """文本 → 九路矩阵。"""
    seed = text_to_seed(text)
    matrices = {}
    channels = ["视觉", "听觉", "触觉", "温度觉", "痛觉", "发声", "红外", "雷达", "LiDAR"]
    for name in channels:
        matrices[name] = seed_to_matrix(name, seed)
    return matrices


# ── 3. 角色 + 覆盖度 ──
CHANNEL_CONFIG = {
    "视觉":    {"role": "evidence"},
    "听觉":    {"role": "evidence"},
    "触觉":    {"role": "sparse"},
    "温度觉":  {"role": "evidence", "baseline": 293.0},
    "痛觉":    {"role": "modulation"},
    "发声":    {"role": "evidence"},
    "红外":    {"role": "evidence"},
    "雷达":    {"role": "sparse"},
    "LiDAR":   {"role": "sparse"},
}


def compute_roles_and_coverage(matrices):
    roles = {}
    coverages = {}
    for name, m in matrices.items():
        roles[name] = CHANNEL_CONFIG[name]["role"]
        coverages[name] = compute_coverage(m)
    return roles, coverages


# ── 4. 整条链路 ──
def run_full_pipeline(text):
    print("=" * 80)
    print("【对话测试 v0.1 — 完整链路】")
    print("=" * 80)

    print(f"\n  「输入」{text}")

    # Step 1: 文本 → 种子
    seed = text_to_seed(text)
    print(f"\n  「种子」前 {len(seed)} 个码点: {seed[:10]}...")

    # Step 2: 种子 → 九路矩阵
    matrices = generate_all_matrices(text)
    print(f"\n  「矩阵生成」")
    for name, m in matrices.items():
        flat = [v for row in m for v in row]
        print(f"    {name:6s}: 值域=[{min(flat):.1f}, {max(flat):.1f}]  "
              f"均值={np.mean(flat):.1f}  coverage={compute_coverage(m):.1%}")

    # Step 3: 跨模态差分（36 对）
    print(f"\n  「跨模态差分 36 对」")
    roles, coverages = compute_roles_and_coverage(matrices)

    diff_results = {}
    channels = list(matrices.keys())
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            na, nb = channels[i], channels[j]
            # 自净：温度觉去基线
            ma, mb = matrices[na], matrices[nb]
            if "baseline" in CHANNEL_CONFIG.get(na, {}):
                ma = remove_baseline(ma, CHANNEL_CONFIG[na]["baseline"])
            if "baseline" in CHANNEL_CONFIG.get(nb, {}):
                mb = remove_baseline(mb, CHANNEL_CONFIG[nb]["baseline"])

            cd = 跨模态差分(
                "dialog_test", ma, mb, na, nb,
                coverage_a=coverages[na], coverage_b=coverages[nb],
                role_a=roles[na], role_b=roles[nb]
            )
            diff_results[f"{na}-{nb}"] = cd

    for k, v in sorted(diff_results.items(), key=lambda x: abs(x[1]["gamma"])):
        print(f"    {k:12s} γ={v['gamma']:.6f}  "
              f"g=[{v['g'][0]:+.4f},{v['g'][1]:+.4f}]  "
              f"cov=({v['coverage_a']:.0%},{v['coverage_b']:.0%})")

    # Step 4: 融合
    print(f"\n  「融合」")
    # 简化：所有 evidence 路的 γ 排序
    evidence_pairs = {k: v for k, v in diff_results.items()
                      if v["role_a"] == "evidence" or v["role_b"] == "evidence"}
    sorted_pairs = sorted(evidence_pairs.items(), key=lambda x: abs(x[1]["gamma"]))

    # 硬门槛
    threshold = 0.05
    over_threshold = [(k, v) for k, v in sorted_pairs if abs(v["gamma"]) > threshold]
    under_threshold = [(k, v) for k, v in sorted_pairs if abs(v["gamma"]) <= threshold]

    print(f"    硬门槛 θ={threshold}")
    print(f"    超过门槛（{len(over_threshold)} 对）：")
    for k, v in over_threshold[:12]:
        print(f"      {k:12s} γ={v['gamma']:.4f}")
    print(f"    低于门槛（{len(under_threshold)} 对，近似相同）：")
    for k, v in under_threshold[:8]:
        print(f"      {k:12s} γ={v['gamma']:.4f}")

    # Step 5: 整句 "对上了没有"
    print(f"\n  「整句判定」")
    if len(over_threshold) > 0:
        top_pair = over_threshold[0]
        print(f"    主差异源：{top_pair[0]}  γ={top_pair[1]['gamma']:.4f}")
        print(f"    判定：文本通道产生了独立特征 ✓")
    else:
        print(f"    判定：所有通道均低于门槛 → 无区分信息 ✗")

    print(f"\n状态：[判读·对话测试]")
    return matrices, diff_results, roles, coverages


if __name__ == "__main__":
    # Min test: one simple sentence
    text = "你好洛神"
    matrices, diff_results, roles, coverages = run_full_pipeline(text)
