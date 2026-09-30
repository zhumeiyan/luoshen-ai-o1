"""36 对跨模态差分通道——9 类传感器 × C(9,2) = 36 对。

九类传感器：
  视觉 / 听觉 / 触觉 / 温度觉 / 痛觉 / 发声 / 红外 / 雷达 / LiDAR

重点验证三对重复性：
  1. 视-听 vs 视-雷达 → γ 是否不同
  2. 听-雷达 vs 听-红外 → γ 是否不同
  3. 雷达-红外 vs 听-雷达 → γ 是否不同

每对输出：γ, g, h, 来源, 状态

纪律：
  - 全部合成数据跑
  - 只验证"通道是否不同"
  - 相同 = 重复通道 = 还需要再加设备
  - 不同 = 通道独立 = 多维扩展成立
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from 矩阵.跨模态 import 帧差分, 跨模态差分, 矩阵转二维
from 矩阵.视觉 import 建测试帧 as vis_frame, 视觉矩阵
from 矩阵.听觉 import 建测试帧 as aud_frame, 听矩阵
from 矩阵.触觉 import 建测试帧 as tac_frame, 触矩阵
from 矩阵.温度觉 import 建测试帧 as th_frame, 温矩阵
from 矩阵.痛觉 import 建测试帧 as pn_frame, 痛矩阵
from 矩阵.发声 import 建测试帧 as vo_frame, 发声矩阵
from 矩阵.红外 import 建测试帧 as ir_frame, 红外矩阵
from 矩阵.雷达 import 建测试帧 as rad_frame, 雷达矩阵
from 矩阵.激光雷达 import 建测试帧 as lid_frame, 激光雷达矩阵

USER = "diff36_demo"


def gen_all_matrices():
    """为每个传感器生成一帧基准矩阵（跨模态用）。"""
    matrices = {}

    # 视觉：gradient（48×64）
    matrices["视觉"] = 视觉矩阵.from_frame(vis_frame(pattern="gradient"), source="test").data

    # 听觉：tone_sweep（48×64）
    matrices["听觉"] = 听矩阵.from_frame(aud_frame(pattern="tone_sweep"), source="test").data

    # 触觉：gradient（8×64）
    matrices["触觉"] = 触矩阵.from_frame(tac_frame("gradient"), sub_modality="压力_kPa").data

    # 温度觉：cold_footprint（8×64）
    matrices["温度觉"] = 温矩阵.from_frame(th_frame("cold_footprint"), sub_modality="冷觉_°C").data

    # 痛觉：gradient（8×64）
    matrices["痛觉"] = 痛矩阵.from_frame(pn_frame("gradient"), sub_modality="机械痛_kPa").data

    # 发声：gradient（8×64）
    matrices["发声"] = 发声矩阵.from_frame(vo_frame("gradient"), sub_modality="基频_Hz").data

    # 红外：thermal_footprint（8×64）
    matrices["红外"] = 红外矩阵.from_frame(ir_frame("thermal_footprint"), sub_modality="红外温度_°C").data

    # 雷达：doppler_footprint（8×64）
    matrices["雷达"] = 雷达矩阵.from_frame(rad_frame("doppler_footprint"), sub_modality="速度_m_s").data

    # LiDAR：point_cloud_footprint（8×64）
    matrices["LiDAR"] = 激光雷达矩阵.from_frame(lid_frame("point_cloud_footprint"), sub_modality="距离_m").data

    return matrices


def run_cross_modal_diffs(matrices):
    """36 对跨模态差分通道。"""
    print("=" * 90)
    print("【36 跨模态差分 — C(9,2)】")
    print("=" * 90)
    senses = list(matrices.keys())
    results = {}
    for i in range(len(senses)):
        for j in range(i + 1, len(senses)):
            na, nb = senses[i], senses[j]
            ma = matrices[na]
            mb = matrices[nb]
            r = 跨模态差分(USER, ma, mb, na, nb)
            key = f"{na}-{nb}"
            results[key] = r
            print(f"  {na:4s}-{nb:4s} | g=[{r['g'][0]:+.6f}, {r['g'][1]:+.6f}]"
                  f" | σ={r['sigma']:.6f} | γ={r['gamma']:.6f}"
                  f" | hash={r['trace_hash']}")
    return results


def compare_pairs(results, pair1_key, pair2_key):
    """对比两对跨模态差分的 γ，判断是否相同。"""
    r1 = results.get(pair1_key)
    r2 = results.get(pair2_key)
    if r1 is None or r2 is None:
        return f"  {pair1_key}: MISSING  {pair2_key}: MISSING"
    g1_gamma = r1["gamma"]
    g2_gamma = r2["gamma"]
    same_gamma = abs(g1_gamma - g2_gamma) < 1e-6
    return (f"  {pair1_key:10s} γ={g1_gamma:.6f}"
            f"  |  {pair2_key:10s} γ={g2_gamma:.6f}"
            f"  |  {'重复?' if same_gamma else '不同 ✓'}")


def run_three_pair_comparison(results):
    """三对重点对比。"""
    print()
    print("=" * 90)
    print("【重点验证：通道重复性】")
    print("=" * 90)

    print("\n  检验1：视-听 vs 视-雷达（同一'视'通道，不同搭配 → 应不同）")
    print(f"    {compare_pairs(results, '视觉-听觉', '视觉-雷达')}")

    print("\n  检验2：听-雷达 vs 听-红外（雷达 vs 红外非同一通道 → 应不同）")
    print(f"    {compare_pairs(results, '听觉-雷达', '听觉-红外')}")

    print("\n  检验3：红-雷 vs 听-雷（两非视觉 vs 听雷 → 应不同）")
    print(f"    {compare_pairs(results, '红外-雷达', '听觉-雷达')}")

    print()


def print_full_summary(results):
    """完整输出所有 36 对的 γ 值。"""
    print("=" * 90)
    print("【完整 36 对 γ 值】")
    print("=" * 90)

    # 视觉 8 对
    print("\n  【视觉 与其他】")
    for k, v in sorted(results.items()):
        if k.startswith("视觉"):
            print(f"    {k:12s} γ={v['gamma']:.6f}")

    # 听觉的后续 7 对（跳过视-听）
    print("\n  【听觉 与 听以后】")
    for k, v in sorted(results.items()):
        if k.startswith("听觉-") and not k.startswith("视觉"):
            print(f"    {k:12s} γ={v['gamma']:.6f}")

    # 三新设备的跨对
    print("\n  【新设备：红外/雷达/LiDAR 跨对】")
    for k, v in sorted(results.items()):
        if any(d in k for d in ["红外", "雷达", "LiDAR"]):
            print(f"    {k:12s} γ={v['gamma']:.6f}")

    print()


if __name__ == "__main__":
    matrices = gen_all_matrices()
    results = run_cross_modal_diffs(matrices)
    run_three_pair_comparison(results)
    print_full_summary(results)
