"""全上融合 v0.2 — 九路全跑 + L2 归一化 + 自净 + 时间对齐。

对照 v0.1：
  - L2 替代 L∞（两维均保留）
  - 温度觉去环境基线（~293 K）
  - 痛觉标为调制角色
  - 神经元标为垫底角色
  - 雷达/LiDAR/触觉标覆盖度
  - 时间对齐（挂基准频率 30 Hz）

输出：36 对 γ（L2 版）+ 角色标注 + 覆盖度。
标 [判读·待核]。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from 矩阵.跨模态 import (
    跨模态差分, compute_coverage, remove_baseline
)
from 矩阵.视觉 import 建测试帧 as vis_frame, 视觉矩阵
from 矩阵.听觉 import 建测试帧 as aud_frame, 听矩阵
from 矩阵.触觉 import 建测试帧 as tac_frame, 触矩阵
from 矩阵.温度觉 import 建测试帧 as th_frame, 温矩阵
from 矩阵.痛觉 import 建测试帧 as pn_frame, 痛矩阵
from 矩阵.发声 import 建测试帧 as vo_frame, 发声矩阵
from 矩阵.红外 import 建测试帧 as ir_frame, 红外矩阵
from 矩阵.雷达 import 建测试帧 as rad_frame, 雷达矩阵
from 矩阵.激光雷达 import 建测试帧 as lid_frame, 激光雷达矩阵

USER = "full_fusion_v02"

# ── 通道配置 ──
# role: evidence / modulation / substrate / sparse
CHANNELS = {
    "视觉":    {"role": "evidence",  "sub": "亮度_cd_m2",    "factory": "视觉矩阵"},
    "听觉":    {"role": "evidence",  "sub": "声压_dB_SPL",   "factory": "听矩阵"},
    "触觉":    {"role": "sparse",    "sub": "压力_kPa",      "factory": "触矩阵"},
    "温度觉":  {"role": "evidence",  "sub": "冷觉_°C",       "factory": "温矩阵",
               "baseline": 293.0},
    "痛觉":    {"role": "modulation","sub": "热痛_°C",       "factory": "痛矩阵"},
    "发声":    {"role": "evidence",  "sub": "基频_Hz",       "factory": "发声矩阵"},
    "红外":    {"role": "evidence",  "sub": "红外温度_°C",   "factory": "红外矩阵"},
    "雷达":    {"role": "sparse",    "sub": "速度_m_s",      "factory": "雷达矩阵"},
    "LiDAR":   {"role": "sparse",    "sub": "距离_m",        "factory": "激光雷达矩阵"},
}

# 通道顺序
ORDER = ["视觉", "听觉", "触觉", "温度觉", "痛觉", "发声", "红外", "雷达", "LiDAR"]


def gen_matrix(name):
    """生成一个通道的测试矩阵，应用自净（温度去基线、稀疏标覆盖度）。"""
    cfg = CHANNELS[name]
    frames = {
        "视觉":  vis_frame(pattern="gradient"),
        "听觉":  aud_frame(pattern="tone_sweep"),
        "触觉":  tac_frame("gradient"),
        "温度觉": th_frame("cold_footprint"),
        "痛觉":  pn_frame("pain_threshold_footprint"),
        "发声":  vo_frame("speech_footprint"),
        "红外":  ir_frame("thermal_footprint"),
        "雷达":  rad_frame("doppler_footprint"),
        "LiDAR": lid_frame("point_cloud_footprint"),
    }

    matrix_cls = {
        "视觉": 视觉矩阵, "听觉": 听矩阵,
        "触觉": 触矩阵, "温度觉": 温矩阵,
        "痛觉": 痛矩阵, "发声": 发声矩阵,
        "红外": 红外矩阵, "雷达": 雷达矩阵, "LiDAR": 激光雷达矩阵,
    }
    # 各矩阵 from_frame 签名不同，分别处理
    from_frame_args = {
        "视觉":  lambda f: 视觉矩阵.from_frame(f, source="device_spec"),
        "听觉":  lambda f: 听矩阵.from_frame(f, source="device_spec"),
        "触觉":  lambda f: 触矩阵.from_frame(f, sub_modality=CHANNELS["触觉"]["sub"]),
        "温度觉": lambda f: 温矩阵.from_frame(f, sub_modality=CHANNELS["温度觉"]["sub"]),
        "痛觉":  lambda f: 痛矩阵.from_frame(f, sub_modality=CHANNELS["痛觉"]["sub"]),
        "发声":  lambda f: 发声矩阵.from_frame(f, sub_modality=CHANNELS["发声"]["sub"]),
        "红外":  lambda f: 红外矩阵.from_frame(f, sub_modality=CHANNELS["红外"]["sub"]),
        "雷达":  lambda f: 雷达矩阵.from_frame(f, sub_modality=CHANNELS["雷达"]["sub"]),
        "LiDAR": lambda f: 激光雷达矩阵.from_frame(f, sub_modality=CHANNELS["LiDAR"]["sub"]),
    }

    m = from_frame_args[name](frames[name]).data

    # 自净：温度去环境基线
    if "baseline" in cfg:
        m = remove_baseline(m, cfg["baseline"])

    return m


def compute_channel_coverage(name, matrix):
    """计算覆盖度（稀疏路）。"""
    return compute_coverage(matrix)


def run_all_pairs():
    """生成九路 + 跑 36 对。"""
    print("=" * 90)
    print("【全上融合 v0.2 — L2 归一化 + 自净 + 角色标注】")
    print("=" * 90)

    # ── 1. 生成各通道矩阵 ──
    matrices = {}
    coverages = {}
    roles = {}

    for name in ORDER:
        m = gen_matrix(name)
        matrices[name] = m
        cov = compute_channel_coverage(name, m)
        coverages[name] = cov
        roles[name] = CHANNELS[name]["role"]

    # ── 2. 角色汇总 ──
    print("\n【角色与覆盖度】")
    for name in ORDER:
        print(f"  {name:6s}  角色={roles[name]:12s}  覆盖度={coverages[name]:.2%}")

    # ── 3. 跑 36 对跨模态 ──
    print("\n" + "=" * 90)
    print("【36 对跨模态差分 — L2 版】")
    print("=" * 90)

    results = {}
    for i in range(len(ORDER)):
        for j in range(i + 1, len(ORDER)):
            na, nb = ORDER[i], ORDER[j]
            cd = 跨模态差分(
                USER, matrices[na], matrices[nb], na, nb,
                coverage_a=coverages[na], coverage_b=coverages[nb],
                role_a=roles[na], role_b=roles[nb]
            )
            results[f"{na}-{nb}"] = cd
            print(f"  {na:5s}-{nb:5s} | γ={cd['gamma']:.6f}"
                  f"  g=[{cd['g'][0]:+.4f}, {cd['g'][1]:+.4f}]"
                  f"  cov=({cd['coverage_a']:.0%}, {cd['coverage_b']:.0%})"
                  f"  role=({cd['role_a'][:1]},{cd['role_b'][:1]})")

    return results, roles, coverages


def analyze_results(results):
    """分析：组块结构 + 0.41 簇是否散开 + 视听 vs 视触。"""
    print("\n" + "=" * 90)
    print("【分析】")
    print("=" * 90)

    gammas = {k: v["gamma"] for k, v in results.items()}
    vals = list(gammas.values())

    import numpy as np
    mean_g = np.mean(vals)
    std_g = np.std(vals)
    min_g = min(vals)
    max_g = max(vals)

    print(f"\n  γ 统计：mean={mean_g:.6f}  std={std_g:.6f}  "
          f"min={min_g:.6f}  max={max_g:.6f}")

    # 0.41 簇检查
    near_041 = {k: v for k, v in gammas.items() if 0.3 < v < 0.5}
    print(f"  0.3<γ<0.5 簇（{len(near_041)} 对）：")
    for k, v in sorted(near_041.items(), key=lambda x: x[1]):
        print(f"    {k:12s} γ={v:.6f}")

    # 视听 vs 视触
    vid_aud = gammas.get("视觉-听觉")
    vid_tac = gammas.get("视觉-触觉")
    print(f"\n  视听 vs 视触：γ={vid_aud:.6f} vs γ={vid_tac:.6f}  "
          f"差异={abs(vid_aud-vid_tac):.6f}  {'散开 ✓' if abs(vid_aud-vid_tac) > 0.01 else '仍撞 ✗'}")

    # 两维是否都非零
    both_nz = 0
    for k, v in results.items():
        if abs(v["g"][0]) > 1e-6 and abs(v["g"][1]) > 1e-6:
            both_nz += 1
    total = len(results)
    print(f"  两维均非零：{both_nz}/{total} ({both_nz/total:.0%})")

    # 组块结构（按 γ 值分桶）
    buckets = {"<0.1": [], "0.1-0.3": [], "0.3-0.5": [], "0.5-1.0": [], ">1.0": []}
    for k, v in gammas.items():
        if v < 0.1:
            buckets["<0.1"].append(k)
        elif v < 0.3:
            buckets["0.1-0.3"].append(k)
        elif v < 0.5:
            buckets["0.3-0.5"].append(k)
        elif v < 1.0:
            buckets["0.5-1.0"].append(k)
        else:
            buckets[">1.0"].append(k)

    print(f"\n  γ 分布：")
    for b, items in buckets.items():
        print(f"    {b:8s} : {len(items):2d} 对 → {', '.join(items[:5])}{'...' if len(items)>5 else ''}")


if __name__ == "__main__":
    results, roles, coverages = run_all_pairs()
    analyze_results(results)
    print(f"\n状态：[判读·待核]")
