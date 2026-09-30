"""21 对差分通道跑通验证——六感官 × (15 跨模态 + 6 帧差分)。

六感官：视觉 / 听觉 / 触觉 / 温度觉 / 痛觉 / 发声

15 对跨模态（C(6,2)）：
  视-听, 视-触, 视-温, 视-痛, 视-发声,
  听-触, 听-温, 听-痛, 听-发声,
  触-温, 触-痛, 触-发声,
  温-痛, 温-发声,
  痛-发声

6 帧差分：
  视觉, 听觉, 触觉, 温度觉, 痛觉, 发声

每对输出：γ, g, h, 来源, 状态

纪律：
  - 全部合成数据跑
  - 只验证"结构通不通"
  - 不验证"真数据准不准"
  - 纯数值，LLM 不参与
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

USER = "diff21_demo"


def gen_all_matrices():
    """为每个感官生成两帧（帧差分用）和一帧基准（跨模态用）。"""
    matrices = {}

    # 视觉：gradient + random（visual 建测试帧 signature: rows, cols, pattern）
    vis_g = 视觉矩阵.from_frame(vis_frame(pattern="gradient"), source="test").data
    vis_r = 视觉矩阵.from_frame(vis_frame(pattern="random"), source="test").data
    matrices["视觉"] = (vis_g, vis_r)

    # 听觉：tone_sweep + white_noise（auditory 建测试帧 signature: rows, cols, pattern）
    aud_s = 听矩阵.from_frame(aud_frame(pattern="tone_sweep"), source="test").data
    aud_v = 听矩阵.from_frame(aud_frame(pattern="white_noise"), source="test").data
    matrices["听觉"] = (aud_s, aud_v)

    # 触觉：gradient + random
    tac_g = 触矩阵.from_frame(tac_frame("gradient"), sub_modality="压力_kPa").data
    tac_r = 触矩阵.from_frame(tac_frame("random"), sub_modality="压力_kPa").data
    matrices["触觉"] = (tac_g, tac_r)

    # 温度觉：cold + hot
    th_c = 温矩阵.from_frame(th_frame("cold_footprint"), sub_modality="冷觉_°C").data
    th_h = 温矩阵.from_frame(th_frame("hot_footprint"), sub_modality="冷觉_°C").data
    matrices["温度觉"] = (th_c, th_h)

    # 痛觉：gradient + pain_threshold
    pn_g = 痛矩阵.from_frame(pn_frame("gradient"), sub_modality="机械痛_kPa").data
    pn_p = 痛矩阵.from_frame(pn_frame("pain_threshold_footprint"), sub_modality="机械痛_kPa").data
    matrices["痛觉"] = (pn_g, pn_p)

    # 发声：gradient + speech
    vo_g = 发声矩阵.from_frame(vo_frame("gradient"), sub_modality="基频_Hz").data
    vo_s = 发声矩阵.from_frame(vo_frame("speech_footprint"), sub_modality="基频_Hz").data
    matrices["发声"] = (vo_g, vo_s)

    return matrices


def run_frame_diffs(matrices):
    """6 条帧差分通道。"""
    print("=" * 80)
    print("【6 帧差分】")
    print("=" * 80)
    results = []
    for name, (f0, f1) in matrices.items():
        r = 帧差分(USER, f0, f1, channel=name)
        特征0 = 矩阵转二维(f0)
        特征1 = 矩阵转二维(f1)
        print(f"  {name:6s} | 特征0=[{特征0[0]:.4f}, {特征0[1]:.4f}] → 特征1=[{特征1[0]:.4f}, {特征1[1]:.4f}]"
              f" | g=[{r['g'][0]:+.4f}, {r['g'][1]:+.4f}]"
              f" | γ={r.get('gamma', 'N/A') if 'gamma' in r else 'N'}"
              f" | hash={r['trace_hash']} | seq={r['trace_seq']}")
        results.append((name, r))
    return results


def run_cross_modal_diffs(matrices):
    """15 对跨模态差分通道。"""
    print()
    print("=" * 80)
    print("【15 跨模态差分】")
    print("=" * 80)
    senses = list(matrices.keys())
    results = []
    for i in range(len(senses)):
        for j in range(i + 1, len(senses)):
            na, nb = senses[i], senses[j]
            # 用各感官的第一帧做跨模态
            ma = matrices[na][0]
            mb = matrices[nb][0]
            r = 跨模态差分(USER, ma, mb, na, nb)
            print(f"  {na:4s}-{nb:4s} | g=[{r['g'][0]:+.6f}, {r['g'][1]:+.6f}]"
                  f" | h=[{r['h'][0]:+.6f}, {r['h'][1]:+.6f}]"
                  f" | σ={r['sigma']:.6f} | γ={r['gamma']:.6f}"
                  f" | g≠h={'是' if any(abs(r['g'][k] - r['h'][k]) > 1e-10 for k in range(2)) else '否'}"
                  f" | hash={r['trace_hash']}")
            results.append(((na, nb), r))
    return results


def print_summary(frame_results, cross_results):
    """总结：结构通不通、γ量级、g≠h 是否成立。"""
    print()
    print("=" * 80)
    print("【总结】")
    print("=" * 80)

    print("\n  帧差分 6 条：")
    for name, r in frame_results:
        g = r.get('g', [None, None])
        print(f"    {name:6s} | g=[{g[0]:+.4f}, {g[1]:+.4f}] | 能跑=✓ | [判读·待核]")

    print("\n  跨模态差分 15 对：")
    all_g_neq_h = True
    all_gamma_ok = True
    for (na, nb), r in cross_results:
        g_ok = any(abs(r['g'][k] - r['h'][k]) > 1e-10 for k in range(2))
        gamma_ok = r['gamma'] is not None and abs(r['gamma']) < 1000
        if not g_ok:
            all_g_neq_h = False
        if not gamma_ok:
            all_gamma_ok = False
        print(f"    {na:4s}-{nb:4s} | g=[{r['g'][0]:+.6f}, {r['g'][1]:+.6f}]"
              f" | h=[{r['h'][0]:+.6f}, {r['h'][1]:+.6f}]"
              f" | γ={r['gamma']:.6f} | g≠h={'✓' if g_ok else '✗'} | [判读·待核]")

    print()
    print(f"  6 帧差分全部能跑：✓")
    print(f"  15 跨模态全部能跑：✓")
    print(f"  全部 γ 量级可控（|γ|<1000）：{'✓' if all_gamma_ok else '✗'}")
    print(f"  全部 g≠h 分离：{'✓' if all_g_neq_h else '✗'}")
    print()
    print(f"  21 条差分通道结构全通。")
    print(f"  状态：[判读·待核]（合成数据，仅验结构，不验真值）")


if __name__ == "__main__":
    matrices = gen_all_matrices()
    frame_results = run_frame_diffs(matrices)
    cross_results = run_cross_modal_diffs(matrices)
    print_summary(frame_results, cross_results)
