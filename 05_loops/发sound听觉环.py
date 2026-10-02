"""发声-听觉连续循环测试 v0.1 — 果：连着跑几百帧，看变化趋势。

两个矩阵形成闭环：
  vocal (8×64) 发声帧 → resample 到 48×64 → auditory 接收并映射到 dB SPL
  auditory (48×64) 输出帧 → resample 回 8×64 → 喂回 vocal 作为下一帧

每帧记录：
  - vocal 帧均值 / 方差 / 值域
  - auditory 帧均值 / 方差 / 值域
  - 帧间 delta (L2 norm)
  - 相邻帧 delta 的 diff (二阶差)

纪律：
  - 不判趋势、不写结论、不标同质化/衰变的判断语
  - 只摆数字
"""

import random, math, sys

# 直接在函数里 import，避免顶层 from...import 触发矩阵类校验
def _import_matrices():
    sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
    import 矩阵.发声 as vocal_m
    import 矩阵.听觉 as auditory_m
    return vocal_m, auditory_m


def _resample_2d(src, dst_rows, dst_cols):
    """把 src (R×C) 重采样到 dst_rows×dst_cols。连续插值。"""
    if not src or not src[0]:
        return [[0.0] * dst_cols for _ in range(dst_rows)]
    src_rows = len(src)
    src_cols = max(len(r) for r in src)
    out = []
    for i in range(dst_rows):
        # 连续坐标映射
        fi = i * (src_rows - 1) / max(dst_rows - 1, 1)
        i0 = int(fi)
        i1 = min(i0 + 1, src_rows - 1)
        wi = fi - i0
        row = []
        for j in range(dst_cols):
            fj = j * (src_cols - 1) / max(dst_cols - 1, 1)
            j0 = int(fj)
            j1 = min(j0 + 1, src_cols - 1)
            wj = fj - j0
            # 双线性
            v00 = src[i0][j0]
            v01 = src[i0][j1]
            v10 = src[i1][j0]
            v11 = src[i1][j1]
            val = (v00 * (1-wi) * (1-wj) + v01 * (1-wi) * wj +
                   v10 * wi * (1-wj) + v11 * wi * wj)
            row.append(val)
        out.append(row)
    return out


def _stats(frame_2d):
    """返回均值、方差、min、max。"""
    flat = [v for r in frame_2d for v in r]
    n = len(flat)
    if n == 0:
        return {"mean": 0.0, "var": 0.0, "mn": 0.0, "mx": 0.0}
    mean = sum(flat) / n
    var = sum((v - mean) ** 2 for v in flat) / n
    return {"mean": mean, "var": var, "mn": min(flat), "mx": max(flat)}


def _l2_delta(frame_a, frame_b):
    """两个帧之间的 L2 距离。不同尺寸先 resample 到同尺寸。"""
    ra, ca = len(frame_a), len(frame_a[0]) if frame_a[0] else 0
    rb, cb = len(frame_b), len(frame_b[0]) if frame_b[0] else 0
    if ra != rb or ca != cb:
        frame_b = _resample_2d(frame_b, ra, ca)
    s = 0.0
    for i in range(ra):
        for j in range(ca):
            d = frame_a[i][j] - frame_b[i][j]
            s += d * d
    return math.sqrt(s / (ra * ca))


def run_loop(n_frames=400, seed=7):
    vocal_m, auditory_m = _import_matrices()

    # 初始帧：随机 (8×64)
    rng = random.Random(seed)
    frame_v = [[rng.uniform(0, 100) for _ in range(64)] for _ in range(8)]

    records = []

    for f in range(n_frames):
        # 1. vocal 帧发声：用 frame_v 做输入，通过 frame_to_f0 映射到基频范围
        #    发声的本质是"把当前状态映射为声学参数"——取三通道的均值代表能量
        vocal_out = VocalMatrixWrapper(frame_v, vocal_m)

        # 2. auditory 接收：把 vocal_out (8×64) resample 到 48×64，映射到 dB SPL
        vocal_as_48 = _resample_2d(vocal_out, 48, 64)
        aud_matrix, _ = auditory_m.frame_to_dbSPL(vocal_as_48)
        # aud_matrix 是 48×64，值域 0–120 dB SPL

        # 3. 反馈：auditory 输出 resample 回 8×64，作为下一帧 vocal 输入
        frame_next = _resample_2d(aud_matrix, 8, 64)

        # 4. 统计
        s_v = _stats(frame_v)
        s_a = _stats(aud_matrix)
        delta = _l2_delta(frame_next, frame_v)
        delta_prev = records[-1]["delta"] if records else None
        d2 = abs(delta - delta_prev) if delta_prev is not None else 0.0

        rec = {
            "frame": f,
            "v_mean": round(s_v["mean"], 4),
            "v_var": round(s_v["var"], 4),
            "v_mn": round(s_v["mn"], 4),
            "v_mx": round(s_v["mx"], 4),
            "a_mean": round(s_a["mean"], 4),
            "a_var": round(s_a["var"], 4),
            "a_mn": round(s_a["mn"], 4),
            "a_mx": round(s_a["mx"], 4),
            "delta": round(delta, 6),
            "delta_prev_diff": round(d2, 6),
        }
        records.append(rec)

        # 交换
        frame_v = frame_next

    return records


class VocalMatrixHelper:
    """把 frame_v 通过发声 factory 映射——但不创建 发声矩阵 实例（避免尺寸校验）。"""
    pass


def VocalMatrixWrapper(frame_2d, vocal_m):
    """直接调 frame_to_f0 取基频范围。返回 8×64。"""
    return vocal_m.frame_to_f0(frame_2d)


def print_slice(records, step=50):
    """每隔 step 帧打印一条。"""
    for r in records:
        if r["frame"] % step == 0 or r["frame"] == len(records) - 1:
            print(f"f{r['frame']:>4} | "
                  f"v μ={r['v_mean']:>8.3f} σ²={r['v_var']:>8.3f} [{r['v_mn']:>7.2f},{r['v_mx']:>7.2f}] | "
                  f"a μ={r['a_mean']:>8.3f} σ²={r['a_var']:>8.3f} [{r['a_mn']:>7.2f},{r['a_mx']:>7.2f}] | "
                  f"δ={r['delta']:>8.5f} | d²={r['delta_prev_diff']:>8.6f}")


def main():
    n = 400
    records = run_loop(n_frames=n, seed=7)
    print(f"发声-听觉循环测试，{n} 帧")
    print("v: vocal 帧 (8×64) 统计 | a: auditory 帧 (48×64) 统计 | δ: 帧间 L2 | d²: 相邻 δ 差")
    print("-" * 120)
    print_slice(records, step=25)

    # 首尾对比
    first = records[0]
    last = records[-1]
    print()
    print(f"[首帧 f{first['frame']}] v_μ={first['v_mean']:.3f} v_σ²={first['v_var']:.3f} a_μ={first['a_mean']:.3f} δ={first['delta']:.5f}")
    print(f"[末帧 f{last['frame']}] v_μ={last['v_mean']:.3f} v_σ²={last['v_var']:.3f} a_μ={last['a_mean']:.3f} δ={last['delta']:.5f}")


if __name__ == "__main__":
    main()
