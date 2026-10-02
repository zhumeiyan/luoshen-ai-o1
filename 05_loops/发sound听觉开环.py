"""发声-听觉开放闭环 v0.1 — 外部输入进来，看还能不能冻。

闭合回路：v_out → resample → aud_in → resample → v_in (= v_out)
  干净闭环，自己吃自己，纤维化不动点。

开放回路：v_out + external(t) → resample → aud_in → resample → v_in
  外部输入每一帧都不同。
  看 δ 还冻不冻。
"""

import random, math, sys

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
import 矩阵.发声 as vocal_m
import 矩阵.听觉 as auditory_m


def _resample_2d(src, dst_rows, dst_cols):
    if not src or not src[0]:
        return [[0.0] * dst_cols for _ in range(dst_rows)]
    src_rows = len(src)
    src_cols = max(len(r) for r in src)
    out = []
    for i in range(dst_rows):
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
            v00 = src[i0][j0]
            v01 = src[i0][j1]
            v10 = src[i1][j0]
            v11 = src[i1][j1]
            val = (v00*(1-wi)*(1-wj) + v01*(1-wi)*wj +
                   v10*wi*(1-wj) + v11*wi*wj)
            row.append(val)
        out.append(row)
    return out


def _external_input(frame_idx, rows=8, cols=64):
    """外部输入帧。每一帧都不同。

    频率用无理数倍（sqrt2, sqrt3, phi），确保永远不重复。
    三个时变分量：空间驻波 + 慢包络 + 拍频。
    """
    phi = (1 + math.sqrt(5)) / 2
    f1 = math.sqrt(2)
    f2 = math.sqrt(3)
    f3 = phi
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            # 空间坐标归一化到 [0,1]
            x = i / max(rows - 1, 1)
            y = j / max(cols - 1, 1)
            t = frame_idx / 100.0  # 时间缩放
            # 三个非谐波振荡 + 空间模式
            v = (50 * math.sin(2 * math.pi * (f1 * t + x)) +
                 30 * math.sin(2 * math.pi * (f2 * t + y)) +
                 20 * math.sin(2 * math.pi * (f3 * t + x * y)))
            row.append(v)
        out.append(row)
    return out


def _stats(frame_2d):
    flat = [v for r in frame_2d for v in r]
    n = len(flat)
    if n == 0:
        return {"mean": 0.0, "var": 0.0, "mn": 0.0, "mx": 0.0}
    mean = sum(flat) / n
    var = sum((v - mean) ** 2 for v in flat) / n
    return {"mean": mean, "var": var, "mn": min(flat), "mx": max(flat)}


def _l2_delta(a, b):
    ra, ca = len(a), len(a[0])
    rb, cb = len(b), len(b[0])
    if ra != rb or ca != cb:
        b = _resample_2d(b, ra, ca)
    s = 0.0
    n = 0
    for i in range(ra):
        for j in range(ca):
            d = a[i][j] - b[i][j]
            s += d * d
            n += 1
    return math.sqrt(s / n)


def _blend(a, b, wa=0.7, wb=0.3):
    """两帧线性混合。wa*a + wb*b，再按对应通道范围缩放。"""
    ra, ca = len(a), len(a[0])
    rb, cb = len(b), len(b[0])
    if ra != rb or ca != cb:
        b = _resample_2d(b, ra, ca)
    out = []
    for i in range(ra):
        row = []
        for j in range(ca):
            row.append(wa * a[i][j] + wb * b[i][j])
        out.append(row)
    return out


def run_open_loop(n_frames=5000, seed=7):
    rng = random.Random(seed)
    frame_v = [[rng.uniform(0, 100) for _ in range(64)] for _ in range(8)]

    records = []

    for f in range(n_frames):
        vocal_out = vocal_m.frame_to_f0(frame_v)  # vocal 输出 (8x64)

        # ━━━ 外部输入 ━━━
        ext = _external_input(f)  # 每帧都不同

        # 混合：vocal 输出 + 外部输入，一起进 auditory
        combined = _blend(vocal_out, ext, wa=0.5, wb=0.5)

        combined_48 = _resample_2d(combined, 48, 64)
        aud_out, _ = auditory_m.frame_to_dbSPL(combined_48)  # 48x64
        frame_next = _resample_2d(aud_out, 8, 64)  # 回到 8x64

        s_v = _stats(frame_v)
        s_a = _stats(aud_out)
        s_e = _stats(ext)
        delta = _l2_delta(frame_next, frame_v)

        # 记录
        records.append({
            "frame": f,
            "v_mean": round(s_v["mean"], 6),
            "v_var": round(s_v["var"], 4),
            "a_mean": round(s_a["mean"], 6),
            "a_var": round(s_a["var"], 4),
            "e_mean": round(s_e["mean"], 4),
            "e_var": round(s_e["var"], 4),
            "delta": round(delta, 10),
        })

        frame_v = frame_next

    return records


def detect_freeze(records, window=20, threshold=1e-6):
    """找 δ 降到 threshold 以下的起始帧。"""
    for i in range(window, len(records)):
        last_n = [r["delta"] for r in records[i - window:i + 1]]
        if all(d < threshold for d in last_n):
            return i - window + 1
    return None


def main():
    records = run_open_loop(n_frames=5000, seed=7)
    freeze_at = detect_freeze(records, window=20, threshold=1e-6)

    print(f"跑：5000 帧开放闭环，外部输入=非谐波三角混合 (sqrt2/sqrt3/phi)")
    print(f"混合比：vocal 0.5 + external 0.5")
    if freeze_at is not None:
        print(f"δ 冻到 <1e-6 的起始帧：{freeze_at}")
    else:
        print(f"δ 从未冻到 <1e-6（整个 5000 帧都保持波动）")
    print()

    # 每 500 帧切片
    step = 500
    print(f"【每 {step} 帧一条】")
    for r in records:
        if r["frame"] % step == 0 or r["frame"] == len(records) - 1:
            print(f"f{r['frame']:>5} | v_μ={r['v_mean']:>10.6f} v_σ²={r['v_var']:>8.4f} | "
                  f"a_μ={r['a_mean']:>10.6f} a_σ²={r['a_var']:>8.4f} | "
                  f"e_μ={r['e_mean']:>8.4f} e_σ²={r['e_var']:>8.4f} | δ={r['delta']:>12.10f}")

    # 首末对比
    first = records[0]
    last = records[-1]
    print()
    print(f"[首帧 f0]    v_μ={first['v_mean']:>10.6f} v_σ²={first['v_var']:>8.4f} "
          f"a_μ={first['a_mean']:>10.6f} a_σ²={first['a_var']:>8.4f} δ={first['delta']:.10f}")
    print(f"[末帧 f4999] v_μ={last['v_mean']:>10.6f} v_σ²={last['v_var']:>8.4f} "
          f"a_μ={last['a_mean']:>10.6f} a_σ²={last['a_var']:>8.4f} δ={last['delta']:.10f}")

    # δ 最小值和最大值
    min_r = min(records, key=lambda r: r["delta"])
    max_r = max(records, key=lambda r: r["delta"])
    print()
    print(f"[δ 最小] f{min_r['frame']} δ={min_r['delta']:.10f} v_μ={min_r['v_mean']:.6f} a_μ={min_r['a_mean']:.6f}")
    print(f"[δ 最大] f{max_r['frame']} δ={max_r['delta']:.10f} v_μ={max_r['v_mean']:.6f} a_μ={max_r['a_mean']:.6f}")


if __name__ == "__main__":
    main()
