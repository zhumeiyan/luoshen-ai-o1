"""发声-听觉连续循环 — 跑到底。

接着 400 帧跑，继续跑到 δ 触底
记录：δ 最低值出现在哪一帧、之后怎么走
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
            val = (v00 * (1-wi)*(1-wj) + v01 * (1-wi)*wj +
                   v10 * wi*(1-wj) + v11 * wi*wj)
            row.append(val)
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
    ra, ca = len(a), len(a[0]) if a[0] else 0
    rb, cb = len(b), len(b[0]) if b[0] else 0
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


def run_loop(n_frames=5000, seed=7):
    # 初始帧：随机 (8×64)，与上一轮同 seed
    rng = random.Random(seed)
    frame_v = [[rng.uniform(0, 100) for _ in range(64)] for _ in range(8)]

    records = []
    min_delta = float('inf')
    min_frame = -1
    plateau_counter = 0
    plateau_threshold = 1e-8  # 连续 N 帧 δ 变化量低于此值视为触底

    for f in range(n_frames):
        # vocal -> resample -> auditory -> resample -> next vocal
        vocal_out = vocal_m.frame_to_f0(frame_v)
        vocal_as_48 = _resample_2d(vocal_out, 48, 64)
        aud_out, _ = auditory_m.frame_to_dbSPL(vocal_as_48)
        frame_next = _resample_2d(aud_out, 8, 64)

        s_v = _stats(frame_v)
        s_a = _stats(aud_out)
        delta = _l2_delta(frame_next, frame_v)

        if delta < min_delta:
            min_delta = delta
            min_frame = f
            plateau_counter = 0
        else:
            plateau_counter += 1

        records.append({
            "frame": f,
            "v_mean": round(s_v["mean"], 6),
            "v_var": round(s_v["var"], 4),
            "a_mean": round(s_a["mean"], 6),
            "a_var": round(s_a["var"], 4),
            "delta": round(delta, 10),
        })

        frame_v = frame_next

    return records, min_frame, min_delta


def detect_plateau(records):
    """找 δ 从'下降'变为'平稳/回升'的转折点。
    用连续10帧的 δ 净变化判断：净变化 < 1e-9 视为触底。"""
    for i in range(50, len(records) - 10):
        window = [r["delta"] for r in records[i:i+10]]
        net_change = abs(window[-1] - window[0])
        if net_change < 1e-9:
            return i, records[i]["delta"]
    return None, None


def main():
    records, min_frame, min_delta = run_loop(n_frames=5000, seed=7)
    plateau_frame, plateau_delta = detect_plateau(records)

    print(f"跑：5000 帧，seed=7")
    print(f"全局最低 δ：frame={min_frame}, δ={min_delta:.10f}")
    print(f"触底区（连续10帧净变化<1e-9）：frame={plateau_frame}, δ={plateau_delta}")
    print()

    # 打印触底附近 ±50 帧
    if plateau_frame:
        center = plateau_frame
        start = max(0, center - 30)
        end = min(len(records), center + 50)
        print(f"【触底附近 frame {start}~{end-1}，每5帧一条】")
        for r in records[start:end]:
            if (r["frame"] - start) % 5 == 0:
                print(f"f{r['frame']:>5} | v_μ={r['v_mean']:>10.6f} v_σ²={r['v_var']:>8.4f} | "
                      f"a_μ={r['a_mean']:>10.6f} a_σ²={r['a_var']:>8.4f} | δ={r['delta']:>12.10f}")

    # 末段
    print()
    print(f"【末段 frame 4980~5000】")
    for r in records[4980:5000]:
        if (r["frame"] - 4980) % 4 == 0:
            print(f"f{r['frame']:>5} | v_μ={r['v_mean']:>10.6f} v_σ²={r['v_var']:>8.4f} | "
                  f"a_μ={r['a_mean']:>10.6f} a_σ²={r['a_var']:>8.4f} | δ={r['delta']:>12.10f}")


if __name__ == "__main__":
    main()
