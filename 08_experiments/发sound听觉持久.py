"""发声-听觉开放回路 · 长期养化 v0.1 — 接上时间流。

不是单批 5000 帧，是多段接续跑。
每段从上一段的末帧接着跑，落一次痕。
可反复执行：每次 --continue 从断点续跑。

状态存：~/.catpaw/state/vaud_loop_state.json
落痕：~/.catpaw/traces/vaud_loop_chain.jsonl
"""

import random, math, json, sys, os, time
from pathlib import Path

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
import 矩阵.发声 as vocal_m
import 矩阵.听觉 as auditory_m

STATE_DIR = Path.home() / ".catpaw" / "state"
TRACE_DIR = Path.home() / ".catpaw" / "traces"
STATE_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "vaud_loop_state.json"
TRACE_FILE = TRACE_DIR / "vaud_loop_chain.jsonl"

ROWS, COLS = 8, 64
WA, WB = 0.5, 0.5
SEGMENT_SIZE = 500


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
            val = (v00 * (1 - wi) * (1 - wj) + v01 * (1 - wi) * wj +
                   v10 * wi * (1 - wj) + v11 * wi * wj)
            row.append(val)
        out.append(row)
    return out


def _external_input(frame_idx, rows=ROWS, cols=COLS):
    phi = (1 + math.sqrt(5)) / 2
    f1 = math.sqrt(2)
    f2 = math.sqrt(3)
    f3 = phi
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            x = i / max(rows - 1, 1)
            y = j / max(cols - 1, 1)
            t = frame_idx / 100.0
            v = (50 * math.sin(2 * math.pi * (f1 * t + x)) +
                 30 * math.sin(2 * math.pi * (f2 * t + y)) +
                 20 * math.sin(2 * math.pi * (f3 * t + x * y)))
            row.append(v)
        out.append(row)
    return out


def _stats(frame_2d):
    flat = [v for r in frame_2d for v in r]
    n = len(flat)
    mean = sum(flat) / n
    var = sum((v - mean) ** 2 for v in flat) / n
    return mean, var, min(flat), max(flat)


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


def _blend(a, b, wa=WA, wb=WB):
    ra, ca = len(a), len(a[0])
    rb, cb = len(b), len(b[0])
    if ra != rb or ca != cb:
        b = _resample_2d(b, ra, ca)
    return [[wa * a[i][j] + wb * b[i][j] for j in range(ca)] for i in range(ra)]


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return None


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append_trace(record):
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_segment(frame_v, start_frame, n_frames):
    deltas = []
    v_means = []
    a_means = []

    for offset in range(n_frames):
        f = start_frame + offset
        vocal_out = vocal_m.frame_to_f0(frame_v)
        ext = _external_input(f)
        combined = _blend(vocal_out, ext)
        combined_48 = _resample_2d(combined, 48, 64)
        aud_out, _ = auditory_m.frame_to_dbSPL(combined_48)
        frame_next = _resample_2d(aud_out, ROWS, COLS)

        vm, vv, _, _ = _stats(frame_v)
        am, av, _, _ = _stats(aud_out)
        delta = _l2_delta(frame_next, frame_v)

        deltas.append(delta)
        v_means.append(vm)
        a_means.append(am)

        frame_v = frame_next

    return frame_v, deltas, v_means, a_means


def init_fresh(seed=7):
    rng = random.Random(seed)
    frame_v = [[rng.uniform(0, 100) for _ in range(COLS)] for _ in range(ROWS)]
    return {
        "frame_v": frame_v,
        "total_frames": 0,
        "segment_count": 0,
        "history_delta_mean": [],
        "history_delta_range": [],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue", dest="cont", action="store_true", help="从上次断点续跑")
    parser.add_argument("--segments", type=int, default=1, help="本次跑几段 (每段500帧)")
    parser.add_argument("--reset", action="store_true", help="重置，从头开始")
    args = parser.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print("状态重置，下一帧从头开始")

    if args.cont and STATE_FILE.exists():
        state = load_state()
        frame_v = state["frame_v"]
        total = state["total_frames"]
        seg_count = state["segment_count"]
        h_delta_mean = state["history_delta_mean"]
        h_delta_range = state["history_delta_range"]
    else:
        state = init_fresh()
        frame_v = state["frame_v"]
        total = 0
        seg_count = 0
        h_delta_mean = []
        h_delta_range = []

    print(f"起始帧: f{total}, 已有段数: {seg_count}")
    print("-" * 100)

    for seg in range(args.segments):
        start_f = total
        new_frame_v, deltas, v_means, a_means = run_segment(frame_v, start_f, SEGMENT_SIZE)

        # 段统计
        d_mean = sum(deltas) / len(deltas)
        d_min = min(deltas)
        d_max = max(deltas)
        v_mean_seg = sum(v_means) / len(v_means)
        a_mean_seg = sum(a_means) / len(a_means)

        # 连续性指标：段起始 δ（第一帧 vs 上一段末帧）
        # 用的是 run_segment 内部第一帧的 δ
        first_delta = deltas[0]

        record = {
            "segment": seg_count,
            "frames": f"{start_f}~{start_f + SEGMENT_SIZE - 1}",
            "delta_mean": round(d_mean, 6),
            "delta_min": round(d_min, 6),
            "delta_max": round(d_max, 6),
            "delta_range": round(d_max - d_min, 6),
            "first_delta": round(first_delta, 10),
            "v_mean": round(v_mean_seg, 4),
            "a_mean": round(a_mean_seg, 4),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        append_trace(record)
        h_delta_mean.append(round(d_mean, 4))
        h_delta_range.append(round(d_max - d_min, 4))

        print(f"段 {seg_count:>3} [{record['frames']:>12}] | "
              f"δ_均={d_mean:.5f} δ_范={d_max - d_min:.5f} δ_首={first_delta:.5f} | "
              f"v_μ={v_mean_seg:.3f} a_μ={a_mean_seg:.3f}")

        frame_v = new_frame_v
        total += SEGMENT_SIZE
        seg_count += 1

    save_state({
        "frame_v": frame_v,
        "total_frames": total,
        "segment_count": seg_count,
        "history_delta_mean": h_delta_mean,
        "history_delta_range": h_delta_range,
    })

    print("-" * 100)
    print(f"存状态: f{total}, 段 {seg_count}, δ_均历史: {h_delta_mean}")
    print(f"痕: {TRACE_FILE}")


if __name__ == "__main__":
    main()
