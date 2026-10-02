"""双对互耦回路 v0.1 — A 的输出喂进 B，B 的输出喂进 A，一起跑。

对A：发声-听觉（8×64 ↔ 48×64）
对B：视觉-触觉（48×64 ↔ 8×64）

耦合方式：
  A 的 vocal_out (8×64) → resample 48×64 → 混入 B 的视觉输入
  B 的 tactile_out (8×64) → 混入 A 的 vocal 输入

外部输入每对各自独立（非谐波），耦合信号是额外加的。
"""

import random, math, json, sys, time
from pathlib import Path

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
import 矩阵.发声 as vocal_m
import 矩阵.听觉 as auditory_m
import 矩阵.视觉 as visual_m
import 矩阵.触觉 as tactile_m

STATE_FILE = Path.home() / ".catpaw" / "state" / "coupled_state.json"
TRACE_FILE = Path.home() / ".catpaw" / "traces" / "coupled_chain.jsonl"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)

ROWS_V, COLS_V = 8, 64
ROWS_A, COLS_A = 48, 64
ROWS_X, COLS_X = 48, 64
ROWS_T, COLS_T = 8, 64
SEG = 500
W_SELF = 0.5   # 自己回路权重
W_EXT = 0.3    # 外部输入权重
W_COUP = 0.2   # 耦合输入权重


def _resample(src, dr, dc):
    if not src or not src[0]:
        return [[0.0] * dc for _ in range(dr)]
    sr, sc = len(src), max(len(r) for r in src)
    out = []
    for i in range(dr):
        fi = i * (sr - 1) / max(dr - 1, 1)
        i0, i1 = int(fi), min(int(fi) + 1, sr - 1)
        wi = fi - i0
        row = []
        for j in range(dc):
            fj = j * (sc - 1) / max(dc - 1, 1)
            j0, j1 = int(fj), min(int(fj) + 1, sc - 1)
            wj = fj - j0
            v = (src[i0][j0] * (1 - wi) * (1 - wj) + src[i0][j1] * (1 - wi) * wj +
                 src[i1][j0] * wi * (1 - wj) + src[i1][j1] * wi * wj)
            row.append(v)
        out.append(row)
    return out


def _blend3(a, b, c, w1, w2, w3):
    ra, ca = len(a), len(a[0])
    if (len(b), len(b[0])) != (ra, ca):
        b = _resample(b, ra, ca)
    if (len(c), len(c[0])) != (ra, ca):
        c = _resample(c, ra, ca)
    return [[w1 * a[i][j] + w2 * b[i][j] + w3 * c[i][j]
             for j in range(ca)] for i in range(ra)]


def _stats(f):
    fl = [v for r in f for v in r]
    m = sum(fl) / len(fl)
    return m, sum((v - m) ** 2 for v in fl) / len(fl), min(fl), max(fl)


def _delta(a, b):
    ra, ca = len(a), len(a[0])
    if (len(b), len(b[0])) != (ra, ca):
        b = _resample(b, ra, ca)
    s = sum((a[i][j] - b[i][j]) ** 2 for i in range(ra) for j in range(ca))
    return math.sqrt(s / (ra * ca))


def _ext(f, rows, cols, seed_offset):
    phi = (1 + math.sqrt(5)) / 2
    o = []
    for i in range(rows):
        r = []
        for j in range(cols):
            x = i / max(rows - 1, 1)
            y = j / max(cols - 1, 1)
            t = (f + seed_offset) / 100.0
            v = (50 * math.sin(2 * math.pi * (math.sqrt(2) * t + x)) +
                 30 * math.sin(2 * math.pi * (math.sqrt(3) * t + y)) +
                 20 * math.sin(2 * math.pi * (phi * t + x * y)))
            r.append(v)
        o.append(r)
    return o


def step_A(vf, feedback_from_B, f):
    """发声-听觉步。输入 = 自己输出 + 外部 + 来自 B 的反馈"""
    vout = vocal_m.frame_to_f0(vf)
    ext_f = _ext(f, ROWS_V, COLS_V, seed_offset=100)
    # feedback_from_B 是 B 的 tactile_out (8×64)，直接混入 A 的 vocal 输入
    mixed_input = _blend3(vout, ext_f, feedback_from_B, W_SELF, W_EXT, W_COUP)
    c48 = _resample(mixed_input, ROWS_A, COLS_A)
    aout, _ = auditory_m.frame_to_dbSPL(c48)
    nf = _resample(aout, ROWS_V, COLS_V)
    # 输出给 B：vocal 输出 resample 到 48×64
    out_to_B = _resample(vout, ROWS_X, COLS_X)
    return nf, out_to_B, vout, aout


def step_B(vf, feedback_from_A, f):
    """视觉-触觉步。输入 = 自己输出 + 外部 + 来自 A 的反馈"""
    v8 = _resample(vf, ROWS_T, COLS_T)
    tout = tactile_m.frame_to_pressure(v8)
    ext_f = _ext(f, ROWS_T, COLS_T, seed_offset=200)
    mixed = _blend3(tout, ext_f, feedback_from_A, W_SELF, W_EXT, W_COUP)
    nf = _resample(mixed, ROWS_X, COLS_X)
    out_to_A = _resample(tout, ROWS_V, COLS_V)
    return nf, out_to_A, tout


def run_coupled(vf_a, vf_b, start, n):
    dA, dB, vA, aA, vB, tB = [], [], [], [], [], []
    fb_a = [[0.0] * COLS_V for _ in range(ROWS_V)]
    fb_b = [[0.0] * COLS_X for _ in range(ROWS_X)]

    for off in range(n):
        f = start + off
        vo_prev = vf_a
        vf_a, out_b, vo, ao = step_A(vf_a, fb_a, f)
        vf_b, out_a, to = step_B(vf_b, fb_b, f)
        fb_a = out_a
        fb_b = out_b

        dA.append(_delta(vf_a, vo_prev))
        dB.append(_delta(vf_b, to))

        vm, _, _, _ = _stats(vf_a)
        am, _, _, _ = _stats(ao)
        vA.append(vm)
        aA.append(am)
        bm, _, _, _ = _stats(vf_b)
        tm, _, _, _ = _stats(to)
        vB.append(bm)
        tB.append(tm)

    return vf_a, vf_b, dA, dB, vA, aA, vB, tB


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--segments", type=int, default=2)
    p.add_argument("--reset", action="store_true")
    a = p.parse_args()

    if a.reset:
        for f in [STATE_FILE, TRACE_FILE]:
            if f.exists():
                f.unlink()

    if STATE_FILE.exists():
        st = json.loads(STATE_FILE.read_text())
        vf_a = st["vf_a"]
        vf_b = st["vf_b"]
        total, seg_c = st["total"], st["seg_count"]
        h_dA, h_dB = st["dA"], st["dB"]
    else:
        rng = random.Random(7)
        vf_a = [[rng.uniform(0, 100) for _ in range(COLS_V)] for _ in range(ROWS_V)]
        vf_b = [[rng.uniform(0, 200) for _ in range(COLS_X)] for _ in range(ROWS_X)]
        total, seg_c, h_dA, h_dB = 0, 0, [], []

    print(f"起始帧 f{total}, 已有段 {seg_c}, 耦合权重 self={W_SELF} ext={W_EXT} coup={W_COUP}")
    for s in range(a.segments):
        s0 = total
        vf_a, vf_b, dA, dB, vA, aA, vB, tB = run_coupled(vf_a, vf_b, s0, SEG)
        mA, mnA, mxA = sum(dA) / len(dA), min(dA), max(dA)
        mB, mnB, mxB = sum(dB) / len(dB), min(dB), max(dB)

        # 逐帧相关
        win = 50
        cors = []
        for k in range(0, len(dA) - win, win):
            xs, ys = dA[k:k + win], dB[k:k + win]
            mx, my = sum(xs) / win, sum(ys) / win
            cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(win)) / win
            vx = sum((x - mx) ** 2 for x in xs) / win
            vy = sum((y - my) ** 2 for y in ys) / win
            if vx > 1e-30 and vy > 1e-30:
                cors.append(cov / math.sqrt(vx * vy))
        rseg = sum(cors) / len(cors) if cors else 0.0

        rec = {"seg": seg_c, "frames": f"{s0}~{s0 + SEG - 1}",
               "A_mean": round(mA, 5), "A_min": round(mnA, 5), "A_max": round(mxA, 5),
               "B_mean": round(mB, 5), "B_min": round(mnB, 5), "B_max": round(mxB, 5),
               "AB_corr": round(rseg, 4),
               "vA_mean": round(sum(vA) / len(vA), 5), "aA_mean": round(sum(aA) / len(aA), 5),
               "vB_mean": round(sum(vB) / len(vB), 5), "tB_mean": round(sum(tB) / len(tB), 5),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(TRACE_FILE, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        h_dA.append(round(mA, 5))
        h_dB.append(round(mB, 5))

        print(f"段{seg_c:>3} [{rec['frames']:>12}] | "
              f"A: δ均={mA:.5f} 范围={mnA:.5f}~{mxA:.5f} | "
              f"B: δ均={mB:.5f} 范围={mnB:.5f}~{mxB:.5f} | "
              f"r={rseg:+.4f} | "
              f"vA={sum(vA)/len(vA):.3f} aA={sum(aA)/len(aA):.3f} vB={sum(vB)/len(vB):.3f} tB={sum(tB)/len(tB):.3f}")

        total += SEG
        seg_c += 1

    STATE_FILE.write_text(json.dumps({
        "vf_a": vf_a, "vf_b": vf_b, "total": total,
        "seg_count": seg_c, "dA": h_dA, "dB": h_dB,
    }, ensure_ascii=False, indent=2))
    print(f"\n存状态: f{total}, A δ 历史: {h_dA}")
    print(f"           B δ 历史: {h_dB}")


if __name__ == "__main__":
    main()
