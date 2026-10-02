"""双对开放回路 v0.1 — 声-听 + 视-触，同时跑，看有没有第三东西。

对A：发声-听觉（8×64 → resample → 48×64 → resample → 8×64）
对B：视觉-触觉（48×64 → resample → 8×64 → resample → 48×64）

两对各自独立外部输入，不共享状态。
跑长，算两对 δ 的时间序列相关性，看有没有耦合。
"""

import random, math, json, sys, time
from pathlib import Path

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
import 矩阵.发声 as vocal_m
import 矩阵.听觉 as auditory_m
import 矩阵.视觉 as visual_m
import 矩阵.触觉 as tactile_m

STATE_FILE = Path.home() / ".catpaw" / "state" / "dual_loop_state.json"
TRACE_FILE = Path.home() / ".catpaw" / "traces" / "dual_loop_chain.jsonl"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)

ROWS_V, COLS_V = 8, 64    # vocal
ROWS_A, COLS_A = 48, 64   # auditory
ROWS_X, COLS_X = 48, 64   # visual
ROWS_T, COLS_T = 8, 64    # tactile
SEG = 500


def _resample(src, dr, dc):
    if not src or not src[0]:
        return [[0.0]*dc for _ in range(dr)]
    sr, sc = len(src), max(len(r) for r in src)
    out = []
    for i in range(dr):
        fi = i * (sr - 1) / max(dr - 1, 1)
        i0, i1 = int(fi), min(int(fi)+1, sr-1)
        wi = fi - i0
        row = []
        for j in range(dc):
            fj = j * (sc - 1) / max(dc - 1, 1)
            j0, j1 = int(fj), min(int(fj)+1, sc-1)
            wj = fj - j0
            v = (src[i0][j0]*(1-wi)*(1-wj) + src[i0][j1]*(1-wi)*wj +
                 src[i1][j0]*wi*(1-wj) + src[i1][j1]*wi*wj)
            row.append(v)
        out.append(row)
    return out


def _stats(f):
    fl = [v for r in f for v in r]
    m = sum(fl)/len(fl)
    return m, sum((v-m)**2 for v in fl)/len(fl), min(fl), max(fl)


def _delta(a, b):
    ra, ca = len(a), len(a[0])
    if (len(b), len(b[0])) != (ra, ca):
        b = _resample(b, ra, ca)
    s = sum((a[i][j]-b[i][j])**2 for i in range(ra) for j in range(ca))
    return math.sqrt(s / (ra*ca))


def _blend(a, b, w1=0.5, w2=0.5):
    ra, ca = len(a), len(a[0])
    if (len(b), len(b[0])) != (ra, ca):
        b = _resample(b, ra, ca)
    return [[w1*a[i][j]+w2*b[i][j] for j in range(ca)] for i in range(ra)]


def _ext_audio(f):
    """对A外部输入：低频拍频 + 空间行波"""
    phi = (1+math.sqrt(5))/2
    o = []
    for i in range(ROWS_V):
        r = []
        for j in range(COLS_V):
            x = i/max(ROWS_V-1,1); y = j/max(COLS_V-1,1); t = f/100.0
            v = (40*math.sin(2*math.pi*(math.sqrt(2)*t + x)) +
                 25*math.sin(2*math.pi*(math.sqrt(3)*t + y)) +
                 15*math.sin(2*math.pi*(phi*t + x*y)))
            r.append(v)
        o.append(r)
    return o


def _ext_visual(f):
    """对B外部输入：非谐波混合（与对A同类），不周期"""
    phi = (1+math.sqrt(5))/2
    o = []
    for i in range(ROWS_X):
        r = []
        for j in range(COLS_X):
            x = i/max(ROWS_X-1,1); y = j/max(COLS_X-1,1); t = f/100.0
            v = (50*math.sin(2*math.pi*(math.sqrt(2)*t + x)) +
                 30*math.sin(2*math.pi*(math.sqrt(3)*t + y)) +
                 20*math.sin(2*math.pi*(phi*t + x*y)))
            r.append(v)
        o.append(r)
    return o


def run_pair_a(vf, start, n):
    """发声-听觉对"""
    ds, vs, ams = [], [], []
    for off in range(n):
        f = start + off
        vout = vocal_m.frame_to_f0(vf)
        ext = _ext_audio(f)
        combined = _blend(vout, ext)
        c48 = _resample(combined, ROWS_A, COLS_A)
        aout, _ = auditory_m.frame_to_dbSPL(c48)
        nf = _resample(aout, ROWS_V, COLS_V)

        d = _delta(nf, vf)
        vm, _, _, _ = _stats(vf)
        am, _, _, _ = _stats(aout)
        ds.append(d); vs.append(vm); ams.append(am)
        vf = nf
    return vf, ds, vs, ams


def run_pair_b(vf, start, n):
    """视觉-触觉对"""
    ds, vs, tms = [], [], []
    for off in range(n):
        f = start + off
        v8 = _resample(vf, ROWS_T, COLS_T)
        tout = tactile_m.frame_to_pressure(v8)
        ext = _ext_visual(f)
        ext8 = _resample(ext, ROWS_T, COLS_T)
        combined = _blend(tout, ext8)
        nf = _resample(combined, ROWS_X, COLS_X)

        d = _delta(nf, vf)
        vm, _, _, _ = _stats(vf)
        tm, _, _, _ = _stats(tout)
        ds.append(d); vs.append(vm); tms.append(tm)
        vf = nf
    return vf, ds, vs, tms


def _corr(xs, ys):
    """皮尔逊相关系数"""
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n)) / n
    vx = sum((x-mx)**2 for x in xs) / n
    vy = sum((y-my)**2 for y in ys) / n
    if vx < 1e-30 or vy < 1e-30:
        return 0.0
    return cov / math.sqrt(vx * vy)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--segments", type=int, default=2)
    p.add_argument("--reset", action="store_true")
    a = p.parse_args()

    if a.reset and STATE_FILE.exists():
        STATE_FILE.unlink()

    if STATE_FILE.exists():
        st = json.loads(STATE_FILE.read_text())
        vf_a, vf_b = st["vf_a"], st["vf_b"]
        total, seg_c = st["total"], st["seg_count"]
        dA_hist = st["dA_hist"]
        dB_hist = st["dB_hist"]
    else:
        rng = random.Random(7)
        vf_a = [[rng.uniform(0,100) for _ in range(COLS_V)] for _ in range(ROWS_V)]
        vf_b = [[rng.uniform(0,200) for _ in range(COLS_X)] for _ in range(ROWS_X)]
        total, seg_c, dA_hist, dB_hist = 0, 0, [], []

    print(f"起始帧 f{total}, 已有段 {seg_c}")
    for s in range(a.segments):
        s0 = total
        vf_a, dA, vA, aA = run_pair_a(vf_a, s0, SEG)
        vf_b, dB, vB, tB = run_pair_b(vf_b, s0, SEG)

        m_A = sum(dA)/len(dA); mn_A = min(dA); mx_A = max(dA)
        m_B = sum(dB)/len(dB); mn_B = min(dB); mx_B = max(dB)

        # 两对 δ 相关性（段内 500 帧滑动窗口每 50 帧一个点，算 r）
        win = 50
        pts = []
        for k in range(0, len(dA)-win, win):
            r = _corr(dA[k:k+win], dB[k:k+win])
            pts.append(r)
        r_seg = sum(pts)/len(pts) if pts else 0.0

        rec = {
            "seg": seg_c, "frames": f"{s0}~{s0+SEG-1}",
            "A_delta_mean": round(m_A,5), "A_delta_min": round(mn_A,5), "A_delta_max": round(mx_A,5),
            "B_delta_mean": round(m_B,5), "B_delta_min": round(mn_B,5), "B_delta_max": round(mx_B,5),
            "AB_corr": round(r_seg,4),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(TRACE_FILE, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        dA_hist.append(round(m_A,4))
        dB_hist.append(round(m_B,4))

        print(f"段{seg_c:>3} [{rec['frames']:>12}] | "
              f"A: δ均={m_A:.5f} 范={mx_A-mn_A:.5f} | "
              f"B: δ均={m_B:.5f} 范={mx_B-mn_B:.5f} | "
              f"r(δA,δB)={r_seg:+.4f}")

        total += SEG; seg_c += 1

    STATE_FILE.write_text(json.dumps({
        "vf_a": vf_a, "vf_b": vf_b, "total": total,
        "seg_count": seg_c, "dA_hist": dA_hist, "dB_hist": dB_hist,
    }, ensure_ascii=False, indent=2))

    print(f"\n存状态: f{total}, 段 {seg_c}")
    print(f"A δ 历史: {dA_hist}")
    print(f"B δ 历史: {dB_hist}")


if __name__ == "__main__":
    main()
