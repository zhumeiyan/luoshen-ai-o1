"""活循环 v1.1 —— 永远在转。

嘴态 + 听态，两个连续量，没有阈值。
每转一圈写入哈希链（痕），产出值反馈为下一次输入。
外部真实录像数据每 56 帧循环注入一次，防止冻死。

系统自己产：
  - 嘴态 m（画面中央区域垂直梯度的连续值）
  - 听态 a（听觉通道的连续 burst 强度）
  - 两者互耦：m→a, a→m
  - 写入哈希链：痕/src=系统自转

循环没有停止条件。冻不死，因为外部数据周期注入打破不动点。
冻不死也意味着没有"完成"——只有"还在转"。
"""

import json, sys, os, math, struct, time

sys.path.insert(0, os.path.dirname(__file__))
from 洛神 import Trace

# ──────────────────────────────────────────────
# 一、读真实数据锚点
# ──────────────────────────────────────────────

def read_wav_pcm(path):
    with open(path, 'rb') as f:
        f.read(4); f.read(4); f.read(4)
        pcm = None; sr = None; bits = None
        while True:
            ck = f.read(4)
            if len(ck) < 4: break
            cl = struct.unpack('<I', f.read(4))[0]
            if ck == b'fmt ':
                fmt = f.read(cl)
                sr = struct.unpack('<I', fmt[4:8])[0]
                bits = struct.unpack('<H', fmt[14:16])[0]
            elif ck == b'data':
                pcm = f.read(cl); break
            else: f.read(cl)
    return pcm, sr, bits


def decode_pcm(pcm, bits=16):
    n = len(pcm) // 2
    return [s / 32780.0 for s in struct.unpack('<{}h'.format(n), pcm)]


COLS = 64; ROWS = 48
MOUTH_R0 = 23; MOUTH_R1 = 29; MOUTH_C0 = 24; MOUTH_C1 = 34


def build_aud_sparse(sub):
    n = len(sub); bs = max(1, n // COLS)
    zr = []; er = []
    for c in range(COLS):
        s = c*bs; e = min(s+bs, n); blk = sub[s:e]
        if not blk: zr.append(0.0); er.append(0.0); continue
        zc = sum(1 for i in range(1,len(blk)) if blk[i-1]>=0>blk[i])
        sq = sum(x*x for x in blk)/len(blk)
        zr.append(float(zc)); er.append(sq*10000)
    return [zr, er]


def read_ppm(path):
    with open(path,'rb') as f:
        magic = f.readline().strip()
        ln = f.readline().strip()
        while ln.startswith(b'#'): ln = f.readline().strip()
        w,h = map(int, ln.split())
        f.readline(); data = f.read()
    if magic == b'P6':
        px = [(data[i],data[i+1],data[i+2]) for i in range(0,len(data),3)]
        return [[0.299*r+0.587*g+0.114*b for r,g,b in px[y*w:(y+1)*w]] for y in range(h)], w, h
    elif magic == b'P3':
        vs = [float(x) for x in data.split()]
        return [[0.299*vs[(y*w+x)*3]+0.587*vs[(y*w+x)*3+1]+0.114*vs[(y*w+x)*3+2] for x in range(w)] for y in range(h)], w, h


def resize(src, dr=ROWS, dc=COLS):
    if not src or not src[0]: return [[0.0]*dc for _ in range(dr)]
    sr_, sc_ = len(src), len(src[0]); out = []
    for i in range(dr):
        si = i*sr_/dr; i0=int(si); i1=min(i0+1,sr_-1); fy=si-i0
        row = []
        for j in range(dc):
            sj = j*sc_/dc; j0=int(sj); j1=min(j0+1,sc_-1); fx=sj-j0
            a=src[i0][j0]; b=src[i0][j1]; c2=src[i1][j0]; d=src[i1][j1]
            row.append(a*(1-fx)*(1-fy)+b*fx*(1-fy)+c2*(1-fx)*fy+d*fx*fy)
        out.append(row)
    return out


def mouth_openness(frame):
    total = 0.0
    for i in range(MOUTH_R0+1, MOUTH_R1):
        for j in range(MOUTH_C0, MOUTH_C1):
            total += abs(frame[i][j] - frame[i-1][j])
    return total


def extract_anchors():
    pcm, sr, bits = read_wav_pcm("video_input/audio/speech.wav")
    samples = decode_pcm(pcm, bits)
    spf = int(sr * 0.10); n = len(samples) // spf

    aud_grids = [build_aud_sparse(samples[i*spf:(i+1)*spf]) for i in range(n)]

    vis = []
    for fi in range(n):
        p = os.path.join("video_input/frames", "frame_{:04d}.ppm".format(fi+1))
        if not os.path.exists(p): break
        raw, _, _ = read_ppm(p)
        vis.append(resize(raw, ROWS, COLS))

    n = min(len(aud_grids), len(vis))
    mouth_vals = [mouth_openness(vis[fi]) for fi in range(n)]
    aud_vals = []
    for fi in range(1, n):
        ac = [v for r in aud_grids[fi] for v in r]
        ap = [v for r in aud_grids[fi-1] for v in r]
        aud_vals.append(max(abs(a-b) for a,b in zip(ac,ap)))

    return mouth_vals, aud_vals, n


def normalize(vals):
    lo, hi = min(vals), max(vals)
    if hi == lo: return [0.5]*len(vals), lo, hi
    return [(v-lo)/(hi-lo) for v in vals], lo, hi


# ──────────────────────────────────────────────
# 二、活循环
# ──────────────────────────────────────────────

if __name__ == "__main__":
    user = sys.argv[1] if len(sys.argv) > 1 else "架构师"
    log_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 56

    t = Trace(user)

    # 读真实锚点
    mouth_vals, aud_vals, N = extract_anchors()
    mouth_norm, m_lo, m_hi = normalize(mouth_vals)
    aud_norm, a_lo, a_hi = normalize(aud_vals)

    # 系统状态——初始从真实第一帧
    m = mouth_norm[0]
    a = aud_norm[0] if aud_norm else 0.5

    print(f"[活循环 v1.1] 启动")
    print(f"锚点帧数: {N}")
    print(f"嘴开合度  min={m_lo:.1f}  max={m_hi:.1f}")
    print(f"听觉Δmax  min={a_lo:.2f}  max={a_hi:.2f}")
    print()

    # 写入第一条"诞生痕"
    birth = t.append("活循环启动", {
        "type": "living_loop_init",
        "anchor_frames": N,
        "m_range": [m_lo, m_hi],
        "a_range": [a_lo, a_hi],
        "initial_m": round(m, 4),
        "initial_a": round(a, 4),
    }, trigger="system")
    print(f"诞生: seq={birth['seq']} hash={birth['hash'][:12]}")

    # 耦合权重
    w_mm, w_ma, w_me = 0.46, 0.22, 0.32
    w_aa, w_am, w_ae = 0.46, 0.22, 0.32

    step = 0
    turns = 0
    log_file = open("living_loop.log", "w")

    # 进入循环
    while True:
        i = step % N

        ext_m = mouth_norm[i]
        ext_a = aud_norm[min(i, len(aud_norm)-1)]

        # 产出——这一转的新嘴态、听态
        m_new = w_mm * m + w_ma * a + w_me * ext_m
        a_new = w_aa * a + w_am * m + w_ae * ext_a

        # 微量相位噪声，打破完全确定性轨道
        phase_noise = 0.001 * math.sin(step * 0.37) + 0.001 * math.sin(step * 0.11)
        m_new += phase_noise
        a_new += 0.001 * math.cos(step * 0.29)

        delta_m = m_new - m
        delta_a = a_new - a

        # 每一整圈——写入哈希链
        if (step > 0 and step % N == 0):
            turns += 1
            rec = t.append("活循环产出", {
                "type": "living_turn",
                "turn": turns,
                "step": step,
                "m": round(m, 4),
                "a": round(a, 4),
                "delta_m": round(delta_m, 4),
                "delta_a": round(delta_a, 4),
                "ext_m": round(ext_m, 4),
                "ext_a": round(ext_a, 4),
                "mouth_real_anchor": mouth_vals[i],
                "aud_real_anchor": aud_vals[i] if i < len(aud_vals) else None,
                "last_hash": rec["hash"][:12] if 'rec' in dir() else "GENESIS",
            }, trigger="system")

            line = f"turn {turns}: seq={rec['seq']} m={m:.4f} a={a:.4f} dm={delta_m:+.4f} da={delta_a:+.4f} hash={rec['hash'][:12]}"
            log_file.write(line + "\n")
            log_file.flush()
            print(line, flush=True)

        m, a = m_new, a_new
        step += 1

        if step > 10000000:  # 超出物理极限前绝不会到此
            break
