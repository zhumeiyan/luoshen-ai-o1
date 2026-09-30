"""视听对齐 v0.9 —— 从脸区里挑出嘴的开合。

步骤：
  1. 脸区固定（质心 row±10, col±12），只取下半（row+2 ~ row+10）= 嘴区
  2. 嘴开合 = 嘴区内垂直方向灰度梯度（行间差分）的帧间变化
  3. 去掉脉冲（眨眼=单帧尖、持续<2帧的）
  4. 嘴开合 vs 听觉 burst 对齐
"""

import struct, os


def read_wav_pcm(path):
    with open(path, 'rb') as f:
        f.read(4); f.read(4); f.read(4)
        pcm = None; sr = None; bits = None; ch = None
        while True:
            ck = f.read(4)
            if len(ck) < 4: break
            cl = struct.unpack('<I', f.read(4))[0]
            if ck == b'fmt ':
                fmt = f.read(cl)
                ch = struct.unpack('<H', fmt[2:4])[0]
                sr = struct.unpack('<I', fmt[4:8])[0]
                bits = struct.unpack('<H', fmt[14:16])[0]
            elif ck == b'data':
                pcm = f.read(cl); break
            else: f.read(cl)
    return pcm, sr, bits, ch


def decode_pcm(pcm, bits=16):
    n = len(pcm) // 2
    return [s / 32780.0 for s in struct.unpack('<{}h'.format(n), pcm)]


COLS = 64; ROWS = 48


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
        maxv = int(f.readline().strip()); data = f.read()
    if magic == b'P6':
        px = [(data[i],data[i+1],data[i+2]) for i in range(0,len(data),3)]
        return [[0.299*r+0.587*g+0.114*b for r,g,b in px[y*w:(y+1)*w]] for y in range(h)],w,h
    elif magic == b'P3':
        vs = [float(x) for x in data.split()]
        return [[0.299*vs[(y*w+x)*3]+0.587*vs[(y*w+x)*3+1]+0.114*vs[(y*w+x)*3+2] for x in range(w)] for y in range(h)],w,h


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


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float('nan')
    mx = sum(xs)/n; my = sum(ys)/n
    dx = [x-mx for x in xs]; dy = [y-my for y in ys]
    num = sum(a*b for a,b in zip(dx,dy))
    dx2 = sum(a*a for a in dx)**0.5; dy2 = sum(b*b for b in dy)**0.5
    if dx2<1e-10 or dy2<1e-10: return float('nan')
    return num/(dx2*dy2)


# ============================================================
# 嘴区检测：脸质心（已知约 row 21.5, col 29.6）
# ============================================================

FACE_CX = 21.5   # 脸质心行（从 v0.8 测得）
FACE_CY = 29.6   # 脸质心列
FACE_HALF_R = 8  # 脸区行 ±8
FACE_HALF_C = 10 # 脸区列 ±10

# 嘴区 = 脸区下半部分（嘴唇所在）
MOUTH_R0 = int(FACE_CX) + 1   # row 23
MOUTH_R1 = int(FACE_CX) + 8   # row 29  （嘴唇到下颚）
MOUTH_C0 = int(FACE_CY) - 5   # col 24
MOUTH_C1 = int(FACE_CY) + 5   # col 35


def mouth_openness(frame):
    """嘴的开合度 = 嘴区内垂直灰度梯度（行间差分）的总和。
    嘴唇开合 → 上下唇分离 → 行间灰度跳变增大。
    """
    total = 0.0
    for i in range(MOUTH_R0+1, MOUTH_R1):
        for j in range(MOUTH_C0, MOUTH_C1):
            total += abs(frame[i][j] - frame[i-1][j])
    return total


def remove_pulse(signal, win=2):
    """去掉单帧脉冲：当前帧的值如果比前后帧都高很多，视为脉冲，
    用前后帧均值代替。"""
    out = list(signal)
    for i in range(win, len(signal)-win):
        neighbors = signal[i-win:i] + signal[i+1:i+win+1]
        nb_mean = sum(neighbors) / len(neighbors)
        if signal[i] > nb_mean * 3:   # 尖峰 > 邻居均值的 3 倍
            out[i] = nb_mean
    return out


if __name__ == "__main__":
    pcm, sr, bits, ch = read_wav_pcm("video_input/audio/speech.wav")
    samples = decode_pcm(pcm, bits)
    spf = int(sr * 0.10); n_frames = len(samples) // spf

    aud = [build_aud_sparse(samples[i*spf:(i+1)*spf]) for i in range(n_frames)]

    vis = []
    for fi in range(n_frames):
        p = os.path.join("video_input/frames", "frame_{:04d}.ppm".format(fi+1))
        if not os.path.exists(p): break
        raw, _, _ = read_ppm(p)
        vis.append(resize(raw, ROWS, COLS))

    n = min(len(aud), len(vis))
    print(f"共同帧数: {n}")
    print(f"嘴区：行 {MOUTH_R0}~{MOUTH_R1}，列 {MOUTH_C0}~{MOUTH_C1}")
    print()

    # 每帧嘴开合度
    mouth_raw = [mouth_openness(vis[fi]) for fi in range(n)]
    # 去掉脉冲（眨眼等）
    mouth_clean = remove_pulse(mouth_raw, win=1)

    # 嘴开合变化率（帧间差分）
    mouth_delta = [mouth_clean[i] - mouth_clean[i-1] for i in range(1, n)]
    mouth_delta_abs = [abs(d) for d in mouth_delta]

    # 听觉 Δmax
    aud_dmax = []
    for fi in range(1, n):
        ac = [v for r in aud[fi] for v in r]
        ap = [v for r in aud[fi-1] for v in r]
        aud_dmax.append(max(abs(a-b) for a,b in zip(ac,ap)))

    # 对齐
    r = pearson(mouth_delta_abs, aud_dmax)

    print("=" * 130)
    print(f"嘴区 (行 {MOUTH_R0}~{MOUTH_R1}, 列 {MOUTH_C0}~{MOUTH_C1}) 的开合度 —— 帧间 vs 听觉 Δmax")
    print("=" * 130)
    print(f"帧      时间 | 嘴开合(脉冲去后) | 嘴Δabs | 听觉Δmax | 听觉burst?")
    print("-" * 130)

    AUD_BURST_THLD = 50.0   # 听觉 burst 阈值
    MOUTH_MOVE_THLD = 30.0  # 嘴动阈值

    both = 0; mouth_only = 0; aud_only = 0; neither = 0

    for fi in range(1, n):
        idx = fi - 1
        m_d = mouth_delta_abs[idx]
        a_d = aud_dmax[idx]
        m_move = m_d > MOUTH_MOVE_THLD
        a_burst = a_d > AUD_BURST_THLD

        if m_move and a_burst: both += 1; tag = "同动"
        elif m_move: mouth_only += 1; tag = "嘴动"
        elif a_burst: aud_only += 1; tag = "听动"
        else: neither += 1; tag = "---"

        mbar = "#" * min(30, int(m_d / 20))
        abar = "*" * min(30, int(a_d / 50))

        print(f"f{fi:02d} {fi*0.10:4.1f}s | {mouth_clean[fi]:12.1f} | {m_d:6.1f} | {a_d:8.2f} | {tag}")
        if m_d > MOUTH_MOVE_THLD or a_d > AUD_BURST_THLD:
            print(f"                    {mbar}")
            print(f"                    {abar}")

    print()
    print("=" * 130)
    print(f"总计 {n-1} 帧: 同动={both}  仅嘴动={mouth_only}  仅听动={aud_only}  同静={neither}")
    print(f"Pearson r(嘴Δabs vs 听觉Δmax) = {r:+.4f}")
    print()

    # Top-5 嘴动帧
    m_sorted = sorted(range(len(mouth_delta_abs)), key=lambda i: -mouth_delta_abs[i])
    a_sorted = sorted(range(len(aud_dmax)), key=lambda i: -aud_dmax[i])

    print("嘴变化 Top-5:")
    for idx in m_sorted[:5]:
        print(f"  f{idx+1:02d} ({(idx+1)*0.10:.1f}s): Δabs={mouth_delta_abs[idx]:.1f}")
    print("听觉 burst Top-5:")
    for idx in a_sorted[:5]:
        print(f"  f{idx+1:02d} ({(idx+1)*0.10:.1f}s): Δmax={aud_dmax[idx]:.2f}")

    # 交集
    m_top = set(i+1 for i in m_sorted[:10])
    a_top = set(i+1 for i in a_sorted[:10])
    print(f"嘴 Top-10 ∩ 听 Top-10: {sorted(m_top & a_top)}")
    print()

    # burst 区间对齐
    burst_intervals = [(10,17), (25,32), (40,49)]  # 听觉 burst 区间
    print("嘴动在 burst 区间内 (f10-f17, f25-f32, f40-f49):")
    for lo, hi in burst_intervals:
        m_in = [i+1 for i in range(lo-1, min(hi-1, len(mouth_delta_abs))) if mouth_delta_abs[i] > MOUTH_MOVE_THLD]
        print(f"  f{lo:02d}-f{hi:02d}: 嘴动帧={m_in if m_in else '无'}")
    print()
    print("状态：[判读·视听对齐·嘴区]")
