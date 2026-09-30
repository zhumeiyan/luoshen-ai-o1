"""视听对齐 v0.10 —— 对不上那三组是什么。

三组：
  A：嘴动没声（17帧）——嘴在动，在干什么？
  B：有声嘴没动（8帧）——声从哪来？
  C：同静（14帧）——基线

只报事实，不判，不升。
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
MOUTH_R0 = 23; MOUTH_R1 = 29; MOUTH_C0 = 24; MOUTH_C1 = 34
AUD_BURST_THLD = 50.0
MOUTH_MOVE_THLD = 30.0


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


def mouth_openness(frame):
    total = 0.0
    for i in range(MOUTH_R0+1, MOUTH_R1):
        for j in range(MOUTH_C0, MOUTH_C1):
            total += abs(frame[i][j] - frame[i-1][j])
    return total


def aud_profile(frame_aud):
    """返回 (max_d, mean_d, changed_cells, zcr_mean, energy_mean)."""
    max_d = max(frame_aud)
    mean_d = sum(frame_aud) / len(frame_aud)
    changed = sum(1 for v in frame_aud if v > 1.0)
    zcr_mean = sum(frame_aud[:COLS]) / COLS  # row 0 = ZCR
    energy_mean = sum(frame_aud[COLS:]) / COLS  # row 1 = energy
    return max_d, mean_d, changed, zcr_mean, energy_mean


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

    # 算嘴开合 + 脉冲去除
    mouth_raw = [mouth_openness(vis[fi]) for fi in range(n)]
    mouth = list(mouth_raw)
    for i in range(1, n-1):
        nb = [mouth_raw[i-1], mouth_raw[i+1]]
        if mouth_raw[i] > sum(nb)/len(nb) * 3:
            mouth[i] = sum(nb) / len(nb)

    mouth_delta = [abs(mouth[i] - mouth[i-1]) for i in range(1, n)]

    # 听觉
    aud_dmax = []; aud_mean = []; aud_changed = []
    for fi in range(1, n):
        ac = [v for r in aud[fi] for v in r]
        ap = [v for r in aud[fi-1] for v in r]
        ds = [abs(a-b) for a,b in zip(ac,ap)]
        aud_dmax.append(max(ds))
        aud_mean.append(sum(ds)/len(ds))
        aud_changed.append(sum(1 for d in ds if d > 1.0))

    # 分组
    both_move = []   # 嘴动 + 有声
    mouth_only = []  # 嘴动没声
    aud_only = []    # 有声嘴没动
    neither = []     # 同静

    for fi in range(1, n):
        idx = fi - 1
        m_move = mouth_delta[idx] > MOUTH_MOVE_THLD
        a_burst = aud_dmax[idx] > AUD_BURST_THLD
        if m_move and a_burst:
            both_move.append(fi)
        elif m_move:
            mouth_only.append(fi)
        elif a_burst:
            aud_only.append(fi)
        else:
            neither.append(fi)

    print("=" * 120)
    print(f"总览：嘴动+有声={len(both_move)}  嘴动没声={len(mouth_only)}  有声嘴没动={len(aud_only)}  同静={len(neither)}")
    print(f"嘴动+有声: {both_move}")
    print(f"嘴动没声: {mouth_only}")
    print(f"有声嘴没动: {aud_only}")
    print(f"同静: {neither}")
    print()

    # ========================================
    # A. 嘴动没声的 17 帧 —— 什么在动
    # ========================================
    print("=" * 120)
    print(f"【A. 嘴动没声】{len(mouth_only)} 帧 —— 嘴在动，没声音")
    print("-" * 120)
    print(f"帧      时间 | 嘴Δabs | 嘴开合度(前后)       | 听觉Δmax | 听觉flat(变化格数)")
    print()

    for fi in mouth_only:
        idx = fi - 1
        m_before = mouth[fi-1]
        m_curr = mouth[fi]
        m_after = mouth[fi+1] if fi+1 < n else mouth[fi]
        delta = mouth[fi] - mouth[fi-1]
        trend = ""
        if m_curr > m_before and m_curr > m_after:
            trend = "尖峰（单帧冲高后回落）"
        elif m_curr > m_before and m_after > m_curr:
            trend = "上升中（还在开）"
        elif m_curr < m_before and m_after > m_curr:
            trend = "谷底（从开到合）"
        elif m_curr < m_before and m_after < m_curr:
            trend = "下降中（正在合）"
        elif abs(delta) < 5:
            trend = "平稳微动"
        else:
            trend = "小波动"

        print(f"f{fi:02d} {fi*0.10:4.1f}s | {mouth_delta[idx]:5.1f}  | {m_before:5.1f} → {m_curr:5.1f} → {m_after:5.1f} | {aud_dmax[idx]:8.2f} | 格变{aud_changed[idx]:2d} | {trend}")

    print()

    # 嘴动没声组汇总
    mo_deltas = [mouth_delta[fi-1] for fi in mouth_only]
    mo_dmax = [aud_dmax[fi-1] for fi in mouth_only]
    print(f"统计：嘴Δabs  mean={sum(mo_deltas)/len(mo_deltas):.1f}  max={max(mo_deltas):.1f}")
    print(f"      听觉Δmax mean={sum(mo_dmax)/len(mo_dmax):.2f}  max={max(mo_dmax):.2f}")
    print()

    # 和说话时的嘴动对比
    both_deltas = [mouth_delta[fi-1] for fi in both_move]
    print(f"对比（说话中的嘴动）：嘴Δabs  mean={sum(both_deltas)/len(both_deltas):.1f}  max={max(both_deltas):.1f}")
    print()

    # ========================================
    # B. 有声嘴没动的 8 帧 —— 声从哪来
    # ========================================
    print("=" * 120)
    print(f"【B. 有声嘴没动】{len(aud_only)} 帧 —— 有声音，嘴视觉没跟上")
    print("-" * 120)
    print(f"帧      时间 | 听觉Δmax | 听觉格变 | 嘴Δabs | 嘴开合度(前后)       | 声音长啥样")
    print()

    for fi in aud_only:
        idx = fi - 1
        a_dmax = aud_dmax[idx]
        a_changed = aud_changed[idx]
        m_d = mouth_delta[idx]
        m_before = mouth[fi-1]
        m_curr = mouth[fi]
        m_after = mouth[fi+1] if fi+1 < n else mouth[fi]

        # 看这一帧的声音是什么形状（哪些格子变了, ZCR还是energy主导）
        ac = [v for r in aud[fi] for v in r]
        ap = [v for r in aud[fi-1] for v in r]
        ds = [abs(a-b) for a,b in zip(ac,ap)]
        zcr_ds = ds[:COLS]       # ZCR row
        eng_ds = ds[COLS:]       # energy row
        zcr_max = max(zcr_ds)
        eng_max = max(eng_ds)
        zcr_changed = sum(1 for d in zcr_ds if d > 1.0)
        eng_changed = sum(1 for d in eng_ds if d > 1.0)

        if eng_max > zcr_max * 10:
            nature = f"energy主导 (engΔ={eng_max:.0f} vs ZCRΔ={zcr_max:.1f})"
        elif zcr_max > eng_max:
            nature = f"ZCR主导 (ZCRΔ={zcr_max:.1f} vs engΔ={eng_max:.0f})"
        else:
            nature = f"混合 (ZCRΔ={zcr_max:.1f} engΔ={eng_max:.0f})"

        print(f"f{fi:02d} {fi*0.10:4.1f}s | {a_dmax:8.2f} | {a_changed:3d}/{128}   | {m_d:5.1f} | {m_before:5.1f} → {m_curr:5.1f} → {m_after:5.1f} | {nature}")

    print()

    ao_dmax = [aud_dmax[fi-1] for fi in aud_only]
    ao_mdelta = [mouth_delta[fi-1] for fi in aud_only]
    print(f"统计：听觉Δmax mean={sum(ao_dmax)/len(ao_dmax):.2f}  max={max(ao_dmax):.2f}")
    print(f"      嘴Δabs    mean={sum(ao_mdelta)/len(ao_mdelta):.1f}  max={max(ao_mdelta):.1f}")
    print()

    # 对比说话时的听觉
    both_a = [aud_dmax[fi-1] for fi in both_move]
    print(f"对比（说话中的听觉）：听觉Δmax mean={sum(both_a)/len(both_a):.2f}  max={max(both_a):.2f}")
    print()

    # ========================================
    # C. 两组差在哪（摆着）
    # ========================================
    print("=" * 120)
    print(f"【C. 两组差异——摆着】")
    print("-" * 120)
    print()
    print(f"嘴动没声组 ({len(mouth_only)} 帧)：")
    print(f"  嘴Δabs mean={sum(mo_deltas)/len(mo_deltas):.1f}，范围 {min(mo_deltas):.1f}~{max(mo_deltas):.1f}")
    print(f"  帧分布：{mouth_only}")
    print(f"  特征：嘴在动但声音 flat——听觉没有 burst")
    print()
    print(f"有声嘴没动组 ({len(aud_only)} 帧)：")
    print(f"  听觉Δmax mean={sum(ao_dmax)/len(ao_dmax):.2f}，范围 {min(ao_dmax):.2f}~{max(ao_dmax):.2f}")
    print(f"  嘴Δabs mean={sum(ao_mdelta)/len(ao_mdelta):.1f}，范围 {min(ao_mdelta):.1f}~{max(ao_mdelta):.1f}")
    print(f"  帧分布：{aud_only}")
    print(f"  特征：声音 burst 但视觉嘴区变化低于阈值")
    print()
    print(f"两组差异：")
    print(f"  嘴动没声 = 视觉信号先走/独行，听觉没跟上")
    print(f"  有声嘴没动 = 听觉有 burst，视觉嘴区没观察到对应的动作")
    print(f"  可能解释：嘴动阈值偏严、嘴动了但垂直梯度不够、非嘴唇发出的声音")
    print()
    print("状态：[判读·视听对齐·不对帧分析]")
