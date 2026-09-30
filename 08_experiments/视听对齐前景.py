"""视听对齐 v0.7 — 视觉前景分离。

步骤：
  1. 视觉帧间差 Δ = |frame[i] - frame[i-1]|
  2. 背景 = Δ 的低频层（大块模糊），前景 = Δ - 背景
  3. 只拿前景去和听觉 burst 对齐
"""

import struct, os


# ============================================================
# 一、WAV 读入（同 v0.6）
# ============================================================

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
            else:
                f.read(cl)
    return pcm, sr, bits, ch


def decode_pcm(pcm, bits=16):
    n = len(pcm) // 2
    return [s / 32780.0 for s in struct.unpack('<{}h'.format(n), pcm)]


COLS = 64


def build_aud_sparse(samples_sub):
    n = len(samples_sub)
    bs = max(1, n // COLS)
    zcr_r = []; eng_r = []
    for c in range(COLS):
        s = c * bs; e = min(s + bs, n)
        blk = samples_sub[s:e]
        if not blk:
            zcr_r.append(0.0); eng_r.append(0.0); continue
        zc = sum(1 for i in range(1, len(blk)) if blk[i-1] >= 0 > blk[i])
        sq = sum(x*x for x in blk) / len(blk)
        zcr_r.append(float(zc))
        eng_r.append(sq * 10000)
    return [zcr_r, eng_r]


# ============================================================
# 二、视觉
# ============================================================

def read_ppm(path):
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        ln = f.readline().strip()
        while ln.startswith(b'#'):
            ln = f.readline().strip()
        w, h = map(int, ln.split())
        maxv = int(f.readline().strip())
        data = f.read()
    if magic == b'P6':
        px = [(data[i], data[i+1], data[i+2]) for i in range(0, len(data), 3)]
        return [[0.299*r + 0.587*g + 0.114*b for r, g, b in px[y*w:(y+1)*w]] for y in range(h)], w, h
    elif magic == b'P3':
        vs = [float(x) for x in data.split()]
        return [[0.299*vs[(y*w+x)*3] + 0.587*vs[(y*w+x)*3+1] + 0.114*vs[(y*w+x)*3+2] for x in range(w)] for y in range(h)], w, h


def resize(src, dr=48, dc=64):
    if not src or not src[0]:
        return [[0.0]*dc for _ in range(dr)]
    sr_, sc_ = len(src), len(src[0])
    out = []
    for i in range(dr):
        si = i * sr_ / dr; i0 = int(si); i1 = min(i0+1, sr_-1); fy = si-i0
        row = []
        for j in range(dc):
            sj = j * sc_ / dc; j0 = int(sj); j1 = min(j0+1, sc_-1); fx = sj-j0
            a = src[i0][j0]; b = src[i0][j1]; c_ = src[i1][j0]; d = src[i1][j1]
            row.append(a*(1-fx)*(1-fy) + b*fx*(1-fy) + c_*(1-fx)*fy + d*fx*fy)
        out.append(row)
    return out


# ============================================================
# 三、前景分离（空间高通）
# ============================================================

def separate_foreground(curr, prev, ksize=3):
    """返回 (foreground_delta, background_delta)，每个是 48×64."""
    r, c = len(curr), len(curr[0])
    k = ksize         # 3 → 7×7 邻域（±3）
    h = k // 2

    fg = [[0.0]*c for _ in range(r)]
    bg = [[0.0]*c for _ in range(r)]

    for i in range(r):
        for j in range(c):
            d = abs(curr[i][j] - prev[i][j])
            # 局部均值（背景层）
            cnt = 0; s = 0.0
            for di in range(-h, h+1):
                for dj in range(-h, h+1):
                    ni, nj = i+di, j+dj
                    if 0 <= ni < r and 0 <= nj < c:
                        s += abs(curr[ni][nj] - prev[ni][nj])
                        cnt += 1
            local_mean = s / cnt if cnt else 0.0
            bg[i][j] = local_mean
            fg[i][j] = max(0.0, d - local_mean)
    return fg, bg


# ============================================================
# 四、Pearson
# ============================================================

def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float('nan')
    mx = sum(xs)/n; my = sum(ys)/n
    dx = [x-mx for x in xs]; dy = [y-my for y in ys]
    num = sum(a*b for a,b in zip(dx,dy))
    dx2 = sum(a*a for a in dx)**0.5
    dy2 = sum(b*b for b in dy)**0.5
    if dx2 < 1e-10 or dy2 < 1e-10: return float('nan')
    return num / (dx2*dy2)


# ============================================================
# 五、主
# ============================================================

if __name__ == "__main__":
    # 读音频
    pcm, sr, bits, ch = read_wav_pcm("video_input/audio/speech.wav")
    samples = decode_pcm(pcm, bits)
    spf = int(sr * 0.10)
    n_frames = len(samples) // spf

    aud = []
    for fi in range(n_frames):
        aud.append(build_aud_sparse(samples[fi*spf : (fi+1)*spf]))

    # 读视觉
    vis = []
    for fi in range(n_frames):
        p = os.path.join("video_input/frames", "frame_{:04d}.ppm".format(fi+1))
        if not os.path.exists(p): break
        raw, _, _ = read_ppm(p)
        vis.append(resize(raw, 48, 64))

    n = min(len(aud), len(vis))
    print(f"共同帧数: {n}")

    # 计算每帧的前景色变化量 + 听觉变化
    print("=" * 130)
    print("阈值：听觉格子变化 > 1.0 | 视觉前景格子变化 > 10.0")
    print("=" * 130)
    print(f"帧      时间 | 前景_变化格  占比   Δ均 | 听觉_变化格 占比   Δ均 | 听Δmax")
    print("-" * 130)

    fg_fracs = []; aud_fracs = []
    fg_means = []; aud_means = []
    both_move = 0

    FG_THLD = 10.0    # 视觉前景格子变化阈值
    AUD_THLD = 1.0    # 听觉格子变化阈值
    FRACT_THLD = 0.05

    for fi in range(1, n):
        fg, bg = separate_foreground(vis[fi], vis[fi-1])

        # 前景格子：fg[i][j] > FG_THLD
        fg_flat = [v for r in fg for v in r]
        fg_changed = sum(1 for v in fg_flat if v > FG_THLD)
        fg_total = len(fg_flat)
        fg_frac = fg_changed / fg_total
        fg_delta_sum = sum(v for v in fg_flat if v > FG_THLD)
        fg_mean_d = fg_delta_sum / fg_changed if fg_changed else 0.0

        # 听觉
        aud_curr = [v for r in aud[fi] for v in r]
        aud_prev = [v for r in aud[fi-1] for v in r]
        aud_changed = sum(1 for a, b in zip(aud_curr, aud_prev) if abs(a-b) > AUD_THLD)
        aud_delta = [abs(a-b) for a, b in zip(aud_curr, aud_prev) if abs(a-b) > AUD_THLD]
        aud_frac = aud_changed / len(aud_curr)
        aud_mean_d = sum(aud_delta) / len(aud_delta) if aud_delta else 0.0
        aud_dmx = max(abs(a-b) for a, b in zip(aud_curr, aud_prev))

        fg_fracs.append(fg_frac); fg_means.append(fg_mean_d)
        aud_fracs.append(aud_frac); aud_means.append(aud_mean_d)

        v_moved = fg_frac > FRACT_THLD
        a_moved = aud_frac > FRACT_THLD

        if v_moved and a_moved:
            both_move += 1; tag = "同动"
        elif v_moved:
            tag = "视动"
        elif a_moved:
            tag = "听动"
        else:
            tag = "同静"

        print(f"f{fi:02d} {fi*0.10:4.1f}s | {fg_changed:4d}/{fg_total}   {fg_frac:5.2f}  {fg_mean_d:7.2f} | {aud_changed:3d}/{len(aud_curr)}   {aud_frac:5.2f}  {aud_mean_d:7.2f} | {aud_dmx:8.2f} | {tag}")

    print()
    r_f = pearson(fg_fracs, aud_fracs)
    r_m = pearson(fg_means, aud_means)
    print(f"帧对数: {len(fg_fracs)}")
    print(f"两路同动帧: {both_move} ({both_move/len(fg_fracs)*100:.1f}%)")
    print(f"Pearson r(前景变化格占比 vs 听觉变化格占比): {r_f:+.4f}")
    print(f"Pearson r(前景Δ均值 vs 听觉Δ均值):           {r_m:+.4f}")
    print()

    # Top-10 前景帧 vs Top-10 听觉帧
    fg_by_frames = []
    aud_by_frames = []
    for fi in range(1, n):
        fg, _ = separate_foreground(vis[fi], vis[fi-1])
        fg_ch = sum(1 for r in fg for v in r if v > FG_THLD)
        fg_by_frames.append((fi, fg_ch))

        aud_curr = [v for r in aud[fi] for v in r]
        aud_prev = [v for r in aud[fi-1] for v in r]
        aud_ch = sum(1 for a, b in zip(aud_curr, aud_prev) if abs(a-b) > AUD_THLD)
        aud_by_frames.append((fi, aud_ch))

    fg_by_frames.sort(key=lambda x: -x[1])
    aud_by_frames.sort(key=lambda x: -x[1])

    print("前景变化格 Top-10:")
    for fi, cnt in fg_by_frames[:10]:
        print(f"  f{fi:02d} ({fi*0.10:.1f}s): {cnt}")
    print("听觉变化格 Top-10:")
    for fi, cnt in aud_by_frames[:10]:
        print(f"  f{fi:02d} ({fi*0.10:.1f}s): {cnt}")
    print()

    # 交集
    v_top = set(fi for fi, c in fg_by_frames if c > 50)
    a_top = set(fi for fi, c in aud_by_frames if c > 20)
    overlap = sorted(v_top & a_top)
    v_only = sorted(v_top - a_top)
    a_only = sorted(a_top - v_top)
    print(f"两路都有显著变化的帧 (前景>50格 且 听觉>20格):")
    print(f"  交集: {len(overlap)} 帧 {overlap}")
    print(f"  仅前景: {len(v_only)} 帧 {v_only[:15]}{'...' if len(v_only)>15 else ''}")
    print(f"  仅听觉: {len(a_only)} 帧 {a_only[:15]}{'...' if len(a_only)>15 else ''}")
    print()

    # ===== 新指标：前景"突发度" vs 听觉"突发度" =====
    # 突发度 = 前景变化格中的 Δ 总和（不是均值，是总量，反映"这次突然动有多大"）
    fg_burst = []
    aud_burst = []
    for fi in range(1, n):
        fg, _ = separate_foreground(vis[fi], vis[fi-1])
        fg_sum = sum(v for r in fg for v in r if v > FG_THLD)
        fg_burst.append(fg_sum)

        aud_curr = [v for r in aud[fi] for v in r]
        aud_prev = [v for r in aud[fi-1] for v in r]
        aud_sum = sum(abs(a-b) for a, b in zip(aud_curr, aud_prev) if abs(a-b) > AUD_THLD)
        aud_burst.append(aud_sum)

    r_burst = pearson(fg_burst, aud_burst)
    fg_burst_sorted = sorted(range(len(fg_burst)), key=lambda i: -fg_burst[i])
    aud_burst_sorted = sorted(range(len(aud_burst)), key=lambda i: -aud_burst[i])

    print(f"突发度相关: r(前景Δ总和 vs 听觉Δ总和) = {r_burst:+.4f}")
    print(f"前景突发度 Top-5: {[(i+1, round(fg_burst[i])) for i in fg_burst_sorted[:5]]}")
    print(f"听觉突发度 Top-5: {[(i+1, round(aud_burst[i])) for i in aud_burst_sorted[:5]]}")
    print()
    print("状态：[判读·视听对齐·前景分离]")
