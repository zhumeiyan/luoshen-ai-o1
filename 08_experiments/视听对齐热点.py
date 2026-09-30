"""视听对齐 v0.8 —— 视觉：找变化最集中的区域，不看整块。

每帧：
  1. 算 Δ = |frame[i] - frame[i-1]|（48×64）
  2. 找 Δ 的"质心"（变化最集中的那个点）
  3. 取质心周围一块小区域（嘴那么大）
  4. 那块区域的变化量 = 这帧的视觉活度
  
对齐：视觉活度 vs 听觉活度 burst
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


COLS = 64
ROWS = 48


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


def resize(src, dr=ROWS, dc=COLS):
    if not src or not src[0]:
        return [[0.0]*dc for _ in range(dr)]
    sr_, sc_ = len(src), len(src[0])
    out = []
    for i in range(dr):
        si = i * sr_ / dr; i0 = int(si); i1 = min(i0+1, sr_-1); fy = si - i0
        row = []
        for j in range(dc):
            sj = j * sc_ / dc; j0 = int(sj); j1 = min(j0+1, sc_-1); fx = sj - j0
            a = src[i0][j0]; b = src[i0][j1]; c_ = src[i1][j0]; d = src[i1][j1]
            row.append(a*(1-fx)*(1-fy) + b*fx*(1-fy) + c_*(1-fx)*fy + d*fx*fy)
        out.append(row)
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float('nan')
    mx = sum(xs)/n; my = sum(ys)/n
    dx = [x-mx for x in xs]; dy = [y-my for y in ys]
    num = sum(a*b for a,b in zip(dx,dy))
    dx2 = sum(a*a for a in dx)**0.5; dy2 = sum(b*b for b in dy)**0.5
    if dx2 < 1e-10 or dy2 < 1e-10: return float('nan')
    return num / (dx2*dy2)


if __name__ == "__main__":
    pcm, sr, bits, ch = read_wav_pcm("video_input/audio/speech.wav")
    samples = decode_pcm(pcm, bits)
    spf = int(sr * 0.10)
    n_frames = len(samples) // spf

    aud = []
    for fi in range(n_frames):
        aud.append(build_aud_sparse(samples[fi*spf : (fi+1)*spf]))

    vis = []
    for fi in range(n_frames):
        p = os.path.join("video_input/frames", "frame_{:04d}.ppm".format(fi+1))
        if not os.path.exists(p): break
        raw, _, _ = read_ppm(p)
        vis.append(resize(raw, ROWS, COLS))

    n = min(len(aud), len(vis))
    print(f"共同帧数: {n}")
    print()

    # 每帧算 Δ 的质心和活度
    HOTSPOT_R = 5  # 质心上下±5行
    HOTSPOT_C = 8  # 质心左右±8列

    cx_list = []; cy_list = []
    v_activity = []
    a_activity = []

    print("=" * 130)
    print(f"帧      时间 | 质心_行 质心_列 | 区域活度 | 听觉活度(max) | 听Δmax")
    print("-" * 130)

    for fi in range(1, n):
        # Δ 矩阵
        delta_matrix = []
        total_delta = 0.0
        for i in range(ROWS):
            row = []
            for j in range(COLS):
                d = abs(vis[fi][i][j] - vis[fi-1][i][j])
                row.append(d)
                total_delta += d
            delta_matrix.append(row)

        if total_delta < 1e-6:
            cx_list.append(None); cy_list.append(None)
            v_activity.append(0.0)
            aud_curr = [v for r in aud[fi] for v in r]
            aud_prev = [v for r in aud[fi-1] for v in r]
            a_activity.append(max(abs(a-b) for a,b in zip(aud_curr, aud_prev)))
            print(f"f{fi:02d} {fi*0.10:4.1f}s |      全静       |    0    |               |")
            continue

        # 质心
        sum_ix = 0.0; sum_jx = 0.0; sum_w = 0.0
        for i in range(ROWS):
            for j in range(COLS):
                sum_ix += i * delta_matrix[i][j]
                sum_jx += j * delta_matrix[i][j]
                sum_w += delta_matrix[i][j]
        cx = sum_ix / sum_w
        cy = sum_jx / sum_w
        cx_list.append(cx); cy_list.append(cy)

        # 区域活度：质心周围 ±HOTSPOT
        r0 = max(0, int(cx) - HOTSPOT_R)
        r1 = min(ROWS, int(cx) + HOTSPOT_R + 1)
        c0 = max(0, int(cy) - HOTSPOT_C)
        c1 = min(COLS, int(cy) + HOTSPOT_C + 1)
        region_sum = sum(delta_matrix[i][j] for i in range(r0, r1) for j in range(c0, c1))
        v_activity.append(region_sum)

        # 听觉活度
        aud_curr = [v for r in aud[fi] for v in r]
        aud_prev = [v for r in aud[fi-1] for v in r]
        a_dmax = max(abs(a-b) for a,b in zip(aud_curr, aud_prev))
        a_activity.append(a_dmax)

        print(f"f{fi:02d} {fi*0.10:4.1f}s |  {cx:5.1f}  {cy:5.1f}  | {region_sum:8.1f} | {a_dmax:8.2f}      |       |")

    print()
    print("=" * 130)
    print(f"质心稳定性：")
    valid_cx = [c for c in cx_list if c is not None]
    valid_cy = [c for i, c in enumerate(cy_list) if cx_list[i] is not None]
    if valid_cx:
        print(f"  行均值={sum(valid_cx)/len(valid_cx):.1f} (std={((sum((c-sum(valid_cx)/len(valid_cx))**2 for c in valid_cx)/len(valid_cx))**0.5):.1f})")
        print(f"  列均值={sum(valid_cy)/len(valid_cy):.1f} (std={((sum((c-sum(valid_cy)/len(valid_cy))**2 for c in valid_cy)/len(valid_cy))**0.5):.1f})")
        print(f"  行范围 {int(min(valid_cx))}~{int(max(valid_cx))}，列范围 {int(min(valid_cy))}~{int(max(valid_cy))}")

        # 质心聚集度（帧间方差）
        cx_var = sum((c-sum(valid_cx)/len(valid_cx))**2 for c in valid_cx)/len(valid_cx)
        cy_var = sum((c-sum(valid_cy)/len(valid_cy))**2 for c in valid_cy)/len(valid_cy)
        print(f"  帧间行方差={cx_var:.2f}，列方差={cy_var:.2f}")
        print(f"  48×64 画面的中心是行 24、列 32")
        print(f"  质心如果在 行 20~30 / 列 28~36 附近 → 画面中央（脸所在）")
        print(f"  质心如果散布在各处 → 变化没有固定位置")
    print()

    # 对齐
    r = pearson(v_activity, a_activity)
    print(f"Pearson r(视觉质心区域活度 vs 听觉Δmax) = {r:+.4f}")
    print()

    # Top-5 视觉活度帧
    v_sorted = sorted(range(len(v_activity)), key=lambda i: -v_activity[i])
    a_sorted = sorted(range(len(a_activity)), key=lambda i: -a_activity[i])

    print("视觉 Top-5 质心区域活度帧:")
    for idx in v_sorted[:5]:
        fi = idx + 1
        cx = cx_list[idx]; cy = cy_list[idx]
        cx_s = f"{cx:.1f}" if cx is not None else "---"
        cy_s = f"{cy:.1f}" if cy is not None else "---"
        print(f"  f{fi:02d} ({fi*0.10:.1f}s): 活度={v_activity[idx]:.1f} 质心=({cx_s},{cy_s})")

    print("听觉 Top-5 活度帧:")
    for idx in a_sorted[:5]:
        fi = idx + 1
        print(f"  f{fi:02d} ({fi*0.10:.1f}s): Δmax={a_activity[idx]:.2f}")

    # 交集
    v_top = set(idx+1 for idx in v_sorted[:10])
    a_top = set(idx+1 for idx in a_sorted[:10])
    overlap = sorted(v_top & a_top)
    print(f"\n视觉 Top-10 ∩ 听觉 Top-10: {overlap}")
    print()
    print("状态：[判读·视听对齐·视觉质心]")
