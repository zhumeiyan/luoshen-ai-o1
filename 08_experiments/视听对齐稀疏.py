"""视听对齐 v0.6 — 稀疏表征 + 变化格子占比。

改：
  去冗余：8×64 → 2×64（ZCR 一行 + 能量一行，row5-7 全 0 删掉）
  delta：Σ|Δ| → "变化格子占比" + "变化格子的均值"
        不动的那一大片不算

标 [判读·视觉-听觉对齐·稀疏版]
"""

import struct


# ============================================================
# 一、读 WAV
# ============================================================

def read_wav_pcm(path):
    with open(path, 'rb') as f:
        riiff = f.read(4)
        f.read(4)
        wave = f.read(4)
        pcm_data = None
        sr = None
        bits = None
        ch = None
        while True:
            chunk = f.read(4)
            if len(chunk) < 4:
                break
            clen = struct.unpack('<I', f.read(4))[0]
            if chunk == b'fmt ':
                fmt = f.read(clen)
                ch = struct.unpack('<H', fmt[2:4])[0]
                sr = struct.unpack('<I', fmt[4:8])[0]
                bits = struct.unpack('<H', fmt[14:16])[0]
            elif chunk == b'data':
                pcm_data = f.read(clen)
                break
            else:
                f.read(clen)
    return pcm_data, sr, bits, ch


def decode_pcm(pcm, bits=16):
    n = len(pcm) // 2
    raw = struct.unpack('<{}h'.format(n), pcm)
    return [s / 32780.0 for s in raw]


# ============================================================
# 二、音频 2×64 稀疏矩阵
# ============================================================

COLS = 64
ROWS_OUT = 2          # row0 = ZCR, row1 = energy


def build_aud_sparse(samples_sub):
    """1600 samples → 2×64.

    row 0 = ZCR density (per block, × block_len)
    row 1 = energy    (sq_mean × 10000)
    """
    n = len(samples_sub)
    block_size = max(1, n // COLS)
    row_zcr = []
    row_energy = []
    for c in range(COLS):
        s = c * block_size
        e = min(s + block_size, n)
        block = samples_sub[s:e]
        if not block:
            row_zcr.append(0.0)
            row_energy.append(0.0)
            continue
        zc = sum(1 for i in range(1, len(block)) if block[i-1] >= 0 > block[i])
        sq = sum(x * x for x in block) / len(block)
        row_zcr.append(zc)
        row_energy.append(sq * 10000)
    return [row_zcr, row_energy]


# ============================================================
# 三、视觉 PPM 解码
# ============================================================

FRAMES_DIR = "video_input/frames"


def read_ppm(path):
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        line2 = f.readline().strip()
        while line2.startswith(b'#'):
            line2 = f.readline().strip()
        dims = line2.split()
        w, h = int(dims[0]), int(dims[1])
        maxval = int(f.readline().strip())
        data = f.read()
    if magic == b'P6':
        pixels = []
        for i in range(0, len(data), 3):
            pixels.append((data[i], data[i+1], data[i+2]))
        rows = []
        for y in range(h):
            row = []
            for x in range(w):
                r, g, b = pixels[y * w + x]
                row.append(0.299 * r + 0.587 * g + 0.114 * b)
            rows.append(row)
        return rows, w, h
    elif magic == b'P3':
        values = data.split()
        pixels = [float(values[i]) for i in range(len(values))]
        rows = []
        for y in range(h):
            row = []
            for x in range(w):
                idx = (y * w + x) * 3
                r, g, b = pixels[idx], pixels[idx+1], pixels[idx+2]
                row.append(0.299 * r + 0.587 * g + 0.114 * b)
            rows.append(row)
        return rows, w, h


def bilinear_resize(src, dst_rows=48, dst_cols=64):
    if not src or not src[0]:
        return [[0.0] * dst_cols for _ in range(dst_rows)]
    src_r = len(src)
    src_c = len(src[0])
    out = []
    for i in range(dst_rows):
        si = i * src_r / dst_rows
        i0 = int(si)
        i1 = min(i0 + 1, src_r - 1)
        fy = si - i0
        row = []
        for j in range(dst_cols):
            sj = j * src_c / dst_cols
            j0 = int(sj)
            j1 = min(j0 + 1, src_c - 1)
            fx = sj - j0
            v00 = src[i0][j0] if i0 < src_r and j0 < src_c else 0
            v01 = src[i0][j1] if i0 < src_r and j1 < src_c else 0
            v10 = src[i1][j0] if i1 < src_r and j0 < src_c else 0
            v11 = src[i1][j1] if i1 < src_r and j1 < src_c else 0
            val = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy + v11 * fx * fy
            row.append(val)
        out.append(row)
    return out


# ============================================================
# 四、稀疏 delta
# ============================================================

DELTA_THLD = 1.0    # 单个格子变化低于此值=没变


def sparse_delta(curr, prev):
    """Return (changed_cells, total_cells, sum_delta_of_changed, mean_delta_of_changed)."""
    flat_c = [v for r in curr for v in r]
    flat_p = [v for r in prev for v in r]
    total = len(flat_c)
    changed = 0
    sum_delta = 0.0
    for a, b in zip(flat_c, flat_p):
        d = abs(a - b)
        if d > DELTA_THLD:
            changed += 1
            sum_delta += d
    frac = changed / total if total else 0.0
    mean_d = sum_delta / changed if changed else 0.0
    return changed, total, frac, mean_d


# ============================================================
# 五、对齐
# ============================================================

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    num = sum(a * b for a, b in zip(dx, dy))
    den_x = sum(a * a for a in dx) ** 0.5
    den_y = sum(b * b for b in dy) ** 0.5
    if den_x < 1e-10 or den_y < 1e-10:
        return float('nan')
    return num / (den_x * den_y)


# ============================================================
# 六、主：读数据 → 算 delta → 对齐 → 报
# ============================================================

if __name__ == "__main__":
    # --- 读音频 ---
    pcm, sr, bits, ch = read_wav_pcm("video_input/audio/speech.wav")
    samples = decode_pcm(pcm, bits)
    spf = int(sr * 0.10)
    n_frames = len(samples) // spf

    aud_grids = []
    for fi in range(n_frames):
        sub = samples[fi * spf : (fi + 1) * spf]
        aud_grids.append(build_aud_sparse(sub))

    # --- 读视觉 ---
    import os
    vis_grids = []
    for fi in range(n_frames):
        path = os.path.join(FRAMES_DIR, "frame_{:04d}.ppm".format(fi + 1))
        if not os.path.exists(path):
            break
        raw, w, h = read_ppm(path)
        small = bilinear_resize(raw, 48, 64)
        vis_grids.append(small)

    n = min(len(aud_grids), len(vis_grids))
    print(f"共同帧数: {n}  (aud={len(aud_grids)}, vis={len(vis_grids)})")
    print()

    # --- 计算稀疏 delta ---
    print(f"阈值(单格子变化量 > {DELTA_THLD} 才算动)")
    print("=" * 130)
    print(f"帧      时间 | 视觉_变化格 占比  Δ均  | 听觉_变化格 占比  Δ均  | 两路同动？")
    print("-" * 130)

    v_frac = []
    v_mean = []
    a_frac = []
    a_mean = []
    both_move = 0

    FRACT_THLD = 0.05  # 变化格占比 > 5% 才算这一帧在动

    for fi in range(1, n):
        vc, vt, vf, vm = sparse_delta(vis_grids[fi], vis_grids[fi - 1])
        ac, at, af, am = sparse_delta(aud_grids[fi], aud_grids[fi - 1])
        v_frac.append(vf)
        v_mean.append(vm)
        a_frac.append(af)
        a_mean.append(am)
        v_moved = vf > FRACT_THLD
        a_moved = af > FRACT_THLD
        if v_moved and a_moved:
            both_move += 1
            tag = "同动"
        elif v_moved:
            tag = "视动"
        elif a_moved:
            tag = "听动"
        else:
            tag = "同静"
        print(f"f{fi:02d} {fi*0.10:4.1f}s | {vc:4d}/{vt}  {vf:5.2f}  {vm:7.2f} | {ac:3d}/{at}  {af:5.2f}  {am:7.2f} | {tag}")

    print()
    print("=" * 130)
    print(f"帧对数: {len(v_frac)}")
    print(f"两路同动帧: {both_move} ({both_move/len(v_frac)*100:.1f}%)")
    print()

    r_frac = pearson(v_frac, a_frac)
    r_mean = pearson(v_mean, a_mean)
    print(f"Pearson r(变化格占比): {r_frac:+.4f}")
    print(f"Pearson r(变化格Δ均值): {r_mean:+.4f}")
    print()

    # --- 逐帧明细：看变化格子的位置是否一致 ---
    print("=" * 130)
    print(f"【格子级 — 视觉 row×64 变化最大的帧、听觉变化最大的帧】")
    print(f"视觉变化格定义为 |delta|>50 (亮度)，听觉变化格定义为 |delta|>1.0 (ZCR/energy)")
    print("-" * 130)

    VIS_CELL_THLD = 50.0
    AUD_CELL_THLD = 1.0

    vis_moved_frames = []
    aud_moved_frames = []
    for fi in range(1, n):
        # 视觉
        flat_c = [v for r in vis_grids[fi] for v in r]
        flat_p = [v for r in vis_grids[fi-1] for v in r]
        vc = sum(1 for a, b in zip(flat_c, flat_p) if abs(a - b) > VIS_CELL_THLD)
        # 听觉
        ac_flat = [v for r in aud_grids[fi] for v in r]
        ap_flat = [v for r in aud_grids[fi-1] for v in r]
        ac = sum(1 for a, b in zip(ac_flat, ap_flat) if abs(a - b) > AUD_CELL_THLD)
        vis_moved_frames.append((fi, vc))
        aud_moved_frames.append((fi, ac))

    vis_moved_frames.sort(key=lambda x: -x[1])
    aud_moved_frames.sort(key=lambda x: -x[1])

    print("视觉变化格 Top-10:")
    for fi, cnt in vis_moved_frames[:10]:
        print(f"  f{fi:02d} ({fi*0.10:.1f}s): {cnt:4d} 格变化 (>50 亮度)")

    print("听觉变化格 Top-10:")
    for fi, cnt in aud_moved_frames[:10]:
        print(f"  f{fi:02d} ({fi*0.10:.1f}s): {cnt:3d} 格变化 (>1.0)")

    print()

    # --- burst 对齐：两路同时有变化的帧 ---
    print("=" * 130)
    print(f"【两路都有显著变化的帧（视觉>50格 且 听觉>10格）】")
    v_top = set(fi for fi, cnt in vis_moved_frames if cnt > 50)
    a_top = set(fi for fi, cnt in aud_moved_frames if cnt > 10)
    overlap = sorted(v_top & a_top)
    v_only = sorted(v_top - a_top)
    a_only = sorted(a_top - v_top)

    print(f"交集: {len(overlap)} 帧 {overlap}")
    print(f"仅视觉: {len(v_only)} 帧 {v_only[:20]}{'...' if len(v_only)>20 else ''}")
    print(f"仅听觉: {len(a_only)} 帧 {a_only[:20]}{'...' if len(a_only)>20 else ''}")

    print()

    # --- 最终变化格占比 vs 听觉变化格占比 对比表 ---
    print("=" * 130)
    print(f"【变化格占比相关 r(全部帧 f1~f{n-1})】")
    print(f"  视觉变化格占比 vs 听觉变化格占比: r = {r_frac:+.4f}")
    print(f"  视觉变化格Δ均值 vs 听觉变化格Δ均值: r = {r_mean:+.4f}")
    print(f"  同动帧占比: {both_move}/{len(v_frac)} = {both_move/len(v_frac)*100:.1f}%")
    print()
    print(f"状态：[判读·视听对齐·稀疏版]")

