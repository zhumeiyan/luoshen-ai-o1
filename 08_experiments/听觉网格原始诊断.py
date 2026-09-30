"""听觉矩阵原始分布诊断 v0.1 — f2~f57，每帧格子值的分布。

不是 mean/max 这种概括。是原始格子逐个看：
  每个格子填了什么值
  哪些格子是"满"的（接近物理最大值）
  哪些格子是"空"的
  帧对比帧，同一位置的格子变没变

标 [判读·听觉矩阵诊断]
"""

import struct, sys, os
import math

# ── 配置 ──
WAV_FILE = "video_input/audio/speech.wav"
ROWS = 8
COLS = 64


# ============================================================
# 一、读 WAV
# ============================================================

def read_wav_pcm(path):
    with open(path, 'rb') as f:
        riiff = f.read(4)
        if riiff != b'RIFF':
            raise ValueError(f"not RIFF: {riiff}")
        f.read(4)  # file size
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"not WAVE: {wave}")
        pcm_data = None
        sample_rate = None
        bits_per_sample = None
        num_channels = None
        while True:
            chunk = f.read(4)
            if len(chunk) < 4:
                break
            clen = struct.unpack('<I', f.read(4))[0]
            if chunk == b'fmt ':
                fmt = f.read(clen)
                audio_fmt = struct.unpack('<H', fmt[0:2])[0]
                num_channels = struct.unpack('<H', fmt[2:4])[0]
                sample_rate = struct.unpack('<I', fmt[4:8])[0]
                bits_per_sample = struct.unpack('<H', fmt[14:16])[0]
            elif chunk == b'data':
                pcm_data = f.read(clen)
                break
            else:
                f.read(clen)
    return pcm_data, sample_rate, bits_per_sample, num_channels


def decode_samples(pcm_data, bits=16, channels=1):
    """PCM bytes → list[float] normalized to [-1, +1]."""
    if bits == 16:
        fmt = '<{}h'.format(len(pcm_data) // 2)
        raw = struct.unpack(fmt, pcm_data)
    else:
        raise ValueError(f"bits={bits} not implemented")
    if channels > 1:
        return [raw[i] / 32780.0 for i in range(0, len(raw), channels)]
    return [s / 32780.0 for s in raw]


# ============================================================
# 二、按帧切分，一帧 = COLS 个 block
# ============================================================

FRAME_DURATION = 0.10  # 100 ms/frame


def build_aud_grid_for_frame( samples_sub ):
    """samples_sub[:1600] for 1 frame → 8×64 (ZCR + energy)."""
    n = len(samples_sub)
    block_size = max(1, n // COLS)
    grid = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = c * block_size
            e = min(s + block_size, n)
            block = samples_sub[s:e]
            if not block:
                row.append(0.0)
                continue
            mean_val = sum(block) / len(block)
            sq_mean = sum(x * x for x in block) / len(block)
            zc = sum(1 for i in range(1, len(block)) if block[i-1] >= 0 and block[i] < 0)
            zcr = zc / len(block)
            # 频段对应：按行映射
            # rows 0-2: zzr_mean * n
            # rows 3-4: energy / n * 100
            # rows 5-7: zeros (unused)
            if r < 3:
                # ZCR in this band
                val = zcr * len(block)
            elif r < 5:
                val = sq_mean * 10000
            else:
                val = 0.0
            row.append(val)
        grid.append(row)
    return grid


# ============================================================
# 三、诊断
# ============================================================

if __name__ == "__main__":
    pcm, sr, bits, ch = read_wav_pcm(WAV_FILE)
    samples = decode_samples(pcm, bits, ch)
    print(f"sr={sr}, bits={bits}, ch={ch}, total_samples={len(samples)}")
    print(f"duration: {len(samples)/sr:.3f}s = {len(samples)/sr*1000:.0f} ms")
    print(f"samples/frame @ 100ms = {int(sr*0.10)}")
    print(f"理论帧数: {len(samples)/ int(sr*0.10):.2f}")
    print()
    
    N_FRAMES = len(samples) // int(sr * 0.10)
    SAMPLES_PER_FRAME = int(sr * 0.10)
    
    print("=" * 120)
    print(f"帧  时间 | row0         row1         row2         row3         row4         row5   row6   row7    | [min..max]")
    print("-" * 120)
    
    per_frame_mins = []
    per_frame_maxs = []
    per_frame_all = []
    
    for fi in range(N_FRAMES):
        start = fi * SAMPLES_PER_FRAME
        end = start + SAMPLES_PER_FRAME
        if end > len(samples):
            break
        sub = samples[start:end]
        grid = build_aud_grid_for_frame(sub)
        flat = [v for r in grid for v in r]
        per_frame_all.append(flat)
        
        mn, mx = min(flat), max(flat)
        per_frame_mins.append(mn)
        per_frame_maxs.append(mx)
        
        # 每行取摘要（min..max of that row）
        row_summaries = []
        for r_grid in grid:
            rmn, rmx = min(r_grid), max(r_grid)
            row_summaries.append(f"{rmn:5.2f}-{rmx:5.2f}")
        
        # 每帧原始格子值（row 全行展示前 16 个格子，其余省略）
        is_diagnostic_range = (2 <= fi <= 57)
        if fi < 3 or fi > 54 or is_diagnostic_range:
            # 展示
            if fi >= 3 and fi <= 54:
                # 紧凑模式：只看 row summary，只报 row 0-4 (低频-中频)
                rs = " ".join(f"{row_summaries[i]:>12}" for i in range(5))
                all_mn = min(flat[:5*COLS])
                all_mx = max(flat[:5*COLS])
                print(f"f{fi:02d} {fi*0.10:4.1f}s | {rs} | [{all_mn:6.2f} .. {all_mx:7.2f}]")
            else:
                rs = " ".join(f"{row_summaries[i]:>12}" for i in range(5))
                all_mn = min(flat[:5*COLS])
                all_mx = max(flat[:5*COLS])
                print(f"f{fi:02d} {fi*0.10:4.1f}s | {rs} | [{all_mn:6.2f} .. {all_mx:7.2f}]")
        elif fi == 3:
            print("  ...")
    
    print()
    print("=" * 120)
    print("【帧间横向对比 — 同一帧行值的 min/max】")
    print("-" * 120)
    print("说明：如果每一帧的 row summary 完全一样 → 映射被压平了")
    print("      如果 row summary 每帧不同 → 信号有差异，再看格子级")
    print()
    
    # 3. ===== 帧间差异：同位置格子的逐帧对比 =====
    print("=" * 120)
    print("【格子级诊断 — 固定帧间对比，取 row 0 (mean energy) 的 64 个格子】")
    print("-" * 120)
    
    # row 0 = zzr_mean × n，这是帧内"平均过零数 × 帧长" = 脉冲密度的一个指标
    # 所有 f2-f57 的 row 0 格子值，每帧 64 个值，贴 raw
    
    print()
    print("f02 row0 (64 values):")
    g2 = per_frame_all[2]  # frame 2
    r0_2 = g2[0:64]  # row 0 of frame 2
    for start in range(0, 64, 16):
        chunk = r0_2[start:start+16]
        print(f"  col{start:2}-{start+len(chunk)-1:2}: " + " ".join(f"{v:6.2f}" for v in chunk))
    
    print()
    print("f30 row0 (64 values):")
    g30 = per_frame_all[30]
    r0_30 = g30[0:64]
    for start in range(0, 64, 16):
        chunk = r0_30[start:start+16]
        print(f"  col{start:2}-{start+len(chunk)-1:2}: " + " ".join(f"{v:6.2f}" for v in chunk))
    
    print()
    print("f50 row0 (64 values):")
    g50 = per_frame_all[50]
    r0_50 = g50[0:64]
    for start in range(0, 64, 16):
        chunk = r0_50[start:start+16]
        print(f"  col{start:2}-{start+len(chunk)-1:2}: " + " ".join(f"{v:6.2f}" for v in chunk))
    
    print()
    print("f55 row0 (64 values):")
    g55 = per_frame_all[55]
    r0_55 = g55[0:64]
    for start in range(0, 64, 16):
        chunk = r0_55[start:start+16]
        print(f"  col{start:2}-{start+len(chunk)-1:2}: " + " ".join(f"{v:6.2f}" for v in chunk))
    
    print()
    print()
    print("=" * 120)
    print("【17 个帧的 row0 64 格汇聚（frame × col）— 全贴】")
    print("行=帧 f2~f18，列=col 0..63。如果格子随帧不变 → 信号没进去")
    print("-" * 120)
    
    SHOW_FRAMES = [2, 5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 45, 48, 50, 53, 55]
    for fi in SHOW_FRAMES:
        flat = per_frame_all[fi]
        row0 = flat[0:64]
        print(f"f{fi:02d}: " + " ".join(f"{v:6.2f}" for v in row0))
    
    print()
    print("=" * 120)
    print("【frame delta — row 0 (col-by-col 差值，相邻帧相减)】")
    print("如果 delta 全为 0 → 信号被压平")
    print("-" * 120)
    
    for fi in [5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 45, 48, 50, 53, 55]:
        a = per_frame_all[fi-1][0:64]
        b = per_frame_all[fi][0:64]
        d = [b[j] - a[j] for j in range(64)]
        smx = max(d)
        smn = min(d)
        nonzero = sum(1 for v in d if abs(v) > 0.1)
        print(f"Δf{fi-1:02d}->f{fi:02d}: min_delta={smn:6.2f}, max_delta={smx:6.2f}, nonzero_cells(>0.1)={nonzero:2}")
    
    print()
    print()
    print("=" * 120)
    print("【3-1 分布摘要：每一帧格子值按大小排序后的 P25/P50/P75/Max】")
    print("-" * 120)
    print("帧  时间 | P0(min)  P25      P50       P75       P100(max) | 单元格数>1的信噪")
    print()
    for fi in SHOW_FRAMES:
        flat = per_frame_all[fi]
        flat_s = sorted(flat)
        n = len(flat_s)
        p0 = flat_s[0]
        p25 = flat_s[round(n*0.25)]
        p50 = flat_s[round(n*0.50)]
        p75 = flat_s[round(n*0.75)]
        p100 = flat_s[-1]
        cells_over_1 = sum(1 for v in flat if v > 1.0)
        cells_over_10 = sum(1 for v in flat if v > 10.0)
        cells_over_100 = sum(1 for v in flat if v > 100.0)
        print(f"f{fi:02d} {fi*0.10:4.1f}s | {p0:7.2f}  {p25:7.2f}  {p50:7.2f}  {p75:7.2f}  {p100:8.2f} | >1:{cells_over_1:3} >10:{cells_over_10:3} >100:{cells_over_100:3}")
