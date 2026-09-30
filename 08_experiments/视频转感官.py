"""视频 → 视觉矩阵 + 听觉矩阵，帧级对齐。

输入：
  video_input/frames/frame_XXXX.ppm (48×64 PPM, 10fps)
  video_input/audio/speech.wav (PCM 16-bit mono, 16kHz)

输出：
  luoshenfu_home/traces/video_sensory_chain.jsonl (每帧一条，视觉+听觉对齐)

解码全部手工：WAV header 跳过 + int16 直接读；PPM 按 decode_image.py 的解码器。
特征提取：零穿越率 + 短时能量，不调现成音频库。
"""

import numpy as np
import json, sys, struct, os, math

sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
import 矩阵.视觉 as visual_m
import 矩阵.听觉 as auditory_m

FPS = 10
AUDIO_SR = 16000
SAMPLES_PER_FRAME = AUDIO_SR // FPS  # 1600 samples per 100ms frame
AUD_ROWS, AUD_COLS = 48, 64
VIS_ROWS, VIS_COLS = 48, 64

FRAMES_DIR = 'video_input/frames'
AUDIO_PATH = 'video_input/audio/speech.wav'
TRACE_PATH = 'luoshenfu_home/traces/video_sensory_chain.jsonl'

# ---------- PPM 解码（复用） ----------

def decode_ppm_p3(path):
    """解码 PPM，支持 P3 (ASCII) 和 P6 (binary)。"""
    with open(path, 'rb') as f:
        raw = f.read()

    # header: magic + width + height + maxval (all ASCII text, newline separated)
    # find the end of header (after maxval line)
    parts = raw[:200].split(b'\n')
    magic = parts[0].strip()
    # Skip comment lines if any
    idx = 1
    while parts[idx].strip().startswith(b'#'):
        idx += 1
    w, h = map(int, parts[idx].strip().split())
    idx += 1
    while parts[idx].strip().startswith(b'#'):
        idx += 1
    maxval = int(parts[idx].strip())
    # Find start of pixel data: after maxval + newline
    header_end = raw.index(b'\n', raw.index(b'\n', raw.index(parts[0])+1)+1)+1
    # skip any additional newlines after maxval
    while raw[header_end:header_end+1] == b'\n':
        header_end += 1
    pixel_data = raw[header_end:]

    if magic == b'P3':
        # ASCII
        values = list(map(int, pixel_data.split()))
        pixels = []
        pos = 0
        for y in range(h):
            row = []
            for x in range(w):
                row.append((values[pos], values[pos+1], values[pos+2]))
                pos += 3
            pixels.append(row)
    elif magic == b'P6':
        # Binary: 3 bytes per pixel
        pixels = []
        pos = 0
        for y in range(h):
            row = []
            for x in range(w):
                row.append((pixel_data[pos], pixel_data[pos+1], pixel_data[pos+2]))
                pos += 3
            pixels.append(row)
    else:
        raise ValueError(f'Unknown PPM magic: {magic}')

    return pixels, w, h

# ---------- WAV 解码（手工） ----------

def decode_wav_pcm(path):
    """读 PCM WAV，跳过 header，返回 int16 样本数组。"""
    with open(path, 'rb') as f:
        data = f.read()
    # RIFF header: 4 + 4 + 4
    # fmt chunk: 4 + 4 + ... 
    # Just find 'data' marker
    pos = 0
    while pos < len(data) - 8:
        if data[pos:pos+4] == b'data':
            break
        pos += 1
    # data chunk: 'data' + 4-byte size + payload
    pos += 4
    n_bytes = struct.unpack_from('<I', data, pos)[0]
    pos += 4
    n_samples = n_bytes // 2
    samples = np.array(
        [struct.unpack_from('<h', data, pos + i*2)[0] for i in range(n_samples)],
        dtype=np.float64
    )
    return samples

# ---------- 音频特征提取 ----------

def audio_to_grid(samples_1600):
    """1600 samples (100ms @16kHz) → 48×64 网格。

    策略：把 1600 samples 切成 64 个时间块（每块 25 samples），
    每块计算局部零穿越率（代表频率）和能量。
    然后映射到 48 行（频率带）× 64 列（时间）。

    映射方式：
    - 列 j = 时间块 j（0~63）
    - 行 i = 零穿越率线性映射到 0~47
    - 值 = 该时间块的能量（RMS）
    """
    grid = np.zeros((AUD_ROWS, AUD_COLS))
    block_size = len(samples_1600) // AUD_COLS  # 25

    energies = []
    zcrs = []
    for j in range(AUD_COLS):
        block = samples_1600[j*block_size:(j+1)*block_size]
        energy = math.sqrt(np.mean(block**2))
        # zero-crossing rate
        zc = np.sum(np.abs(np.diff(np.sign(block)))) / (2 * len(block))
        energies.append(energy)
        zcrs.append(zc)

    # Map ZCR to row and fill energy
    max_zc = max(zcrs) if max(zcrs) > 0 else 1.0
    max_e = max(energies) if max(energies) > 0 else 1.0

    for j in range(AUD_COLS):
        row = int(zcrs[j] / max_zc * (AUD_ROWS - 1))
        row = min(row, AUD_ROWS - 1)
        # Also spread energy to neighboring rows for a blur-like spread
        for dr in [-1, 0, 1]:
            rr = row + dr
            if 0 <= rr < AUD_ROWS:
                w = 0.5 if dr == 0 else 0.25
                grid[rr, j] = max(grid[rr, j], energies[j] * w)

    return grid

# ---------- 视频帧处理 ----------

def frame_to_visual_mat(ppm_path):
    pixels, w, h = decode_ppm_p3(ppm_path)
    A, B = VIS_ROWS, VIS_COLS
    # downsample to 48×64 using bilinear
    src_h, src_w = h, w
    grid = np.zeros((A, B))
    for i in range(A):
        fi = i * (src_h - 1) / max(A - 1, 1)
        i0, i1 = int(fi), min(int(fi)+1, src_h-1)
        wi = fi - i0
        for j in range(B):
            fj = j * (src_w - 1) / max(B - 1, 1)
            j0, j1 = int(fj), min(int(fj)+1, src_w-1)
            wj = fj - j0
            r00,g00,b00 = pixels[i0][j0]
            r01,g01,b01 = pixels[i0][j1]
            r10,g10,b10 = pixels[i1][j0]
            r11,g11,b11 = pixels[i1][j1]
            def l(r,g,b): return 0.299*r+0.587*g+0.114*b
            v = (l(r00,g00,b00)*(1-wi)*(1-wj) + l(r01,g01,b01)*(1-wi)*wj +
                 l(r10,g10,b10)*wi*(1-wj) + l(r11,g11,b11)*wi*wj)
            grid[i,j] = v

    result = visual_m.帧转亮度(grid.tolist())
    mat = result[0] if isinstance(result, tuple) else result
    flat = [v for r in mat for v in r]
    return mat, {'mean': round(float(np.mean(flat)),2), 'max': round(float(np.max(flat)),2), 'min': round(float(np.min(flat)),2)}

# ---------- 主流程 ----------

def main():
    frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.ppm')])
    samples = decode_wav_pcm(AUDIO_PATH)
    n_frames = len(frame_files)

    print(f"Frame files: {n_frames}, Audio samples: {len(samples)}")
    os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
    deposited = 0

    for idx, fname in enumerate(frame_files):
        # visual
        vis_mat, vis_stats = frame_to_visual_mat(os.path.join(FRAMES_DIR, fname))

        # audio (corresponding 100ms chunk)
        a_start = idx * SAMPLES_PER_FRAME
        a_end = min(a_start + SAMPLES_PER_FRAME, len(samples))
        a_chunk = samples[a_start:a_end]
        if len(a_chunk) < SAMPLES_PER_FRAME:
            a_chunk = np.pad(a_chunk, (0, SAMPLES_PER_FRAME - len(a_chunk)))

        aud_grid = audio_to_grid(a_chunk)
        result = auditory_m.frame_to_dbSPL(aud_grid.tolist())
        aud_mat = result[0] if isinstance(result, tuple) else result
        aud_flat = [v for r in aud_mat for v in r]
        aud_stats = {'mean': round(float(np.mean(aud_flat)),2), 'max': round(float(np.max(aud_flat)),2), 'min': round(float(np.min(aud_flat)),2)}

        rec = {
            'frame_idx': idx,
            'timestamp': f'{idx / FPS:.2f}s',
            'visual': vis_stats,
            'auditory': aud_stats,
            'source': f'{fname} + audio_chunk[{a_start}:{a_end}]',
        }

        with open(TRACE_PATH, 'a', encoding='utf-8') as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
        deposited += 1

        if idx % 10 == 0:
            print(f"  f{idx:>3} [{rec['timestamp']}] vis_μ={vis_stats['mean']:>8.2f} vis_max={vis_stats['max']:>8.2f} | aud_μ={aud_stats['mean']:>8.2f} aud_max={aud_stats['max']:>8.2f}")

    print(f"\n落痕 {deposited} 条 → {TRACE_PATH}")
    return deposited

if __name__ == '__main__':
    main()
