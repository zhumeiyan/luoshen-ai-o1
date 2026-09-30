"""音视频帧间变化对齐 — 不取值，取 delta（帧 i 与帧 i−1 的差）。

两路：
  视觉：相邻 PPM 帧的像素绝对差之和 → vis_delta[i] = Σ|frame[i] - frame[i-1]|
  听觉：相邻音频网格帧的绝对差之和 → aud_delta[i] = Σ|grid[i] - grid[i-1]|

对齐：delta[i] 都是帧 i 与帧 i−1 的差，一一对应。
tag: delta 高 = "动"，delta 低 = "静"。
"""

import numpy as np
import os, sys, struct, math

sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
import 矩阵.听觉 as auditory_m

FRAMES_DIR = 'video_input/frames'
AUDIO_PATH = 'video_input/audio/speech.wav'
FPS = 10
SR = 16000
BLOCK = SR // FPS  # 1600

# ---------- 读 PPM ----------
def read_ppm(path):
    with open(path, 'rb') as f:
        raw = f.read()
    p = raw[:200].split(b'\n')
    idx = 1
    while p[idx].strip().startswith(b'#'): idx += 1
    w, h = map(int, p[idx].strip().split()); idx += 1
    while p[idx].strip().startswith(b'#'): idx += 1
    maxval = int(p[idx].strip())
    hdr_end = raw.index(b'\n', raw.index(b'\n', raw.index(p[0])+1)+1)+1
    while raw[hdr_end:hdr_end+1] == b'\n': hdr_end += 1
    pixel_data = raw[hdr_end:]
    grid = np.zeros((h, w), dtype=np.float64)
    pos = 0
    for y in range(h):
        for x in range(w):
            r, g, b = pixel_data[pos], pixel_data[pos+1], pixel_data[pos+2]
            grid[y, x] = 0.299*r + 0.587*g + 0.114*b
            pos += 3
    return grid

def read_audio_chunks():
    with open(AUDIO_PATH, 'rb') as f:
        raw = f.read()
    pos = 0
    while raw[pos:pos+4] != b'data': pos += 1
    pos += 4; n_bytes = struct.unpack_from('<I', raw, pos)[0]; pos += 4
    n_samp = n_bytes // 2
    samples = np.array([struct.unpack_from('<h', raw, pos + i*2)[0] for i in range(n_samp)], dtype=np.float64)
    chunks = []
    for i in range(0, n_samp, BLOCK):
        c = samples[i:i+BLOCK]
        if len(c) < BLOCK:
            c = np.pad(c, (0, BLOCK - len(c)))
        chunks.append(c)
    return chunks

def chunk_to_aud_grid(chunk):
    g = np.zeros((48, 64))
    bs = len(chunk) // 64  # 25
    for j in range(64):
        blk = chunk[j*bs:(j+1)*bs]
        energy = math.sqrt(np.mean(blk**2))
        zc = np.sum(np.abs(np.diff(np.sign(blk)))) / (2 * len(blk))
        # simple: use zc for row and energy for magnitude
        row = min(int(zc / max(0.01, (np.sum(np.abs(np.diff(np.sign(chunk))))/len(chunk))) * 47), 47)
        g[row, j] = energy
    r = auditory_m.frame_to_dbSPL(g.tolist())
    return np.array(r[0] if isinstance(r, tuple) else r)


def main():
    frame_files = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith('.ppm'))
    N = len(frame_files)

    # 视觉帧序列
    vis_frames = [read_ppm(os.path.join(FRAMES_DIR, f)) for f in frame_files]
    # 听觉网格序列
    chunks = read_audio_chunks()
    aud_grids = [chunk_to_aud_grid(chunks[min(i, len(chunks)-1)]) for i in range(N)]

    # 帧间差
    vis_delta = np.zeros(N)
    aud_delta = np.zeros(N)
    for i in range(1, N):
        vis_delta[i] = np.sum(np.abs(vis_frames[i] - vis_frames[i-1]))
        aud_delta[i] = np.sum(np.abs(aud_grids[i] - aud_grids[i-1]))
    vis_delta[0] = vis_delta[1]
    aud_delta[0] = aud_delta[1]

    # 归一化
    def norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else x*0
    vn = norm(vis_delta)
    an = norm(aud_delta)

    # 阈值判断动/静
    thr = 0.3
    tag_map = {
        (True, True):  '动-动 两路同动',
        (True, False): '动-静 视觉动听觉静',
        (False, True): '静-静 两路同静',  # (actually means aud move only)
        (False, False): '静-静 两路同静',
    }

    print(f'{"帧":>4} {"时间":>7} | {"vis_Δ":>10} {"vis_n":>7} {"V动":>4} | {"aud_Δ":>10} {"aud_n":>7} {"A动":>4} | 两路状态')
    print('-' * 105)

    stats = {'动-动': 0, '动-静': 0, '静-动': 0, '静-静': 0}
    both_big_deltas = []

    for i in range(N):
        v_move = vn[i] > thr
        a_move = an[i] > thr
        v = '动' if v_move else '静'
        a = '动' if a_move else '静'

        if v_move and a_move:
            tag = '动-动 两路同动'
            stats['动-动'] += 1
            both_big_deltas.append(i)
        elif v_move and not a_move:
            tag = '动-静 视觉动听觉静'
            stats['动-静'] += 1
        elif not v_move and a_move:
            tag = '静-动 视觉静听觉动'
            stats['静-动'] += 1
        else:
            tag = '静-静 两路同静'
            stats['静-静'] += 1

        ts = f'{i/FPS:.2f}s'
        print(f'{i:>4} {ts:>7} | {vis_delta[i]:>10.2f} {vn[i]:>7.4f} {v:>4} | {aud_delta[i]:>10.2f} {an[i]:>7.4f} {a:>4} | {tag}')

    # 相关性
    r = np.corrcoef(vis_delta, aud_delta)[0, 1] if N > 2 else 0.0

    print(f'\n【帧间变化相关性】r = {r:+.4f}')
    print(f'\n【统计】阈值={thr}')
    for k, v in stats.items():
        print(f'  {k}: {v} 帧')

    print(f'\n【两路同动（帧间变化同时超阈值）】：')
    if both_big_deltas:
        print(f'  帧: {both_big_deltas}')
        print(f'  时间: {[f"{i/FPS:.2f}s" for i in both_big_deltas]}')

    # 找delta变化大的视觉帧对应的听觉delta，反之亦然
    print(f'\n【视觉变化最大的 5 帧，听觉对应多大：】')
    top_vis = np.argsort(vis_delta)[-5:][::-1]
    for i in top_vis:
        print(f'  f{i} ({i/FPS:.2f}s): vis_delta={vis_delta[i]:.2f}(n={vn[i]:.3f}), aud_delta={aud_delta[i]:.2f}(n={an[i]:.3f})')

    print(f'\n【听觉变化最大的 5 帧，视觉对应多大：】')
    top_aud = np.argsort(aud_delta)[-5:][::-1]
    for i in top_aud:
        print(f'  f{i} ({i/FPS:.2f}s): aud_delta={aud_delta[i]:.2f}(n={an[i]:.3f}), vis_delta={vis_delta[i]:.2f}(n={vn[i]:.3f})')


if __name__ == '__main__':
    main()
