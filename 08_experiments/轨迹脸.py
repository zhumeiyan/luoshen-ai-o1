"""trajectory_face.py — 帧间 sim 轨迹, 标断点。

轨迹判据: delta[t] = sim(帧 t-1, 帧 t)
  - 同人: delta 一直小 (~0.99+), 连续
  - 换人: delta 突然变大, 断点

规矩: 不加 if, 不判对错, 原值。
"""
import sys, os, math, struct
from pathlib import Path
from face_signature import full_signature, signature_similarity

sys.path.insert(0, ".")
FRAMES_DIR = Path("video_input/frames")
AUDIO_PATH = Path("video_input/audio/speech.wav")

def decode_ppm(path):
    with open(path, "rb") as f:
        raw = f.read()
    parts = raw[:200].split(b"\n"); idx = 1
    while parts[idx].strip().startswith(b"#"):
        idx += 1
    w, h = map(int, parts[idx].strip().split()); idx += 1
    while parts[idx].strip().startswith(b"#"):
        idx += 1
    int(parts[idx].strip())
    he = raw.index(b"\n", raw.index(b"\n", raw.index(parts[0]) + 1) + 1) + 1
    while raw[he : he + 1] == b"\n":
        he += 1
    pd = raw[he:]
    magic = parts[0].strip()
    rows = []
    if magic == b"P3":
        vals = list(map(int, pd.split()))
        for y in range(h):
            rows.append([(vals[(y * w + x) * 3 + c] for c in range(3)) for x in range(w)])
    else:
        for y in range(h):
            rows.append([(pd[(y * w + x) * 3 + c] for c in range(3)) for x in range(w)])
    return rows, h, w

def decode_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data) - 8 and data[pos : pos + 4] != b"data":
        pos += 1
    pos += 4
    n_bytes = struct.unpack_from("<I", data, pos)[0]; pos += 4
    n = n_bytes // 2
    return [struct.unpack_from("<h", data, pos + i * 2)[0] for i in range(n)]

def lum_grid(ppm, h, w, A=48, B=64):
    g = [[0.0] * B for _ in range(A)]
    for i in range(A):
        for j in range(B):
            i0, i1 = i * h // A, max((i + 1) * h // A, 1)
            j0 = j * w // B; j1 = max((j + 1) * w // B, 1)
            s = 0.0; c = 0
            for y in range(i0, min(i1, h)):
                for x in range(j0, min(j1, w)):
                    r, gg, b = ppm[y][x]
                    s += 0.299 * r + 0.587 * gg + 0.114 * b
                    c += 1
            g[i][j] = s / max(1, c)
    return g

def aud_grid(samples, idx, A=48, B=64):
    a0 = idx * 1600; a1 = a0 + 1600
    blk = samples[a0:a1]
    g = [[0.0] * B for _ in range(A)]
    bs = max(1, len(blk) // B)
    bx = min(B * bs, len(blk))
    for j in range(B):
        b2 = blk[j * bs : bx]
        bn = len(b2)
        e = math.sqrt(sum(x * x for x in b2) / max(1, bn)) * min(1, bn)
        ed = max(0.0, min(60.0, 20.0 * math.log10(e + 1.0)))
        for i in range(A):
            rf = max(0.05, min(1.0, 1.0 - abs(i - 12) / 48.0))
            g[i][j] = ed * rf
    return g

def mirror(g):
    return [row[::-1] for row in g]

def shift_vis(g, d):
    return [[max(0.0, v + d) for v in row] for row in g]

def trajectory(sigs):
    """返回帧间 sim: [sim(sigs[0], sigs[1]), sim(sigs[1], sigs[2]), ...]"""
    return [signature_similarity(sigs[i], sigs[i + 1]) for i in range(len(sigs) - 1)]

def delta(traj):
    return [traj[i + 1] - traj[i] for i in range(len(traj) - 1)]

def mean(lst):
    return sum(lst) / max(1, len(lst))

# ---- 同脸签名序列 ----
frames = sorted(p for p in os.listdir(FRAMES_DIR) if p.endswith(".ppm"))
samples = decode_wav(AUDIO_PATH)
same_sigs = []
for idx in range(len(frames)):
    ppm_, h, w = decode_ppm(FRAMES_DIR / frames[idx])
    vis = lum_grid(ppm_, h, w)
    aud = aud_grid(samples, idx)
    same_sigs.append(full_signature(vis, aud))

# ---- 异脸签名序列 (不同人完全异图: mirror + 亮度 +300 + dB-30) ----
diff_sigs = []
for idx in range(len(frames)):
    ppm_, h, w = decode_ppm(FRAMES_DIR / frames[idx])
    vis_diff = mirror(shift_vis(lum_grid(ppm_, h, w), 300.0))
    aud_diff = [[max(0.0, min(60.0, v - 30.0)) for v in row] for row in aud_grid(samples, idx)]
    diff_sigs.append(full_signature(vis_diff, aud_diff))

# ---- A: 同人轨迹 ----
traj_A = trajectory(same_sigs)

# ---- B: 换人轨迹: [同人 1..14] + [异脸 15..] ----
mixed_sigs = []
for i in range(len(frames)):
    # 前 15 帧同人, 后续换异脸 (模拟换人事件)
    g = min(1.0, max(0.0, 15 - i + 0.5))
    # g>0.5 同人, else 异脸
    same_g = max(0.0, min(1.0, 14.5 - i + 0.5))
    diff_g = 1.0 - same_g
    # 连续加权签名 (不是 if-then)
    sig_blend = [same_sigs[i][k] * same_g + diff_sigs[i][k] * diff_g for k in range(len(same_sigs[i]))]
    mixed_sigs.append(sig_blend)
traj_B = trajectory(mixed_sigs)

# ---- 报告: A ----
print("=" * 60)
print(f"一. 同人轨迹 (帧间 sim, {len(traj_A)} d)")
print(f"{'frame_pair':>12s} | {'sim(t-1,t)':>12s}")
print("-" * 28)
for i, s in enumerate(traj_A):
    print(f"{i:4d}→{i+1:<4d} | {s:12.6f}")
print()
print(f"min={min(traj_A):.6f}  max={max(traj_A):.6f}  mean={mean(traj_A):.6f}  std={math.sqrt(sum((s-mean(traj_A))**2 for s in traj_A)/len(traj_A)):.6f}")

# ---- 报告: B ----
print()
print("=" * 60)
print(f"二. 换人轨迹 (帧 0-14 同人, 帧 15+ 异脸) ({len(traj_B)} d)")
print(f"{'frame_pair':>12s} | {'sim(t-1,t)':>12s}")
print("-" * 28)
for i, s in enumerate(traj_B):
    tag = " 断" * (i == 14)
    print(f"{i:4d}→{i+1:<4d} | {s:12.6f}{tag}")
print()
print(f"min={min(traj_B):.6f}  max={max(traj_B):.6f}  mean={mean(traj_B):.6f}  std={math.sqrt(sum((s-mean(traj_B))**2 for s in traj_B)/len(traj_B)):.6f}")

# ---- B: 断点前后均值 ----
pre = traj_B[:14]
post = traj_B[14:]
print()
print("=" * 60)
print("三. 断点分析")
print()
print(f"  断点前 (同人段)   mean={mean(pre):.6f}  std={math.sqrt(sum((s-mean(pre))**2 for s in pre)/max(1,len(pre))):.6f}")
print(f"  断点后 (换人段)   mean={mean(post):.6f}  std={math.sqrt(sum((s-mean(post))**2 for s in post)/max(1,len(post))):.6f}")
print(f"  断差 (前-后)      = {mean(pre) - mean(post):.6f}")
