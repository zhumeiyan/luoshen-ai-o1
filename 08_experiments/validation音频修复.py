"""verify_audio_fix.py — 音频通道解锁验证。
跑两测:
  1. 同人轨迹: 帧间 sim, 逐帧原始值 + std/min/max
  2. 静音轨迹: 静音→有声跳变 vs 同人连续段 jump 对比

用 face_signature.aud_grid (新版, 逐帧 48 行实际 dB) + face_auditory_signature (7 维, 6 段+zcr)
"""
import sys, math, struct
from pathlib import Path
sys.path.insert(0, ".")
from face_signature import aud_grid, face_auditory_signature, full_signature, signature_similarity

FRAMES_DIR = Path("video_input/frames")
WAV_PATH  = Path("video_input/audio/speech.wav")

# ──── 解码 ────
def decode_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data) - 8 and data[pos:pos+4] != b"data":
        pos += 1
    pos += 4
    nb = struct.unpack_from("<I", data, pos)[0]; pos += 4
    n = nb // 2
    return [struct.unpack_from("<h", data, pos + i*2)[0] for i in range(n)]

def decode_ppm(path):
    with open(path, "rb") as f:
        raw = f.read()
    parts = raw[:200].split(b"\n")
    magic = parts[0].strip()
    idx = 1
    while parts[idx].strip().startswith(b"#"):
        idx += 1
    w, h = map(int, parts[idx].strip().split()); idx += 1
    while parts[idx].strip().startswith(b"#"):
        idx += 1
    maxv = int(parts[idx].strip())
    he = raw.index(b"\n", raw.index(b"\n", raw.index(parts[0])+1)+1) + 1
    while raw[he:he+1] == b"\n":
        he += 1
    pd = raw[he:]
    def _lum(r, g, b):
        return 0.299*r + 0.587*g + 0.114*b
    rows = []
    if magic == b"P3":
        vals = list(map(int, pd.split()))
        for y in range(h):
            rows.append([_lum(*tuple(vals[(y*w+x)*3+c] for c in range(3))) for x in range(w)])
    else:
        for y in range(h):
            rows.append([_lum(*tuple(pd[(y*w+x)*3+c] for c in range(3))) for x in range(w)])
    return rows

def rms_db(samples):
    n = len(samples)
    e = math.sqrt(sum(x*x for x in samples) / max(1, n)) * min(1, n)
    return 20 * math.log10(e + 1.0)

# ──── 主 ────
samples = decode_wav(WAV_PATH)
frame_files = sorted([f for f in FRAMES_DIR.iterdir() if f.suffix == ".ppm"])
n_frames = len(frame_files)

print(f"总帧数: {n_frames} | WAV samples: {len(samples)} ({len(samples)/16000:.3f}s)\n")

sig_list = []       # 每帧 13 维签名
aud_sig_list = []   # 每帧 7 维听觉
rms_list = []
for idx in range(n_frames):
    vis = decode_ppm(frame_files[idx])
    aud = aud_grid(samples, idx)
    sig_list.append(full_signature(vis, aud))
    aud_sig_list.append(face_auditory_signature(aud))
    blk = samples[idx*1600:(idx+1)*1600]
    rms_list.append(rms_db(blk))

# ──── 帧间 sim ────
frame_sims = []
for i in range(1, n_frames):
    frame_sims.append(signature_similarity(sig_list[i-1], sig_list[i]))

# ──── 静音分界: 找第一个 rms>15 与最后一个 rms>15 ────
silent_thresh = 15.0
first_sound = 0
while first_sound < n_frames and rms_list[first_sound] < silent_thresh:
    first_sound += 1
last_sound = n_frames - 1
while last_sound >= 0 and rms_list[last_sound] < silent_thresh:
    last_sound -= 1

# ──── 报告一: 原始 sim 序列 + rms + aud 分布 ────
print("=" * 90)
print("一. 逐帧: rms_dB | sim(t-1,t) | aud 6 段 (0.158=均匀)")
print("=" * 90)
print(f"{'frame':>5s} | {'rms_dB':>8s} | {'sim':>10s} | aud[0]   aud[1]   aud[2]   aud[3]   aud[4]   aud[5]   zcr")
print("-" * 90)
for i in range(n_frames):
    si = frame_sims[i-1] if i >= 1 else None
    s_str = f"{si:10.6f}" if si is not None else "     —"
    a = aud_sig_list[i]
    print(f"{i:5d} | {rms_list[i]:8.3f} | {s_str} |"
          f" {a[0]:.4f}  {a[1]:.4f}  {a[2]:.4f}  {a[3]:.4f}  {a[4]:.4f}  {a[5]:.4f}  {a[6]:.4f}")

# ──── 报告二: 同人轨迹统计 ────
print("\n" + "=" * 90)
print("二. 同人轨迹 sim 统计")
print("=" * 90)

def stats(arr):
    n = len(arr)
    mn = min(arr); mx = max(arr)
    mean = sum(arr)/n
    var = sum((s-mean)**2 for s in arr)/n
    return mn, mx, mean, math.sqrt(var)

# 全序列
mn, mx, mean, std = stats(frame_sims)
print(f"全序列    : min={mn:.6f}  max={mx:.6f}  mean={mean:.6f}  std={std:.6f}")

# 初跳 (静音→有声) - 是否在 first_sound-1 → first_sound
if first_sound >= 1 and first_sound < n_frames:
    start_jump = signature_similarity(sig_list[first_sound-1], sig_list[first_sound])
    print(f"初跳({first_sound-1}→{first_sound}): sim={start_jump:.6f}  跳幅={1-start_jump:.6f}")

# 尾跳 (有声→静音)
if last_sound >= 0 and last_sound+1 < n_frames:
    end_jump = signature_similarity(sig_list[last_sound], sig_list[last_sound+1])
    print(f"尾跳({last_sound}→{last_sound+1}): sim={end_jump:.6f}  跳幅={1-end_jump:.6f}")

# 连续有声段 (first_sound+1 .. last_sound)
cont_sims = []
for i in range(first_sound+1, last_sound+1):
    if i >= 1 and i-1 >= 1:
        cont_sims.append(signature_similarity(sig_list[i-1], sig_list[i]))

if cont_sims:
    mn2, mx2, mean2, std2 = stats(cont_sims)
    print(f"连续有声段: min={mn2:.6f}  max={mx2:.6f}  mean={mean2:.6f}  std={std2:.6f}  (n={len(cont_sims)})")

# ──── 报告三: 音频帧间变化 ────
print("\n" + "=" * 90)
print("三. 音频通道验证: 相邻帧听觉签名 aud[0] 差值 (应非全零)")
print("=" * 90)
aud_diffs = []
for i in range(1, n_frames):
    d = abs(aud_sig_list[i][0] - aud_sig_list[i-1][0])
    aud_diffs.append(d)
n_zero = sum(1 for d in aud_diffs if d == 0.0)
n_nonzero = len(aud_diffs) - n_zero
print(f"帧对数: {len(aud_diffs)} | aud[0] 全等帧: {n_zero} | aud[0] 有变化帧: {n_nonzero}")
if aud_diffs:
    print(f"差值: min={min(aud_diffs):.7f}  max={max(aud_diffs):.7f}  mean={sum(aud_diffs)/len(aud_diffs):.7f}")
