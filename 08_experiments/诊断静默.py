"""diag_silence.py — 静音段 sim 行为诊断。
查:0.909 跳变是输入静音还是距离函数低能量饱和。
"""
import sys, os, math, struct
from pathlib import Path
from face_signature import full_signature, signature_similarity

sys.path.insert(0, ".")
FRAMES_DIR = Path("video_input/frames")
AUDIO_PATH = Path("video_input/audio/speech.wav")

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

def rms(samples):
    n = len(samples)
    e = math.sqrt(sum(x * x for x in samples) / max(1, n)) * min(1, n)
    return 20 * math.log10(e + 1.0)

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

def is_silence(aud_grid_val, thresh_db=15.0):
    """连续判断静音: 全帧 dB < thresh → 静音帧。"""
    return all(v < thresh_db for row in aud_grid_val for v in row)

samples = decode_wav(AUDIO_PATH)
print(f"WAV 总 samples: {len(samples)} ({len(samples)/16000:.3f}s @16kHz)")

# 逐帧: 计算它的 RMS dB, 静音状态, 帧间 sim (sim(t-1, t))
frames = list(range(len(samples) // 1600 + 1))
rms_list = []
silent_list = []
grid_list = []
for idx in frames:
    a0 = idx * 1600; a1 = a0 + 1600
    blk = samples[a0:a1]
    rms_list.append(rms(blk))
    ag = aud_grid(samples, idx)
    grid_list.append(ag)
    silent_list.append(is_silence(ag))

# 帧间 sim
sim_list = [None]  # 帧 0 无前帧
for i in range(1, len(grid_list)):
    sig_a = [0.0] * 6 + [grid_list[i-1][r][c] for r in range(48) for c in range(64)]
    # 直接算听觉分布: 用 full_signature 里的 vis(sig=占位)+aud
    # 为了只比听觉:
    from face_signature import face_auditory_signature
    aud_a = face_auditory_signature(grid_list[i-1])
    aud_b = face_auditory_signature(grid_list[i])
    sim_list.append(signature_similarity(aud_a + [0.0]*6, aud_b + [0.0]*6))

print(f"总帧数: {len(frames)}")
print(f"静音帧 (all dB < 15): {[i for i,s in enumerate(silent_list) if s]}")
print(f"有声帧: {[i for i,s in enumerate(silent_list) if not s]}\n")

print(f"{'frame':>5s} | {'rms_dB':>8s} | {'静音':>4s} | {'sim(t-1,t)':>12s} | aud_low/mid/high")
print("-" * 70)
for i in range(len(frames)):
    aud_sig = face_auditory_signature(grid_list[i]) if i < len(grid_list) else [0,0,0,0]
    tag = " ← 断" if sim_list[i] is not None and sim_list[i] < 0.95 else ""
    print(f"{i:5d} | {rms_list[i]:8.3f} | {'S' if silent_list[i] else 'A':>4s} |"
          f"{('%.6f'%sim_list[i] if sim_list[i] is not None else '    —'):>12s} |"
          f" {aud_sig[0]:.3f} {aud_sig[1]:.3f} {aud_sig[2]:.3f}{tag}")
