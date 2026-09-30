"""方案A：光压成1个值；方案B：声扩成16个值。"""
import os, sys, json, numpy as np, wave
from PIL import Image
import imageio

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

# ============================================================
# 视频帧 → 帧列表（100帧 × 16×16）
# ============================================================
vid_p = "/mnt/data/catpaw/home/workspace/洛神赋1.1/video_input/teacher_speak.mp4"
rdr = imageio.get_reader(vid_p, 'ffmpeg')
FRAMES = []
for i in range(100):
    f = rdr.get_data(i)
    img = Image.fromarray(f).resize((16, 16), Image.LANCZOS)
    FRAMES.append(np.mean(np.array(img), axis=2))
rdr.close()
N_FRAMES = 16
REPS = 50

# ============================================================
# 音频 → 16×4 矩阵
# ============================================================
wav_p = "/mnt/data/catpaw/home/workspace/洛神赋1.1/video_input/teacher_audio.wav"
wf = wave.open(wav_p, 'rb')
sr = wf.getframerate()
raw = wf.readframes(wf.getnframes())
wf.close()
samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, wf.getnchannels())
mono = samples.mean(axis=1).astype(np.float64) / 32768.0
seg_len = len(mono) // N_FRAMES
audio_mat = np.zeros((N_FRAMES, 4))
for fi in range(N_FRAMES):
    seg = mono[fi*seg_len:(fi+1)*seg_len]
    spectrum = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0/sr)
    band_edges = np.linspace(0, sr//2, 5)
    for bi in range(4):
        mask = (freqs >= band_edges[bi]) & (freqs < band_edges[bi+1])
        audio_mat[fi, bi] = np.mean(spectrum[mask]) if mask.any() else 0.0
audio_mat /= audio_mat.max() if audio_mat.max() > 0 else 1.0

# 音频按帧 → 16段细分（用于方案B）
audio_sub = np.zeros((N_FRAMES * 16, 4))  # 256个时段
seg_len2 = len(mono) // (N_FRAMES * 16)
for si in range(N_FRAMES * 16):
    seg = mono[si*seg_len2:(si+1)*seg_len2]
    spectrum = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0/sr)
    band_edges = np.linspace(0, sr//2, 5)
    for bi in range(4):
        mask = (freqs >= band_edges[bi]) & (freqs < band_edges[bi+1])
        audio_sub[si, bi] = np.mean(spectrum[mask]) if mask.any() else 0.0
audio_sub /= audio_sub.max() if audio_sub.max() > 0 else 1.0

def 跑实验(引擎, 光脑_set, 声脑_set, 上轮, 光喂法, 声喂法):
    for frame_i in range(N_FRAMES):
        for rep in range(REPS):
            light_val = 光喂法(frame_i, rep)
            sound_val = 声喂法(frame_i, rep)
            引擎.环境["光"] = light_val + 引擎.随机器.uniform(-1, 1)
            引擎.环境["声"] = sound_val + 引擎.随机器.uniform(-1, 1)
            引擎.环境["力"] = 10.0
            引擎.环境["温"] = 295.0
            引擎.推进()
            当前 = np.array(引擎.激活)
            if 上轮 is not None:
                差 = 当前 - 上轮
                for idx in range(96):
                    if abs(差[idx]) > 0.02:
                        nb = idx // 24
                        if nb == 0: 光脑_set.add(idx)
                        elif nb == 1: 声脑_set.add(idx)
            上轮 = 当前.copy()
    return 上轮

def 光方案A(frame_i, rep):
    # 整帧均值（16×16 → 1值）
    return float(np.mean(FRAMES[frame_i]))

def 光方案B(frame_i, rep):
    # 按行循环
    row_idx = rep if rep < 16 else 15
    return float(np.mean(FRAMES[frame_i][row_idx, :]))

def 声方案A(frame_i, rep):
    # 帧均值×200
    return float(np.mean(audio_mat[frame_i, :])) * 200

def 声方案B(frame_i, rep):
    # 16个时段遍历
    sub_idx = rep if rep < 16 else 15
    return float(np.mean(audio_sub[frame_i * 16 + sub_idx, :])) * 200

# ============================================================
# 方案A：光压1值
# ============================================================
print("=" * 60)
print("方案A：光压成1个值")
print("=" * 60)

痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"
if os.path.exists(痕文件): os.remove(痕文件)

引擎A = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
初筋 = len(引擎A.筋树)
光A = set(); 声A = set()
跑实验(引擎A, 光A, 声A, None, 光方案A, 声方案A)
末筋A = len(引擎A.筋树)
脑间A = sum(1 for (j,i) in 引擎A.筋树 if j//24 != i//24)

print(f"净增 = {末筋A - 初筋}")
print(f"光脑亮 = {sorted(光A)}")
print(f"光脑亮数 = {len(光A)}")
print(f"声脑亮 = {sorted(声A)}")
print(f"声脑亮数 = {len(声A)}")
print(f"光声重合 = {sorted(光A & 声A)}")
print(f"重合数 = {len(光A & 声A)}")
print(f"脑间筋 = {脑间A}")

# ============================================================
# 方案B：声扩16值
# ============================================================
print()
print("=" * 60)
print("方案B：声扩成16个值")
print("=" * 60)

if os.path.exists(痕文件): os.remove(痕文件)

引擎B = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
初筋B = len(引擎B.筋树)
光B = set(); 声B = set()
跑实验(引擎B, 光B, 声B, None, 光方案B, 声方案B)
末筋B = len(引擎B.筋树)
脑间B = sum(1 for (j,i) in 引擎B.筋树 if j//24 != i//24)

print(f"净增 = {末筋B - 初筋B}")
print(f"光脑亮 = {sorted(光B)}")
print(f"光脑亮数 = {len(光B)}")
print(f"声脑亮 = {sorted(声B)}")
print(f"声脑亮数 = {len(声B)}")
print(f"光声重合 = {sorted(光B & 声B)}")
print(f"重合数 = {len(光B & 声B)}")
print(f"脑间筋 = {脑间B}")

# ============================================================
# 对照
# ============================================================
print()
print("=" * 60)
print("对照")
print("=" * 60)
print(f"上一轮不对称喂法：净增=659, 脑间=1592, 光脑=4, 声脑=13, 重合=0")
print(f"方案A：净增={末筋A - 初筋}, 脑间={脑间A}, 光脑={len(光A)}, 声脑={len(声A)}, 重合={len(光A & 声A)}")
print(f"方案B：净增={末筋B - 初筋B}, 脑间={脑间B}, 光脑={len(光B)}, 声脑={len(声B)}, 重合={len(光B & 声B)}")
