"""第一次真正喂声。
音频来自 teacher_speak.mp4 解出的 teacher_audio.wav。
"""
import os, sys, json, numpy as np, wave
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

# ============================================================
# 一、读音频
# ============================================================
wav_path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/video_input/teacher_audio.wav"
wf = wave.open(wav_path, 'rb')
nchn = wf.getnchannels()
sr = wf.getframerate()
nframes = wf.getnframes()
raw = wf.readframes(nframes)
wf.close()
samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, nchn)
mono = samples.mean(axis=1).astype(np.float64) / 32768.0
print(f"采样率 = {sr}, 声道 = {nchn}, 帧 = {nframes}, 采样点 = {len(mono)}")
print(f"时长 = {len(mono)/sr:.3f}s")

# ============================================================
# 二、音频转 16×4 矩阵
# ============================================================
# 分16个等长时段，每段做FFT分4频带
N_BANDS = 4
N_FRAMES = 16

segment_len = len(mono) // N_FRAMES
matrix = np.zeros((N_FRAMES, N_BANDS))

for fi in range(N_FRAMES):
    seg = mono[fi*segment_len:(fi+1)*segment_len]
    spectrum = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0/sr)
    # 4 个等宽频带（Hz）：0-6000, 6000-12000, 12000-18000, 18000-24000
    band_edges = np.linspace(0, sr//2, N_BANDS+1)
    for bi in range(N_BANDS):
        lo, hi = band_edges[bi], band_edges[bi+1]
        mask = (freqs >= lo) & (freqs < hi)
        matrix[fi, bi] = np.mean(spectrum[mask]) if mask.any() else 0.0

# 归一化到 [0,1]
mx = matrix.max()
if mx > 0:
    matrix /= mx

print(f"矩阵 max={matrix.max():.4f}, mean={matrix.mean():.4f}")

def 标量(矩阵):
    return float(np.mean(矩阵))

# ============================================================
# 三、喂声通道
# ============================================================
痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
初筋 = len(引擎.筋树)

脑大小 = 24
光脑_002 = set()
声脑_002 = set()
上轮激活 = None

# 每帧作为一拍（共16拍）
for fi in range(N_FRAMES):
    # 声强 = 矩阵行均值 × 倍数（用200，保过阈值25）
    行均值 = float(np.mean(matrix[fi, :]))
    声强 = 行均值 * 200

    引擎.环境["光"] = 50.0
    引擎.环境["声"] = 声强 + 引擎.随机器.uniform(-1, 1)
    引擎.环境["力"] = 10.0
    引擎.环境["温"] = 295.0
    引擎.推进()

    当前 = np.array(引擎.激活)
    if 上轮激活 is not None:
        差 = 当前 - 上轮激活
        for idx in range(96):
            if abs(差[idx]) > 0.02:
                nb = idx // 脑大小
                if nb == 0:
                    光脑_002.add(idx)
                elif nb == 1:
                    声脑_002.add(idx)
    上轮激活 = 当前.copy()

末筋 = len(引擎.筋树)
脑间筋 = sum(1 for (j,i) in 引擎.筋树 if j//脑大小 != i//脑大小)

print()
print("=*= 结果 =*=")
print(f"初筋 = {初筋}")
print(f"末筋 = {末筋}")
print(f"净增 = {末筋 - 初筋}")
print(f"光脑亮(|差|>0.02) = {sorted(光脑_002)}")
print(f"光脑亮数 = {len(光脑_002)}")
print(f"声脑亮(|差|>0.02) = {sorted(声脑_002)}")
print(f"声脑亮数 = {len(声脑_002)}")
print(f"脑间筋 = {脑间筋}")
print(f"喂拍数 = {N_FRAMES}")
print(f"声强倍数 = 200")
print(f"阈值未动 = 25")
