"""无上限(999999)，声倍数300+声末梢低槛0.015，光声同时喂8000拍。"""
import os, sys, importlib.util, numpy as np, wave
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

spec = importlib.util.spec_from_file_location("eng", "/mnt/data/catpaw/home/workspace/洛神赋1.1/_无上限_引擎.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
身体 = mod.身体

import imageio
from PIL import Image
vid_p = "/mnt/data/catpaw/home/workspace/洛神赋1.1/video_input/teacher_speak.mp4"
rdr = imageio.get_reader(vid_p, 'ffmpeg')
FRAMES = []
for i in range(100):
    f = rdr.get_data(i)
    img = Image.fromarray(f).resize((16, 16), Image.LANCZOS)
    arr = np.array(img)
    grey = np.mean(arr, axis=2)
    FRAMES.append(grey)
rdr.close()

wav_p = "/mnt/data/catpaw/home/workspace/洛神赋1.1/video_input/teacher_audio.wav"
wf = wave.open(wav_p, 'rb')
sr = wf.getframerate()
nframes_w = wf.getnframes()
raw = wf.readframes(nframes_w)
wf.close()
samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, wf.getnchannels())
mono = samples.mean(axis=1).astype(np.float64) / 32768.0
N_FRAMES_SRC = 16
N_BANDS = 4
seg_len = len(mono) // N_FRAMES_SRC
audio_mat = np.zeros((N_FRAMES_SRC, N_BANDS))
for fi in range(N_FRAMES_SRC):
    seg = mono[fi*seg_len:(fi+1)*seg_len]
    spectrum = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0/sr)
    band_edges = np.linspace(0, sr//2, N_BANDS+1)
    for bi in range(N_BANDS):
        lo, hi = band_edges[bi], band_edges[bi+1]
        mask = (freqs >= lo) & (freqs < hi)
        audio_mat[fi, bi] = np.mean(spectrum[mask]) if mask.any() else 0.0
mx = audio_mat.max()
if mx > 0:
    audio_mat /= mx

if os.path.exists("/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"):
    os.remove("/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl")

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
脑大小 = 24
初筋 = len(引擎.筋树)

activations = [[] for _ in range(4)]  # 4个脑
点火记录 = []
门控触发 = 0
拍号 = 0

TOTAL_FRAMES = 160
REPS = 50
总拍数 = TOTAL_FRAMES * REPS
倍数 = 300

for fi in range(TOTAL_FRAMES):
    frame_i = fi % N_FRAMES_SRC
    light_frame = FRAMES[frame_i]
    sound_row_mean = float(np.mean(audio_mat[frame_i, :]))
    for rep in range(REPS):
        row_idx = rep if rep < 16 else 15
        light_val = float(np.mean(light_frame[row_idx, :]))
        sound_val = sound_row_mean * 倍数
        引擎.环境["光"] = light_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["声"] = sound_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["力"] = 10.0
        引擎.环境["温"] = 295.0
        引擎.推进()

        np_a = np.array(引擎.激活)
        for nb in range(4):
            activations[nb].extend(np_a[nb*脑大小:(nb+1)*脑大小].tolist())
        for idx in range(96):
            if abs(np_a[idx]) > 0.5:
                点火记录.append(idx // 脑大小)

        缩 = 引擎.信号记录["门控缩放"]
        if 缩 < 1.0:
            门控触发 += 1

        拍号 += 1

末筋 = len(引擎.筋树)

光 = np.array(activations[0])
声 = np.array(activations[1])
力 = np.array(activations[2])
温 = np.array(activations[3])

光火 = sum(1 for nb in 点火记录 if nb == 0)
声火 = sum(1 for nb in 点火记录 if nb == 1)
力火 = sum(1 for nb in 点火记录 if nb == 2)
温火 = sum(1 for nb in 点火记录 if nb == 3)

print(f"跑法 = 上限=999999声倍数300声末梢低槛0.015光声同时喂8000拍种子=1")
print(f"光脑节点激活最大 = {float(np.max(光)):.6f}")
print(f"声脑节点激活最大 = {float(np.max(声)):.6f}")
print(f"力脑节点激活最大 = {float(np.max(力)):.6f}")
print(f"温脑节点激活最大 = {float(np.max(温)):.6f}")
print(f"光脑点火次数 = {光火}")
print(f"声脑点火次数 = {声火}")
print(f"力脑点火次数 = {力火}")
print(f"温脑点火次数 = {温火}")
print(f"净增筋数 = {末筋 - 初筋}")
print(f"门控触发次数 = {门控触发}")
