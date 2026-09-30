"""门槛原值，光声同时喂800拍。逐节点激活差最大/最小+分布。"""
import os, sys, importlib.util, numpy as np, wave
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

spec = importlib.util.spec_from_file_location("engine_原版", "/mnt/data/catpaw/home/workspace/洛神赋1.1/_跨脑门槛实验_引擎_原版.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
身体_原版 = mod.身体

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
N_FRAMES = 16
N_BANDS = 4
seg_len = len(mono) // N_FRAMES
audio_mat = np.zeros((N_FRAMES, N_BANDS))
for fi in range(N_FRAMES):
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

痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)

引擎 = 身体_原版(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
脑大小 = 24

# 逐节点差值记录
节点差记录 = [[] for _ in range(96)]
上轮激活 = None

for frame_i in range(N_FRAMES):
    light_frame = FRAMES[frame_i]
    sound_row_mean = float(np.mean(audio_mat[frame_i, :]))
    for rep in range(50):
        row_idx = rep if rep < 16 else 15
        light_val = float(np.mean(light_frame[row_idx, :]))
        sound_val = sound_row_mean * 200
        引擎.环境["光"] = light_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["声"] = sound_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["力"] = 10.0
        引擎.环境["温"] = 295.0
        引擎.推进()

        当前 = np.array(引擎.激活)
        if 上轮激活 is not None:
            差 = 当前 - 上轮激活
            for i in range(96):
                节点差记录[i].append(差[i])
        上轮激活 = 当前.copy()

# 按脑聚合
for nb, name in [(0,"光"), (1,"声"), (2,"力"), (3,"温")]:
    所有差 = []
    for idx in range(nb*脑大小, (nb+1)*脑大小):
        所有差.extend(节点差记录[idx])
    所有差 = np.array(所有差)
    print(f"{name}脑激活差最大 = {float(np.max(所有差)):.6f}")
    print(f"{name}脑激活差最小 = {float(np.min(所有差)):.6f}")

# 全节点分布统计
全差 = []
for i in range(96):
    全差.extend(节点差记录[i])
全差 = np.array(全差)

print()
档边界 = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, float('inf')]
档标签 = ["0", "0.001", "0.005", "0.01", "0.02", "0.05", "0.1", "0.5", "1.0"]
print(f"全节点差值总样本 = {len(全差)}")
print(f"全节点|差|>0.01的样本 = {int(np.sum(np.abs(全差) > 0.01))}")
print(f"全节点|差|>0.02的样本 = {int(np.sum(np.abs(全差) > 0.02))}")
print(f"全节点|差|<=0.01的样本 = {int(np.sum(np.abs(全差) <= 0.01))}")
print()
for i in range(len(档标签)):
    hi = 档边界[i+1] if i+1 < len(档边界) else float('inf')
    lo = 档边界[i]
    cnt = int(np.sum((np.abs(全差) > lo) & (np.abs(全差) <= hi)))
    hi_s = 档标签[i+1] if i+1 < len(档标签) else "inf"
    print(f"|差|在{档标签[i]}到{hi_s}之间 = {cnt}")
