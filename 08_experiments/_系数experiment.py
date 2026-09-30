"""系数0.3，光声同时喂800拍。"""
import os, sys, importlib.util, numpy as np, wave
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

spec = importlib.util.spec_from_file_location("eng", "/mnt/data/catpaw/home/workspace/洛神赋1.1/_信号追踪_引擎.py")
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

if os.path.exists("/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"):
    os.remove("/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl")

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
脑大小 = 24

新激活前_all = []
光激活 = []
声激活 = []
力激活 = []
温激活 = []
点火次数 = 0

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

        新激活前_all.extend(引擎.信号记录["新激活前"])
        np_a = np.array(引擎.激活)
        光激活.extend(np_a[0:脑大小].tolist())
        声激活.extend(np_a[脑大小:2*脑大小].tolist())
        力激活.extend(np_a[2*脑大小:3*脑大小].tolist())
        温激活.extend(np_a[3*脑大小:4*脑大小].tolist())

        # 点火 = 门控后 abs(激活)>0.5 的节点数
        点火次数 += sum(1 for v in np_a if abs(v) > 0.5)

新激活前 = np.array(新激活前_all)
光 = np.array(光激活)
声 = np.array(声激活)
力 = np.array(力激活)
温 = np.array(温激活)

光过05 = int(np.sum(np.abs(光) > 0.5))
声过05 = int(np.sum(np.abs(声) > 0.5))
力过05 = int(np.sum(np.abs(力) > 0.5))
温过05 = int(np.sum(np.abs(温) > 0.5))

print(f"跑法 = 系数0.3，光声同时喂，门槛原值0.02/0.05，800拍，种子=1")
print(f"新激活门控前最大 = {float(np.max(新激活前)):.6f}")
print(f"新激活门控前最小 = {float(np.min(新激活前)):.6f}")
print(f"光脑节点激活最大 = {float(np.max(光)):.6f}")
print(f"声脑节点激活最大 = {float(np.max(声)):.6f}")
print(f"光脑过0.5节点数 = {光过05}")
print(f"声脑过0.5节点数 = {声过05}")
print(f"力脑过0.5节点数 = {力过05}")
print(f"温脑过0.5节点数 = {温过05}")
print(f"点火次数 = {点火次数}")
