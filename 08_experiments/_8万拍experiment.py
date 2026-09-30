"""上限999999，声倍数300，声末梢低槛0.015，光声同时喂，80000拍。逐2万拍报点火。"""
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

总光最大 = 0.0
总声最大 = 0.0
总力最大 = 0.0
总温最大 = 0.0
总点火 = [0]*4
门控触发总数 = 0
段点火 = []

STAGE = 20000
总拍 = 80000
段数 = 总拍 // STAGE

for 段 in range(段数):
    段点火计数 = [0]*4
    for 拍 in range(STAGE):
        global拍 = 段 * STAGE + 拍
        frame_i = 拍 % N_FRAMES_SRC
        light_frame = FRAMES[frame_i]
        sound_row_mean = float(np.mean(audio_mat[frame_i, :]))
        rep = 拍 % 50
        row_idx = rep if rep < 16 else 15
        light_val = float(np.mean(light_frame[row_idx, :]))
        sound_val = sound_row_mean * 300
        引擎.环境["光"] = light_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["声"] = sound_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["力"] = 10.0
        引擎.环境["温"] = 295.0
        引擎.推进()

        np_a = np.array(引擎.激活)
        g = 引擎.信号记录["门控缩放"]
        光最 = float(np.max(np.abs(np_a[0:脑大小])))
        声最 = float(np.max(np.abs(np_a[脑大小:2*脑大小])))
        力最 = float(np.max(np.abs(np_a[2*脑大小:3*脑大小])))
        温最 = float(np.max(np.abs(np_a[3*脑大小:4*脑大小])))
        if 光最 > 总光最大: 总光最大 = 光最
        if 声最 > 总声最大: 总声最大 = 声最
        if 力最 > 总力最大: 总力最大 = 力最
        if 温最 > 总温最大: 总温最大 = 温最

        for idx in range(96):
            if abs(np_a[idx]) > 0.5:
                nb = idx // 脑大小
                总点火[nb] += 1
                段点火计数[nb] += 1

        if g < 1.0:
            门控触发总数 += 1

    段点火.append(段点火计数)

末筋 = len(引擎.筋树)

print(f"跑法 = 上限999999声倍数300声末梢低槛0.015光声同时喂80000拍种子=1")
print(f"光脑节点激活最大 = {总光最大:.6f}")
print(f"声脑节点激活最大 = {总声最大:.6f}")
print(f"力脑节点激活最大 = {总力最大:.6f}")
print(f"温脑节点激活最大 = {总温最大:.6f}")
print(f"光脑总点火次数 = {总点火[0]}")
print(f"声脑总点火次数 = {总点火[1]}")
print(f"力脑总点火次数 = {总点火[2]}")
print(f"温脑总点火次数 = {总点火[3]}")
print(f"净增筋数 = {末筋 - 初筋}")
print(f"门控触发次数 = {门控触发总数}")
print()
for i, 计数 in enumerate(段点火):
    段名 = (i+1) * 2
    print(f"第{段名}万拍光脑点火次数 = {计数[0]}")
    print(f"第{段名}万拍声脑点火次数 = {计数[1]}")
    print(f"第{段名}万拍力脑点火次数 = {计数[2]}")
    print(f"第{段名}万拍温脑点火次数 = {计数[3]}")
