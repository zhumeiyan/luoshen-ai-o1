"""自己点机制观测。上限999999，声倍数300，声末梢低槛0.015。"""
import os, sys, importlib.util, numpy as np, wave
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

# 用无上限引擎
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

痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
N = 96

# 自己点记录
自己点记录 = []

# 重写_自己点, 插入记录
import types
原始自己点 = 引擎._自己点

def 自己点带记录(记录=None):
    if 引擎.上轮激活 is None:
        return
    回接集 = set(引擎._回接节点)
    候选 = []
    for (j,i) in 引擎.筋树:
        if j in 回接集 and i in 回接集:
            候选.append(i)
    if not 候选:
        return
    上次 = getattr(引擎, '_上次自己点点', None)
    if 上次 is not None and 上次 in 候选:
        沉淀 = abs(引擎.激活[上次] - 引擎.上轮激活[上次]) if 引擎.上轮激活 else 0
        if 沉淀 >= 0.1:
            目标 = 上次
        else:
            目标 = 引擎.随机器.choice(候选)
    else:
        目标 = 引擎.随机器.choice(候选)
    引擎.激活[目标] += 0.5
    引擎._上次自己点点 = 目标
    自己点记录.append(目标)

引擎._自己点 = 自己点带记录

STAGE = 10000
段点火 = []
脑大小 = 24

总拍 = 1000000
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
        for idx in range(96):
            if abs(np_a[idx]) > 0.5:
                nb = idx // 脑大小
                段点火计数[nb] += 1

    段点火.append(段点火计数)

# 统计
from collections import Counter
计数 = Counter(自己点记录)
不同节点 = sorted(计数.keys())

print(f"跑拍数 = {总拍}")
print(f"自己点总次数 = {len(自己点记录)}")
print(f"不同节点数 = {len(不同节点)}")
print()
for 节点 in 不同节点:
    print(f"节点{节点}被点次数 = {计数[节点]}")
print()
print(f"被点最多节点 = {计数.most_common(1)[0][0]}")
print(f"被点最多节点次数 = {计数.most_common(1)[0][1]}")
print(f"被点最少节点次数 = {计数.most_common()[-1][1]}")
只点一次 = sum(1 for v in 计数.values() if v == 1)
超100 = sum(1 for v in 计数.values() if v > 100)
print(f"只被点一次节点数 = {只点一次}")
print(f"被点超100次节点数 = {超100}")
