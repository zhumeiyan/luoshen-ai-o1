"""脑亮节点计数+对照。门槛减半 vs 原版，光声同时喂800拍。"""
import os, sys, importlib.util, numpy as np, wave
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

spec1 = importlib.util.spec_from_file_location("engine_原版", "/mnt/data/catpaw/home/workspace/洛神赋1.1/_跨脑门槛实验_引擎_原版.py")
mod1 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(mod1)
身体_原版 = mod1.身体

spec2 = importlib.util.spec_from_file_location("engine_低半", "/mnt/data/catpaw/home/workspace/洛神赋1.1/_跨脑门槛实验_引擎_低半.py")
mod2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mod2)
身体_低半 = mod2.身体

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

脑大小 = 24

def 跑(引擎类, 阈值, 标签):
    痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"
    if os.path.exists(痕文件):
        os.remove(痕文件)

    引擎 = 引擎类(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)

    光亮集 = set()
    声亮集 = set()
    力亮集 = set()
    温亮集 = set()

    # 记录：力温同时亮的时候，光声是否在亮
    力亮时_光在亮 = 0
    力亮时_声在亮 = 0
    温亮时_光在亮 = 0
    温亮时_声在亮 = 0
    力温同时亮拍数 = 0

    # 记录：力/温亮的拍序号
    力亮拍 = []
    温亮拍 = []

    上轮激活 = None
    拍号 = 0

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
                拍光 = False
                拍声 = False
                拍力 = False
                拍温 = False
                for idx in range(96):
                    if abs(差[idx]) > 阈值:
                        nb = idx // 脑大小
                        if nb == 0:
                            光亮集.add(idx)
                            拍光 = True
                        elif nb == 1:
                            声亮集.add(idx)
                            拍声 = True
                        elif nb == 2:
                            力亮集.add(idx)
                            拍力 = True
                        elif nb == 3:
                            温亮集.add(idx)
                            拍温 = True

                if 拍力 and 拍温:
                    力温同时亮拍数 += 1
                    力亮拍.append(拍号)
                    温亮拍.append(拍号)
                    if 拍光:
                        力亮时_光在亮 += 1
                    if 拍声:
                        力亮时_声在亮 += 1
                elif 拍力:
                    力亮拍.append(拍号)
                elif 拍温:
                    温亮拍.append(拍号)

            上轮激活 = 当前.copy()
            拍号 += 1

    print(f"{标签}脑亮阈值 = {阈值}")
    print(f"{标签}光脑亮数 = {len(光亮集)}")
    print(f"{标签}声脑亮数 = {len(声亮集)}")
    print(f"{标签}力脑亮数 = {len(力亮集)}")
    print(f"{标签}温脑亮数 = {len(温亮集)}")
    print(f"{标签}光脑亮节点列表 = {sorted(光亮集)}")
    print(f"{标签}声脑亮节点列表 = {sorted(声亮集)}")
    print(f"{标签}力脑亮节点列表 = {sorted(力亮集)}")
    print(f"{标签}温脑亮节点列表 = {sorted(温亮集)}")
    print(f"{标签}力脑亮拍 = {力亮拍}")
    print(f"{标签}温脑亮拍 = {温亮拍}")
    print(f"{标签}力温同时亮拍数 = {力温同时亮拍数}")
    if 力温同时亮拍数 > 0:
        print(f"{标签}力温同时亮时光也在亮 = {力亮时_光在亮}")
        print(f"{标签}力温同时亮时声也在亮 = {力亮时_声在亮}")
    print()

跑(身体_低半, 0.01, "低半")
跑(身体_原版, 0.02, "原版")
