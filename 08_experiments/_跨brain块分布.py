"""门槛减半后，跨脑筋块分布。只跑低半，只报块对。"""
import os, sys, importlib.util, numpy as np, wave
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

# 低半引擎
spec = importlib.util.spec_from_file_location("engine_低半", "/mnt/data/catpaw/home/workspace/洛神赋1.1/_跨脑门槛实验_引擎_低半.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
身体_低半 = mod.身体

# 读取视频帧 → 16×16
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

# 读取音频 → 16×4
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

# 初筋快照
引擎初 = 身体_低半(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
初筋集 = set(引擎初.筋树)
初跨脑 = sum(1 for (j,i) in 初筋集 if j//24 != i//24)

# 跑实验
REPS = 50
痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)

引擎 = 身体_低半(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
for frame_i in range(N_FRAMES):
    light_frame = FRAMES[frame_i]
    sound_row_mean = float(np.mean(audio_mat[frame_i, :]))
    for rep in range(REPS):
        row_idx = rep if rep < 16 else 15
        light_val = float(np.mean(light_frame[row_idx, :]))
        sound_val = sound_row_mean * 200
        引擎.环境["光"] = light_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["声"] = sound_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["力"] = 10.0
        引擎.环境["温"] = 295.0
        引擎.推进()

末筋集 = set(引擎.筋树)
新增跨脑 = [(j,i) for (j,i) in 末筋集 if j//24 != i//24 and (j,i) not in 初筋集]
末跨脑 = sum(1 for (j,i) in 末筋集 if j//24 != i//24)

脑名 = {0:"光", 1:"声", 2:"力", 3:"温"}
有向计数 = {}
光声新增 = 0
其他新增 = 0

for (j,i) in 新增跨脑:
    bj = j // 24
    bi = i // 24
    有向计数[(bj,bi)] = 有向计数.get((bj,bi), 0) + 1
    if (bj,bi) == (0,1) or (bj,bi) == (1,0):
        光声新增 += 1
    else:
        其他新增 += 1

print(f"跑法 = 门槛减半，光声同时喂，800拍，种子=1")
print(f"初跨脑 = {初跨脑}")
print(f"末跨脑 = {末跨脑}")
print(f"跨脑增长 = {末跨脑 - 初跨脑}")
print(f"新增跨脑总数 = {len(新增跨脑)}")
print()

光到声 = 有向计数.get((0,1), 0)
声到光 = 有向计数.get((1,0), 0)
光到力 = 有向计数.get((0,2), 0)
光到温 = 有向计数.get((0,3), 0)
声到力 = 有向计数.get((1,2), 0)
声到温 = 有向计数.get((1,3), 0)
力到温 = 有向计数.get((2,3), 0)

已知有向 = {(0,1),(1,0),(0,2),(0,3),(1,2),(1,3),(2,3)}
其余 = sum(v for k,v in 有向计数.items() if k not in 已知有向)

print(f"光到声 = {光到声}")
print(f"声到光 = {声到光}")
print(f"光到力 = {光到力}")
print(f"光到温 = {光到温}")
print(f"声到力 = {声到力}")
print(f"声到温 = {声到温}")
print(f"力到温 = {力到温}")
print(f"其余块对 = {其余}")
print()

print(f"新增跨脑中光声块 = {光声新增}")
print(f"新增跨脑中其他块 = {其他新增}")
比例分子 = 光声新增
分母 = len(新增跨脑)
print(f"光声块占新增跨脑比例 = {比例分子}/{分母}")
