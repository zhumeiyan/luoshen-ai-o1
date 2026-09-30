"""跨脑门槛对照。光声同时喂，800拍，种子=1。只报跨脑筋增长。"""
import os, sys, json, numpy as np, wave
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

# 原版引擎 (阈值0.02, 概率0.05)
import importlib.util
spec1 = importlib.util.spec_from_file_location("engine_原版", "/mnt/data/catpaw/home/workspace/洛神赋1.1/_跨脑门槛实验_引擎_原版.py")
mod1 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(mod1)
身体_原版 = mod1.身体

# 低半引擎 (阈值0.01, 概率0.025)
spec2 = importlib.util.spec_from_file_location("engine_低半", "/mnt/data/catpaw/home/workspace/洛神赋1.1/_跨脑门槛实验_引擎_低半.py")
mod2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mod2)
身体_低半 = mod2.身体

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

REPS = 50
总拍数 = N_FRAMES * REPS  # 800

def 跑实验(引擎类, 标签):
    痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"
    if os.path.exists(痕文件):
        os.remove(痕文件)

    引擎 = 引擎类(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
    初筋 = len(引擎.筋树)
    脑大小 = 24

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

    末筋 = len(引擎.筋树)
    脑间筋数 = sum(1 for (j,i) in 引擎.筋树 if j//脑大小 != i//脑大小)

    print(f"{标签}跨脑差阈值 = {0.02 if '原版' in 标签 else 0.01}")
    print(f"{标签}跨脑出筋概率 = {0.05 if '原版' in 标签 else 0.025}")
    print(f"{标签}初筋 = {初筋}")
    print(f"{标签}末筋 = {末筋}")
    print(f"{标签}净增 = {末筋 - 初筋}")

    # 重新算初筋跨脑数
    引擎初 = 引擎类(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
    初跨脑 = sum(1 for (j,i) in 引擎初.筋树 if j//脑大小 != i//脑大小)
    print(f"{标签}初跨脑 = {初跨脑}")
    print(f"{标签}末跨脑 = {脑间筋数}")
    print(f"{标签}跨脑筋增长 = {脑间筋数 - 初跨脑}")

print("=== 跨脑门槛对照 ===")
print(f"总拍数 = {总拍数}")
print()

跑实验(身体_原版, "原版")
print()
跑实验(身体_低半, "低半")
