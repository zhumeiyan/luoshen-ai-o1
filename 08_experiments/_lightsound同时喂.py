"""光声同时喂 + 光单独喂对照。参数写死，不改。"""
import os, sys, json, numpy as np, wave
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

# ============================================================
# 读取视频帧 → 16×16
# ============================================================
import imageio
vid_p = "/mnt/data/catpaw/home/workspace/洛神赋1.1/video_input/teacher_speak.mp4"
rdr = imageio.get_reader(vid_p, 'ffmpeg')
FRAMES = []
for i in range(100):
    f = rdr.get_data(i)
    from PIL import Image
    img = Image.fromarray(f).resize((16, 16), Image.LANCZOS)
    arr = np.array(img)
    grey = np.mean(arr, axis=2)
    FRAMES.append(grey)  # 16×16
rdr.close()

# ============================================================
# 读取音频 → 16×4
# ============================================================
wav_p = "/mnt/data/catpaw/home/workspace/洛神赋1.1/video_input/teacher_audio.wav"
wf = wave.open(wav_p, 'rb')
sr = wf.getframerate()
nframes = wf.getnframes()
raw = wf.readframes(nframes)
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

# ============================================================
# 实验一：光声同时喂
# ============================================================
print("=" * 60)
print("实验一：光声同时喂")
print("=" * 60)

痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
初筋 = len(引擎.筋树)
脑大小 = 24
光脑_002 = set()
声脑_002 = set()
上轮激活 = None

REPS = 50  # 每帧声重复数
总拍数 = 0

for frame_i in range(N_FRAMES):
    light_frame = FRAMES[frame_i]
    sound_row_mean = float(np.mean(audio_mat[frame_i, :]))

    for rep in range(REPS):
        # 光：16行循环，超过15行就保持第15行
        row_idx = rep if rep < 16 else 15
        light_val = float(np.mean(light_frame[row_idx, :]))

        # 声：每帧固定
        sound_val = sound_row_mean * 200

        引擎.环境["光"] = light_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["声"] = sound_val + 引擎.随机器.uniform(-1, 1)
        引擎.环境["力"] = 10.0
        引擎.环境["温"] = 295.0
        引擎.推进()
        总拍数 += 1

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
脑间筋数 = sum(1 for (j,i) in 引擎.筋树 if j//脑大小 != i//脑大小)

print(f"喂拍数 = {总拍数}")
print(f"初筋 = {初筋}")
print(f"末筋 = {末筋}")
print(f"净增 = {末筋 - 初筋}")
print(f"光脑亮(|差|>0.02) = {sorted(光脑_002)}")
print(f"光脑亮数 = {len(光脑_002)}")
print(f"声脑亮(|差|>0.02) = {sorted(声脑_002)}")
print(f"声脑亮数 = {len(声脑_002)}")
重合 = sorted(光脑_002 & 声脑_002)
print(f"光声重合 = {重合}")
print(f"重合数 = {len(重合)}")
print(f"脑间筋 = {脑间筋数}")

# ============================================================
# 实验二：只喂光，不喂声
# ============================================================
print()
print("=" * 60)
print("实验二：只喂光，声=0")
print("=" * 60)

if os.path.exists(痕文件):
    os.remove(痕文件)

引擎2 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
光脑_002_2 = set()
声脑_002_2 = set()
上轮激活_2 = None

for frame_i in range(N_FRAMES):
    light_frame = FRAMES[frame_i]
    for rep in range(REPS):
        row_idx = rep if rep < 16 else 15
        light_val = float(np.mean(light_frame[row_idx, :]))
        引擎2.环境["光"] = light_val + 引擎2.随机器.uniform(-1, 1)
        引擎2.环境["声"] = 0.0
        引擎2.环境["力"] = 10.0
        引擎2.环境["温"] = 295.0
        引擎2.推进()

        当前 = np.array(引擎2.激活)
        if 上轮激活_2 is not None:
            差 = 当前 - 上轮激活_2
            for idx in range(96):
                if abs(差[idx]) > 0.02:
                    nb = idx // 脑大小
                    if nb == 0:
                        光脑_002_2.add(idx)
                    elif nb == 1:
                        声脑_002_2.add(idx)
        上轮激活_2 = 当前.copy()

print(f"光脑亮(|差|>0.02) = {sorted(光脑_002_2)}")
print(f"光脑亮数 = {len(光脑_002_2)}")
print(f"声脑亮(|差|>0.02) = {sorted(声脑_002_2)}")
print(f"声脑亮数 = {len(声脑_002_2)}")

# ============================================================
# 三、逐帧光声参数
# ============================================================
print()
print("=" * 60)
print("逐帧光声参数")
print("=" * 60)
print("帧 | 光行均值(行0均值) | 声行均值 | 声强(×200) | 声原归一(= (声强-25)/50, <0截0)")
for fi in range(N_FRAMES):
    light_val = float(np.mean(FRAMES[fi][0, :]))  # 行均值
    sound_row_mean = float(np.mean(audio_mat[fi, :]))
    sound_strong = sound_row_mean * 200
    sound_norm = max(0.0, (sound_strong - 25) / 50)
    print(f"{fi:2d} | {light_val:8.2f} | {sound_row_mean:.4f} | {sound_strong:7.2f} | {sound_norm:.4f}")
