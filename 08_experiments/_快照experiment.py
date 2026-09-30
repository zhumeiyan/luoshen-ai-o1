"""100万拍快照实验。上限999999，声增强，种子=1。每20万拍存快照。"""
import time, pickle, os, json
from collections import Counter
import numpy as np
from _加速引擎 import 身体

WORK = "/mnt/data/catpaw/home/workspace/洛神赋1.1"
SNAP = os.path.join(WORK, "snapshots")
os.makedirs(SNAP, exist_ok=True)

import wave
wav_p = os.path.join(WORK, "video_input/teacher_audio.wav")
wf = wave.open(wav_p, 'rb')
sr = wf.getframerate()
raw = wf.readframes(wf.getnframes())
wf.close()
samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, wf.getnchannels())
mono = samples.mean(axis=1).astype(np.float64) / 32768.0
N_FRAMES_SRC, N_BANDS = 16, 4
seg_len = len(mono) // N_FRAMES_SRC
audio_mat = np.zeros((N_FRAMES_SRC, N_BANDS))
for fi in range(N_FRAMES_SRC):
    seg = mono[fi*seg_len:(fi+1)*seg_len]
    spectrum = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0/sr)
    band_edges = np.linspace(0, sr//2, N_BANDS+1)
    for bi in range(N_BANDS):
        mask = (freqs >= band_edges[bi]) & (freqs < band_edges[bi+1])
        audio_mat[fi, bi] = np.mean(spectrum[mask]) if mask.any() else 0.0
if audio_mat.max() > 0: audio_mat /= audio_mat.max()

痕文件 = os.path.join(WORK, "body_trace.jsonl")
if os.path.exists(痕文件): os.remove(痕文件)

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1, 上限满值=999999.0)
引擎.末梢["声"]["低"] = 0.015  # 声末梢门槛降低

脑大小 = 24
初筋 = len(引擎.筋树)
总拍, 段拍, 段数 = 1000000, 200000, 5
段点火 = []
点火总 = [0]*4

# 自己的自己点记录
原始自己点 = 引擎._自己点
自己点记录 = []
def 自己点带记录(记录=None):
    回接集 = set(引擎._回接节点)
    候选 = [i for (j,i) in 引擎.筋树 if j in 回接集 and i in 回接集]
    if not 候选: return
    上次 = getattr(引擎, '_上次自己点点', None)
    if 上次 is not None and 上次 in 候选:
        沉淀 = abs(引擎.激活[上次] - 引擎.上轮激活[上次]) if 引擎.上轮激活 else 0
        目标 = 上次 if 沉淀 >= 0.1 else 引擎.随机器.choice(候选)
    else: 目标 = 引擎.随机器.choice(候选)
    引擎.激活[目标] += 0.5
    引擎._上次自己点点 = 目标
    自己点记录.append(目标)
引擎._自己点 = 自己点带记录

t0 = time.time()
for 段 in range(段数):
    段计数 = [0]*4
    for 拍 in range(段拍):
        fi = 拍 % N_FRAMES_SRC
        # 16x17简化：light只采中央区域均值
        拍聲 = float(np.mean(audio_mat[fi, :])) * 300
        引擎.环境["光"] = 200.0 + 引擎.随机器.uniform(-1, 1)
        引擎.环境["声"] = 拍聲 + 引擎.随机器.uniform(-1, 1)
        引擎.环境["力"] = 10.0
        引擎.环境["温"] = 295.0
        引擎.推进()
        np_a = np.array(引擎.激活)
        for idx in range(96):
            if abs(np_a[idx]) > 0.5:
                nb = idx // 脑大小
                点火总[nb] += 1
                段计数[nb] += 1
    # 快照
    快照数据 = {
        "段": 段, "段尾拍": (段+1)*段拍,
        "激活": list(引擎.激活), "不应期": list(引擎.不应期),
        "筋树": {str(k):v for k,v in 引擎.筋树.items()},
        "自己点记录": list(自己点记录),
        "传导差": list(引擎.传导差),
        "当前上限": 引擎.当前上限, "脑共火计数": list(引擎.脑共火计数),
        "筋数": len(引擎.筋树), "段点火": list(段计数),
    }
    with open(os.path.join(SNAP, f"段{段}_拍{(段+1)*段拍}.pkl"), 'wb') as f:
        pickle.dump(快照数据, f)
    段点火.append(段计数)

t1 = time.time()
耗时 = t1 - t0
末筋 = len(引擎.筋树)
计数 = Counter(自己点记录)

print(f"耗时秒数 = {耗时:.2f}")
print(f"每秒拍数 = {总拍/耗时:.1f}")
print(f"末筋 = {末筋}")
print(f"初筋 = {初筋}")
print(f"净增筋数 = {末筋 - 初筋}")
print(f"光脑点火总次数 = {点火总[0]}")
print(f"声脑点火总次数 = {点火总[1]}")
print(f"力脑点火总次数 = {点火总[2]}")
print(f"温脑点火总次数 = {点火总[3]}")
print(f"自己点总次数 = {len(自己点记录)}")
print(f"自己点选过的不同节点数 = {len(计数)}")
print()
for i, 计数段 in enumerate(段点火):
    print(f"第{(i+1)*20}万拍光脑点火 = {计数段[0]}")
    print(f"第{(i+1)*20}万拍声脑点火 = {计数段[1]}")
    print(f"第{(i+1)*20}万拍力脑点火 = {计数段[2]}")
    print(f"第{(i+1)*20}万拍温脑点火 = {计数段[3]}")
