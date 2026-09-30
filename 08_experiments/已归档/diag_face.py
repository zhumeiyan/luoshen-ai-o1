# [已撤回] 2026-09-19 推演者归档：测旧身体接口，新架构已换
"""diag_face.py — 三件诊断: 同脸输入字节 vs 身体 state vs 异脸反应。
   规矩: 不加 if, 不判对错, 原值。
"""
import sys, os, json, math, hashlib, struct, copy
from pathlib import Path

sys.path.insert(0, ".")
from 身体稳态 import 身体
from face_signature import (
    full_signature, signature_distance, signature_similarity,
)

DIR = Path(".")
FRAMES_DIR = DIR / "video_input/frames"
AUDIO_PATH = DIR / "video_input/audio/speech.wav"
FPS = 10
SPF = 1600  # samples per frame

# ---- 手工解码 PPM + WAV ----
def decode_ppm(path):
    with open(path, "rb") as f:
        raw = f.read()
    parts = raw[:200].split(b"\n"); idx = 1
    while parts[idx].strip().startswith(b"#"):
        idx += 1
    w, h = map(int, parts[idx].strip().split()); idx += 1
    while parts[idx].strip().startswith(b"#"):
        idx += 1
    int(parts[idx].strip())  # maxval
    he = raw.index(b"\n", raw.index(b"\n", raw.index(parts[0]) + 1) + 1) + 1
    while raw[he : he + 1] == b"\n":
        he += 1
    pd = raw[he:]
    magic = parts[0].strip()
    rows = []
    if magic == b"P3":
        vals = list(map(int, pd.split()))
        for y in range(h):
            rows.append([(vals[(y * w + x) * 3 + c] for c in range(3)) for x in range(w)])
    else:
        for y in range(h):
            rows.append([(pd[(y * w + x) * 3 + c] for c in range(3)) for x in range(w)])
    return rows, h, w

def decode_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data) - 8 and data[pos : pos + 4] != b"data":
        pos += 1
    pos += 4
    n_bytes = struct.unpack_from("<I", data, pos)[0]; pos += 4
    n = n_bytes // 2
    return [struct.unpack_from("<h", data, pos + i * 2)[0] for i in range(n)]

def lum_grid(ppm, h, w, A=48, B=64):
    g = [[0.0] * B for _ in range(A)]
    for i in range(A):
        for j in range(B):
            i0, i1 = i * h // A, max((i + 1) * h // A, 1)
            j0, j1 = j * B // B, max((j + 1) * B // B, 1)
            s = 0.0; c = 0
            for y in range(i0, min(i1, h)):
                for x in range(j0, min(j1, w)):
                    r, gg, b = ppm[y][x]
                    s += 0.299 * r + 0.587 * gg + 0.114 * b
                    c += 1
            g[i][j] = s / max(1, c)
    return g

def aud_grid(samples, A=48, B=64):
    n = len(samples); bs = max(1, n // B)
    g = [[0.0] * B for _ in range(A)]
    for j in range(B):
        blk = samples[j * bs : (j + 1) * bs]
        bn = len(blk)
        e = math.sqrt(sum(x * x for x in blk) / max(1, bn)) * min(1, bn)
        e_db = max(0.0, min(60.0, 20.0 * math.log10(e + 1.0)))
        for i in range(A):
            rf = max(0.05, min(1.0, 1.0 - abs(i - 12) / 48.0))
            g[i][j] = e_db * rf
    return g

# ---- 取参考签名 ----
frames = sorted(p for p in os.listdir(FRAMES_DIR) if p.endswith(".ppm"))
samples_all = decode_wav(AUDIO_PATH)

def sig_at_frame(frame_idx):
    ppm, h, w = decode_ppm(FRAMES_DIR / frames[frame_idx])
    vis = lum_grid(ppm, h, w)
    a0 = frame_idx * SPF; a1 = a0 + SPF
    aud = aud_grid(samples_all[a0:a1])
    return full_signature(vis, aud)

# ---- 任务一: 同脸 3 轮帧签名逐轮比对 ----
print("=" * 60)
print("一. 同脸 3 轮输入签名比对")
print(f"frame count: {len(frames)}")
print()
round_sigs_0 = [sig_at_frame(i) for i in range(len(frames))]
round_sigs_1 = [sig_at_frame(i) for i in range(len(frames))]
round_sigs_2 = [sig_at_frame(i) for i in range(len(frames))]

def sigs_equal(a, b, tol=1e-9):
    return all(abs(a[k] - b[k]) < tol for k in range(min(len(a), len(b))))

print(f"轮 0 vs 轮 1 (同脸): len0={len(round_sigs_0)} len1={len(round_sigs_1)}")
frames_equal_01 = [sigs_equal(round_sigs_0[i], round_sigs_1[i]) for i in range(len(frames))]
print(f"  逐帧 sig 相等: {sum(1 for x in frames_equal_01 if x)}/{len(frames)}")
print(f"  frame[0] sig 轮0: {[round(v, 6) for v in round_sigs_0[0]]}")
print(f"  frame[0] sig 轮1: {[round(v, 6) for v in round_sigs_1[0]]}")
diff_01 = sum((round_sigs_0[0][k] - round_sigs_1[0][k]) ** 2 for k in range(10))
print(f"  frame[0] L2 diff 轮0↔轮1: {diff_01:.2e}")

print()
print(f"轮 1 vs 轮 2 (同脸):")
frames_equal_12 = [sigs_equal(round_sigs_1[i], round_sigs_2[i]) for i in range(len(frames))]
print(f"  逐帧 sig 相等: {sum(1 for x in frames_equal_12 if x)}/{len(frames)}")

# ---- 任务二: 异脸 — 用异张照片(镜像翻转+亮度大幅移动) ----
def mirror_lum_grid(grid):
    return [row[::-1] for row in grid]

def shift_lum_grid(grid, delta):
    return [[max(0.0, v + delta) for v in row] for row in grid]

# 伪造一个完全不同的脸: 视觉亮度向下平移 800 cd/m², 左半翻到右半
def face_sig_different():
    """构造一张跟作者完全异的脸: 作者的翻镜像 + 亮度平移 + 听觉静音翻反向。"""
    ppm, h, w = decode_ppm(FRAMES_DIR / frames[10])  # 取中间帧
    vis = lum_grid(ppm, h, w)
    vis_diff = mirror_lum_grid(shift_lum_grid(vis, 800.0))
    aud_diff = [[60.0 - v for v in row] for row in aud_grid(samples_all[10*SPF:11*SPF])]
    return full_signature(vis_diff, aud_diff)

diff_sig = face_sig_different()
ref_sig = round_sigs_0[10]
print()
print("=" * 60)
print("二. 异脸反应")
print()
print(f"  同脸 ref (frame[10]):      {[round(v, 4) for v in ref_sig]}")
print(f"  异脸 fake (mirror+shift):  {[round(v, 4) for v in diff_sig]}")
sim_same = signature_similarity(round_sigs_0[9], round_sigs_0[11])  # 同脸近帧
sim_diff = signature_similarity(ref_sig, diff_sig)
dist_diff = signature_distance(ref_sig, diff_sig)
print(f"  同脸远近帧 sim: {sim_same:.6f}")
print(f"  异脸 vs 同脸 sim: {sim_diff:.6f}")
print(f"  异脸 vs 同脸 dist: {dist_diff:.6f}")

# ---- 任务三: std=0 来路 —— 用身体跑了看 state 变不变 ----
print()
print("=" * 60)
print("三. std=0 来路")
print()
body_a = 身体(节点数=24, 种子=1)
body_b = 身体(节点数=24, 种子=1)
# 给同一 sig 三次
boost = signature_similarity(ref_sig, ref_sig) * 0.3  # 完全同 sig → sim=1.0
print(f"  参考 sig 自相似 = 1.0 → boost = {boost:.4f}")
states_a = []
states_b = []
for _ in range(3):
    body_a._social_heat = min(1.0, body_a._social_heat + boost)
    body_a.tick()
    body_b._social_heat = min(1.0, body_b._social_heat + boost)
    body_b.tick()
    states_a.append({
        "sh": round(body_a._social_heat, 7),
        "ss": round(body_a.state["social"], 7),
        "th": round(body_a.state["thermal"], 7),
        "vo": round(body_a.voice_aperture(), 7),
        "sig_mean": round(sum(ref_sig) / len(ref_sig), 7),
    })
    states_b.append({
        "sh": round(body_b._social_heat, 7),
        "ss": round(body_b.state["social"], 7),
        "th": round(body_b.state["thermal"], 7),
        "vo": round(body_b.voice_aperture(), 7),
        "sig_mean": round(sum(ref_sig) / len(ref_sig), 7),
    })

print("  Body A (克隆):")
for i, s in enumerate(states_a):
    print(f"    tick {i+1}: {s}")
print("  Body B (克隆, 同样输入):")
for i, s in enumerate(states_b):
    print(f"    tick {i+1}: {s}")

# 比对 A vs B (输入完全一样, 克隆起点一样 → state 也应完全一样)
def state_eq(a, b, tol=1e-9):
    return all(abs(a[k] - b[k]) < tol for k in a)
all_eq = [state_eq(states_a[i], states_b[i]) for i in range(3)]
print(f"  A[i]==B[i] (同输入同起点): {all_eq}")

# 关键: 输入变一点点, state 变多少?
body_c = 身体(节点数=24, 种子=1)
ref_states = []
sigs_to_try = [round_sigs_0[0], round_sigs_0[1], round_sigs_0[2]]  # 同脸 3 帧
for sig in sigs_to_try:
    sim = signature_similarity(sig, ref_sig)
    boost = sim * 0.3
    body_c._social_heat = min(1.0, body_c._social_heat + boost)
    body_c.tick()
    ref_states.append({
        "sim": round(sim, 7),
        "boost": round(boost, 7),
        "sh": round(body_c._social_heat, 7),
        "ss": round(body_c.state["social"], 7),
        "vo": round(body_c.voice_aperture(), 7),
    })
print()
print("  同脸 3 帧微差别输入 (body_c):")
for i, s in enumerate(ref_states):
    print(f"    frame {i}: {s}")

# compute std of sim
sims = [s["sim"] for s in ref_states]
mean_s = sum(sims) / len(sims)
var_s = sum((s - mean_s) ** 2 for s in sims) / len(sims)
print(f"  sim mean={mean_s:.7f} var={var_s:.2e} std={math.sqrt(var_s):.2e}")

# sh std
shs = [s["sh"] for s in ref_states]
mean_sh = sum(shs) / len(shs)
var_sh = sum((s - mean_sh) ** 2 for s in shs) / len(shs)
print(f"  social_heat std across 3 frames: {math.sqrt(var_sh):.2e}")
