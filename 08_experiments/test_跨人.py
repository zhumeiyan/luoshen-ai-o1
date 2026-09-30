"""test_cross_person.py — 同人 vs 换真人 交叉 sim 测试。

输入:
  video_input/frames/frame_0001..0058.ppm + audio/speech.wav        旧同人基准
  video_input/different_person/frames/frame_001..058.ppm + audio/speech.wav  teacher_speak

同人内部 + 换人内部 + 交叉 Δ_cross + 最大内部跳幅
"""
import sys, math, struct, json
from pathlib import Path
sys.path.insert(0, ".")
from face_signature import aud_grid, full_signature, signature_similarity, face_visual_signature, face_auditory_signature

AUTHOR_FRAMES = Path("video_input/frames")
AUTHOR_WAV   = Path("video_input/audio/speech.wav")
DIFF_FRAMES  = Path("video_input/different_person/frames")
DIFF_WAV     = Path("video_input/different_person/audio/speech.wav")

def decode_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data) - 8 and data[pos:pos+4] != b"data":
        pos += 1
    pos += 4
    nb = struct.unpack_from("<I", data, pos)[0]; pos += 4
    n = nb // 2
    return [struct.unpack_from("<h", data, pos + i*2)[0] for i in range(n)]

def decode_ppm_lum(path):
    with open(path, "rb") as f:
        raw = f.read()
    parts = raw[:200].split(b"\n")
    magic = parts[0].strip()
    idx = 1
    while parts[idx].strip().startswith(b"#"):
        idx += 1
    w, h = map(int, parts[idx].strip().split()); idx += 1
    while parts[idx].strip().startswith(b"#"):
        idx += 1
    maxv = int(parts[idx].strip())
    he = raw.index(b"\n", raw.index(b"\n", raw.index(parts[0])+1)+1) + 1
    while raw[he:he+1] == b"\n":
        he += 1
    pd = raw[he:]
    rows = []
    if magic == b"P3":
        vals = list(map(int, pd.split()))
        for y in range(h):
            rows.append([0.299*vals[(y*w+x)*3]+0.587*vals[(y*w+x)*3+1]+0.114*vals[(y*w+x)*3+2] for x in range(w)])
    else:
        for y in range(h):
            rows.append([0.299*pd[(y*w+x)*3]+0.587*pd[(y*w+x)*3+1]+0.114*pd[(y*w+x)*3+2] for x in range(w)])
    return rows

def build_signatures(frames_dir, wav_path):
    samples = decode_wav(wav_path)
    frames = sorted([f for f in frames_dir.iterdir() if f.suffix == ".ppm"])
    sigs = []
    vis_only = []
    aud_only = []
    for idx, f in enumerate(frames):
        vis = decode_ppm_lum(f)
        aud = aud_grid(samples, idx)
        sigs.append(full_signature(vis, aud))
        vis_only.append(face_visual_signature(vis))
        aud_only.append(face_auditory_signature(aud))
    return sigs, vis_only, aud_only

def internal_stats(sigs):
    """帧间 sim 列表"""
    sims = [signature_similarity(sigs[i], sigs[i+1]) for i in range(len(sigs)-1)]
    return sims

def sim_list_stats(sims):
    n = len(sims)
    mn = min(sims)
    mx = max(sims)
    mean = sum(sims)/n
    std = math.sqrt(sum((s-mean)**2 for s in sims)/n)
    return mn, mx, mean, std

def max_internal_drop(sims):
    """相邻 sim 最小值 = 最大跳幅"""
    return min(sims)

print("=" * 90)
print("素材")
print("=" * 90)
author_sigs, author_vis, author_aud = build_signatures(AUTHOR_FRAMES, AUTHOR_WAV)
diff_sigs, diff_vis, diff_aud = build_signatures(DIFF_FRAMES, DIFF_WAV)
print(f"旧 author 帧数: {len(author_sigs)}  签名维: {len(author_sigs[0])}")
print(f"new diff   帧数: {len(diff_sigs)}  签名维: {len(diff_sigs[0])}")

# ── 作者内部 ──
print("\n" + "=" * 90)
print("一. 同人轨迹 (author 内部 frame_i → frame_i+1)")
print("=" * 90)
a_sims = internal_stats(author_sigs)
a_mn, a_mx, a_mean, a_std = sim_list_stats(a_sims)
print(f"min={a_mn:.6f}  max={a_mx:.6f}  mean={a_mean:.6f}  std={a_std:.6f}")
print("逐帧:")
for i, s in enumerate(a_sims):
    print(f"  {i:2d}→{i+1:2d}  {s:.6f}")

# ── 换人内部 ──
print("\n" + "=" * 90)
print("二. 换人轨迹 (teacher_speak 内部 frame_i → frame_i+1)")
print("=" * 90)
d_sims = internal_stats(diff_sigs)
d_mn, d_mx, d_mean, d_std = sim_list_stats(d_sims)
print(f"min={d_mn:.6f}  max={d_mx:.6f}  mean={d_mean:.6f}  std={d_std:.6f}")
print("逐帧:")
for i, s in enumerate(d_sims):
    print(f"  {i:2d}→{i+1:2d}  {s:.6f}")

# ── 交叉 ──
print("\n" + "=" * 90)
print("三. 交叉 sim: author_N → different_0 (断点)")
print("=" * 90)
cross_sim = signature_similarity(author_sigs[-1], diff_sigs[0])
print(f"author帧{len(author_sigs)-1} → different帧0:   sim = {cross_sim:.6f}")
# 同时报 visual-only 与 auditory-only 交叉
cross_vis = signature_similarity(author_vis[-1], diff_vis[0])
cross_aud = signature_similarity(author_aud[-1], diff_aud[0])
print(f"  仅有 visual:  {cross_vis:.6f}")
print(f"  仅有 audio:   {cross_aud:.6f}")

# ── 最大跳幅 ──
print("\n" + "=" * 90)
print("四. 最大内部跳幅")
print("=" * 90)
print(f"同人内部最大跳(最小 sim){{frame}}: {max_internal_drop(a_sims):.6f}")
print(f"换人内部最大跳(最小 sim){{frame}}: {max_internal_drop(d_sims):.6f}")

# ── 关键比较 ──
print("\n" + "=" * 90)
print("五. 关键比较 (不判,只列)")
print("=" * 90)
delta_cross = 1.0 - cross_sim
delta_a = 1.0 - max_internal_drop(a_sims)
delta_d = 1.0 - max_internal_drop(d_sims)
print(f"Δ_cross = 1 - sim(author末帧 → diff首帧) = {delta_cross:.6f}")
print(f"Δ_a_max = 1 - sim(同人内部最大跳)           = {delta_a:.6f}")
print(f"Δ_d_max = 1 - sim(换人内部最大跳)           = {delta_d:.6f}")

# ── 视觉 / 听觉两维度各自对比 ──
print("\n" + "=" * 90)
print("六. 视觉 / 听觉 维度独立比较 (末帧)")
print("=" * 90)
# 视觉: author末帧 vs diff首帧 — 同人情况下 sim 应更高
# 听觉: author末帧 vs diff首帧
print(f"visual(author末 vs diff首):  sim = {cross_vis:.6f}")
print(f"audio (author末 vs diff首):  sim = {cross_aud:.6f}")
print()
# 同维度同人连续段
print(f"visual 同人平均帧间sim: {sim_list_stats([signature_similarity(author_vis[i], author_vis[i+1]) for i in range(len(author_vis)-1)])[2]:.6f}")
print(f"audio  同人平均帧间sim: {sim_list_stats([signature_similarity(author_aud[i], author_aud[i+1]) for i in range(len(author_aud)-1)])[2]:.6f}")

# ── teacher_speak 内的 aud 分布变化 (检查是否有状态跳) ──
print("\n" + "=" * 90)
print("七. teacher_speak 段内 aud[0] 帧间差值 (查是否有状态跳)")
print("=" * 90)
aud_diffs = []
for i in range(1, len(diff_aud)):
    d = abs(diff_aud[i][0] - diff_aud[i-1][0])
    aud_diffs.append(d)
print(f"aud[0] 帧间差值: min={min(aud_diffs):.6f}  max={max(aud_diffs):.6f}  mean={sum(aud_diffs)/len(aud_diffs):.6f}")
