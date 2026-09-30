# [已撤回] 2026-09-19 推演者归档：旧引用，新架构已换
"""run_new_tests.py — 4 段新素材同人 / 换人测试。

4 段 (文叔叔下载):
  VID_20260918_213421 (57f)   VID_20260918_213427 (48f)   — 9/18 21:34 录制
  VID_20260919_025853 (51f)   VID_20260919_025859 (45f)   — 9/19 02:58 录制

测试:
  一. 每段内部轨迹: 帧间 sim 列表 + std/min/max
  二. sim(S_i末帧 → S_j首帧) 矩阵 (i,j ∈ 4 段) — 段间切换跳幅
  三. 仿射矩阵: 段_i 末 vs 段_{i+1} 首 sim, 同人 / 异人分离
输出原始 sim 数值, 不判。
"""
import sys, math, struct
from pathlib import Path
sys.path.insert(0, ".")
from face_signature import aud_grid, full_signature, face_visual_signature, face_auditory_signature, signature_similarity

RAW = Path("video_input/raw_frames")
VIDS = ["VID_20260918_213421", "VID_20260918_213427",
        "VID_20260919_025853", "VID_20260919_025859"]

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

def build_signatures(vid_name):
    frames_dir = RAW / vid_name
    wav_path = RAW / vid_name / "audio.wav"
    samples = decode_wav(wav_path)
    frames = sorted([f for f in frames_dir.iterdir() if f.suffix == ".ppm"])
    sig_list = []
    for idx, f in enumerate(frames):
        vis = decode_ppm_lum(f)
        aud = aud_grid(samples, idx)
        sig_list.append(full_signature(vis, aud))
    return sig_list

def frame_trajectory(sigs):
    """帧间 sim"""
    return [signature_similarity(sigs[i], sigs[i+1]) for i in range(len(sigs)-1)]

def sim_stats(sims):
    n = len(sims)
    mn = min(sims); mx = max(sims)
    mean = sum(sims)/n
    std = math.sqrt(sum((s-mean)**2 for s in sims)/n)
    return mn, mx, mean, std

# ──── 载入 4 段签名 ────
all_sigs = {}
for v in VIDS:
    all_sigs[v] = build_signatures(v)
    print(f"{v}: {len(all_sigs[v])} frames, sig_dim={len(all_sigs[v][0])}")

# ──── 一. 段内轨迹 ────
print("\n" + "=" * 100)
print("一. 段内轨迹 (frame_i → frame_i+1)")
print("=" * 100)
for v in VIDS:
    traj = frame_trajectory(all_sigs[v])
    mn, mx, mean, std = sim_stats(traj)
    print(f"\n--- {v} ({len(traj)} pairs) ---")
    print(f"min={mn:.6f}  max={mx:.6f}  mean={mean:.6f}  std={std:.6f}")
    # 逐帧
    for i, s in enumerate(traj):
        marker = "  ★" if s < 0.95 else ""
        print(f"  {i:2d}→{i+1:2d}  {s:.6f}{marker}")

# ──── 二. 段间 sim 矩阵 ────
print("\n" + "=" * 100)
print("二. 段间 sim: segment_i末帧 → segment_j首帧")
print("=" * 100)
print(f"{'':>25s}", end="")
for v2 in VIDS:
    print(f"  {v2[-6:]:>8s}", end="")
print()
for v1 in VIDS:
    print(f"{v1:>25s}", end="")
    for v2 in VIDS:
        s = signature_similarity(all_sigs[v1][-1], all_sigs[v2][0])
        print(f"  {s:8.6f}", end="")
    print()

# ──── 三. 关键跳幅 (段 i 末 → 段 i+1 首, 相邻段) ────
print("\n" + "=" * 100)
print("三. 关键跳幅 (末→首)")
print("=" * 100)
# 按录制顺序: 8/18 两段是 i=0→1; 9/19 两段是 i=2→3; 跨日期 1→2
order_pairs = [
    ("9/18 #1→#2 (同人不同状态?)", VIDS[0], VIDS[1]),
    ("9/19 #3→#4 (同人不同状态?)", VIDS[2], VIDS[3]),
    ("跨日期 #2→#3", VIDS[1], VIDS[2]),
    ("跨日期 #1→#4", VIDS[0], VIDS[3]),
]
for desc, va, vb in order_pairs:
    s = signature_similarity(all_sigs[va][-1], all_sigs[vb][0])
    print(f"  {desc:40s}  sim = {s:.6f}   Δ = {1-s:.6f}")

# ──── 四. 段内最大跳 (排除静音边界第一帧) ────
print("\n" + "=" * 100)
print("四. 段内最大跳幅 (不含帧0→1静音起)")
print("=" * 100)
for v in VIDS:
    traj = frame_trajectory(all_sigs[v])
    inner = traj[1:]  # 去掉 0→1
    if inner:
        mn_i = min(inner)
        idx_i = traj.index(mn_i)
        print(f"  {v}: min_inner={mn_i:.6f} (frame {idx_i}→{idx_i+1})  跳幅Δ={1-mn_i:.6f}")
