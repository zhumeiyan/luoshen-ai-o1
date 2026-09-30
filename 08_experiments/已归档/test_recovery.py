# [已撤回] 2026-09-19 推演者归档：旧引用，新架构已换
"""test_recovery.py — 跌后恢复基线测试。

判据: 跌完 N 帧后, sim 是否回到切换前基线。
同人换状态 → 跌完回原基线
换真人 → 跌完在新基线重建, 不回原基线
"""
import sys, math, struct
from pathlib import Path
sys.path.insert(0, ".")
from face_signature import aud_grid, full_signature, signature_similarity

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
    return [full_signature(decode_ppm_lum(f), aud_grid(samples, i)) for i, f in enumerate(frames)]

# 载入 4 段
all_sigs = {v: build_signatures(v) for v in VIDS}

# ──── 1. 全量 concatenated sim 序列 ────
print("=" * 100)
print("全量 sim 序列 (concatenated: 213421 | 213427 | 025853 | 025859)")
print("=" * 100)

# 段内 sim 列表(各自独立)
def intra_sigs(sigs):
    return [(i, i+1, signature_similarity(sigs[i], sigs[i+1])) for i in range(len(sigs)-1)]

# 段间 sim: 段_i 末 → 段_j 首
def cross_sigs(sigs_a, sigs_b):
    return signature_similarity(sigs_a[-1], sigs_b[0])

# 构建全局帧偏移
offsets = {}
cursor = 0
for v in VIDS:
    offsets[v] = cursor
    cursor += len(all_sigs[v])

# 每个切换点(全局帧号, 标记)
# 段间切换点 = offsets[VIDS[i]] + len(VIDS[i]) - 1 (末帧)
print("\n切换点位置 (全局帧号):")
for i, v in enumerate(VIDS):
    end_global = offsets[v] + len(all_sigs[v]) - 1
    print(f"  {v}: 全局帧 {offsets[v]}..{end_global}")

# 每个切换点之后 N 帧的 sim 序列
RECOVERY_NS = [5, 10, 20]
SWITCH_POINTS = [
    ("9/18 段1末→213427首", VIDS[0], VIDS[1]),
    ("9/19 段3末→025859首", VIDS[2], VIDS[3]),
    ("跨日期 段2末→025853首", VIDS[1], VIDS[2]),
    ("段1末→025859首", VIDS[0], VIDS[3]),
]

for desc, va, vb in SWITCH_POINTS:
    base_a = all_sigs[va]
    base_b = all_sigs[vb]
    # 基准: 切换前最后一帧对切换后第一帧 sim
    base_sim = signature_similarity(base_a[-1], base_b[0])
    print(f"\n{'=' * 100}")
    print(f"切换: {desc}")
    print(f"基线(末帧{va} → 首帧{vb}) = {base_sim:.6f}")
    # 切换后 N 帧: 取 vb[1..N+1], 每帧算与 vb[0] 的 sim
    for n in RECOVERY_NS:
        print(f"\n  往后 {n} 帧 (与 {vb}[0] 的 sim):")
        for k in range(1, min(n+1, len(base_b))):
            s = signature_similarity(base_b[0], base_b[k])
            print(f"    {vb}[0]→{vb}[{k}]  sim = {s:.6f}  Δ_vs_base = {s - base_sim:+.6f}")

# ──── 2. 异常点 025859 帧43→44, 跌到 0.860 ────
print("\n" + "=" * 100)
print("异常点: 025859 帧43→44 sim=0.859534 (Δ=-0.140466)")
print("=" * 100)
abn = all_sigs["VID_20260919_025859"]
abn_drop_idx = 43  # 帧43→44 异常
abn_baseline = signature_similarity(abn[abn_drop_idx], abn[abn_drop_idx+1])
print(f"基线 sim({abn_drop_idx}→{abn_drop_idx+1}) = {abn_baseline:.6f}")
# 往后 N 帧, 每帧对 abn[44] 的 sim
for n in RECOVERY_NS:
    print(f"\n  往后 {n} 帧 (与 {abn_drop_idx+1} 的 sim):")
    for k in range(2, min(n+2, len(abn))):
        s = signature_similarity(abn[abn_drop_idx+1], abn[abn_drop_idx+k])
        print(f"    [44]→[{44+k-1}]  sim = {s:.6f}  Δ_vs_drop = {s - abn_baseline:+.6f}")

# ──── 3. 同晚 9/18 切换后, 基线恢复测试 ────
print("\n" + "=" * 100)
print("同晚9/18 213421→213427: 切换后恢复")
print("=" * 100)
sA = all_sigs["VID_20260918_213421"]
sB = all_sigs["VID_20260918_213427"]
base_cross = signature_similarity(sA[-1], sB[0])
print(f"切换点 sim = {base_cross:.6f}")
for n in RECOVERY_NS:
    print(f"\n  往后 {n} 帧 (与 213427[0]):")
    for k in range(1, min(n+1, len(sB))):
        s = signature_similarity(sB[0], sB[k])
        print(f"    [0]→[{k}]  sim = {s:.6f}  Δ_vs_cross = {s - base_cross:+.6f}")

# ──── 4. 跨日期换人恢复 ────
print("\n" + "=" * 100)
print("跨日期 213427→025853: 切换后恢复")
print("=" * 100)
sC = all_sigs["VID_20260919_025853"]
base_cross = signature_similarity(sB[-1], sC[0])
print(f"切换点 sim = {base_cross:.6f}")
for n in RECOVERY_NS:
    print(f"\n  往后 {n} 帧 (与 025853[0]):")
    for k in range(1, min(n+1, len(sC))):
        s = signature_similarity(sC[0], sC[k])
        print(f"    [0]→[{k}]  sim = {s:.6f}  Δ_vs_cross = {s - base_cross:+.6f}")

# ──── 同晚9/19 切换后恢复 ────
print("\n" + "=" * 100)
print("同晚9/19 025853→025859: 切换后恢复")
print("=" * 100)
sD = all_sigs["VID_20260919_025859"]
base_cross = signature_similarity(sC[-1], sD[0])
print(f"切换点 sim = {base_cross:.6f}")
for n in RECOVERY_NS:
    print(f"\n  往后 {n} 帧 (与 025859[0]):")
    for k in range(1, min(n+1, len(sD))):
        s = signature_similarity(sD[0], sD[k])
        print(f"    [0]→[{k}]  sim = {s:.6f}  Δ_vs_cross = {s - base_cross:+.6f}")
