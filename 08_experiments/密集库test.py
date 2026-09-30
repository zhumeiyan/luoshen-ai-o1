import sys, os, json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from 矩阵.跨模态 import 归一化N维
from modality_specific_test import visual_specific, auditory_specific

GRAY = Path(__file__).parent / "luoshenfu_home" / "gray"
PENDING = GRAY / "pending" / "dense_lib_test"
PENDING.mkdir(parents=True, exist_ok=True)

# ── 摆差异：不称异常 ──
def raw_diffs(results):
    diffs = []
    seed_set_v = set()
    seed_set_a = set()
    for lvl in results:
        for t in lvl["trials"]:
            seed_set_v.add(t.get("seed_v"))
            seed_set_a.add(t.get("seed_a"))
    only_v = sorted(set(seed_set_v) - set(seed_set_a))
    only_a = sorted(set(seed_set_a) - set(seed_set_v))
    both = sorted(set(seed_set_v) & set(seed_set_a))
    diffs.append(("seed_intersection", {"v_only_first3": only_v[:3], "a_only_first3": only_a[:3], "both_count": len(both)}))
    for lvl in results:
        for t in lvl["trials"]:
            if "[事实" in str(t.get("mark", "")):
                diffs.append(("mark_contains_事实", t["noise"], t["trial"], t["mark"]))
    return diffs

def interp(a, b, t):
    rows = len(a); cols = len(a[0])
    return [[a[r][c]*(1-t)+b[r][c]*t for c in range(cols)] for r in range(rows)]

def _build_thin_vis(offset, amp):
    rows, cols = 8, 64
    m = [[0.0]*cols for _ in range(rows)]
    for c in range(cols):
        for r in range(2,5):
            d = abs(c-offset)/12.0
            m[r][c] = max(0,(1-d))*amp
        for r in range(5,8): m[r][c]=200
    return m

def _build_double_vis():
    rows, cols = 8, 64
    m = [[0.0]*cols for _ in range(rows)]
    for c in range(cols):
        for r in range(2,5):
            d1=abs(c-20)/10.0; d2=abs(c-44)/10.0
            m[r][c]=max(0,(1-d1))*5000+max(0,(1-d2))*4000
        for r in range(5,8): m[r][c]=200
    return m

def _build_thin_aud(freq_shift):
    rows, cols = 8, 64
    m = [[0.0]*cols for _ in range(rows)]
    for c in range(6,50):
        for r in range(3): m[r][c]=800+freq_shift
        for r in range(3,5): m[r][c]=1300+freq_shift
        for r in range(5,8): m[r][c]=2600+freq_shift
    return m

def _build_double_aud(freq_shift):
    rows, cols = 8, 64
    m = [[0.0]*cols for _ in range(rows)]
    for c in range(6,50):
        for r in range(3): m[r][c]=750+freq_shift
        for r in range(3,5): m[r][c]=1400+freq_shift
        for r in range(5,8): m[r][c]=2800+freq_shift
    return m

# 解析轨迹名 → 起点/终点帧
def _parse_target(name):
    parts = name.replace("_to_","|").split("|")
    a_name, b_name = parts[0], parts[1]
    if a_name.startswith("classic_"):
        b = a_name[8:]
        a_vis = {"thin": _build_thin_vis(32,6000), "double": _build_double_vis(), "wide": _build_thin_vis(32,4500)}.get(b, _build_thin_vis(32,6000))
        a_aud = {"thin": _build_thin_aud(0), "double": _build_double_aud(0), "wide": _build_thin_aud(0)}.get(b, _build_thin_aud(0))
    elif a_name.startswith("off"):
        off_val = int(a_name[3:])
        a_vis = _build_thin_vis(off_val, 6000)
        a_aud = _build_thin_aud((32-off_val)*5)
    else:
        a_vis = {"thin": _build_thin_vis(32,6000), "double": _build_double_vis(), "wide": _build_thin_vis(32,4500)}.get(a_name, _build_thin_vis(32,6000))
        a_aud = {"thin": _build_thin_aud(0), "double": _build_double_aud(0), "wide": _build_thin_aud(0)}.get(a_name, _build_thin_aud(0))
    if b_name.startswith("classic_"):
        b = b_name[8:]
        b_vis = {"thin": _build_thin_vis(32,6000), "double": _build_double_vis(), "wide": _build_thin_vis(32,4500)}.get(b, _build_thin_vis(32,6000))
        b_aud = {"thin": _build_thin_aud(0), "double": _build_double_aud(0), "wide": _build_thin_aud(0)}.get(b, _build_thin_aud(0))
    elif b_name.startswith("off"):
        off_val = int(b_name[3:])
        b_vis = _build_thin_vis(off_val, 6000)
        b_aud = _build_thin_aud((32-off_val)*5)
    elif b_name == "double":
        b_vis = _build_double_vis(); b_aud = _build_double_aud(0)
    else:
        b_vis = {"thin": _build_thin_vis(32,6000), "double": _build_double_vis(), "wide": _build_thin_vis(32,4500)}.get(b_name, _build_thin_vis(32,6000))
        b_aud = {"thin": _build_thin_aud(0), "double": _build_double_aud(0), "wide": _build_thin_aud(0)}.get(b_name, _build_thin_aud(0))
    return a_vis, a_aud, b_vis, b_aud

def add_noise(frames, level, seed, amp):
    rng = np.random.RandomState(seed)
    std = level * amp
    out = []
    for f in frames:
        n = rng.normal(0, std, (8,64))
        out.append([[f[r][c]+n[r][c] for c in range(64)] for r in range(8)])
    return out

N = 12  # 帧数
offsets = [18, 22, 26, 30, 34, 38, 42, 46]

library = []
for off in offsets:
    vis_f = [interp(_build_thin_vis(32, 6000), _build_thin_vis(off, 6000), t/max(1,N-1)) for t in range(N)]
    aud_f = [interp(_build_thin_aud(0), _build_thin_aud((32-off)*5), t/max(1,N-1)) for t in range(N)]
    library.append({"name": f"thin_to_off{off}",
        "vis_raw": [visual_specific(f) for f in vis_f],
        "aud_raw": [auditory_specific(f) for f in aud_f]})
for off in offsets:
    vis_f = [interp(_build_thin_vis(off, 6000), _build_double_vis(), t/max(1,N-1)) for t in range(N)]
    aud_f = [interp(_build_thin_aud((32-off)*5), _build_double_aud(0), t/max(1,N-1)) for t in range(N)]
    library.append({"name": f"off{off}_to_double",
        "vis_raw": [visual_specific(f) for f in vis_f],
        "aud_raw": [auditory_specific(f) for f in aud_f]})
for a,b in [("thin","wide"),("thin","double"),("wide","double"),("thin","off32")]:
    va = _build_thin_vis(32, 6000) if a=="off32" else {"thin": _build_thin_vis(32,6000),"double": _build_double_vis(),"wide": _build_thin_vis(32,4500)}[a]
    vb = _build_thin_vis(32, 6000) if b=="off32" else {"thin": _build_thin_vis(32,6000),"double": _build_double_vis(),"wide": _build_thin_vis(32,4500),"off32": _build_thin_vis(32,6000)}[b]
    aa = _build_thin_aud(0) if a=="off32" else {"thin": _build_thin_aud(0),"double": _build_double_aud(0),"wide": _build_thin_aud(0)}[a]
    ab = _build_thin_aud(0) if b=="off32" else {"thin": _build_thin_aud(0),"double": _build_double_aud(0),"wide": _build_thin_aud(0),"off32": _build_thin_aud(0)}[b]
    vis_f = [interp(va, vb, t/max(1,N-1)) for t in range(N)]
    aud_f = [interp(aa, ab, t/max(1,N-1)) for t in range(N)]
    library.append({"name": f"classic_{a}_to_{b}",
        "vis_raw": [visual_specific(f) for f in vis_f],
        "aud_raw": [auditory_specific(f) for f in aud_f]})
print(f"[待核] 库大小: {len(library)} 条轨迹")

noise_levels = [0.0, 0.05, 0.15, 0.30]
n_trials = 10
prefix_len = 2

def distance(a, b):
    if not a or not b: return float('inf')
    dim = min(len(a), len(b))
    na, nb = 归一化N维(a), 归一化N维(b)
    return sum((na[j]-nb[j])**2 for j in range(dim))**0.5

def match_single(feats, library, key):
    best_name=None; best_dist=float('inf')
    for entry in library:
        if len(entry[key]) < prefix_len: continue
        d = sum(distance(feats[i], entry[key][i]) for i in range(prefix_len))
        d_pair = (d, entry["name"])
        best_pair = (best_dist, best_name)
        ordered = sorted([d_pair, best_pair])
        best_dist, best_name = ordered[0]
    return best_name, best_dist

print(f"[待核] 噪声档位: {noise_levels}, 每档 {n_trials} 次试验")
print("="*70)

results = []

for noise in noise_levels:
    level_results = []
    for trial in range(n_trials):
        target_idx = trial % len(library)
        target = library[target_idx]
        a_vis, a_aud, b_vis, b_aud = _parse_target(target["name"])
        vis_frames = [interp(a_vis, b_vis, t/max(1,N-1)) for t in range(N)]
        aud_frames = [interp(a_aud, b_aud, t/max(1,N-1)) for t in range(N)]

        seed_v = trial * 2 + 1000
        seed_a = trial * 2 + 2000
        vis_frames = add_noise(vis_frames, noise, seed_v, 6000)
        aud_frames = add_noise(aud_frames, noise, seed_a, 2800)

        vis_feats = [visual_specific(f) for f in vis_frames]
        aud_feats = [auditory_specific(f) for f in aud_frames]

        v_name, v_dist = match_single(vis_feats[:prefix_len], library, "vis_raw")
        a_name_match, a_dist = match_single(aud_feats[:prefix_len], library, "aud_raw")

        record = {
            "trial": trial, "noise": noise,
            "seed_v": seed_v, "seed_a": seed_a,
            "target": target["name"],
            "v_name": v_name, "v_dist": round(v_dist,6),
            "a_name": a_name_match, "a_dist": round(a_dist,6),
            "mark": "[待核]"
        }
        level_results.append(record)
        print(f"[待核] {noise*100:>5.0f}% {trial:>3} | v={v_dist:.6f} {v_name[:15]} | a={a_dist:.6f} {a_name_match[:15]} | tgt={target['name'][:15]}")

    results.append({
        "noise": noise,
        "total": len(level_results),
        "trials": level_results,
        "mark": "[待核]"
    })
    print(f"[待核] 共有 {len(level_results)} 条记录")
    print("-"*70)

# ── 摆差异 ──
diffs = raw_diffs(results)
print("\n── 差异对（摆着，不称异常）──")
for d in diffs:
    print(f"  {d[0]}: {d[1] if len(d)==2 else d[1:]}")

# ── 落 gray/pENDING/ ──
ts_str = datetime.now().isoformat(timespec="seconds").replace(":", "-")
pending_file = PENDING / f"{ts_str}-dense_library.json"
out = {
    "id": f"{ts_str}-dense_library",
    "ts": datetime.now().isoformat(timespec="seconds"),
    "source": "dense_library_test.py",
    "title": "库加密测试 (V/A 独立种子, 不标对错, 只摆原始值)",
    "mark": "[待核]",
    "config": {
        "library_size": len(library), "prefix_len": prefix_len,
        "noise_levels": noise_levels, "n_trials": n_trials,
        "N_frames": N, "offsets": offsets,
        "vis_amp": 6000, "aud_amp": 2800,
        "seed_v_formula": "trial*2+1000", "seed_a_formula": "trial*2+2000"
    },
    "raw_diffs": diffs,
    "results": results
}
pending_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[待核] 落盘: {pending_file}")
print("[待核] 未升主链")
