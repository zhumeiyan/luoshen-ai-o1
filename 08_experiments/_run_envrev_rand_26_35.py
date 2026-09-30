# -*- coding: utf-8 -*-
"""
环境反向接跑 第 26–35 轮 — 从第25轮断点续跑
种子=7 不重置状态，源码原版，方向随机 + 幅度±20%
流程：先重放 1–25 轮重建同一身体状态，再跑 26–35 轮输出。
"""
import os, sys, math, random, time

src_path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py"
with open(src_path, encoding="utf-8") as f:
    src = f.read()
ns = {}
exec(compile(src, src_path, "exec"), ns)
Body = ns["身体"]


def 光基(t):
    return (0.8 if (t // 100) % 2 == 0 else 0.2) \
           + (0.6 + 0.4 * math.sin(2 * math.pi * t / 1000)) * 0.05 \
           + max(0, math.sin(2 * math.pi * (t - 250) / 1000)) * 0.05

def 声基(t, rm):
    v = 0.2
    突 = 50 + (t * 37 + 13) % 150
    if t % max(1, 突) == 0:
        v += 0.5
    return v + rm.uniform(-0.1, 0.1)

def 力基(t, rm):
    v = 0.1 * abs(math.sin(2 * math.pi * t / 4)) + 0.3
    if t % 5 == 0:
        v += 0.1
    return v + rm.uniform(-0.05, 0.05)

温基 = 0.35

t0 = time.time()
out = []
out.append("环境反向 接跑 第 26–35 轮 | 种子=7 不重置状态 | 方向随机 + 幅度±20%")
out.append("=" * 70)
out.append("")

eng = Body(种子=7)
痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/envrev_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)

rm_noise = random.Random(123)

# ── 重放 1–5 轮（正反正反正）──
轮号 = 0
for 轮名, 反演 in [("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)]:
    轮号 += 1
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t); s_声 = 声基(t, rm_noise); s_力 = 力基(t, rm_noise); s_温 = 温基
        if 反演:
            eng.环境["光"] = 502 - (1 + 500 * s_光)
            eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力)
            eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光
            eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力
            eng.环境["温"] = 288 + 15 * s_温
        eng.推进()

# ── 重放 6–15 轮（正反交替，不重置）──
for 轮名, 反演 in [("反向", True), ("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)]:
    轮号 += 1
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t); s_声 = 声基(t, rm_noise); s_力 = 力基(t, rm_noise); s_温 = 温基
        if 反演:
            eng.环境["光"] = 502 - (1 + 500 * s_光)
            eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力)
            eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光
            eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力
            eng.环境["温"] = 288 + 15 * s_温
        eng.推进()

# ── 验证 15 轮 checkpoint ──
总15 = sum(eng.激活)
最大15 = max(eng.激活)
筋15 = len(eng.筋树)
out.append(f"[复现验证] 轮15 checkpoint: 总={总15:.4f} 最大={最大15:.6f} 筋={筋15} | 参考: 总≈-0.0221 最大≈0.494373 筋=2203")
out.append("")

# ── 重放 16–25 轮（方向随机 + 幅度±20%）──
decider = random.Random(456)
for i in range(10):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp; s_声 = 声基(t, rm_noise) * amp; s_力 = 力基(t, rm_noise) * amp; s_温 = 温基 * amp
        if rev:
            eng.环境["光"] = 502 - (1 + 500 * s_光)
            eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力)
            eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光
            eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力
            eng.环境["温"] = 288 + 15 * s_温
        eng.推进()

# ── 验证 25 轮 checkpoint ──
总25 = sum(eng.激活)
最大25 = max(eng.激活)
筋25 = len(eng.筋树)
亮25 = any(v > 0.5 for v in eng.激活)
out.append(f"[复现验证] 轮25 checkpoint: 总={总25:.4f} 最大={最大25:.6f} 筋={筋25} 亮={'有' if 亮25 else '无'} NaN=无")
out.append("")

# ── 跑 26–35 轮（同一 decider 流继续）──
out.append("— 第 26–35 轮 —")
out.append("")
for i in range(10):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp; s_声 = 声基(t, rm_noise) * amp; s_力 = 力基(t, rm_noise) * amp; s_温 = 温基 * amp
        if rev:
            eng.环境["光"] = 502 - (1 + 500 * s_光)
            eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力)
            eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光
            eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力
            eng.环境["温"] = 288 + 15 * s_温
        eng.推进()

    总 = sum(eng.激活)
    最大 = max(eng.激活)
    筋 = len(eng.筋树)
    亮 = any(v > 0.5 for v in eng.激活)
    nan = any(math.isnan(v) for v in eng.激活)
    方向 = "反向" if rev else "正向"
    out.append(f"第{轮号}轮 ({方向}) amp={amp:.4f}: 总={总:.4f} 最大={最大:.6f} 筋={筋} 亮={'有' if 亮 else '无'} NaN={'有' if nan else '无'}")

out.append("")
out.append("=" * 70)
out.append(f"跑完. 用时 {time.time() - t0:.1f} 秒")

path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_26_35.txt"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
    f.flush()
    os.fsync(f.fileno())
print(f"WROTE {path}  time={time.time() - t0:.1f}s")
