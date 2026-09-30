# -*- coding: utf-8 -*-
"""
环境反向接跑 第 36–45 轮 — 从第35轮断点续跑
种子=7 不重置状态，源码原版，方向随机 + 幅度±20%
流程：先重放 1–35 轮重建同一身体状态，再跑 36–45 轮输出。
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

def 跑一轮(eng, t, rm, rev, amp, decider=None):
    for _ in range(10000):
        tt = eng.时刻
        s_光 = 光基(tt) * amp; s_声 = 声基(tt, rm) * amp; s_力 = 力基(tt, rm) * amp; s_温 = 温基 * amp
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
    return eng

def 取状态(eng):
    总 = sum(eng.激活)
    最大 = max(eng.激活)
    筋 = len(eng.筋树)
    亮 = any(v > 0.5 for v in eng.激活)
    nan = any(math.isnan(v) for v in eng.激活)
    return 总, 最大, 筋, 亮, nan

t0 = time.time()
out = []
out.append("环境反向 接跑 第 36–45 轮 | 种子=7 不重置状态 | 方向随机 + 幅度±20%")
out.append("=" * 70)
out.append("")

eng = Body(种子=7)
痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/envrev_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)

rm_noise = random.Random(123)

# ── 重放 1–5 轮（正反正反正）──
轮号 = 0
patterns_1_5 = [("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)]
for 轮名, 反演 in patterns_1_5:
    轮号 += 1
    amp = 1.0
    eng = 跑一轮(eng, eng.时刻, rm_noise, 反演, amp)

# ── 重放 6–15 轮（正反交替，不重置）──
patterns_6_15 = [("反向", True), ("正向", False), ("反向", True), ("正向", False), ("反向", True),
                 ("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)]
for 轮名, 反演 in patterns_6_15:
    轮号 += 1
    amp = 1.0
    eng = 跑一轮(eng, eng.时刻, rm_noise, 反演, amp)

# ── 验证 15 轮 checkpoint ──
总15, 最大15, 筋15, 亮15, nan15 = 取状态(eng)
out.append(f"[复现验证] 轮15 checkpoint: 总={总15:.4f} 最大={最大15:.6f} 筋={筋15} 亮={'有' if 亮15 else '无'} NaN={'有' if nan15 else '无'} | 参考: 总≈-0.0221 最大≈0.494373 筋=2203")
out.append("")

# ── 重放 16–25 轮（方向随机 + 幅度±20%）──
decider = random.Random(456)
for i in range(10):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    eng = 跑一轮(eng, eng.时刻, rm_noise, rev, amp)

# ── 验证 25 轮 checkpoint ──
总25, 最大25, 筋25, 亮25, nan25 = 取状态(eng)
out.append(f"[复现验证] 轮25 checkpoint: 总={总25:.4f} 最大={最大25:.6f} 筋={筋25} 亮={'有' if 亮25 else '无'} NaN={'有' if nan25 else '无'} | 参考: 总≈-0.2179 最大≈0.554730 筋=2128 亮=有")
out.append("")

# ── 重放 26–35 轮（同一decider流继续）──
for i in range(10):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    eng = 跑一轮(eng, eng.时刻, rm_noise, rev, amp)

# ── 验证 35 轮 checkpoint ──
总35, 最大35, 筋35, 亮35, nan35 = 取状态(eng)
out.append(f"[复现验证] 轮35 checkpoint: 总={总35:.4f} 最大={最大35:.6f} 筋={筋35} 亮={'有' if 亮35 else '无'} NaN={'有' if nan35 else '无'} | 参考: 总≈-0.0588 最大≈0.403646 筋=2656")
out.append("")

# ── 跑 36–45 轮（同一decider流继续）──
out.append("— 第 36–45 轮 —")
out.append("")
for i in range(10):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    eng = 跑一轮(eng, eng.时刻, rm_noise, rev, amp)

    总, 最大, 筋, 亮, nan = 取状态(eng)
    方向 = "反向" if rev else "正向"
    out.append(f"第{轮号}轮 ({方向}) amp={amp:.4f}: 总={总:.4f} 最大={最大:.6f} 筋={筋} 亮={'有' if 亮 else '无'} NaN={'有' if nan else '无'}")

out.append("")
out.append("=" * 70)
out.append(f"跑完. 用时 {time.time() - t0:.1f} 秒")

path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_36_45.txt"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
    f.flush()
    os.fsync(f.fileno())
print(f"WROTE {path}  time={time.time() - t0:.1f}s")
