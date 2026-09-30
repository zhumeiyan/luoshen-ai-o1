# -*- coding: utf-8 -*-
"""
环境反向 36-45 轮 — 断点续跑版
- 每 5 轮保存 dill checkpoint
- 每轮追加写结果到文件
- 重启时自动从最后一个 checkpoint 续跑
"""
import os, sys, math, random, time, dill, shutil

src_path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py"
with open(src_path, encoding="utf-8") as f:
    src = f.read()
ns = {}
exec(compile(src, src_path, "exec"), ns)
Body = ns["身体"]

CKPT_PATH = "/mnt/data/catpaw/home/workspace/洛神赋1.1/_ckpt_envrev.pkl"
OUT_PATH  = "/mnt/data/catpaw/home/workspace/洛神赋1.1/_partial_36_45.txt"

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

def 取状态(eng):
    return sum(eng.激活), max(eng.激活), len(eng.筋树), any(v > 0.5 for v in eng.激活), any(math.isnan(v) for v in eng.激活)

# 初始化或加载 checkpoint
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH, "rb") as f:
        state = dill.load(f)
    eng = state["eng"]
    rm_noise = state["rm_noise"]
    decider = state["decider"]
    轮号 = state["轮号"]
    print(f"[续跑] 从轮{轮号}继续 t={eng.时刻}")
else:
    eng = Body(种子=7)
    rm_noise = random.Random(123)
    decider = None  # 1-15 之后才初始化
    轮号 = 0

t0 = time.time()

def 写结果(line):
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

def 保存检查点():
    with open(CKPT_PATH, "wb") as f:
        dill.dump({"eng": eng, "rm_noise": rm_noise, "decider": decider, "轮号": 轮号}, f)
    print(f"[checkpoint] 轮{轮号} t={eng.时刻}")

# 如果刚启动，写标题
if 轮号 == 0 and not os.path.exists(OUT_PATH):
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("环境反向 36-45 | 断点续跑\n" + "=" * 70 + "\n\n")

# ── 重放 1–5 轮（正反正反正）──
patterns_1_5 = [("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)]
for i, (轮名, 反演) in enumerate(patterns_1_5):
    if 轮号 >= i + 1:
        continue
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t); s_声 = 声基(t, rm_noise); s_力 = 力基(t, rm_noise); s_温 = 温基
        if 反演:
            eng.环境["光"] = 502 - (1 + 500 * s_光); eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力); eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光; eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力; eng.环境["温"] = 288 + 15 * s_温
        eng.推进()
    轮号 = i + 1

保存检查点()
print(f"[轮5完成] t={eng.时刻}")

# ── 6–15 ──
patterns_6_15 = [("反向", True), ("正向", False), ("反向", True), ("正向", False), ("反向", True),
                 ("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)]
for i, (轮名, 反演) in enumerate(patterns_6_15):
    this_round = 6 + i
    if 轮号 >= this_round:
        continue
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t); s_声 = 声基(t, rm_noise); s_力 = 力基(t, rm_noise); s_温 = 温基
        if 反演:
            eng.环境["光"] = 502 - (1 + 500 * s_光); eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力); eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光; eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力; eng.环境["温"] = 288 + 15 * s_温
        eng.推进()
    轮号 = this_round

保存检查点()
总15, 最大15, 筋15, 亮15, nan15 = 取状态(eng)
写结果(f"[验证] 轮15: 总={总15:.4f} 最大={最大15:.6f} 筋={筋15} 亮={'有' if 亮15 else '无'} NaN={'有' if nan15 else '无'}")

# ── 16–25 随机 ──
decider = random.Random(456)
for i in range(10):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp; s_声 = 声基(t, rm_noise) * amp; s_力 = 力基(t, rm_noise) * amp; s_温 = 温基 * amp
        if rev:
            eng.环境["光"] = 502 - (1 + 500 * s_光); eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力); eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光; eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力; eng.环境["温"] = 288 + 15 * s_温
        eng.推进()

保存检查点()
总25, 最大25, 筋25, 亮25, _ = 取状态(eng)
写结果(f"[验证] 轮25: 总={总25:.4f} 最大={最大25:.6f} 筋={筋25} 亮={'有' if 亮25 else '无'}")

# ── 26–35 随机 ──
for i in range(10):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp; s_声 = 声基(t, rm_noise) * amp; s_力 = 力基(t, rm_noise) * amp; s_温 = 温基 * amp
        if rev:
            eng.环境["光"] = 502 - (1 + 500 * s_光); eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力); eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光; eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力; eng.环境["温"] = 288 + 15 * s_温
        eng.推进()

保存检查点()
总35, 最大35, 筋35, 亮35, _ = 取状态(eng)
写结果(f"[验证] 轮35: 总={总35:.4f} 最大={最大35:.6f} 筋={筋35} 亮={'有' if 亮35 else '无'}")

# ── 36–45 随机 ──
写结果("\n— 第 36–45 轮 —\n")
for i in range(10):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp; s_声 = 声基(t, rm_noise) * amp; s_力 = 力基(t, rm_noise) * amp; s_温 = 温基 * amp
        if rev:
            eng.环境["光"] = 502 - (1 + 500 * s_光); eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力); eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光; eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力; eng.环境["温"] = 288 + 15 * s_温
        eng.推进()
    总, 最大, 筋, 亮, nan = 取状态(eng)
    方向 = "反向" if rev else "正向"
    写结果(f"第{轮号}轮 ({方向}) amp={amp:.4f}: 总={总:.4f} 最大={最大:.6f} 筋={筋} 亮={'有' if 亮 else '无'} NaN={'有' if nan else '无'}")

写结果(f"\n{'=' * 70}\n跑完. 用时 {time.time() - t0:.1f} 秒")
保存检查点()

# 最终输出也写到目标路径
shutil.copy(OUT_PATH, "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_36_45.txt")
print(f"DONE 轮45. time={time.time()-t0:.1f}s")
