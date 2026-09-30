# -*- coding: utf-8 -*-
"""
环境反向 61-65 v7: 直跑，带内存管理
重放 1-60 轮后输出 61-65
内存优化:
  1. 将 痕文件 重定向到 /dev/null 减少磁盘IO
  2. 每1000次迭代清空 eng.日志 释放内存
"""
import os, sys, math, random, time

src_path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py"
with open(src_path, encoding="utf-8") as f:
    src = f.read()
ns = {}
exec(compile(src, src_path, "exec"), ns)
Body = ns["身体"]

if '痕文件' in ns:
    ns['痕文件'] = "/dev/null"

def 光基(t):
    return (0.8 if (t // 100) % 2 == 0 else 0.2) \
           + (0.6 + 0.4 * math.sin(2 * math.pi * t / 1000)) * 0.05 \
           + max(0, math.sin(2 * math.pi * (t - 250) / 1000)) * 0.05

def 声基(t, rm):
    v = 0.2; 突 = 50 + (t * 37 + 13) % 150
    if t % max(1, 突) == 0: v += 0.5
    return v + rm.uniform(-0.1, 0.1)

def 力基(t, rm):
    v = 0.1 * abs(math.sin(2 * math.pi * t / 4)) + 0.3
    if t % 5 == 0: v += 0.1
    return v + rm.uniform(-0.05, 0.05)

温基 = 0.35

def 取状态(eng):
    return sum(eng.激活), max(eng.激活), len(eng.筋树), any(v > 0.5 for v in eng.激活), any(math.isnan(v) for v in eng.激活)

t0 = time.time()
out = []
out.append("环境反向 接跑 第 61–65 轮 | 种子=7 不重置状态 | 方向随机 + 幅度±20%")
out.append("=" * 70)
out.append("")

eng = Body(种子=7)
rm_noise = random.Random(123)
轮号 = 0

def 跑固定轮(patterns):
    global 轮号, eng, rm_noise
    for 轮名, 反演 in patterns:
        轮号 += 1
        for it in range(10000):
            t = eng.时刻
            s_光 = 光基(t); s_声 = 声基(t, rm_noise); s_力 = 力基(t, rm_noise); s_温 = 温基
            if 反演:
                eng.环境["光"] = 502 - (1 + 500 * s_光); eng.环境["声"] = 100 - (25 + 50 * s_声)
                eng.环境["力"] = 32 - (1 + 30 * s_力); eng.环境["温"] = 591 - (288 + 15 * s_温)
            else:
                eng.环境["光"] = 1 + 500 * s_光; eng.环境["声"] = 25 + 50 * s_声
                eng.环境["力"] = 1 + 30 * s_力; eng.环境["温"] = 288 + 15 * s_温
            eng.推进()
            if it % 1000 == 0 and hasattr(eng, '日志'):
                eng.日志.clear()

跑固定轮([("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)])
跑固定轮([("反向", True), ("正向", False), ("反向", True), ("正向", False), ("反向", True),
         ("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)])

总15, 最大15, 筋15, _, _ = 取状态(eng)
out.append(f"[验证] 轮15: 总={总15:.4f} 最大={最大15:.6f} 筋={筋15} | 参考: -0.0221, 0.494373, 2203")
out.append("")

decider = random.Random(456)
for i in range(45):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    for it in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp; s_声 = 声基(t, rm_noise) * amp; s_力 = 力基(t, rm_noise) * amp; s_温 = 温基 * amp
        if rev:
            eng.环境["光"] = 502 - (1 + 500 * s_光); eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力); eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光; eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力; eng.环境["温"] = 288 + 15 * s_温
        eng.推进()
        if it % 1000 == 0 and hasattr(eng, '日志'):
            eng.日志.clear()

总60, 最大60, 筋60, _, _ = 取状态(eng)
out.append(f"[验证] 轮60: 总={总60:.4f} 最大={最大60:.6f} 筋={筋60} | 参考: -0.2232, 0.412294, 2491")
out.append("")

out.append("— 第 61–65 轮 —")
out.append("")
for i in range(5):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    for it in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp; s_声 = 声基(t, rm_noise) * amp; s_力 = 力基(t, rm_noise) * amp; s_温 = 温基 * amp
        if rev:
            eng.环境["光"] = 502 - (1 + 500 * s_光); eng.环境["声"] = 100 - (25 + 50 * s_声)
            eng.环境["力"] = 32 - (1 + 30 * s_力); eng.环境["温"] = 591 - (288 + 15 * s_温)
        else:
            eng.环境["光"] = 1 + 500 * s_光; eng.环境["声"] = 25 + 50 * s_声
            eng.环境["力"] = 1 + 30 * s_力; eng.环境["温"] = 288 + 15 * s_温
        eng.推进()
        if it % 1000 == 0 and hasattr(eng, '日志'):
            eng.日志.clear()
    总, 最大, 筋, 亮, nan = 取状态(eng)
    方向 = "反向" if rev else "正向"
    out.append(f"第{轮号}轮 ({方向}) amp={amp:.4f}: 总={总:.4f} 最大={最大:.6f} 筋={筋} 亮={'有' if 亮 else '无'} NaN={'有' if nan else '无'}")
    with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/_partial_61_65.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
        f.flush()
        os.fsync(f.fileno())

out.append("")
out.append("=" * 70)
out.append(f"跑完. 用时 {time.time() - t0:.1f} 秒")

path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_61_65.txt"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
    f.flush()
    os.fsync(f.fileno())
print(f"WROTE {path}  time={time.time()-t0:.1f}s")
