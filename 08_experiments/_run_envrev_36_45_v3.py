# -*- coding: utf-8 -*-
"""
环境反向 36-45 v3: 直跑，无 checkpoint，带内存管理
重放 1-35 轮后输出 36-45
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

# 禁用trace写入: 交换模块级痕文件变量
import importlib
body_mod = sys.modules[Body.__module__] if hasattr(Body, '__module__') else None
# 实际上Body是exec创建的,没有__module__,我们直接重定向文件级别的痕文件
# 通过猴子补丁:替换open("/path/to/痕文件")为open("/dev/null")
# 更简单:在exec前替换源码中的痕文件路径
# 已在exec后处理:直接猴子patch 身体稳态 模块的痕文件
# 由于exec创建的ns不是标准模块,我们在ns层面找不到痕文件
# 最简:每次推进后清空日志列表

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
out.append("环境反向 接跑 第 36–45 轮 | 种子=7 不重置状态 | 方向随机 + 幅度±20%")
out.append("=" * 70)
out.append("")

eng = Body(种子=7)

# 关键优化1: 将痕文件重定向到 /dev/null
# 身体稳态.py的痕文件是模块级变量,在exec的ns中
# 我们直接从ns中获取并修改
ns_mod = sys.modules.get('__main__', None)
# 由于身体稳态.py中痕文件 = "..." 是在模块顶层,exec中会成为ns的全局变量
# 我们通过修改ns来影响
# 更简单: 利用Body类内部的痕文件变量通过hp机制影响
# Body的模块是 __main__ (因为我们用exec在一个脚本里)
# 让我们尝试直接替换open调用的路径

# 实际上最简单: 将痕文件名改为/dev/null通过替换ns
if '痕文件' in ns:
    ns['痕文件'] = "/dev/null"

rm_noise = random.Random(123)

轮号 = 0

def 跑固定轮(patterns, label=""):
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
            # 关键优化2: 每1000次迭代清空日志
            if it % 1000 == 0 and hasattr(eng, '日志'):
                eng.日志.clear()

# 1-5
跑固定轮([("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)])

# 6-15
跑固定轮([("反向", True), ("正向", False), ("反向", True), ("正向", False), ("反向", True),
         ("正向", False), ("反向", True), ("正向", False), ("反向", True), ("正向", False)])

总15, 最大15, 筋15, 亮15, _ = 取状态(eng)
out.append(f"[验证] 轮15: 总={总15:.4f} 最大={最大15:.6f} 筋={筋15} | 参考: -0.0221, 0.494373, 2203")
out.append("")

# 16-35 随机（同 decider 流继续）
decider = random.Random(456)
for i in range(20):
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

总35, 最大35, 筋35, 亮35, _ = 取状态(eng)
out.append(f"[验证] 轮35: 总={总35:.4f} 最大={最大35:.6f} 筋={筋35} 亮={'有' if 亮35 else '无'} | 参考: -0.0588, 0.403646, 2656")
out.append("")

# 36-45 随机
out.append("— 第 36–45 轮 —")
out.append("")
for i in range(10):
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
    # 每轮后写盘（防掐）
    mid_path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/_partial_36_45.txt"
    with open(mid_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
        f.flush()
        os.fsync(f.fileno())

out.append("")
out.append("=" * 70)
out.append(f"跑完. 用时 {time.time() - t0:.1f} 秒")

path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_36_45.txt"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
    f.flush()
    os.fsync(f.fileno())
print(f"WROTE {path}  time={time.time()-t0:.1f}s")
