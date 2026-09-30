# -*- coding: utf-8 -*-
"""
环境反向 接跑 10 轮（第 6–15 轮）
- 同一个身体对象、不重置状态
- 源码原版只 read + compile（内存执行），不写回文件
- 反向 = 环境反演 env_rev = 2*offset + scale - env_fwd
- 前 5 轮作为状态预热段（不输出），第 6–15 轮输出
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

eng = Body(种子=7)
痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/envrev_more_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)

def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    print(msg, flush=True)

log(f"START pid={os.getpid()}")

rm_noise = random.Random(123)   # 与前 5 轮同噪声序列起点

# 15 轮：前 5 轮仅推进（状态预热），后 10 轮每轮输出
轮组 = [("正向", False), ("反向", True)] * 7 + [("正向", False)]

轮号 = 0
snap_buf = []
for 轮名, 反演 in 轮组:
    轮号 += 1
    log(f"轮{轮号} ({轮名}) 反演={反演} 开始 t={eng.时刻}")
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

    if 轮号 >= 6:
        总 = sum(eng.激活)
        最大 = max(eng.激活)
        节点亮 = any(v > 0.5 for v in eng.激活)
        nan = any(math.isnan(v) for v in eng.激活)
        筋 = len(eng.筋树)
        line = (f"第{轮号}轮 ({轮名}): 总={总:.4f} 最大={最大:.6f} 筋={筋} "
                f"亮={'有' if 节点亮 else '无'} NaN={'有' if nan else '无'}")
        snap_buf.append(line)
        log(line)
    if 轮号 in (1, 5, 10, 15):
        log(f"  中间 checkpoint 轮{轮号} t={eng.时刻} 筋树={len(eng.筋树)}")

log("开始写盘")
path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_10more.txt"
with open(path, "w", encoding="utf-8", buffering=1) as f:
    f.write("环境反向 接跑 10 轮（第 6–15 轮）种子=7 不重置状态\n")
    f.write("=" * 70 + "\n\n")
    for line in snap_buf:
        f.write(line + "\n")
    f.write("\n" + "=" * 70 + "\n")
    f.write(f"跑完. 用时 {time.time() - t0:.1f} 秒\n")
    f.flush()
    os.fsync(f.fileno())

log(f"WROTE {path} time={time.time() - t0:.1f}s")
