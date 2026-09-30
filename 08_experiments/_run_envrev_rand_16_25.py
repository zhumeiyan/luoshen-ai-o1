# -*- coding: utf-8 -*-
"""
环境反向 接跑 第 16–25 轮
- 同一身体（种子=7），不重置状态
- 源码原版只 read + compile（内存执行），不写回文件
- 反向 = 环境反演 env_rev = 2*offset + scale - env_fwd
- 1-15 轮 精确复现（方向交替 + 幅度=1.0），第 15 轮 checkpoint 验证复现正确
- 第 16-25 轮：每轮方向 50/50 随机，幅度 uniform(0.8, 1.2)
- 环境信号先乘振幅因子 amp，引擎内部 clamp
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
    burst = 50 + (t * 37 + 13) % 150
    if t % max(1, burst) == 0:
        v += 0.5
    return v + rm.uniform(-0.1, 0.1)

def 力基(t, rm):
    v = 0.1 * abs(math.sin(2 * math.pi * t / 4)) + 0.3
    if t % 5 == 0:
        v += 0.1
    return v + rm.uniform(-0.05, 0.05)

温基 = 0.35


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


t0 = time.time()
log(f"START pid={os.getpid()}")

eng = Body(种子=7)
痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/envrev_rand_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)

rm_noise = random.Random(123)   # 环境噪声引擎（复现 1-15 轮 + 继续给 16-25 轮用）
decider  = random.Random(456)   # 独立流，仅用于 16-25 轮的方向/幅度决策

out = []
out.append("环境反向 接跑 第 16–25 轮 | 种子=7 不重置状态 | 方向随机 + 幅度 ±20%")
out.append("=" * 70)
out.append("")

# ===== 第 1–15 轮 精确复现 =====
轮号 = 0
for i in range(15):
    rev = (i % 2 == 1)
    轮号 += 1
    log(f"REP 轮{轮号} rev={rev} start t={eng.时刻}")
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t); s_声 = 声基(t, rm_noise); s_力 = 力基(t, rm_noise); s_温 = 温基
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
    if 轮号 == 15:
        总 = sum(eng.激活); 最大 = max(eng.激活)
        亮 = any(v > 0.5 for v in eng.激活); nan = any(math.isnan(v) for v in eng.激活)
        筋 = len(eng.筋树)
        ck = (f"[复现验证] 轮15 checkpoint: 总={总:.4f} 最大={最大:.6f} 筋={筋} "
              f"亮={'有' if 亮 else '无'} NaN={'有' if nan else '无'} | 参考: 总≈-0.0221 最大≈0.494373 筋=2203")
        log(ck); out.append(ck); out.append("")

# ===== 第 16–25 轮 随机方向和幅度 =====
for i in range(10):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    轮名 = "反向" if rev else "正向"
    log(f"RAND 轮{轮号} {轮名} amp={amp:.4f} start t={eng.时刻}")
    for _ in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp; s_声 = 声基(t, rm_noise) * amp
        s_力 = 力基(t, rm_noise) * amp; s_温 = 温基 * amp
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
    总 = sum(eng.激活); 最大 = max(eng.激活)
    亮 = any(v > 0.5 for v in eng.激活); nan = any(math.isnan(v) for v in eng.激活)
    筋 = len(eng.筋树)
    line = (f"第{轮号}轮 ({轮名}) amp={amp:.4f}: 总={总:.4f} 最大={最大:.6f} 筋={筋} "
            f"亮={'有' if 亮 else '无'} NaN={'有' if nan else '无'}")
    log(line); out.append(line)

out.append("")
out.append("=" * 70)
out.append(f"跑完. 用时 {time.time() - t0:.1f} 秒")

log("WRITING FILE...")
path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_16_25.txt"
with open(path, "w", encoding="utf-8", buffering=1) as f:
    f.write("\n".join(out))
    f.flush()
    os.fsync(f.fileno())
log(f"WROTE {path} time={time.time() - t0:.1f}s")
