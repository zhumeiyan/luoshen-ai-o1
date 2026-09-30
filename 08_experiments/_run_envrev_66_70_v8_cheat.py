# -*- coding: utf-8 -*-
"""
环境反向 66-70 v8: "骗"跑法
不按固定节奏，按身体状态：
- 上轮筋涨了 → 她舒服了 → 抽走，给反向
- 上轮筋跌了 → 她饿了 → 给一口，给正向
目的：看她会不会因为"够不到"，自己长出什么
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

def 跑一轮(eng, rm_noise, rev, amp):
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

t0 = time.time()
out = []
out.append("环境反向 第 66–70 轮 | 骗跑法：上轮涨→反向抽走，上轮跌→正向给一口")
out.append("=" * 70)
out.append("")

eng = Body(种子=7)
rm_noise = random.Random(123)
decider = random.Random(456)  # 延续原有随机流
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

# 16-65: 已有的40轮随机（延续decider流）
for i in range(40):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    跑一轮(eng, rm_noise, rev, amp)

总65, 最大65, 筋65, _, _ = 取状态(eng)
out.append(f"[验证] 轮65: 总={总65:.4f} 最大={最大65:.6f} 筋={筋65} | 参考: -0.1301, 0.492911, 2288")
out.append("")

# === 骗跑法开始 ===
# 轮65: 筋2288 (上轮64筋2161, 涨了+127), 所以她"舒服了" → 抽走，给反向
out.append("— 第 66–70 轮（骗跑法） —")
out.append("")

prev_筋 = 筋65
# 轮65时筋比轮64涨了(2288>2161) → 她"舒服了" → 反向抽走
第一轮反向 = True

for i in range(5):
    轮号 += 1
    # 决定方向：上轮涨→反向(抽走)，上轮跌→正向(给一口)
    if i == 0:
        rev = 第一轮反向
    else:
        rev = (筋变化 > 0)  # 上轮涨了就反向，跌了就正向
    
    amp = decider.uniform(0.8, 1.2)
    跑一轮(eng, rm_noise, rev, amp)
    
    总, 最大, 筋, 亮, nan = 取状态(eng)
    筋变化 = 筋 - prev_筋
    方向 = "反向" if rev else "正向"
    动作 = "抽走" if rev else "给一口"
    out.append(f"轮{轮号}: {方向}({动作}) amp={amp:.4f} 总={总:.4f} 最大={最大:.6f} 筋={筋} 筋变={筋变化:+5} 亮={'有' if 亮 else '无'} NaN={'有' if nan else '无'}")
    
    with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/_partial_66_70_cheat.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
        f.flush()
        os.fsync(f.fileno())
    
    prev_筋 = 筋

out.append("")
out.append("=" * 70)
out.append(f"跑完. 用时 {time.time() - t0:.1f} 秒")

path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_66_70_cheat.txt"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
    f.flush()
    os.fsync(f.fileno())
print(f"WROTE {path}  time={time.time()-t0:.1f}s")
