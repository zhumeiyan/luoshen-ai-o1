# -*- coding: utf-8 -*-
"""
环境反向 76-80 v10: 骗跑法 + 更激进声脑刺激
改动：
1. 状态机：亮1轮就切WITHDRAW（不再等2轮）
2. 声音更激进：大幅度变化、停顿、突发，模仿人类说话节奏
3. 继续观察声脑响应
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

声脑起点 = 24
声脑终点 = 48

def 取状态(eng):
    总 = sum(eng.激活)
    最大 = max(eng.激活)
    筋 = len(eng.筋树)
    亮 = any(v > 0.5 for v in eng.激活)
    nan = any(math.isnan(v) for v in eng.激活)
    声最大 = max(abs(eng.激活[i]) for i in range(声脑起点, 声脑终点))
    return 总, 最大, 筋, 声最大, 亮, nan

# 更激进的人声模拟：大幅度、有停顿、突发
def 人声(t, 轮内时刻):
    # 短语结构：短-停-长-停-短
    短语位 = (轮内时刻 % 300) / 300  # 300拍一个短语，更快节奏
    
    if 短语位 < 0.15:
        # 第一声：短促
        包络 = 0.9
    elif 短语位 < 0.25:
        包络 = 0.03  # 停顿
    elif 短语位 < 0.55:
        # 第二声：长，有起伏
        起伏 = 0.5 + 0.5 * math.sin(2 * math.pi * 轮内时刻 / 70)
        包络 = 0.7 * 起伏
    elif 短语位 < 0.65:
        包络 = 0.03  # 停顿
    elif 短语位 < 0.85:
        # 第三声：颤音
        颤 = 0.3 + 0.7 * abs(math.sin(2 * math.pi * 轮内时刻 / 20))
        包络 = 0.8 * 颤
    else:
        包络 = 0.02  # 收尾长停
    
    # 三个频率叠加
    w1 = math.sin(2 * math.pi * 轮内时刻 / 100)   # 慢
    w2 = math.sin(2 * math.pi * 轮内时刻 / 41)    # 中
    w3 = math.sin(2 * math.pi * 轮内时刻 / 11)    # 快
    
    原始 = 0.5 + 0.35 * w1 + 0.25 * w2 + 0.15 * w3
    return max(0.01, min(0.99, 原始 * 包络))

def 跑一轮正向(eng, rm_noise, amp, 加声音):
    for it in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp
        if 加声音:
            s_声 = 人声(t, it) * amp
        else:
            s_声 = 0.04 * amp  # 静音
        s_力 = 力基(t, rm_noise) * amp
        s_温 = 温基 * amp
        eng.环境["光"] = 1 + 500 * s_光
        eng.环境["声"] = 25 + 50 * s_声
        eng.环境["力"] = 1 + 30 * s_力
        eng.环境["温"] = 288 + 15 * s_温
        eng.推进()
        if it % 1000 == 0 and hasattr(eng, '日志'):
            eng.日志.clear()

def 跑一轮反向(eng, rm_noise, amp, 加声音):
    for it in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp
        if 加声音:
            s_声 = 人声(t, it) * amp
        else:
            s_声 = 0.04 * amp
        s_力 = 力基(t, rm_noise) * amp
        s_温 = 温基 * amp
        eng.环境["光"] = 502 - (1 + 500 * s_光)
        eng.环境["声"] = 100 - (25 + 50 * s_声)
        eng.环境["力"] = 32 - (1 + 30 * s_力)
        eng.环境["温"] = 591 - (288 + 15 * s_温)
        eng.推进()
        if it % 1000 == 0 and hasattr(eng, '日志'):
            eng.日志.clear()

t0 = time.time()
out = []
out.append("环境反向 76-80 v10 | 骗跑法v2(亮1轮即切)+激进声音")
out.append("=" * 70)
out.append("")

eng = Body(种子=7)
rm_noise = random.Random(123)
decider = random.Random(456)
轮号 = 0

def 跑固定轮(patterns):
    global 轮号, eng, rm_noise
    for 轮名, 反演 in patterns:
        轮号 += 1
        for it in range(10000):
            t = eng.时刻
            s_光 = 光基(t); s_声 = 声基(t, rm_noise); s_力 = 力基(t, rm_noise); s_温 = 温基
            if 反演:
                eng.环境["光"]=502-(1+500*s_光); eng.环境["声"]=100-(25+50*s_声)
                eng.环境["力"]=32-(1+30*s_力); eng.环境["温"]=591-(288+15*s_温)
            else:
                eng.环境["光"]=1+500*s_光; eng.环境["声"]=25+50*s_声
                eng.环境["力"]=1+30*s_力; eng.环境["温"]=288+15*s_温
            eng.推进()
            if it % 1000 == 0 and hasattr(eng, '日志'):
                eng.日志.clear()

跑固定轮([("正向",False),("反向",True),("正向",False),("反向",True),("正向",False)])
跑固定轮([("反向",True),("正向",False),("反向",True),("正向",False),("反向",True),
         ("正向",False),("反向",True),("正向",False),("反向",True),("正向",False)])

总15,最大15,筋15,_,_,_ = 取状态(eng)
out.append(f"[验证] 轮15: 总={总15:.4f} 最大={最大15:.6f} 筋={筋15} | 参考: -0.0221, 0.494373, 2203")
out.append("")

# 重放 16-75: 60轮随机
for i in range(60):
    轮号 += 1
    amp = decider.uniform(0.8, 1.2)
    跑一轮正向(eng, rm_noise, amp, 加声音=False)  # 历史轮次无声音

总75,最大75,筋75,声75,亮75,_ = 取状态(eng)
out.append(f"[验证] 轮75: 总={总75:.4f} 最大={最大75:.6f} 筋={筋75} 声脑={声75:.6f} 亮={'有' if 亮75 else '无'} | 参考: -0.1641, 0.567098, 2936")
out.append("")

# === 骗跑法v2开始 ===
out.append("— 第 76–80 轮（骗跑法v2+激进声音） —")
out.append("")

状态 = "FEED"           # 给一口开始
前一轮亮 = False

for i in range(5):
    轮号 += 1
    amp = decider.uniform(0.8, 1.2)
    
    # 状态转移：亮1轮就切
    if 状态 == "FEED" and 前一轮亮:
        状态 = "WITHDRAW"
        前一轮亮 = False
    elif 状态 == "WITHDRAW" and not 前一轮亮:
        # 抽走1轮后就给
        状态 = "FEED"
        前一轮亮 = False
    
    if 状态 == "FEED":
        跑一轮正向(eng, rm_noise, amp, 加声音=True)
        方向标签 = "正向+声"
        动作 = "给一口"
    else:
        跑一轮反向(eng, rm_noise, amp, 加声音=False)
        方向标签 = "反向+静音"
        动作 = "抽走"
    
    总, 最大, 筋, 声最大, 亮, nan = 取状态(eng)
    前一轮亮 = 亮
    
    out.append(f"轮{轮号}: {方向标签}({动作}) amp={amp:.4f} 总={总:.4f} 最大={最大:.6f} 筋={筋} 声脑={声最大:.6f} 亮={'有' if 亮 else '无'} NaN={'有' if nan else '无'}")
    
    with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/_partial_76_80_v10.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
        f.flush()
        os.fsync(f.fileno())

out.append("")
out.append("=" * 70)
out.append(f"跑完. 用时 {time.time() - t0:.1f} 秒")

path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_76_80_v10.txt"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
    f.flush()
    os.fsync(f.fileno())
print(f"WROTE {path}  time={time.time()-t0:.1f}s")
