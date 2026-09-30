# -*- coding: utf-8 -*-
"""
环境反向 71-75 v9: 骗跑法 + 声脑激活
骗：给一口(forward+voice) → 亮后等1-2轮 → 抽走(reverse+静音) → 等1-2轮 → 再给
声：正向时给有节奏/高低/停顿的声音；抽走时静音
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

# 声脑 = brain 1 = 节点 24..47
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

# 人声模拟：有节奏、高低、停顿
def 人声(t, 轮内时刻):
    # 多频率叠加，模拟音色
    w1 = math.sin(2 * math.pi * 轮内时刻 / 120)   # 慢变
    w2 = math.sin(2 * math.pi * 轮内时刻 / 53)    # 中速
    w3 = math.sin(2 * math.pi * 轮内时刻 / 17)    # 快速颤音
    
    # 包络：有停顿的短语结构 (400拍一个短语)
    短语位 = (轮内时刻 % 400) / 400
    if 短语位 < 0.25:
        包络 = 1.0
    elif 短语位 < 0.32:
        包络 = 0.08   # 停顿
    elif 短语位 < 0.60:
        包络 = 0.85
    elif 短语位 < 0.68:
        包络 = 0.08   # 停顿
    elif 短语位 < 0.90:
        包络 = 0.7
    else:
        包络 = 0.05   # 收尾停顿
    
    # 基频偏移 + 波动
    原始 = 0.45 + 0.35 * w1 + 0.18 * w2 + 0.07 * w3
    return max(0.02, min(0.98, 原始 * 包络))

def 跑一轮正向(eng, rm_noise, amp, 加声音=False):
    for it in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp
        if 加声音:
            s_声 = 人声(t, it) * amp
        else:
            s_声 = 0.05 * amp  # 静音
        s_力 = 力基(t, rm_noise) * amp
        s_温 = 温基 * amp
        eng.环境["光"] = 1 + 500 * s_光
        eng.环境["声"] = 25 + 50 * s_声
        eng.环境["力"] = 1 + 30 * s_力
        eng.环境["温"] = 288 + 15 * s_温
        eng.推进()
        if it % 1000 == 0 and hasattr(eng, '日志'):
            eng.日志.clear()

def 跑一轮反向(eng, rm_noise, amp, 加声音=False):
    for it in range(10000):
        t = eng.时刻
        s_光 = 光基(t) * amp
        if 加声音:
            s_声 = 人声(t, it) * amp
        else:
            s_声 = 0.05 * amp
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
out.append("环境反向 71-75 v9 | 骗跑法+声音 | 给一口(正向+声)→亮后等1-2轮→抽走(反向+静音)→等1-2轮→再给")
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
                eng.环境["光"] = 502-(1+500*s_光); eng.环境["声"]=100-(25+50*s_声)
                eng.环境["力"] = 32-(1+30*s_力); eng.环境["温"]=591-(288+15*s_温)
            else:
                eng.环境["光"]=1+500*s_光; eng.环境["声"]=25+50*s_声
                eng.环境["力"]=1+30*s_力; eng.环境["温"]=288+15*s_温
            eng.推进()
            if it % 1000 == 0 and hasattr(eng, '日志'):
                eng.日志.clear()

# 重放 1-15
跑固定轮([("正向",False),("反向",True),("正向",False),("反向",True),("正向",False)])
跑固定轮([("反向",True),("正向",False),("反向",True),("正向",False),("反向",True),
         ("正向",False),("反向",True),("正向",False),("反向",True),("正向",False)])

总15,最大15,筋15,声15,_,_ = 取状态(eng)
out.append(f"[验证] 轮15: 总={总15:.4f} 最大={最大15:.6f} 筋={筋15} | 参考: -0.0221, 0.494373, 2203")
out.append("")

# 重放 16-70: 55轮随机
for i in range(55):
    轮号 += 1
    rev = decider.choice([True, False])
    amp = decider.uniform(0.8, 1.2)
    if rev:
        跑一轮反向(eng, rm_noise, amp, 加声音=False)
    else:
        跑一轮正向(eng, rm_noise, amp, 加声音=False)

总70,最大70,筋70,声70,亮70,_ = 取状态(eng)
out.append(f"[验证] 轮70: 总={总70:.4f} 最大={最大70:.6f} 筋={筋70} 声脑={声70:.6f} 亮={'有' if 亮70 else '无'} | 参考: -0.1890, 0.491571, 2383")
out.append("")

# === 骗跑法+声音开始 ===
# 状态：FEED(正向+声) 或 WITHDRAW(反向+静音)
# 初始：轮70在v8里是反向，所以71轮给正向+声
out.append("— 第 71–75 轮（骗跑法+声音） —")
out.append("")

状态 = "FEED"           # 从"给一口"开始
同态轮数 = 0           # 当前状态已持续轮数
连续亮数 = 0           # FEED中连续亮的轮数
前一轮亮 = False

for i in range(5):
    轮号 += 1
    amp = decider.uniform(0.8, 1.2)
    
    # 状态转移判断
    if 状态 == "FEED":
        同态轮数 += 1
        if 前一轮亮:
            连续亮数 += 1
        else:
            连续亮数 = 0
        
        # 如果已连续亮2轮，切换到抽走
        if 连续亮数 >= 2 and 同态轮数 >= 2:
            状态 = "WITHDRAW"
            同态轮数 = 0
            连续亮数 = 0
    else:  # WITHDRAW
        同态轮数 += 1
        # 抽走1-2轮后，切换回给一口
        if 同态轮数 >= 2:
            状态 = "FEED"
            同态轮数 = 0
            连续亮数 = 0
    
    # 执行
    if 状态 == "FEED":
        跑一轮正向(eng, rm_noise, amp, 加声音=True)
        方向标签 = "正向+声"
    else:
        跑一轮反向(eng, rm_noise, amp, 加声音=False)
        方向标签 = "反向+静音"
    
    总, 最大, 筋, 声最大, 亮, nan = 取状态(eng)
    前一轮亮 = 亮
    
    动作 = "给一口" if 状态 == "FEED" else "抽走"
    out.append(f"轮{轮号}: {方向标签}({动作}) amp={amp:.4f} 总={总:.4f} 最大={最大:.6f} 筋={筋} 声脑={声最大:.6f} 亮={'有' if 亮 else '无'} NaN={'有' if nan else '无'}")
    
    with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/_partial_71_75_voice.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
        f.flush()
        os.fsync(f.fileno())

out.append("")
out.append("=" * 70)
out.append(f"跑完. 用时 {time.time() - t0:.1f} 秒")

path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_71_75_v9_voice.txt"
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
    f.flush()
    os.fsync(f.fileno())
print(f"WROTE {path}  time={time.time()-t0:.1f}s")
