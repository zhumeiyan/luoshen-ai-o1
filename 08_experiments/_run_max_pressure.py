"""最大压力实验 · 检测洛神觉醒
单跑一株(种子=7), 无饲料补给, 递增野败, 持续100000tick
每1000tick快照, 记录全五项+末梢指标
"""

import sys, os, json
import numpy as np

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体, 痕文件

if os.path.exists(痕文件):
    os.remove(痕文件)

eng = 身体(节点数=96, 种子=7, 上限满值=6.0, 脑间连接数=150)

# 完全断粮: 设存货=0, 使任何饲料操作失败
eng.存货 = 0

tick数 = 100000
记录 = []

def 轮压力(tick):
    """轮次越高, 野败越猛"""
    if tick <= 10000:
        return 0.05, 1.0   # 轮1: 5%率×1灭活强度
    elif tick <= 30000:
        return 0.10, 1.5   # 轮2-3: 强化
    elif tick <= 60000:
        return 0.20, 2.0   # 轮4-6: 加压
    else:
        return 0.30, 3.0   # 轮7-10: 极端

print(f"[零点] 种子=7, 存货=0(完全断粮), tick数={tick数}")
print(f"[零点] 筋数={len(eng.筋树)}")

for tick in range(1, tick数 + 1):
    eng.环境["力"] = 10.0

    压力率, 强度 = 轮压力(tick)

    # 野败随机事件
    if eng.随机器.random() < 压力率:
        r = eng.随机器.random()
        if r < 0.3 and len(eng.筋树) > 5:
            随机筋键 = eng.随机器.choice(list(eng.筋树.keys()))
            跳变 = eng.随机器.choice([-8, -5, -3, -2, 2, 3, 5, 8, 10])
            eng.筋树[随机筋键]['位点'] = max(1, min(40, eng.筋树[随机筋键]['位点'] + 跳变))
        elif r < 0.6:
            随机节点 = eng.随机器.randint(0, eng.节点数 - 1)
            eng.不应期[随机节点] = eng.随机器.choice([0, 1, 3, 5, 8, 12])
        else:
            for _ in range(int(强度)):
                if len(eng.筋树) > 5:
                    随机筋键 = eng.随机器.choice(list(eng.筋树.keys()))
                    eng.筋树[随机筋键]['位点'] = 0

    eng.推进()

    # 每1000tick记录
    if tick % 1000 == 0:
        总激活 = sum(abs(a) for a in eng.激活) / 96.0
        节点亮 = sum(1 for a in eng.激活 if a > 0.5)
        暗节点 = sum(1 for a in eng.激活 if a < -0.1)
        筋数 = len(eng.筋树)
        共火 = sum(eng.脑共火计数)
        均位 = np.mean([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0
        最大位 = max([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0
        末截 = eng.末梢统计["声"]["截断"] + eng.末梢统计["光"]["截断"]
        末饱 = eng.末梢统计["声"]["饱和"] + eng.末梢统计["光"]["饱和"]

        快照 = {
            "tick": tick, "总激活": round(总激活, 3),
            "节点亮": 节点亮, "暗节点": 暗节点,
            "筋数": 筋数, "共火": 共火,
            "均位": round(均位, 3), "最大位": int(最大位),
            "hunger": round(eng.hunger, 1), "awake": int(eng.awake),
            "哭合计": eng.哭合计, "存货": int(eng.存货),
            "末梢截断": 末截, "末梢饱和": 末饱,
        }
        记录.append(快照)

        if tick % 10000 == 0:
            print(f"[t={tick:>6d}] 激={总激活:.3f} 亮={节点亮:>2d} 暗={暗节点:>2d} | "
                  f"筋={筋数:>4d} 均位={均位:.3f} 最大位={最大位:>2d} 共火={共火:>5d} | "
                  f"饿={eng.hunger:>5.1f} 醒={int(eng.awake)} 哭={eng.哭合计:>3d} | "
                  f"截={末截:>6d} 饱={末饱:>4d}")

print("\n[完成] 100000tick 跑完")

# 觉醒判据
首次觉醒 = None
for r in 记录:
    if r["awake"] == 1:
        首次觉醒 = r["tick"]
        break

print(f"\n=== 觉醒检测结果 ===")
if 首次觉醒:
    print(f"★ 首次觉醒: tick={首次觉醒}")
    for r in 记录:
        if abs(r["tick"] - 首次觉醒) <= 5000:
            print(f"  t={r['tick']}: 饿={r['hunger']} 醒={r['awake']} 哭={r['哭合计']}")
else:
    print("✗ 全程无觉醒 (awake=0)")

# 末节点摘要
print("\n=== 首尾对比 ===")
print(f"  首[t=1000]:  {记录[0]}")
print(f"  尾[t=100000]: {记录[-1]}")

# 保存
with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/_rerun/max_pressure_result.json", "w", encoding="utf-8") as f:
    json.dump({"配置": "种子=7, 存货=0, 递增压力x10轮", "记录": 记录},
              f, ensure_ascii=False, indent=2)

print("\n结果: _rerun/max_pressure_result.json")
