"""涌现探测脚本 · 扰动对比
阶段1: 力=慢振荡 (周期~5000tick)
阶段2: 力=快振荡 (周期~2500tick)
阶段3: 力=恒定

追踪筋树传导结构是否自组织出差异。
"""

import sys, os, json, random, time
import numpy as np

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体, 痕文件

随机种子 = 7
引擎 = 身体(节点数=96, 种子=随机种子, 上限满值=6.0, 脑间连接数=150)

# 清空痕文件
if os.path.exists(痕文件):
    os.remove(痕文件)

总tick数 = 60000
轮长 = 10000
报告频率 = 1000

# 阶段边界
阶段边界 = {
    1: (1, 20000),
    2: (20001, 40000),
    3: (40001, 60000),
}

历史 = {
    "轮快照": [],
    "筋树统计": [],
    "末梢统计": [],
}

def 取阶段(tick):
    for 阶段名, (起, 止) in 阶段边界.items():
        if 起 <= tick <= 止:
            return 阶段名
    return 0

def 取力输入(tick):
    phase = 取阶段(tick)
    if phase == 1:
        return 10.0 + 5.0 * np.sin(tick / 800.0 * 2 * np.pi)
    elif phase == 2:
        return 10.0 + 5.0 * np.sin(tick / 400.0 * 2 * np.pi)
    else:
        return 10.0

print(f"[零点] 引擎初始化完毕, 种子={随机种子}")
print(f"[零点] 节点数={引擎.节点数}, 筋数={len(引擎.筋树)}")
print(f"[零点] 上限满值={引擎.上限满值}")

t0 = time.time()

for tick in range(1, 总tick数 + 1):
    引擎.环境["力"] = 取力输入(tick)
    引擎.推进()

    if tick % 报告频率 == 0:
        总激活 = sum(abs(a) for a in 引擎.激活)
        活筋数 = sum(1 for j, i in 引擎.筋树 if 引擎.筋树[(j,i)].get('传导次数', 0) > 0)
        总位点 = sum(筋['位点'] for 筋 in 引擎.筋树.values())
        平均位点 = 总位点 / len(引擎.筋树) if 引擎.筋树 else 0
        总疲劳 = sum(筋['疲劳'] for 筋 in 引擎.筋树.values())
        平均疲劳 = 总疲劳 / len(引擎.筋树) if 引擎.筋树 else 0
        节点亮 = sum(1 for a in 引擎.激活 if abs(a) > 0.5)
        末截 = 引擎.末梢统计["声"]["截断"] + 引擎.末梢统计["光"]["截断"]
        末饱 = 引擎.末梢统计["声"]["饱和"] + 引擎.末梢统计["光"]["饱和"]

        if tick % 轮长 == 0:
            round_num = tick // 轮长
            # 筋树统计: 位点分布
            位点分布 = {}
            for 筋 in 引擎.筋树.values():
                p = 筋['位点']
                位点分布[p] = 位点分布.get(p, 0) + 1
            位点分布 = dict(sorted(位点分布.items()))

            轮数据 = {
                "轮": round_num,
                "tick": tick,
                "阶段": 取阶段(tick),
                "总激活": round(总激活, 2),
                "活筋数": 活筋数,
                "筋总数": len(引擎.筋树),
                "平均位点": round(平均位点, 3),
                "平均疲劳": round(平均疲劳, 3),
                "节点亮": 节点亮,
                "hunger": round(引擎.hunger, 2),
                "awake": int(引擎.awake),
                "哭合计": 引擎.哭合计,
                "存货": int(引擎.存货),
                "末梢截断": 末截,
                "末梢饱和": 末饱,
                "位点分布": 位点分布,
            }
            历史["轮快照"].append(轮数据)

            print(f"[轮{round_num}] 阶段{轮数据['阶段']} tick={tick} | "
                  f"激活={轮数据['总激活']} | 活筋={活筋数}/{len(引擎.筋树)} | "
                  f"均位={轮数据['平均位点']} | 均疲={轮数据['平均疲劳']} | "
                  f"亮={节点亮} | 饿={轮数据['hunger']} | 醒={轮数据['awake']} | "
                  f"哭={轮数据['哭合计']} | 存={轮数据['存货']} | "
                  f"截={末截} 饱={末饱}")
            print(f"  └位点分布: {位点分布}")

elapsed = time.time() - t0
print(f"\n[完成] 耗时={elapsed:.1f}秒, 共{总tick数}tick, 速度={总tick数/elapsed:.0f}tick/s")

# 阶段对比
print("\n=== 阶段结构对比 ===")
for rd in 历史["轮快照"]:
    print(f"轮{rd['轮']}(阶段{rd['阶段']}): 筋{rd['筋总数']}条, 均位{rd['平均位点']}, "
          f"均疲{rd['平均疲劳']}, 亮{rd['节点亮']}, 饿{rd['hunger']}, 哭{rd['哭合计']}")

# 保存历史
with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/_rerun/emergence_probe_history.json", "w", encoding="utf-8") as f:
    json.dump({"随机种子": 随机种子, "总tick数": 总tick数, "轮快照": 历史["轮快照"]},
              f, ensure_ascii=False, indent=2)
print("[完成] 历史已保存到 _rerun/emergence_probe_history.json")
