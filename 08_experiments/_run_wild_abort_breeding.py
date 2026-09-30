"""袁隆平法 · 洛神育种
阶段1: 播20株(种子1-20),每株1000tick,筛结构异常
阶段2: 对最异常的3株施加野败压力跑10000tick,逐值报告
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体, 痕文件

if os.path.exists(痕文件):
    os.remove(痕文件)

N株 = 20
取种tick = 1000
压力轮数 = 10
压力轮长 = 1000
总压力tick = 压力轮数 * 压力轮长

# ============================================================
# 阶段1: N株各取1000tick, 记录结构指纹
# ============================================================
print(f"[阶段1] 播{N株}株, 每株{取种tick}tick")

株档案 = []
for seed in range(1, N株 + 1):
    eng = 身体(节点数=96, 种子=seed, 上限满值=6.0, 脑间连接数=150)
    for t in range(1, 取种tick + 1):
        eng.环境["力"] = 10.0
        eng.推进()

    总激活 = sum(abs(a) for a in eng.激活)
    节点亮 = sum(1 for a in eng.激活 if abs(a) > 0.5)
    筋数 = len(eng.筋树)
    共火 = sum(eng.脑共火计数)
    平均位点 = np.mean([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0
    最大位点 = max([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0

    株档案.append({
        "种子": seed, "总激活": round(总激活, 3), "节点亮": 节点亮,
        "筋数": 筋数, "共火": 共火, "平均位点": round(平均位点, 3),
        "最大位点": int(最大位点), "hunger": round(eng.hunger, 2),
        "awake": int(eng.awake),
    })

    print(f"  株{seed}: 激活={总激活:.2f} 亮={节点亮} 筋={筋数} 共火={共火} 均位={平均位点:.3f} 最大位={最大位点} 饿={eng.hunger:.1f}")

# ============================================================
# 筛败株: Z-score综合偏离度, 取前3
# ============================================================
def zscore(档案, key):
    vals = [a[key] for a in 株档案]
    m, s = np.mean(vals), np.std(vals) + 1e-9
    return abs((档案[key] - m) / s)

for 株 in 株档案:
    株["偏离分"] = round(
        zscore(株, "筋数") + zscore(株, "最大位点") +
        zscore(株, "平均位点") + zscore(株, "节点亮") +
        zscore(株, "总激活") * 0.5, 3
    )

株档案_sorted = sorted(株档案, key=lambda x: -x["偏离分"])
候选株 = 株档案_sorted[:3]

print(f"\n[筛败株] 偏离度前3:")
for i, 株 in enumerate(候选株, 1):
    print(f"  第{i}: 种子{株['种子']} 偏离={株['偏离分']} 筋{株['筋数']} 最大位{株['最大位点']} 均位{株['平均位点']} 亮{株['节点亮']} 共火{株['共火']}")

# ============================================================
# 阶段2: 候选株 + 随机野败压力
# ============================================================
print(f"\n[阶段2] 对3株各施加压力{总压力tick}tick, {压力轮数}轮")

t0 = time.time()

for rank, 候选 in enumerate(候选株, 1):
    seed = 候选["种子"]
    eng = 身体(节点数=96, 种子=seed, 上限满值=6.0, 脑间连接数=150)
    if os.path.exists(痕文件):
        os.remove(痕文件)

    轮记录 = []
    灭绝 = False

    for tick in range(1, 总压力tick + 1):
        eng.环境["力"] = 10.0

        # === 野败随机压力 (5%概率触发) ===
        if eng.随机器.random() < 0.05:
            r = eng.随机器.random()
            if r < 0.33 and len(eng.筋树) > 0:
                随机筋键 = eng.随机器.choice(list(eng.筋树.keys()))
                跳变 = eng.随机器.choice([-3, -2, -1, 1, 2, 3, 5])
                eng.筋树[随机筋键]['位点'] = max(1, min(40, eng.筋树[随机筋键]['位点'] + 跳变))
            elif r < 0.66:
                随机节点 = eng.随机器.randint(0, eng.节点数 - 1)
                eng.不应期[随机节点] = eng.随机器.choice([0, 1, 2, 3, 5, 8])
            elif len(eng.筋树) > 10:
                随机筋键 = eng.随机器.choice(list(eng.筋树.keys()))
                eng.筋树[随机筋键]['位点'] = 0  # 不育: 会被死筋清掉

        eng.推进()

        if tick % 压力轮长 == 0:
            round_num = tick // 压力轮长
            总激活 = sum(abs(a) for a in eng.激活)
            节点亮 = sum(1 for a in eng.激活 if abs(a) > 0.5)
            筋数 = len(eng.筋树)
            共火 = sum(eng.脑共火计数)
            平均位点 = np.mean([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0
            最大位点 = max([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0
            if 筋数 == 0:
                灭绝 = True

            轮记录.append({
                "轮": round_num, "种子": seed, "tick": tick,
                "总激活": round(总激活, 2), "节点亮": 节点亮,
                "筋数": 筋数, "共火": 共火, "平均位点": round(平均位点, 3),
                "最大位点": int(最大位点), "hunger": round(eng.hunger, 2),
                "awake": int(eng.awake), "存货": int(eng.存货), "哭合计": eng.哭合计,
            })

            print(f"  [败株{rank}] 种子{seed} 轮{round_num}: "
                  f"激活={总激活:.2f} 亮={节点亮} 筋={筋数} 均位={平均位点:.3f} 最大位={最大位点} "
                  f"共火={共火} 饿={eng.hunger:.1f} 醒={int(eng.awake)} 哭={eng.哭合计} 存={int(eng.存货)}")

    # 阶段2单候选总结
    print(f"\n[压力总结] 败株{rank} (种子{seed}):")
    if 灭绝:
        print(f"  灭绝 (筋数归零)")
    else:
        首轮, 末轮 = 轮记录[0], 轮记录[-1]
        变化 = "增长" if 末轮["筋数"] > 首轮["筋数"] else "衰退"
        print(f"  筋数: {首轮['筋数']}→{末轮['筋数']} ({变化})")
        print(f"  均位: {首轮['平均位点']}→{末轮['平均位点']}")
        print(f"  最大位: {首轮['最大位点']}→{末轮['最大位点']}")
        print(f"  亮: {首轮['节点亮']}→{末轮['节点亮']}")
        print(f"  共火趋势: {[r['共火'] for r in 轮记录]}")

    候选["压力结果"] = 轮记录
    候选["灭绝"] = 灭绝

elapsed = time.time() - t0
print(f"\n[完成] 耗时={elapsed:.1f}秒")

# 保存
with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/_rerun/wild_abort_breeding_result.json", "w", encoding="utf-8") as f:
    json.dump({
        "说明": "袁隆平野败育种: 阶段1(20株1000tick) + 阶段2(3株+压力10000tick)",
        "阶段1_全株档案": 株档案,
        "偏离度前3": [{"种子": c["种子"], "偏离分": c["偏离分"]} for c in 候选株],
    }, f, ensure_ascii=False, indent=2, default=str)

print("\n=== 最终推荐 ===")
存活株 = [c for c in 候选株 if not c.get("灭绝", False)]
if 存活株:
    最优 = max(存活株, key=lambda c: c.get("压力结果", [{}])[-1]["筋数"] if c.get("压力结果") else 0)
    print(f"推荐保留: 种子{最优['种子']} (压力后筋数最多)")
else:
    print("结论: 3株全败, 需调整压力强度或播种更多株")
print("结果: _rerun/wild_abort_breeding_result.json")
