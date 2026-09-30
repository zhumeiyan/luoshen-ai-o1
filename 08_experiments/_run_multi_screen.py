"""多轮野败筛选测试 · 三种策略对比
策略A: 短筛(1500tick) × 末梢截断变化率
策略B: 短筛(1500tick) × 脉冲响应敏感度
策略C: 中筛(3000tick) × 末梢敏感 + 脉冲响应综合
每策略3株进压力阶段(10000tick + 野败)
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体, 痕文件

if os.path.exists(痕文件):
    os.remove(痕文件)

N株 = 20
压力tick = 10000
压力轮长 = 1000
总压力轮 = 压力tick // 压力轮长

def 跑一株(seed, tick数, 脉冲时刻=None):
    """跑单株, 返回结构指纹 + 末梢轨迹"""
    eng = 身体(节点数=96, 种子=seed, 上限满值=6.0, 脑间连接数=150)
    末梢轨迹 = []
    for t in range(1, tick数 + 1):
        eng.环境["力"] = 10.0
        if 脉冲时刻 and t == 脉冲时刻:
            eng.环境["力"] = 15.0  # 微脉冲 +5
        eng.推进()
        if t % 100 == 0:
            末梢轨迹.append({
                "t": t, "截断": eng.末梢统计["声"]["截断"] + eng.末梢统计["光"]["截断"],
                "饱和": eng.末梢统计["声"]["饱和"] + eng.末梢统计["光"]["饱和"],
            })

    总激活 = sum(abs(a) for a in eng.激活)
    节点亮 = sum(1 for a in eng.激活 if abs(a) > 0.5)
    筋数 = len(eng.筋树)
    共火 = sum(eng.脑共火计数)
    平均位点 = np.mean([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0
    最大位点 = max([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0

    return 总激活, 节点亮, 筋数, 共火, 平均位点, int(最大位点), 末梢轨迹, eng


def 筛策略(策略名, 筛长, 脉冲位置, 偏离算法):
    print(f"\n===== 策略{策略名}: 筛长={筛长}, 脉冲@{脉冲位置} =====")
    档案 = []
    for seed in range(1, N株 + 1):
        总激活, 节点亮, 筋数, 共火, 均位, 最大位, 轨迹, _ = 跑一株(seed, 筛长, 脉冲位置)
        档案.append({
            "种子": seed, "总激活": 总激活, "节点亮": 节点亮, "筋数": 筋数,
            "共火": 共火, "均位": round(均位, 3), "最大位": 最大位,
            "末梢轨迹": 轨迹, "脉冲截断": 轨迹[-1]["截断"] if 轨迹 else 0,
        })
        print(f"  株{seed}: 激活={总激活:.2f} 亮={节点亮} 筋={筋数} 共火={共火} 均位={均位:.3f} 最大位={最大位} 末截={轨迹[-1]['截断'] if 轨迹 else 0}")

    候选 = 偏离算法(档案)
    return 档案, 候选


def 偏离A(档案):
    """末梢截断变化率(截断增速)"""
    for 株 in 档案:
        轨迹 = 株["末梢轨迹"]
        if len(轨迹) >= 2:
            d = (轨迹[-1]["截断"] - 轨迹[len(轨迹)//2]["截断"]) / max(1, 株["筋数"])
        else:
            d = 0
        株["偏离A"] = round(d, 4)
    return sorted(档案, key=lambda x: -x["偏离A"])[:3]


def 偏离B(档案):
    """脉冲响应敏感度(脉冲前后末梢截断变化)"""
    for 株 in 档案:
        轨迹 = 株["末梢轨迹"]
        株["偏离B"] = 株["脉冲截断"]  # 末截越高 = 末梢越敏感
    return sorted(档案, key=lambda x: -x["偏离B"])[:3]


def 偏离C(档案):
    """末梢敏感 + 最大位点综合"""
    vals_b = [x["脉冲截断"] for x in 档案]
    vals_m = [x["最大位"] for x in 档案]
    mean_b, std_b = np.mean(vals_b), np.std(vals_b) + 1e-9
    mean_m, std_m = np.mean(vals_m), np.std(vals_m) + 1e-9
    for 株 in 档案:
        株["偏离C"] = round(abs((株["脉冲截断"]-mean_b)/std_b) + abs((株["最大位"]-mean_m)/std_m), 3)
    return sorted(档案, key=lambda x: -x["偏离C"])[:3]


# ============================================================
# 三种策略筛选
# ============================================================
结果汇总 = {}

for 策略, 偏离算法, 筛长, 脉冲位置 in [
    ("A_截断变化率", 偏离A, 1500, None),
    ("B_脉冲敏感度", 偏离B, 1500, 1200),
    ("C_综合敏感度", 偏离C, 3000, 2400),
]:
    档案, 候选 = 筛策略(策略, 筛长, 脉冲位置, 偏离算法)
    种子列表 = [c['种子'] for c in 候选]
    print(f"  → 策略{策略}选: {种子列表}")
    结果汇总[策略] = {"档案": 档案, "候选": 候选}

# ============================================================
# 对每种策略的候选进行压力测试
# ============================================================
print("\n\n############## 压力阶段  ##############")

压力模式 = ("递增野败", {
    "轮1_1": {"概率": 0.02, "灭活率": 0.5},
    "轮2_3": {"概率": 0.05, "灭活率": 1.0},
    "轮4_6": {"概率": 0.10, "灭活率": 1.0},
    "轮7_10": {"概率": 0.15, "灭活率": 2.0},
})

for 策略名, 策略结果 in 结果汇总.items():
    print(f"\n\n===== 压力测试: 策略{策略名} =====")
    for rank, 候选 in enumerate(策略结果["候选"], 1):
        seed = 候选["种子"]
        eng = 身体(节点数=96, 种子=seed, 上限满值=6.0, 脑间连接数=150)
        if os.path.exists(痕文件):
            os.remove(痕文件)

        轮记录 = []
        灭绝 = False

        for tick in range(1, 压力tick + 1):
            eng.环境["力"] = 10.0

            # 根据轮数确定压力强度
            轮次 = (tick - 1) // 压力轮长 + 1
            if 轮次 <= 1:
                压力概率, 压力强度 = 0.02, 0.5
            elif 轮次 <= 3:
                压力概率, 压力强度 = 0.05, 1.0
            elif 轮次 <= 6:
                压力概率, 压力强度 = 0.10, 1.0
            else:
                压力概率, 压力强度 = 0.15, 2.0

            野败触发 = eng.随机器.random() < 压力概率

            if 野败触发 and len(eng.筋树) > 5:
                r = eng.随机器.random()
                if r < 0.25:
                    随机筋键 = eng.随机器.choice(list(eng.筋树.keys()))
                    跳变 = eng.随机器.choice([-5, -3, -2, -1, 1, 2, 3, 5, 8])
                    eng.筋树[随机筋键]['位点'] = max(1, min(40, eng.筋树[随机筋键]['位点'] + 跳变))
                elif r < 0.5:
                    随机节点 = eng.随机器.randint(0, eng.节点数 - 1)
                    eng.不应期[随机节点] = eng.随机器.choice([0, 1, 2, 3, 5, 8, 10])
                elif r < 0.75:
                    for _ in range(int(压力强度)):
                        if len(eng.筋树) > 5:
                            随机筋键 = eng.随机器.choice(list(eng.筋树.keys()))
                            eng.筋树[随机筋键]['位点'] = max(1,  eng.筋树[随机筋键]['位点'] - eng.随机器.choice([1, 2, 3]))
                else:
                    for _ in range(int(压力强度)):
                        if len(eng.筋树) > 5:
                            随机筋键 = eng.随机器.choice(list(eng.筋树.keys()))
                            eng.筋树[随机筋键]['位点'] = 0

            eng.推进()

            if tick % 压力轮长 == 0:
                round_num = tick // 压力轮长
                总激活 = sum(abs(a) for a in eng.激活)
                节点亮 = sum(1 for a in eng.激活 if abs(a) > 0.5)
                筋数 = len(eng.筋树)
                共火 = sum(eng.脑共火计数)
                均位 = np.mean([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0
                最大位 = max([筋['位点'] for 筋 in eng.筋树.values()]) if eng.筋树 else 0
                if 筋数 == 0:
                    灭绝 = True

                轮记录.append({
                    "轮": round_num, "种子": seed, "阶段": 策略名,
                    "筋数": 筋数, "均位": round(均位, 3), "最大位": int(最大位),
                    "共火": 共火, "总激活": round(总激活, 2), "节点亮": 节点亮,
                    "饿": round(eng.hunger, 1), "醒": int(eng.awake),
                    "哭合计": eng.哭合计,
                })

                if 轮次 in [1, 3, 6, 10]:
                    print(f"  [{策略名}] 株{rank} 种子{seed} 轮{round_num}: "
                          f"筋={筋数} 均位={均位:.3f} 最大位={最大位} 共火={共火} "
                          f"亮={节点亮} 饿={eng.hunger:.1f} 醒={int(eng.awake)} 哭={eng.哭合计}")

        if 灭绝:
            print(f"  [{策略名}] 株{rank} 种子{seed} 灭绝")
        首轮, 末轮 = 轮记录[0], 轮记录[-1]
        print(f"  [{策略名}] 株{rank} 总结: 筋{首轮['筋数']}→{末轮['筋数']} "
              f"均位{首轮['均位']}→{末轮['均位']} 最大位{首轮['最大位']}→{末轮['最大位']}")

        候选["压力结果"] = 轮记录
        候选["灭绝"] = 灭绝

    # 策略对比
    存活 = [c for c in 策略结果["候选"] if not c.get("灭绝", False)]
    if 存活:
        最优 = max(存活, key=lambda c: c.get("压力结果", [{}])[-1].get("筋数", 0))
        print(f"\n  ★ 策略{策略名} 推荐: 种子{最优['种子']} (存活+筋数最多)")
    else:
        print(f"\n  ✗ 策略{策略名}: 候选全灭")

# 保存结果
with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/_rerun/multi_screen_result.json", "w", encoding="utf-8") as f:
    json.dump({
        "说明": "多轮野败筛选: 策略A(截断变化率) B(脉冲敏感度) C(综合)",
        "各策略结果": {
            k: {"偏离算法": k, "候选种子": [c["种子"] for c in v["候选"]]}
            for k, v in 结果汇总.items()
        }
    }, f, ensure_ascii=False, indent=2, default=str)

print("\n\nDONE. 结果已保存到 _rerun/multi_screen_result.json")
