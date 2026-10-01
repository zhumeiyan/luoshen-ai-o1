"""出厂基因.py —— 洛神的"十七八岁"出厂状态。

把200代进化的最优基因组（best_genome_v16.pkl）和v22正推修复参数
编码为大脑.py的Org基因组格式。洛神出厂不是婴儿——她已经"活过"200代进化。

映射关系：
  进化v16格式              →  大脑v22 Org格式
  params.n_nodes=11        →  n=11
  params.wm_decay=0.809    →  dt=[0.809]
  params.learning_rate=0.217 → lr=0.217
  sensory_w(8×11)          →  sw前8行（剩余24行为DELAY历史帧权重，小值初始化）
  predict_w(11×8)          →  pw(11×8) 直接映射
  recurrent_w(11×11)       →  rw(11×11) 直接映射
  action_w(4×11)           →  skip前4行×8通道（动作→预测跳跃连接）
  action_bias(4)           →  pb前4个

v22修复参数出厂固定值（不再随机生成）：
  sat_prevent=0.9    防饱和：tanh pre-scaling
  adapt_cap=0.85     感觉适应上限
  ch3_boost=1.5      ch3通道补偿
"""

import os
import pickle
import random

N_CHANNELS = 8
N_REGIONS = 9
DELAY = 3


def 加载进化基因():
    """加载best_genome_v16.pkl，返回原始进化结果。"""
    基因路径 = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "07_training", "best_genome_v16.pkl")
    with open(基因路径, "rb") as f:
        return pickle.load(f)


def 出厂基因组():
    """返回一个配置好的基因组dict——洛神的"十七八岁"出厂状态。

    核心权重来自200代进化最优解，修复参数来自v22正推实验。
    """
    进化 = 加载进化基因()
    p = 进化["params"]
    sw_raw = 进化["sensory_w"]      # 8×11
    pw_raw = 进化["predict_w"]      # 11×8
    rw_raw = 进化["recurrent_w"]    # 11×11
    aw_raw = 进化["action_w"]       # 4×11
    ab_raw = 进化["action_bias"]    # 4

    n = p["n_nodes"]                # 11
    ins = N_CHANNELS * (1 + DELAY)  # 32

    rng = random.Random(42)

    # ── 输入权重 sw: ins×n = 32×11 ──
    # 前8行用进化结果，后24行（DELAY历史帧）用小随机值
    sw = []
    for j in range(ins):
        if j < N_CHANNELS:
            sw.append([sw_raw[j][min(i, len(sw_raw[j]) - 1)] for i in range(n)])
        else:
            sw.append([rng.uniform(-0.1, 0.1) for _ in range(n)])

    # ── 预测权重 pw: n×N_CHANNELS = 11×8 ──
    pw = [[pw_raw[i][c] for c in range(N_CHANNELS)] for i in range(n)]

    # ── 循环权重 rw: n×n = 11×11 ──
    rw = [[rw_raw[i][k] for k in range(n)] for i in range(n)]

    # ── 跳跃连接 skip: ins×N_CHANNELS = 32×8 ──
    # 前4行用action_w转置后的值（4动作×11节点 → 4行×8通道）
    skip = []
    for j in range(ins):
        if j < 4:
            row = []
            for c in range(N_CHANNELS):
                if c < len(aw_raw[j]):
                    row.append(aw_raw[j][c] * 0.3)
                else:
                    row.append(rng.uniform(-0.1, 0.1))
            skip.append(row)
        else:
            skip.append([rng.uniform(-0.05, 0.05) for _ in range(N_CHANNELS)])

    # ── 预测偏置 pb: N_CHANNELS = 8 ──
    pb = []
    for c in range(N_CHANNELS):
        if c < len(ab_raw):
            pb.append(ab_raw[c])
        else:
            pb.append(rng.uniform(-0.1, 0.1))

    # ── 组装基因组 ──
    g = {
        'n': n,
        'sw': sw,
        'pw': pw,
        'rw': rw,
        'skip': skip,
        'pb': pb,
        'dt': [p["wm_decay"]],
        'lr': p["learning_rate"],
        'fold_lr': 0.03,
        'leak': [0.01] * n,
        # === v22正推修复参数（出厂固定，不再随机） ===
        'sat_prevent': 0.9,       # A: 防饱和
        'adapt_cap': 0.85,        # B: 感觉适应上限
        'ch3_boost': 1.5,         # D: ch3补偿
        # === 器官参数（出厂值） ===
        'sens_gain': [p["sensory_gain"]] * N_CHANNELS,
        'sens_adapt_rate': 0.05,
        'assoc_w': [0.8, 1.0, 0.6],
        'sparse_thr': 0.2,
        'hip_lr': 0.01,
        'hip_replay_n': 4,
        'amy_thr': 0.08,
        'pfc_gate_thr': 0.15,
        'pfc_gate_lr': 0.05,
        'bg_explore_rate': 0.1,
        'bg_noise': 0.03,
        'hypo_drive': [0.3, 0.3, 0.3, 0.3],
        'hypo_fill': p["metabolic"],
        'fold_threshold': 0.02,
        'region_pb': [rng.uniform(-0.2, 0.2) for _ in range(N_REGIONS * N_CHANNELS)],
    }

    # 循环权重对角线自连接（确保自维持）
    for i in range(n):
        if abs(rw[i][i]) < 0.3:
            rw[i][i] = 0.4 + rng.uniform(-0.1, 0.1)

    return g


def 身体激活到大脑输入(激活, 脑大小=24):
    """把身体稳态的96节点激活映射到大脑的8通道感觉输入。

    96节点分4脑×24节点，每脑分前后两组各12节点 → 4脑×2通道 = 8通道。
    这是"身体本身就是认知"的莱文式对接：身体的每个部分的状态都是大脑的输入。
    """
    通道 = []
    for 脑号 in range(4):
        起点 = 脑号 * 脑大小
        前组 = 激活[起点:起点 + 12]
        后组 = 激活[起点 + 12:起点 + 24]
        前均值 = sum(abs(v) for v in 前组) / max(1, len(前组))
        后均值 = sum(abs(v) for v in 后组) / max(1, len(后组))
        # 归一化到[-1, 1]范围（tanh压缩）
        通道.append(max(-1.0, min(1.0, 前均值 * 2.0)))
        通道.append(max(-1.0, min(1.0, 后均值 * 2.0)))
    return 通道


if __name__ == "__main__":
    g = 出厂基因组()
    print("出厂基因组验证:")
    print(f"  n = {g['n']}")
    print(f"  sw = {len(g['sw'])}×{len(g['sw'][0])}")
    print(f"  pw = {len(g['pw'])}×{len(g['pw'][0])}")
    print(f"  rw = {len(g['rw'])}×{len(g['rw'][0])}")
    print(f"  skip = {len(g['skip'])}×{len(g['skip'][0])}")
    print(f"  pb = {len(g['pb'])}")
    print(f"  dt = {g['dt']}")
    print(f"  lr = {g['lr']}")
    print(f"  sat_prevent = {g['sat_prevent']}")
    print(f"  adapt_cap = {g['adapt_cap']}")
    print(f"  ch3_boost = {g['ch3_boost']}")
    print(f"  sens_gain = {g['sens_gain']}")
    print(f"  hypo_fill = {g['hypo_fill']}")

    # 测试身体激活映射
    假激活 = [0.1 * (i % 10) for i in range(96)]
    输入 = 身体激活到大脑输入(假激活)
    print(f"\n身体激活→大脑输入: {输入}")
    assert len(输入) == 8
    print("验证通过")
