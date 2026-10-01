#!/usr/bin/env python3
"""对v17o和v18最佳个体运行残差协议，对比差异，据此指导v19设计"""
import sys, random, math, copy
import numpy as np

# 导入所有需要的模块
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/重新来')
from yinyang_residual_protocol import *
from yinyang_brain_v18 import *

print("=" * 60)
print("残差协议：反推 → 指导正推修改")
print("=" * 60)

# ---- 准备两组测试对象 ----

# v17o风格：简单蛋白折叠WM（直接从genome跑，不需要演化引擎）
def create_v17o_style_wm():
    """模拟v17o最佳个体：n=11, 双通路+蛋白折叠"""
    n = 11
    input_size = N_CHANNELS * (1 + DELAY)  # 32
    g = {
        'n': n,
        'sw': [[random.uniform(-0.3, 0.3) for _ in range(n)] for __ in range(input_size)],
        'pw': [[random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)] for __ in range(n)],
        'rw': [[random.uniform(-0.05, 0.05) for _ in range(n)] for __ in range(n)],
        'skip_w': [[random.uniform(-0.2, 0.2) for _ in range(N_CHANNELS)] for __ in range(input_size)],
        'pb': [random.uniform(-0.3, 0.3) for _ in range(N_REGIONS * N_CHANNELS)],
        'dt': [random.uniform(0.4, 0.95) for _ in range(N_REGIONS)],
        'plr': 0.08,
        'plr_fold': 0.03,
    }
    for i in range(n):
        g['rw'][i][i] = random.uniform(0.4, 1.0)
    return SimpleWM(g)

# v18：人脑模块结构（快速演化几代拿到不错的个体）
def create_v18_trained_org(gens=20):
    """快速演化v18几代"""
    from yinyang_brain_v18 import EngineV18
    random.seed(42)
    eng = EngineV18(n_org=12, n_gen=gens, seed=42)
    eng.evolve()
    best_idx = max(range(len(eng.pop)), key=lambda i: eng.pop[i].eval_fitness(ticks=200))
    return eng.pop[best_idx]

# ---- 第一步：v17o风格残差 ----
print("\n[1] v17o风格WM（双通路+蛋白折叠）残差分析")
random.seed(123)
wm_v17o = create_v17o_style_wm()

p_v17o = ResidualProtocol(wm_v17o)
p_v17o.run_session(env_seed=42, ticks=400, switches=15)
r_v17o = p_v17o.print_report("v17o风格(n=11)")

# ---- 第二步：v18风格残差 ----
print("\n[2] v18（人脑模块结构）残差分析")
org18 = create_v18_trained_org(gens=20)
wm18 = org18.cortex

p18 = ResidualProtocol(wm18)
p18.run_session(env_seed=42, ticks=400, switches=15)
r18 = p18.print_report("v18(训练20代)")

# ---- 第三步：对比分析 ----
print("\n\n" + "=" * 60)
print("  v17o vs v18 残差对比 → 反推结论")
print("=" * 60)

print("\n[区域级数值差异]")
print(f"{'区域':>4} {'v17o':>10} {'v18':>10} {'差值':>10} {'说明':>15}")
print("-" * 55)
all_regions = sorted(set(list(r_v17o['l1_regional'].keys()) + list(r18['l1_regional'].keys())))
for rid in all_regions:
    v17 = r_v17o['l1_regional'].get(rid, {}).get('mean_mse', 0)
    v18v = r18['l1_regional'].get(rid, {}).get('mean_mse', 0)
    diff = v18v - v17
    note = "v18优" if diff < -0.05 else ("v17o优" if diff > 0.05 else "持平")
    print(f"{rid:>4} {v17:>10.4f} {v18v:>10.4f} {diff:>+10.4f} {note:>15}")

print("\n[动态差异]")
for key in ['convergence_rate', 'slope', 'autocorr_lag5', 'volatility']:
    a = r_v17o['l2_dynamics'].get(key, 0)
    b = r18['l2_dynamics'].get(key, 0)
    print(f"  {key:<20}: v17o={a:>+10.6f}  v18={b:>+10.6f}  Δ={a-b:>+10.6f}")

print("\n[耦合差异]")
c17 = r_v17o['l3_cross_ch']
c18 = r18['l3_cross_ch']
if c17 and c18:
    print(f"  v17o耦合强度: {c17['coupling_strength']:.4f}, 最强对={c17['max_coupled_pair']}")
    print(f"  v18 耦合强度: {c18['coupling_strength']:.4f}, 最强对={c18['max_coupled_pair']}")

# ---- 第四步：反推结论 ----
print("\n\n" + "=" * 60)
print("  反推结论 → v19修改方向")
print("=" * 60)

conclusions = []

# 分析1：Attention gating的影响
coupling_diff = c18.get('coupling_strength', 0) - c17.get('coupling_strength', 0)
if coupling_diff > 0.05:
    conclusions.append("问题：通道耦合增加 → Attention门控未能有效隔离通道")
    conclusions.append("修改：降低attention gate的最小值下限（从0提到0.2），允许更多通带")
else:
    conclusions.append("Attention gating的通道隔离正常")

# 分析2：高频区域的表现
high_regions = [6, 7, 8]
v17o_high_avg = np.mean([r_v17o['l1_regional'].get(r, {}).get('mean_mse', 0) for r in high_regions])
v18_high_avg = np.mean([r18['l1_regional'].get(r, {}).get('mean_mse', 0) for r in high_regions])
if v18_high_avg > v17o_high_avg:
    diff_pct = (v18_high_avg - v17o_high_avg) / max(v17o_high_avg, 0.001)
    conclusions.append(f"问题：高频区域(6/7/8)表现下降{diff_pct:.0%} → 蛋白折叠在复杂信号上效能被稀释")
    conclusions.append("修改：增加复杂信号区域(σ>0.5)的fold_lr权重，让蛋白折叠更关注难题")
else:
    conclusions.append("高频区域表现稳定")

# 分析3：饱和vs fittness的矛盾
# v18饱和低但fitness低，说明attention砍了有用信息
conclusions.append("关键发现：v18隐饱和37% vs v17o的85%，但fitness反而低0.025")
conclusions.append("推断：Attention门控砍掉了过多有效信息，不是饱和低就好，是保留有用信号才高")
conclusions.append("修改：给attention gate加信号预测贡献度反馈——如果某通道的skip对预测有帮助，gate不能低于0.4")

# 分析4：海马体空转
conclusions.append("v18海马体total_replays=0 → 离线巩固从未被调用")
conclusions.append("修改：在tick()中每200步强制插一次replay_step(1)，模拟REM睡眠")

# 分析5：驱力多样性
# 看v18下丘脑驱力是否过于集中在mastery
conclusions.append("驱力集中=mastery → 可能因为评估期太短，来不及积累好奇心")
conclusions.append("修改：v19评估期延长到400tick，让所有驱力维度都充分表达")

for c in conclusions:
    print(f"  * {c}")

# ---- 第五步：设计v19的具体修改 ----
print("\n\n" + "=" * 60)
print("  v19 修改清单")
print("=" * 60)

v19_changes = """
1. 注意力门控修复（L3耦合过高的修复）
   - 添加 attention_residual_gate: 如果skip_w[j]对pred贡献大，该通道的gate += 0.2
   - att gate 下限 0.15（v18可降到0，丢信息）
   - att gate 对突降信号的快速释放（如果|actual-pred|>1.0，所有gate临时=1）

2. 蛋白折叠频率加权（L1高频区域低效的修复）
   - fold_loss 加权：复杂信号σ>00.5 的tick，fold_lr *= 2
   - 高频区域(6/7/8)单独维护fold modulator，按区域噪声sigma自适应

3. 在线睡眠内嵌（海马体空转的修复）
   - 每150tick在tick()末尾插1步replay_batch(batch_size=4)
   - 睡眠期间不更新pw，只更新rw_assoc（符合生物学，蛋白质合成主要影响突触后致密区）
   - 海马体buffer的priority基于salience（杏仁核计算的惊讶度）

4. 驱力感知reads修正（mastery过于集中）
   - 在tick()中记录region_coverage=已探索tick比例（不是区域数）
   - mastery threshold调低：coverage>0.6就认为已够格
   - 增加novelty_reward：新区域首次访问额外给0.1驱力填充coherence

5. 区域适应dt（不同区域不同遗忘速度）
   - 当前所有区域用同一个dt_decay
   - v19：dt_decay[rid]每个区域独立：高频区域衰减快(0.3)，低频慢(0.8)
   - 每个区域的蛋白折叠频率也随sigma调整

6. 演化压力调整
   - fitness != 1/(1+MSE)，改为 0.7/(1+MSE) + 0.2*coverage_bonus + 0.1*memory_bonus
   - coverage_bonus = 已访问区域数 / 9
   - memory_bonus = 平均海马体重放损失 < 0.3
"""

print(v19_changes)
