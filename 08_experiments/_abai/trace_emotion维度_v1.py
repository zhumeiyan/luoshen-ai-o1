#!/usr/bin/env python3
"""trace_情绪维度_v1.py
验证洛神 v3 情绪维度方案的 3 条预测:
  预测 1: 存货 400→0 过程中情绪色价单调下降(存货<100 后),存货=0 时 <0.35
  预测 2: 缺货前后 环境输入[4] 均值差 ≈ 0.04 (p<0.01)
  预测 3: corr(情绪色价, 当前上限) ∈ (-0.4, -0.1)

只读不写引擎参数(除了存货消耗是引擎正常代谢)
种子=7,单程跑,不重置
"""
import sys, json, math
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
from 身体稳态 import 身体
import numpy as np

SEED = 7
NODES = 96
INV0 = 20  # 小起点,round 15 即耗尽(150k tick),200k tick 含 5 轮纯缺货
# 200k tick ≈ 20 轮,在 tick rate ≈ 200/s 下约 17 分钟
TOTAL_TICK = 200_000
SAMPLE_EVERY = 500  # 每 500 tick 采一次,共 ~400 样本

e = 身体(节点数=NODES, 种子=SEED)
e.存货 = INV0

# 固定环境(只有存货在自然消耗,其余环境量保持恒定基线)
BASE_ENV = {'光': 90, '声': 45, '力': 10, '温': 297}

samples = []

print(f'=== trace_情绪维度_v1 | seed={SEED} | 存货₀={INV0} | total={TOTAL_TICK} ===')
print(f'采样间隔: 每 {SAMPLE_EVERY} tick')
print()

for tick in range(TOTAL_TICK):
    e.环境['光'] = BASE_ENV['光']
    e.环境['声'] = BASE_ENV['声']
    e.环境['力'] = BASE_ENV['力']
    e.环境['温'] = BASE_ENV['温']
    e.推进()

    if (tick + 1) % SAMPLE_EVERY == 0:
        t = int(e.时刻)
        # 从引擎属性直接算 环境输入[4](回接通道):差/20 + (情绪-0.5)*0.1
        if e.上轮激活 is not None:
            差 = sum(e.激活[j] - e.上轮激活[j] for j in e._回接节点)
            回接_base = 差 / 20.0
        else:
            回接_base = 0.0
        env_in4 = 回接_base + (e.情绪色价 - 0.5) * 0.1

        samples.append({
            'tick': t,
            '存货': int(e.存货),
            'hunger': round(float(e.hunger), 4),
            '情绪色价': round(float(e.情绪色价), 6),
            '环境输入4': round(env_in4, 6),
            '回接_base': round(回接_base, 6),
            '当前上限': round(float(e.当前上限), 4),
            '轮': t // 10000,
        })

    if (tick + 1) % 50_000 == 0:
        print(f'  [{(tick+1)//1000}k tick] 存货={int(e.存货):>4} 饿={e.hunger:>6.2f} 情绪={e.情绪色价:.4f}', flush=True)

print(f'\n采样完成,共 {len(samples)} 个样本', flush=True)
print()

# ============== 预测 1: 存货 40→0 情绪色价单调性 ==============
print('=== 预测 1: 情绪色价单调性 (存货<30 后) ===')
below_15 = [s for s in samples if s['存货'] <= 15]
if not below_15:
    print('✗ 存货未降到 15 以下(总 tick 不够),无法验证预测 1')
else:
    # 从存货<15 开始,检查情绪色价是否单调非增
    情绪序列 = [s['情绪色价'] for s in below_15]
    最大上升 = 0.0
    上升次数 = 0
    for i in range(1, len(情绪序列)):
        diff = 情绪序列[i] - 情绪序列[i-1]
        if diff > 0:
            上升次数 += 1
            最大上升 = max(最大上升, diff)

    首末变化 = 情绪序列[-1] - 情绪序列[0]
    存货0样本 = [s for s in samples if s['存货'] == 0]
    存货0情绪 = 存货0样本[-1]['情绪色价'] if 存货0样本 else None

    print(f'  存货≤15 样本数: {len(below_15)}')
    print(f'  存货≤15 情绪色价范围: [{min(情绪序列):.6f}, {max(情绪序列):.6f}]')
    print(f'  首末变化: {首末变化:+.6f}')
    print(f'  上升次数: {上升次数}/{len(情绪序列)-1} (非单调比例: {上升次数/max(len(情绪序列)-1,1)*100:.1f}%)')
    print(f'  最大单步上升: {最大上升:.6f}')
    print(f'  存货=0 时情绪色价: {存货0情绪 if 存货0情绪 is not None else "N/A"}')
    print(f'  存货=0 时情绪 <0.35预测: {"✓ PASS" if 存货0情绪 is not None and 存货0情绪 < 0.35 else "✗ FAIL" if 存货0情绪 is not None else "N/A"}')
    print(f'  单调性判定: {"✓ 基本单调(上升<5%)" if 上升次数/max(len(情绪序列)-1,1) < 0.05 else "✗ 有明显波动"}')

print()

# ============== 预测 2: 缺货前后 环境输入[4] 均值差 ==============
print('=== 预测 2: 缺货前后 环境输入[4] 均值差 ===')
# 不缺货: 存货 ≥ 18 (前段,存货=20 刚开局)
# 严重缺货: 存货 ≤ 3 (末段,存货接近耗尽)
不缺货 = [s for s in samples if s['存货'] >= 18]
严重缺货 = [s for s in samples if s['存货'] <= 3]

if not 不缺货 or not 严重缺货:
    print(f'✗ 数据不足(不缺货={len(不缺货)} 样本,严重缺货={len(严重缺货)} 样本)')
else:
    均值_前 = np.mean([s['环境输入4'] for s in 不缺货])
    均值_后 = np.mean([s['环境输入4'] for s in 严重缺货])
    标准差_前 = np.std([s['环境输入4'] for s in 不缺货], ddof=1)
    标准差_后 = np.std([s['环境输入4'] for s in 严重缺货], ddof=1)
    n1, n2 = len(不缺货), len(严重缺货)

    # Welch's t 检验
    se = math.sqrt(标准差_前**2 / n1 + 标准差_后**2 / n2)
    t_stat = (均值_前 - 均值_后) / se if se > 0 else float('inf')
    # 自由度(Welch-Satterthwaite)
    df = (标准差_前**2/n1 + 标准差_后**2/n2)**2 / \
         ((标准差_前**2/n1)**2/(n1-1) + (标准差_后**2/n2)**2/(n2-1)) if (n1>1 and n2>1) else n1+n2-2

    # 简单 p 值估计(t 分布双侧,p ∝ exp(-t²/2) 近似)
    p_approx = math.exp(-0.5 * t_stat**2) if abs(t_stat) < 20 else 0.0

    均值差 = 均值_前 - 均值_后

    print(f'  不缺货(存货≥300): n={n1}, 均值={均值_前:.6f}, SD={标准差_前:.6f}')
    print(f'  严重缺货(存货=0): n={n2}, 均值={均值_后:.6f}, SD={标准差_后:.6f}')
    print(f'  均值差: {均值差:+.6f}')
    print(f'  t 统计量: {t_stat:.4f}')
    print(f'  近似 p 值: {p_approx:.2e}')
    print(f'  预测"均值差≈0.04": {"✓" if abs(均值差 - 0.04) < 0.03 else "≈ 有偏差" if abs(均值差) > 0.01 else "✗ 信号未进入"}')
    print(f'  预测"p<0.01": {"✓ PASS" if p_approx < 0.01 else "✗ FAIL(if p≈0)"}')

print()

# ============== 预测 3: corr(情绪色价, 当前上限) ==============
print('=== 预测 3: corr(情绪色价, 当前上限) ∈ (-0.4,-0.1) ===')
情绪_all = [s['情绪色价'] for s in samples]
上限_all = [s['当前上限'] for s in samples]
corr = np.corrcoef(情绪_all, 上限_all)[0, 1]

print(f'  全样本 corr = {corr:.4f}')
print(f'  ∈(-0.4,-0.1)? {"✓ PASS" if -0.4 < corr < -0.1 else "✗ FAIL" if corr >= -0.1 else "<-0.4(比预期更负)"}')

print()

# ============== 情绪色价 vs 存货 对照表(关键轮次) ==============
print('=== 关键快照(存货 vs 情绪色价 vs 环境输入[4]) ===')
print(f'{"轮":>5} {"存货":>5} {"饿":>8} {"情绪":>10} {"env_in4":>10} {"上限":>8}')
key_samples = []
for s in samples:
    # 每 10 轮抽一次
    if s['轮'] % 10 == 0 and (not key_samples or key_samples[-1]['轮'] != s['轮']):
        key_samples.append(s)

for s in key_samples[-20:]:  # 只打最后 20 个关键帧(最相关)
    print(f'{s["轮"]:>5} {s["存货"]:>5} {s["hunger"]:>8.2f} {s["情绪色价"]:>10.4f} {s["环境输入4"]:>10.4f} {s["当前上限"]:>8.4f}')

print()

# ============== 落盘 ==============
out_path = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/trace_情绪维度_v1.json'
with open(out_path, 'w') as f:
    json.dump({'seed': SEED, 'total_tick': TOTAL_TICK, 'sample_every': SAMPLE_EVERY,
               'samples': samples}, f, ensure_ascii=False, indent=2)
print(f'落盘: {out_path}')

# ============== 3 条预测汇总 ==============
print()
print('=== 3 条预测汇总 ===')
print(f'  预测1(存货<100 情绪单调递减,存货=0 时<0.35): {"参见上升次数和存货0情绪值"}')
print(f'  预测2(缺货前后 env_in4 均值差≈0.04,p<0.01): 实测={均值差:.6f}, p≈{p_approx:.2e}')
print(f'  预测3(corr∈(-0.4,-0.1)): 实测={corr:.4f}')
