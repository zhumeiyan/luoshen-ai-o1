#!/usr/bin/env python3
"""trace_情绪_v5_猫爪版.py —— 预跑密封卷公式,落盘实测值,与手算对照"""
import sys, os, json, math
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
from 身体稳态 import 身体
import numpy as np

SEED = 7
TOTAL_TICK = 200_000
SAMPLE_EVERY = 500
INV0 = 20  # 喵

e = 身体(节点数=96, 种子=SEED)
e.存货 = INV0

samples = []
monotonic_violations = 0
mono_direction_samples = []

print(f'=== trace_情绪_v5_猫爪版 | seed={SEED} | 存货₀={INV0} | total={TOTAL_TICK} ===')
print()

for tick in range(TOTAL_TICK):
    e.环境['光'] = 90; e.环境['声'] = 45; e.环境['力'] = 10; e.环境['温'] = 297
    e.推进()

    if (tick + 1) % SAMPLE_EVERY == 0:
        t = int(e.时刻)
        samples.append({
            'tick': t, '存货': int(e.存货), 'hunger': round(float(e.hunger), 4),
            '情绪色价': round(float(e.情绪色价), 6), '轮': t // 10000,
        })

    if (tick + 1) % 50_000 == 0:
        print(f'  [{(tick+1)//1000}k tick] 存货={int(e.存货):>4} 饿={e.hunger:>6.2f} 情绪={e.情绪色价:.4f}', flush=True)

print(f'\n采样完成,共 {len(samples)} 个样本\n')

# === 预测 1: 存货≤15 后单调性 ===
print('=== 预测 1: 存货≤15 后单调性 ===')
below_15 = [s for s in samples if s['存货'] <= 15]
if below_15:
    情绪_seq = [s['情绪色价'] for s in below_15]
    ups = sum(1 for i in range(1,len(情绪_seq)) if 情绪_seq[i] > 情绪_seq[i-1])
    total = len(情绪_seq) - 1
    up_pct = ups/total*100 if total else 0
    print(f'  存货≤15 样本数: {len(below_15)}')
    print(f'  情绪色价范围: [{min(情绪_seq):.6f}, {max(情绪_seq):.6f}]')
    print(f'  上升步数: {ups}/{total} = {up_pct:.1f}%')
    print(f'  非单调 < 5%: {"PASS" if up_pct < 5 else "FAIL"}')
    print()

# === 预测 2: 存货=0 时情绪 < 0.40 ===
print('=== 预测 2: 存货=0 情绪阈值 < 0.40 ===')
inv0_samples = [s for s in samples if s['存货'] == 0]
if inv0_samples:
    情绪_at_0 = inv0_samples[-1]['情绪色价']
    情绪_mean_0 = np.mean([s['情绪色价'] for s in inv0_samples[-10:]])
    print(f'  存货=0 样本数: {len(inv0_samples)}')
    print(f'  存货=0 最终条情绪色价: {情绪_at_0:.6f}')
    print(f'  存货=0 末 10 条均值: {情绪_mean_0:.6f}')
    print(f'  < 0.40 预测: {"PASS" if 情绪_mean_0 < 0.40 else "FAIL"}')
    print()

# === 预测 3: 喂事件响应幅度 > 0.03 ===
print('=== 预测 3: 喂事件响应幅度 > 0.03 ===')
feed_events = []
for i in range(1, len(samples)):
    if samples[i]['存货'] > samples[i-1]['存货']:
        continue  # 存货升了 = 不是喂
    # 找到从"高→低"跳变点 = 喂(扣 4 存货)
    if samples[i-1]['存货'] - samples[i]['存货'] >= 4:
        # 找到喂前 10 条和喂后 10 条的情绪均值差
        pre = samples[max(0,i-10):i]
        post = samples[i:min(len(samples),i+10)]
        pre_mean = np.mean([s['情绪色价'] for s in pre])
        post_mean = np.mean([s['情绪色价'] for s in post])
        delta = abs(post_mean - pre_mean)
        feed_events.append({
            'tick': samples[i]['tick'], '存货': samples[i]['存货'],
            'pre_mean': round(pre_mean, 6), 'post_mean': round(post_mean, 6),
            'delta': round(delta, 6), 'PASS': delta > 0.03
        })
        print(f'  feed @ tick={samples[i]["tick"]}: Δ={delta:.4f} {"PASS" if delta > 0.03 else "FAIL"}')

if feed_events:
    pass_rate = sum(1 for f in feed_events if f['PASS']) / len(feed_events)
    print(f'  喂事件通过率: {pass_rate:.0%}')
print()

# === 关键快照 ===
print('=== 关键快照(存货 vs 情绪) ===')
print(f'{"轮":>5} {"存货":>5} {"饿":>8} {"情绪":>10}')

# 每 5 轮抽一次
seen_rounds = set()
for s in samples:
    if s['轮'] % 5 == 0 and s['轮'] not in seen_rounds:
        seen_rounds.add(s['轮'])
        print(f'{s["轮"]:>5} {s["存货"]:>5} {s["hunger"]:>8.2f} {s["情绪色价"]:>10.4f}')

# === 与密封卷手算值对照 ===
print()
print('=== 密封卷手算 v.s. 引擎实测对照 ===')
hand_calc = {
    '刚开局(存货20)': {'存货': 20, '手算v5': 0.536, 'v3实测': 0.5253},
    '半饥饿(存货8)': {'存货': 8, '手算v5': 0.492, 'v3实测': 0.5383},
    '纯缺(存货0)': {'存货': 0, '手算v5': 0.411, 'v3实测': 0.3662},
}
for name, vals in hand_calc.items():
    if vals['存货'] == 20:
        inv_match = [s for s in samples if s['存货'] == 20]
    elif vals['存货'] == 8:
        inv_match = [s for s in samples if 7 <= s['存货'] <= 9]
    else:
        inv_match = [s for s in samples if s['存货'] == 0]
    if inv_match:
        eng_mean = np.mean([s['情绪色价'] for s in inv_match[:5]])
        print(f'  {name}:手算={vals["手算v5"]:.3f}, 引擎={eng_mean:.4f}, 偏差={abs(eng_mean-vals["手算v5"])*100:.1f}%')

# === 落盘 ===
out_path = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/trace_情绪_v5_猫爪实测.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({'seed': SEED, 'total_tick': TOTAL_TICK, 'run_type': '猫爪密封卷预跑',
               'samples': samples, 'feed_events': feed_events,
               'predictions': {'non_mono_pct': up_pct if below_15 else None,
                               'pure_hunger_threshold': 情绪_mean_0 if inv0_samples else None,
                               'feed_pass_rate': pass_rate if feed_events else None}},
              f, ensure_ascii=False, indent=2)
print(f'\n落盘: {out_path}')
