#!/usr/bin/env python3
"""v5 关一（学）：多次哭-停循环，看哭是否分化
设计:
  - TOTAL=140000 (560帧等效)
  - 存货=0, seed=7
  - 外部喂时刻: [55000, 85000, 115000, 135000]
  - 每次喂: 饿-=8 (模拟大人响应)
  - 预期: 4 次哭启动，若速率一致 → 节拍器成立，0.5月龄
"""
import sys, json
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')

from 身体稳态 import 身体

FRAMES = 560
TICKS_PER_FRAME = 250
TOTAL = FRAMES * TICKS_PER_FRAME  # 140000

e = 身体(节点数=96, 种子=7)
e.存货 = 0

喂时刻列表 = [55000, 85000, 115000, 135000]
喂事件 = []
哭事件 = []  # 每一次哭启动/停止
轮内日志 = []

上次哭合计 = 0
上次饿 = 0.0
哭状态 = False  # 当前是否在哭
当前哭起始_tick = None
当前哭起始_合计 = 0

def 记录(tick, tag=''):
    return {
        'tick': int(tick),
        '饿': round(float(e.hunger),2),
        '醒': int(e.awake),
        '哭合计': int(e.哭合计),
        '存货': int(e.存货),
        '哭状态': int(哭状态),
        'tag': tag,
    }

print('[v5_gate1] seed=7 总tick=', TOTAL)
print('喂时刻:', 喂时刻列表)

for i in range(FRAMES):
    e.环境['光'] = 90
    e.环境['温'] = 297

    for j in range(TICKS_PER_FRAME):
        e.推进()
        tick = int(e.时刻)

        # === 喂 ===
        if tick in 喂时刻列表 and tick not in [fe['tick'] for fe in 喂事件]:
            if e.hunger >= 8.0:
                前 = float(e.hunger)
                e.hunger = max(0.0, e.hunger - 8.0)
                后 = float(e.hunger)
                喂事件.append({'tick': tick, '饿降前': 前, '饿降后': 后})

        # === 哭状态检测 ===
        在哭 = (e.awake == 1 and e.hunger >= 12.0)
        if 在哭 and not 哭状态:
            # 哭启动
            哭状态 = True
            当前哭起始_tick = tick
            当前哭起始_合计 = int(e.哭合计)
        elif not 在哭 and 哭状态:
            # 哭停止
            哭状态 = False
            哭事件.append({
                '起始_tick': 当前哭起始_tick,
                '停止_tick': tick,
                '起始_合计': 当前哭起始_合计,
                '停止_合计': int(e.哭合计),
                '哭次数': int(e.哭合计 - 当前哭起始_合计),
                '持续tick': tick - 当前哭起始_tick,
            })

        # === 采样 ===
        if tick % 5000 == 0 or (tick % 5 == 0 and 50000 <= tick <= 52000) or (tick % 5 == 0 and 哭状态):
            轮内日志.append(记录(tick, '哭' if 哭状态 else '静'))

# 若结束时还在哭
if 哭状态:
    哭事件.append({
        '起始_tick': 当前哭起始_tick,
        '停止_tick': int(e.时刻),
        '起始_合计': 当前哭起始_合计,
        '停止_合计': int(e.哭合计),
        '哭次数': int(e.哭合计 - 当前哭起始_合计),
        '持续tick': int(e.时刻) - 当前哭起始_tick,
    })

# === 分期分析 - 按哭轮次 ===
print('\n=== 喂事件原文 ===')
for fe in 喂事件:
    print(json.dumps(fe, ensure_ascii=False))

print('\n=== 哭轮次原文 ===')
for i, ce in enumerate(哭事件):
    print(f'哭轮次{i+1}: {json.dumps(ce, ensure_ascii=False)}')

print('\n=== 哭轮次统计 ===')
真实哭 = [ce for ce in 哭事件 if ce['持续tick'] > 0]  # 过滤 0-tick 伪轮次
if 真实哭:
    哭次数列 = [ce['哭次数'] for ce in 真实哭]
    持续列 = [ce['持续tick'] for ce in 真实哭]
    速率列 = [ce['哭次数'] / ce['持续tick'] for ce in 真实哭]
    print(f'哭轮次数(真实/全部): {len(真实哭)}/{len(哭事件)}')
    print(f'哭次/轮: {哭次数列}')
    print(f'持续tick/轮: {持续列}')
    print(f'哭速率(次/tick): {[round(r,4) for r in 速率列]}')
    print(f'速率均值: {round(sum(速率列)/len(速率列),4)}')
    print(f'速率标准差: {round((sum((r-sum(速率列)/len(速率列))**2 for r in 速率列)/len(速率列))**0.5,4)}')
    print(f'速率范围: [{min(速率列)}, {max(速率列)}]')
    if max(速率列) - min(速率列) < 0.001:
        print('判决: 速率无分化 → 节拍器成立 → 0.5月龄一致')
    else:
        print('判决: 速率有分化 → 可能有学习成分 → 需更多实验')

# 落盘
落盘 = {
    'version': 'v5_gate1',
    '喂时刻列表': 喂时刻列表,
    '喂事件': 喂事件,
    '哭轮次': 哭事件,
    '轮内日志_count': len(轮内日志),
}

out = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v5_gate1.json'
with open(out, 'w') as f:
    json.dump(落盘, f, ensure_ascii=False, indent=2)
print(f'\n落盘: {out}')
