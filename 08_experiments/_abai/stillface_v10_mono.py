#!/usr/bin/env python3
"""v10 单曲验证: 同一身体, 温297(基线)→温267(±20%), 看5项指标断脐前后
只允许改环境输入, 不动引擎
"""
import sys, json, math
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
from 身体稳态 import 身体, HUNG_CRY

FRAMES = 390; TPF = 250; TOTAL = FRAMES * TPF  # 97500
e = 身体(节点数=96, 种子=7)
e.存货 = 0

def snap(tick):
    总 = sum(abs(x) for x in e.激活)
    最大 = max(abs(x) for x in e.激活)
    筋 = len(e.筋树)
    亮 = sum(1 for x in e.激活 if abs(x) > 0.5)
    nan = any(math.isnan(x) for x in e.激活)
    return {'t':tick,'总激活':round(总,4),'最大激活':round(最大,4),'筋树':筋,'节点亮':亮,'NaN':nan,
            '饿':round(e.hunger,3),'醒':int(e.awake),'哭合计':int(e.哭合计),'存货':int(e.存货)}

samples = []
last_env = None
for f in range(FRAMES):
    # === 每 tick 环境输入 ===
    # R0: tick 0..50000 光90/声45/力10/温297
    # R1: tick 50000..97500 光90/声45/力10/温267 (-10%, >288 不归零)
    tick_start = f * TPF
    if tick_start < 50000:
        温 = 297; 标签 = 'R0'
    else:
        温 = 267; 标签 = 'R1'
    e.环境['光'] = 90
    e.环境['声'] = 45
    e.环境['力'] = 10
    e.环境['温'] = 温

    for _ in range(TPF):
        e.推进()
        tick = int(e.时刻)
        if tick % 2000 == 0 or (tick > 49500 and tick < 50500) or tick == TOTAL-1:
            s = snap(tick); s['env'] = 标签; s['温输入'] = 温
            samples.append(s)

print('=== v10 (基线 297 → 断脐至 267, 种子=7) ===')
print(f'total tick = {TOTAL}')
for s in samples: print(json.dumps(s, ensure_ascii=False))

# 50000 前后的 总激活 对比
pre = [s['总激活'] for s in samples if s['t'] < 50000][-3:]
post = [s['总激活'] for s in samples if s['t'] >= 50000][:3]
print(f'\n50000前3采样: {pre}')
print(f'50000后3采样: {post}')
print(f'50000前饿: {[s["饿"] for s in samples if s["t"]<50000][-1]}')
print(f'50000后饿: {[s["饿"] for s in samples if s["t"]>=50000][0]}')

out_path = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v10_mono.json'
with open(out_path, 'w') as f: json.dump(samples, f, ensure_ascii=False, indent=2)
print(f'落盘: {out_path}')
