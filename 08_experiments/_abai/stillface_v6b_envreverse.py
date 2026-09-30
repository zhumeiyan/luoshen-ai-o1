#!/usr/bin/env python3
"""v6b 环境反向实验 · 两轮对照 · 缩短版
结构同 v6，唯一改动: 每轮 40000 tick (160 帧 x 250)
目的: 两轮串行可被合理时间内完成
落盘分两步: R0 完成即落，R1 完成即落
"""
import sys, json
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
from 身体稳态 import 身体

e = 身体(节点数=96, 种子=7)
e.存货 = 0

FRAMES_R0 = 160  # 40000 tick
FRAMES_R1 = 160  # 40000 tick
TPF = 250

def 五项():
    act = e.激活
    return {
        '总激活': round(sum(act), 4),
        '最大激活': round(max(act) if act else 0.0, 4),
        '筋树数': len(e.筋树),
        '节点亮': sum(1 for x in act if x > 0.5),
        'NaN': sum(1 for x in act if x != x),
    }

def 跑轮(光, 温, 轮名, 帧数):
    日志 = []
    for i in range(帧数):
        e.环境['光'] = 光
        e.环境['温'] = 温
        for _ in range(TPF):
            e.推进()
        tick = int(e.时刻)
        if tick % 500 == 0:
            r = 五项()
            r['tick'] = tick
            r['轮'] = 轮名
            r['饿'] = round(float(e.hunger), 2)
            r['醒'] = int(e.awake)
            r['哭合计'] = int(e.哭合计)
            r['存货'] = int(e.存货)
            日志.append(r)
        if i % 40 == 0:
            print(f'{轮名} frame {i}/{帧数} tick={tick}', flush=True)
    return 日志

OUT = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v6b_envreverse.json'

print('[v6b] seed=7 存货=0 R0=40000tick R1=40000tick', flush=True)

# ===== R0 =====
print('\n=== R0 基线 光=90 温=297 ===', flush=True)
R0 = 跑轮(90, 297, 'R0_基线', FRAMES_R0)
print(f'R0 done tick={e.时刻} 饿={e.hunger} 醒={e.awake} 哭={e.哭合计}', flush=True)

# 中间落盘（R0 已确定）
中间 = {
    'version': 'v6b_envreverse',
    'seed': 7, '存货': 0,
    'tick_per_round': FRAMES_R0 * TPF,
    'R0_光90_温297': {
        '末条': R0[-1] if R0 else None,
        '日志条数': len(R0),
    }
}
with open(OUT, 'w') as f:
    json.dump(中间, f, ensure_ascii=False, indent=2)
print(f'R0 已落盘: {OUT}', flush=True)

# ===== R1 =====
print('\n=== R1 扰动 光=72 温=237 (-20%) ===', flush=True)
R1 = 跑轮(72, 237, 'R1_扰动', FRAMES_R1)
print(f'R1 done tick={e.时刻} 饿={e.hunger} 醒={e.awake} 哭={e.哭合计}', flush=True)

落盘 = {
    'version': 'v6b_envreverse',
    'seed': 7, '存货': 0,
    'tick_per_round': FRAMES_R0 * TPF,
    'rounds': [
        {'轮名': 'R0_基线', '光': 90, '温': 297, 'tick数': FRAMES_R0 * TPF},
        {'轮名': 'R1_扰动', '光': 72, '温': 237, 'tick数': FRAMES_R1 * TPF},
    ],
    'R0_末条': R0[-1] if R0 else None,
    'R1_末条': R1[-1] if R1 else None,
    'R0_日志条数': len(R0),
    'R1_日志条数': len(R1),
}
with open(OUT, 'w') as f:
    json.dump(落盘, f, ensure_ascii=False, indent=2)
print(f'落盘完成: {OUT}', flush=True)

# 末态对比
if R0 and R1:
    print('\n=== 末态对比 ===')
    for k in ['总激活', '最大激活', '筋树数', '节点亮', 'NaN', '饿', '醒', '哭合计']:
        基 = R0[-1][k]
        扰 = R1[-1][k]
        diff = (扰 - 基) if isinstance(基, (int, float)) else 'N/A'
        print(f'  {k}: R0={基}, R1={扰}, 差={diff}')
