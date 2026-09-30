#!/usr/bin/env python3
"""v6 环境反向实验 · 两轮对照
设计:
  - 同一身体 (seed=7, 存货=0) 不重置
  - 第 0 轮 (基线): 光=90 温=297
  - 第 1 轮 (扰动): 光=72 温=237 (各 -20%)
  - 每轮 86500 tick, 每 500 tick 记录五项
  - 只改环境输入，不碰引擎代码
"""
import sys, json, math
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')

from 身体稳态 import 身体

e = 身体(节点数=96, 种子=7)
e.存货 = 0

FRAMES = 346
TICKS_PER_FRAME = 250
TOTAL = FRAMES * TICKS_PER_FRAME  # 86500

def 五项():
    """总激活/最大激活/筋树数/节点亮(>0.5)/NaN"""
    act = e.激活
    总 = sum(act)
    最 = max(act) if act else 0.0
    筋 = len(e.筋树)
    亮 = sum(1 for x in act if x > 0.5)
    nan = sum(1 for x in act if x != x)
    return {
        '总激活': round(总, 4),
        '最大激活': round(最, 4),
        '筋树数': 筋,
        '节点亮': 亮,
        'NaN': nan,
    }

def 跑一轮(光, 温, 轮名):
    """跑 86500 tick，每 500 tick 记一轮"""
    日志 = []
    for i in range(FRAMES):
        e.环境['光'] = 光
        e.环境['温'] = 温
        for _ in range(TICKS_PER_FRAME):
            e.推进()
            tick = int(e.时刻)
            if tick % 500 == 0:
                记录 = 五项()
                记录['tick'] = tick
                记录['轮'] = 轮名
                记录['饿'] = round(float(e.hunger), 2)
                记录['醒'] = int(e.awake)
                记录['哭合计'] = int(e.哭合计)
                记录['存货'] = int(e.存货)
                日志.append(记录)
    return 日志

print('[v6_envreverse] seed=7 存货=0')

print('\n=== 第 0 轮: 基线 光=90 温=297 ===')
基线 = 跑一轮(90, 297, 'R0_基线')
print(f'第0轮完成, 日志条数={len(基线)}')
print(f'基线末条: {json.dumps(基线[-1], ensure_ascii=False)}')

# 保存断点状态用于审计
断点_饿 = float(e.hunger)
断点_醒 = int(e.awake)
断点_哭整 = int(e.哭合计)
断点_时 = int(e.时刻)
print(f'断点状态: 饿={断点_饿} 醒={断点_醒} 哭合计={断点_哭整} 时刻={断点_时}')

print('\n=== 第 1 轮: 扰动 光=72 温=237 (-20%) ===')
扰动 = 跑一轮(72, 237, 'R1_扰动')
print(f'第1轮完成, 日志条数={len(扰动)}')
print(f'扰动末条: {json.dumps(扰动[-1], ensure_ascii=False)}')

# === 差异分析 ===
print('\n=== 末态对比 ===')
基线末 = 基线[-1]
扰动末 = 扰动[-1]
for k in ['总激活', '最大激活', '筋树数', '节点亮', 'NaN', '饿', '醒', '哭合计']:
    基 = 基线末[k]
    扰 = 扰动末[k]
    diff = 扰 - 基 if isinstance(基, (int, float)) else 'N/A'
    if isinstance(diff, float):
        diff = round(diff, 4)
    pct = round(diff/基*100, 2) if isinstance(基, (int, float)) and 基 != 0 else 'N/A'
    print(f'  {k}: 基线={基}, 扰动={扰}, 差={diff}, 变化%={pct}')

print('\n=== 饿轨迹对比 (每 5000 tick 末值) ===')
for t in [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000, 86500]:
    基 = next((r for r in 基线 if r['tick'] == t), None)
    扰 = next((r for r in 扰动 if r['tick'] == t), None)
    if 基 and 扰:
        diff_h = round(扰['饿'] - 基['饿'], 2)
        diff_c = 扰['哭合计'] - 基['哭合计']
        print(f'  tick={t}: 基饿={基["饿"]} 扰饿={扰["饿"]} 饿差={diff_h}, 哭差={diff_c}')

# 全部日志落盘
落盘 = {
    'version': 'v6_envreverse',
    'engine_seed': 7,
    'engine_存货': 0,
    'frame_count': FRAMES,
    'ticks_per_frame': TICKS_PER_FRAME,
    'total_ticks_per_round': TOTAL,
    'rounds': [
        {'轮名': 'R0_基线', '光': 90, '温': 297},
        {'轮名': 'R1_扰动', '光': 72, '温': 237},
    ],
    '断点状态_轮0末_轮1前': {
        '饿': 断点_饿,
        '醒': 断点_醒,
        '哭合计': 断点_哭整,
        '时刻': 断点_时,
    },
    '基线日志_条数': len(基线),
    '扰动日志_条数': len(扰动),
    '基线_首条': 基线[0],
    '基线_末条': 基线末,
    '扰动_末条': 扰动末,
}

out = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v6_envreverse.json'
with open(out, 'w') as f:
    json.dump(落盘, f, ensure_ascii=False, indent=2)
print(f'\n落盘: {out}')
