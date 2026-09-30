#!/usr/bin/env python3
"""v7b 反例检验 · 温差拉至 ±50%
逻辑: 若诊断"代谢层不感知环境"正确 → 即使 50% 差, 饿/醒/哭仍应全等
      若饿/醒/哭出现差异 → 诊断被推翻
"""
import sys, json
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
from 身体稳态 import 身体

FRAMES = 240  # 60000 tick
TPF = 250
TOTAL = FRAMES * TPF

def 五项(e):
    act = e.激活
    return {
        '总激活': round(sum(act), 4),
        '最大激活': round(max(act) if act else 0.0, 4),
        '筋树数': len(e.筋树),
        '节点亮': sum(1 for x in act if x > 0.5),
        'NaN': sum(1 for x in act if x != x),
    }

def 跑引擎(光, 温, 标签):
    e = 身体(节点数=96, 种子=7)
    e.存货 = 0
    日志 = []
    for i in range(FRAMES):
        e.环境['光'] = 光
        e.环境['温'] = 温
        for _ in range(TPF):
            e.推进()
        tick = int(e.时刻)
        if tick % 500 == 0:
            r = 五项(e)
            r['tick'] = tick
            r['轮'] = 标签
            r['饿'] = round(float(e.hunger), 2)
            r['醒'] = int(e.awake)
            r['哭合计'] = int(e.哭合计)
            日志.append(r)
        if i % 60 == 0:
            print(f'{标签} frame {i}/{FRAMES} t={tick}', flush=True)
    return 日志, e

print('[v7b_falsify] 反例检验: ±50% 温差', flush=True)

# A: 基线 (v7 同款)
print('\n=== A 基线 光=90 温=297 ===', flush=True)
日志A, A = 跑引擎(90, 297, 'A_v7b_基线')

# B: 极端差 (50%)
print('\n=== B 极端 光=45 温=267 (-50%) ===', flush=True)
日志B, B = 跑引擎(45, 267, 'B_v7b_极端')

末A, 末B = 日志A[-1], 日志B[-1]

print('\n=== 末态 (t=60000) ===')
print(f'A: {json.dumps(末A, ensure_ascii=False)}')
print(f'B: {json.dumps(末B, ensure_ascii=False)}')
print()
for k in ['总激活','最大激活','筋树数','节点亮','NaN','饿','醒','哭合计']:
    a, b = 末A[k], 末B[k]
    diff = (b - a) if isinstance(a,(int,float)) else 'N/A'
    pct = round(diff/abs(a)*100, 2) if isinstance(a,(int,float)) and a != 0 else 'N/A'
    flag = '  ← 同' if (isinstance(diff,(int,float)) and abs(diff) < 0.001) else '  ✗ 异'
    print(f'  {k}: A={a}, B={b}, 差={diff}, 变化%={pct}{flag}')

# 逐回合饿/哭对比
print('\n=== 轨迹对比 ===')
any_diff = False
for t in [10000, 20000, 30000, 40000, 50000, 60000]:
    a = next((r for r in 日志A if r['tick']==t), None)
    b = next((r for r in 日志B if r['tick']==t), None)
    if a and b:
        dh = round(b['饿']-a['饿'], 2)
        dc = b['哭合计']-a['哭合计']
        da = round(b['总激活']-a['总激活'], 4)
        dw = int(b['醒']-a['醒'])
        if dh or dc or dw:
            any_diff = True
        print(f'  t={t}: 饿差={dh:+5.1f} 哭差={dc:+5d} 醒差={dw:+2d} 激活差={da:+.4f}')

判决 = '诊断得到强化 (代谢层仍不感知)' if not any_diff else '诊断被推翻 (代谢层对极端环境有响应)'
print(f'\n判决: {判决}')

落盘 = {
    'version': 'v7b_falsify',
    '设计': '反例检验: ±50% 温差, 若饿/醒/哭仍同则诊断成立, 若有差异则推翻',
    'A_末': 末A,
    'B_末': 末B,
    '轨迹_饿哭醒_有差异': any_diff,
    '判决': 判决,
}
OUT = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v7b_falsify.json'
with open(OUT, 'w') as f:
    json.dump(落盘, f, ensure_ascii=False, indent=2)
print(f'落盘: {OUT}')
