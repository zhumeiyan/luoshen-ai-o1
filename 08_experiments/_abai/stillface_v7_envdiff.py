#!/usr/bin/env python3
"""v7 环境差异实验 · 两独立引擎同初始状态对照
设计:
  - 引擎 A (基线): 光=90 温=297, TICK=60000
  - 引擎 B (扰动): 光=72 温=237 (-20%), TICK=60000
  - 共同: seed=7, 存货=0, 推进节奏同
  - 每 500 tick 采五项
  - 唯一变量 = 环境光/温
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
        if i % 40 == 0:
            print(f'{标签} frame {i}/{FRAMES} t={tick}', flush=True)
    return 日志, e

print('[v7_envdiff] TICK=60000 seed=7 存货=0', flush=True)

print('\n=== A 基线 光=90 温=297 ===', flush=True)
日志A, 引擎A = 跑引擎(90, 297, 'A_基线')
print(f'A done tick={引擎A.时刻} 饿={引擎A.hunger} 醒={引擎A.awake} 哭={引擎A.哭合计}', flush=True)

print('\n=== B 扰动 光=72 温=237 ===', flush=True)
日志B, 引擎B = 跑引擎(72, 237, 'B_扰动')
print(f'B done tick={引擎B.时刻} 饿={引擎B.hunger} 醒={引擎B.awake} 哭={引擎B.哭合计}', flush=True)

# === 末态对比 ===
print('\n=== 末态对比 ===')
末A = 日志A[-1]
末B = 日志B[-1]
print(f'A末: {json.dumps(末A, ensure_ascii=False)}')
print(f'B末: {json.dumps(末B, ensure_ascii=False)}')
print()
for k in ['总激活','最大激活','筋树数','节点亮','NaN','饿','醒','哭合计']:
    a, b = 末A[k], 末B[k]
    diff = (b - a) if isinstance(a, (int,float)) else 'N/A'
    pct = round(diff/abs(a)*100, 2) if isinstance(a,(int,float)) and a != 0 else 'N/A'
    print(f'  {k}: A={a} B={b} 差={diff} 变化%={pct}')

# 饿轨迹对比
print('\n=== 饿轨迹对比 (回合末) ===')
for t in [10000, 20000, 30000, 40000, 50000, 60000]:
    a = next((r for r in 日志A if r['tick']==t), None)
    b = next((r for r in 日志B if r['tick']==t), None)
    if a and b:
        diff_h = round(b['饿'] - a['饿'], 2)
        diff_c = b['哭合计'] - a['哭合计']
        diff_act = round(b['总激活'] - a['总激活'], 4)
        print(f'  t={t}: A饿={a["饿"]:5.1f} B饿={b["饿"]:5.1f} 饿差={diff_h:+5.1f} | A哭={a["哭合计"]:5d} B哭={b["哭合计"]:5d} 哭差={diff_c:+5d} | A激活={a["总激活"]:7.3f} B激活={b["总激活"]:7.3f} 激活差={diff_act:+.3f}')

落盘 = {
    'version': 'v7_envdiff',
    '设计': '两独立引擎 seed=7 存货=0 TICK=60000 唯一变量=环境',
    'A_基线_光90_温297_末': 末A,
    'B_扰动_光72_温237_末': 末B,
    'A_日志条数': len(日志A),
    'B_日志条数': len(日志B),
}

OUT = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v7_envdiff.json'
with open(OUT, 'w') as f:
    json.dump(落盘, f, ensure_ascii=False, indent=2)
print(f'\n落盘: {OUT}')
