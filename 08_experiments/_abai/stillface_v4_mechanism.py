#!/usr/bin/env python3
"""v4 机制层：哭停后饿再升 → 看哭是否重启
与 v3 的区别：
  - v3 在 tick=50000 提前喂 (哭从未启动)
  - v4 在 tick=60001 喂，此时哭已积累至 2000
  - v4 之后完全放开，不再喂，看饿自然回升
  - v4 在关键窗口 logging 每 5 tick（非仅回合边界）
"""
import sys, json
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')

from 身体稳态 import 身体

FRAMES = 346
TICKS_PER_FRAME = 250
TOTAL = FRAMES * TICKS_PER_FRAME  # 86500

e = 身体(节点数=96, 种子=7)
e.存货 = 0  # 清掉出厂400，模拟无供给

轮内日志 = []
喂事件 = []
上次哭合计 = 0
已喂 = False

def 记录(tick, tag=''):
    轮内日志.append({
        'tick': int(tick),
        '饿': round(float(e.hunger),2),
        '醒': int(e.awake),
        '哭合计': int(e.哭合计),
        '存货': int(e.存货),
        '_哭计数': int(getattr(e,'_哭计数',0)),
        'Δ哭': int(e.哭合计 - 上次哭合计),
        'tag': tag,
    })
    return e.哭合计

print('[v4_mechanism] seed=7 总tick=', TOTAL)
print('阶段1: 基线 50000tick 不喂，看哭自然积累')
print('阶段2: tick=60001 喂一次')
print('阶段3: 完全放开，每5tick采样')

for i in range(FRAMES):
    # 感知驱动 (简化)
    e.环境['光'] = 90
    e.环境['温'] = 297

    for j in range(TICKS_PER_FRAME):
        e.推进()
        tick = int(e.时刻)

        # === 喂：阶段2，tick=60001 一次 ===
        if 60000 < tick <= 60010 and not 已喂:
            if e.hunger >= 11.0:
                前 = float(e.hunger)
                e.hunger = max(0.0, e.hunger - 8.0)
                后 = float(e.hunger)
                喂事件.append({'tick': int(tick), '饿降前': 前, '饿降后': 后, '原因': '轮6团圆'})
                已喂 = True
                print(f'[喂] tick={tick} 饿 {前}→{后}')

        # === 采样节奏 ===
        if tick <= 40000:
            # 基线：每2000tick记一次
            if tick % 2000 == 0:
                ret = 记录(tick, '基线')
                上次哭合计 = ret
        elif tick <= 60000:
            # 哭积累期：每500tick记一次
            if tick % 500 == 0:
                ret = 记录(tick, '哭积累')
                上次哭合计 = ret
        elif tick <= 60010:
            # 喂前喂后密集：记全部
            记录(tick, '喂窗口')
            上次哭合计 = e.哭合计
        else:
            # 哭停后观察：每5tick记
            if tick % 5 == 0:
                ret = 记录(tick, '观察')
                # 实时报告哭重启
                if ret > 上次哭合计:
                    print(f'[哭重启!] tick={tick} 饿={e.hunger} 哭合计={ret} Δ={ret-上次哭合计}')
                上次哭合计 = ret

# === 输出 ===
print('\n=== 喂事件原文 ===')
for fe in 喂事件:
    print(json.dumps(fe, ensure_ascii=False))

print('\n=== 轮内日志关键片段 ===')
关键 = [r for r in 轮内日志 if (
    (r['tick']>=59900 and r['tick']<=60100) or
    (r['tick']>=69900 and r['tick']<=70100) or
    (r['tick']>=79900 and r['tick']<=80100) or
    (r['tick']>=50000 and r['tick']<=50020) or
    r['Δ哭'] != 0
)]
for r in 关键:
    print(json.dumps(r, ensure_ascii=False))

# 分期统计
分期 = {'基线(1-40000)':[r for r in 轮内日志 if '基线' in r['tag']],
        '哭积累(40001-60000)':[r for r in 轮内日志 if '哭积累' in r['tag']],
        '喂窗口(60001-60100)':[r for r in 轮内日志 if '喂' in r['tag']],
        '观察(60011-86500)':[r for r in 轮内日志 if '观察' in r['tag']]}

print('\n=== 分期统计 ===')
for k,v in 分期.items():
    if not v:
        print(f'{k}: 无数据')
        continue
    饿序列 = [r['饿'] for r in v]
    哭序列 = [r['哭合计'] for r in v]
    Δ哭 = [r['Δ哭'] for r in v if r['Δ哭']!=0]
    print(f'{k}: tick数={len(v)} 饿范围=[{min(饿序列)},{max(饿序列)}] 哭合计范围=[{min(哭序列)},{max(哭序列)}] Δ哭非零次数={len(Δ哭)}')

落盘 = {
    'version': 'v4_mechanism',
    '喂事件': 喂事件,
    '轮内日志_count': len(轮内日志),
    '轮内日志_关键片段': 关键,
    '分期统计': {k: {
        'tick数': len(v),
        '饿_min': min((r['饿'] for r in v), default=None),
        '饿_max': max((r['饿'] for r in v), default=None),
        '哭_min': min((r['哭合计'] for r in v), default=None),
        '哭_max': max((r['哭合计'] for r in v), default=None),
    } for k,v in 分期.items()},
}

out_path = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v4_mechanism.json'
with open(out_path, 'w') as f:
    json.dump(落盘, f, ensure_ascii=False, indent=2)
print(f'\n落盘: {out_path}')
