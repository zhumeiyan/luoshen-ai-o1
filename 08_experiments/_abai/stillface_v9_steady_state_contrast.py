#!/usr/bin/env python3
"""v9 稳态对比验证: 力/温是有效通道 → 应该能测出稳态差异
设计: 4 条件 (基线 / 力×1.5 / 温-6 / 力+温同变), 每条件独立引擎种子=7
每条件跑 20000 tick, 汇报 total/max激活/筋树数/节点亮/NaN
"""
import sys, json, math
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
from 身体稳态 import 身体, HUNG_CRY

TICKS = 20000
CONDITIONS = {
    'A=基线(光90/声45/力10/温297)': dict(光=90, 声=45, 力=10, 温=297),
    'B=力×1.5(光90/声45/力15/温297)': dict(光=90, 声=45, 力=15, 温=297),
    'C=温-6(光90/声45/力10/温291)': dict(光=90, 声=45, 力=10, 温=291),
    'D=力×1.5且温+9(光90/声45/力15/温306)': dict(光=90, 声=45, 力=15, 温=306),
}

def run_condition(标签, env):
    e = 身体(节点数=96, 种子=7)
    e.存货 = 0
    总激活系列 = []
    最大激活系列 = []
    筋树数系列 = []
    节点亮系列 = []
    nan_hit = False
    nan_detail = None

    for t in range(TICKS):
        e.环境['光'] = env['光']
        e.环境['声'] = env['声']
        e.环境['力'] = env['力']
        e.环境['温'] = env['温']
        e.推进()
        tick = int(e.时刻)
        总 = sum(abs(x) for x in e.激活)
        最大 = max(abs(x) for x in e.激活)
        筋 = len(e.筋树)
        亮 = sum(1 for x in e.激活 if abs(x) > 0.5)
        总激活系列.append(总)
        最大激活系列.append(最大)
        筋树数系列.append(筋)
        节点亮系列.append(亮)
        if math.isnan(总) or math.isnan(最大):
            if not nan_hit:
                nan_hit = True
                nan_detail = f'tick={tick}'

    return {
        '标签': 标签,
        'TICKS': TICKS,
        '总激活_last5k_avg': round(sum(总激活系列[-5000:])/5000, 4),
        '总激活_first5k_avg': round(sum(总激活系列[:5000])/5000, 4),
        '总激活_min': round(min(总激活系列), 4),
        '总激活_max': round(max(总激活系列), 4),
        '最大激活_last': round(最大激活系列[-1], 4),
        '筋树数_final': 筋树数系列[-1],
        '节点亮_final': 节点亮系列[-1],
        '节点亮_max': max(节点亮系列),
        'NaN': nan_hit,
        'NaN_detail': nan_detail,
        '饿_final': round(e.hunger, 4),
        '醒_final': int(e.awake),
        '哭合计_final': int(e.哭合计),
        'tick_100_snapshot': {
            '总激活': round(sum(abs(x) for x in e.激活), 4) if False else None
        },
    }

print('=== v9 稳态对比验证 ===')
print(f'每条件 tick={TICKS}, 独立引擎种子=7 (不延续)')
results = {}
for 标签, env in CONDITIONS.items():
    r = run_condition(标签, env)
    results[标签] = r
    print(f'\n[{r["标签"]}]')
    print(f'  总激活 last5k avg = {r["总激活_last5k_avg"]}')
    print(f'  总激活 first5k avg= {r["总激活_first5k_avg"]}')
    print(f'  总激活 min~max    = [{r["总激活_min"]}, {r["总激活_max"]}]')
    print(f'  最大激活 (final)  = {r["最大激活_last"]}')
    print(f'  筋树数 (final)    = {r["筋树数_final"]}')
    print(f'  节点亮 (final)    = {r["节点亮_final"]}   max={r["节点亮_max"]}')
    print(f'  NaN               = {r["NaN"]}' + (f' @ {r["NaN_detail"]}' if r["NaN"] else ''))
    print(f'  饿(final)={r["饿_final"]} 醒={r["醒_final"]} 哭合计={r["哭合计_final"]}')

print('\n=== 横向对比 ===')
A = results['A=基线(光90/声45/力10/温297)']
B = results['B=力×1.5(光90/声45/力15/温297)']
C = results['C=温-6(光90/声45/力10/温291)']
D = results['D=力×1.5且温+9(光90/声45/力15/温306)']
print(f'A→B 力×1.5: 总激活 {A["总激活_last5k_avg"]}→{B["总激活_last5k_avg"]}  筋{A["筋树数_final"]}→{B["筋树数_final"]}')
print(f'A→C 温-6:   总激活 {A["总激活_last5k_avg"]}→{C["总激活_last5k_avg"]}  筋{A["筋树数_final"]}→{C["筋树数_final"]}')
print(f'A→D 力+温:  总激活 {A["总激活_last5k_avg"]}→{D["总激活_last5k_avg"]}  筋{A["筋树数_final"]}→{D["筋树数_final"]}')
print(f'A vs B 激活差= {round(B["总激活_last5k_avg"]-A["总激活_last5k_avg"],4)}')
print(f'A vs D 激活差= {round(D["总激活_last5k_avg"]-A["总激活_last5k_avg"],4)}')

out_path = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v9_steady_state_contrast.json'
with open(out_path, 'w') as f: json.dump(results, f, ensure_ascii=False, indent=2)
print(f'\n落盘: {out_path}')
