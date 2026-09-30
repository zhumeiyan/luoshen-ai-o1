#!/usr/bin/env python3
"""v8 信号审计: 逐通道抓tick级原值→归一→末梢差→感受器增益→激活贡献
只读不动引擎代码
"""
import sys, math, json
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
from 身体稳态 import 身体, HEARTBEAT_AMP, HEARTBEAT_PERIOD

TOTAL = 300  # tick
SKIP  = 20   # 只取前20tick + 每50tick

def audit(光, 声, 力, 温, 标签):
    e = 身体(节点数=96, 种子=7)
    e.存货 = 0
    samples = []
    env_input_log = []
    for t in range(TOTAL):
        e.环境['光']=光; e.环境['声']=声; e.环境['力']=力; e.环境['温']=温
        e.推进()
        sr = e.信号记录
        tick = int(e.时刻)
        if tick <= SKIP or tick % 50 == 0:
            samples.append({
                't': tick,
                '原归一': [round(v,6) for v in sr['原归一']],
                '末光': round(sr['末梢差光'],6),
                '末声': round(sr['末梢差声'],6),
            })
        # 环境输入 (5ch): 需重算
        原归一 = list(sr['原归一'])
        末光 = sr['末梢差光']
        末声 = sr['末梢差声']
        力值 = 原归一[2]
        温值 = 原归一[3]
        回接 = 0 if e.传导差 is None else sum(e.传导差[j] for j in e._回接节点)/20.0
        env_input_log.append([末光, 末声, 力值, 温值, 回接])

    # 通个统计 per 通道
    通 = [0]*5
    for row in env_input_log:
        for k in range(5):
            if abs(row[k]) > 1e-6: 通[k] += 1

    # 48节点的感受器增益(脑0+脑2,共48节点接"力")
    力增益节点 = [e.感受器[i][2] for i in range(96) if e.感受器[i][2] != 0]
    光增益节点 = [e.感受器[i][0] for i in range(96) if e.感受器[i][0] != 0]
    温增益节点 = [e.感受器[i][3] for i in range(96) if e.感受器[i][3] != 0]
    声增益节点 = [e.感受器[i][1] for i in range(96) if e.感受器[i][1] != 0]

    print(f'\n=== {标签} ===')
    print(f'  环境输入 k=非零tick数 ({TOTAL}tick):')
    print(f'    光(0)={通[0]}' + ('  <- 末梢差变换 on 光(KILL稳态)' if 通[0]<=5 else ''))
    print(f'    声(1)={通[1]}' + ('  <- 末梢差变换 on 声(心跳ac通过)' if 通[1]>5 else ''))
    print(f'    力(2)={通[2]}' + ('  <- 直接通过' if 通[2]>TOTAL-5 else '  <- 力=10→归一=0.3 UNCHANGED'))
    print(f'    温(3)={通[3]}' + ('  <- 直接通过' if 通[3]>TOTAL-5 else f'  <- 温={温}→归一={round(max(0,(温-288)/15),4)} CLIPS'))
    print(f'    回接(4)={通[4]}')
    print(f'  感受器维度: 光{len(光增益节点)}节点 声{len(声增益节点)}节点 力{len(力增益节点)}节点 温{len(温增益节点)}节点')
    print(f'  前{SKICK if (SKICK:=20) else 20}tick:')
    for s in samples[:10]:
        print(f'    t={s["t"]:3d} 原归一={s["原归一"]} 末光={s["末光"]} 末声={s["末声"]}')
    print(f'  tick 150:')
    mid = next((s for s in samples if s['t']==150), samples[-1])
    print(f'    {mid}')
    return 标签, 通

if __name__ == '__main__':
    print('=== v8 信号审计 ===')
    print(f'末梢槛: 身体(96,7).末梢')
    e_tmp = 身体(节点数=96, 种子=7)
    for k,v in e_tmp.末梢.items():
        print(f'  {k}: 低={v["低"]}, 高={v["高"]}')
    print()

    results = {}
    for 光, 声, 力, 温, 标签 in [
        (90, 45, 10, 297, 'A=基线(90/45/10/297)'),
        (72, 45, 10, 237, 'B=扰1±20%(72/45/10/237)'),
        (90, 45, 20, 297, 'C=力加倍(90/45/20/297)'),
        (90, 45, 10, 300, 'D=温+3(90/45/10/300)'),
        (90, 45, 10, 308, 'E=温+20(90/45/10/308)'),
        (90, 45, 1000, 297, 'F=力×100(90/45/1000/297)'),
    ]:
        tag, nz = audit(光, 声, 力, 温, 标签)
        results[tag] = nz

    print('\n=== 汇总表(通道非零tick数/300) ===')
    print(f'{"标签":<40} {"光":>4} {"声":>4} {"力":>4} {"温":>4} {"回":>4}')
    for tag, nz in results.items():
        print(f'{tag:<40} {nz[0]:>4} {nz[1]:>4} {nz[2]:>4} {nz[3]:>4} {nz[4]:>4}')

    print('\n=== 数值结论 ===')
    A_nz = results['A=基线(90/45/10/297)']
    B_nz = results['B=扰1±20%(72/45/10/237)']
    C_nz = results['C=力加倍(90/45/20/297)']
    D_nz = results['D=温+3(90/45/10/300)']
    E_nz = results['E=温+20(90/45/10/308)']
    F_nz = results['F=力×100(90/45/1000/297)']
    print(f'A→B 温由297→237: 温非零 {A_nz[3]}→{B_nz[3]}, 声 {A_nz[1]}→{B_nz[1]}')
    print(f'  → 温 237<288 clips to 0 (归一={round(max(0,(237-288)/15),4)}); A/B 两侧温信号均无稳态对比(差变换虽on但实无)')
    print(f'A→F 力由10→1000: 力非零 {A_nz[2]}→{F_nz[2]}, 归一 {round((10-1)/30,4)}→{round(min(1,(1000-1)/30),4)}')
    print(f'A→E 温由297→308: 温非零 {A_nz[3]}→{E_nz[3]}, 归一 {round((297-288)/15,4)}→{round((308-288)/15,4)}')
    print(f'  → 温是唯一能稳态接入引擎的通道(归一不clip+力不经差变换)')

