"""静脸范式 v2 — 真正能喂的团圆阶段

修v1 bug:
  v1: if 存>=4 才喂 → 存=0时永远喂不了
  v2: 外部喂(母答回路) → 直接降饿, 绕过引擎存货判据
     引擎内部每3轮的喂继续走但存货=0会失败 → 只有外部喂才有效
"""
import sys; sys.path.insert(0,'.')
import numpy as np, json, time
from datetime import datetime
from 身体稳态 import 身体
from _abai.perception import PerceptionProcessor

e = 身体(节点数=96, 种子=7)
e.存货 = 0
p = PerceptionProcessor()

总帧 = 346
ticks_per_frame = 250

print(f'[静脸v2] 存货=0, 外部喂绕过存货判据')
print(f'基线=轮1-2, 板脸=轮3-5, 团圆=轮6-7(强制喂), 观察=轮8')
print()
print('%4s|%6s|%5s|%4s|%5s|%7s|%4s|%6s|%5s|%5s|%4s|%4s'%('轮','tick','饿','醒','筋','哭合计','存','阶段','温K','皮肤%','最位','喂'))
print('-'*85)

喂次数 = 0
轮记录 = []
阶段表 = {1:'基线',2:'基线',3:'板脸',4:'板脸',5:'板脸',6:'团',7:'团',8:'观'}

def 快速感知(帧):
    小 = 帧[::10, ::10, :]
    灰 = np.mean(小.astype(np.float32), axis=2)/255
    光 = float(np.mean(灰))
    力 = 0.0
    if p.上一帧灰 is not None:
        力 = float(np.mean(np.abs(灰-p.上一帧灰)))
    p.上一帧灰 = 灰
    rgb = 小.astype(np.float32)/255
    V=np.max(rgb,axis=2); mn=np.min(rgb,axis=2); d=V-mn
    ds=np.where(d<1e-8,1e-8,d); S=np.where(V<1e-8,0,ds/V)
    mr=rgb[:,:,0]==V; G=rgb[:,:,1]; B=rgb[:,:,2]
    H=np.where(mr,(G-B)/np.maximum(ds,1e-8),np.where(rgb[:,:,1]==V,2+(B-rgb[:,:,0])/np.maximum(ds,1e-8),4+(rgb[:,:,0]-G)/np.maximum(ds,1e-8)))*60
    H=np.where(d<1e-8,0,H)%180
    lo,hi=p.skin_hsv_range
    skin=float(np.mean((H>=lo[0])&(H<=hi[0])&(S>=lo[1])&(S<=hi[1])&(V>=lo[2])&(V<=hi[2])))
    return 光,力,skin

def 外部喂(引擎, 降饿量=8.0):
    """母答回路喂：直接降饿, 绕过引擎存货判据"""
    引擎.hunger = max(0.0, 引擎.hunger - 降饿量)
    return True

for i in range(总帧):
    帧 = p.帧读器.get_data(i)
    光,力,皮肤 = 快速感知(帧)
    e.环境['光'] = 10 + 90*光
    e.环境['力'] = min(40, 10 + 200*力)
    e.环境['温'] = 280 + 30*皮肤
    
    for _ in range(ticks_per_frame):
        e.推进()
    
    if e.时刻 > 0 and e.时刻 % 10000 == 0:
        活筋 = [j for j in e.筋树.values() if j.get('位点',0)>0]
        饿 = float(e.hunger); 醒 = bool(e.awake); 筋 = len(活筋)
        哭 = int(getattr(e,'哭合计',0)); 存 = int(e.存货)
        最大位 = max((j['位点'] for j in 活筋),default=0) if 活筋 else 0
        轮次 = e.时刻 // 10000
        阶段 = 阶段表.get(轮次, '?')
        本轮喂 = 0
        
        # v2: 轮6和轮7都强制外部喂（母亲的回应）
        if 轮次 in (6, 7) and 饿 >= 10.0:
            外部喂(e, 8.0)
            喂次数 += 1
            本轮喂 = 1
        
        轮记录.append({
            '轮':轮次, 'tick':int(e.时刻), '饿':round(饿,1), '醒':醒,
            '筋':筋, '最大位':最大位, '哭合计':哭, '存货':存,
            '阶段':阶段, '皮肤':round(皮肤,3), '温':round(e.环境['温'],1), '外部喂':本轮喂,
        })
        print('%4d|%6d|%5.1f|%4s|%5d|%7d|%4d|%6s|%5.1f|%5.1f|%4d|%4d'%(
            轮次, e.时刻, 饿, 'W' if 醒 else 'S', 筋, 哭, 存,
            阶段, e.环境['温'], 皮肤*100, 最大位, 本轮喂))

p.close()

# 判据
哭序列 = [r['哭合计'] for r in 轮记录]
print()
print('=== 判定 ===')
if len(哭序列) >= 6:
    板脸哭峰值 = 哭序列[4]  # 轮5哭
    团后哭增量_轮6 = 哭序列[5] - 哭序列[4]
    团后哭增量_轮7 = 哭序列[6] - 哭序列[5]
    print(f'轮5哭: {板脸哭峰值}')
    print(f'轮6哭增量(喂1次后): {团后哭增量_轮6} (vs 无喂时=2000)')
    print(f'轮7哭增量(喂第2次后): {团后哭增量_轮7}')
    if 团后哭增量_轮6 < 2000 or 团后哭增量_轮7 < 团后哭增量_轮6:
        print('判决: 哭减速 → 关三(止)候选 = 喷→呼喊转化启动')
    else:
        print('判决: 哭未减速 → 喷泉疑似, 应答强度/频率可能不够')

out = {'范式':'静脸v2','存货':0,'外部喂':True,'判据':'轮6-7强制喂','轮记录':轮记录}
路径 = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v2.json'
with open(路径,'w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print(f'落盘: {路径}')
