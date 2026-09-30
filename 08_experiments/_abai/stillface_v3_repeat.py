"""静脸范式 v3 — 重复哭启动测试"""
import sys
sys.path.insert(0, '.')
import numpy as np
import json

from 身体稳态 import 身体
from _abai.perception import PerceptionProcessor

e = 身体(节点数=96, 种子=7)
e.存货 = 0
p = PerceptionProcessor()

轮记录 = []
喂事件列表 = []


def 快速感知(帧):
    小 = 帧[::10, ::10, :]
    灰 = np.mean(小.astype(np.float32), axis=2) / 255
    光 = float(np.mean(灰))
    力 = 0.0
    if p.上一帧灰 is not None:
        力 = float(np.mean(np.abs(灰 - p.上一帧灰)))
    p.上一帧灰 = 灰
    rgb = 小.astype(np.float32) / 255
    V = np.max(rgb, axis=2); mn = np.min(rgb, axis=2); d = V - mn
    ds = np.where(d < 1e-8, 1e-8, d); S = np.where(V < 1e-8, 0, ds / V)
    mr = rgb[:,:,0] == V; G = rgb[:,:,1]; B = rgb[:,:,2]
    H = np.where(mr, (G - B) / np.maximum(ds, 1e-8),
                 np.where(rgb[:,:,1] == V, 2 + (B - rgb[:,:,0]) / np.maximum(ds, 1e-8),
                          4 + (rgb[:,:,0] - G) / np.maximum(ds, 1e-8))) * 60
    H = np.where(d < 1e-8, 0, H) % 180
    lo, hi = p.skin_hsv_range
    skin = float(np.mean((H >= lo[0]) & (H <= hi[0]) & (S >= lo[1]) & (S <= hi[1]) & (V >= lo[2]) & (V <= hi[2])))
    return 光, 力, skin


print('[v3_repeat] 重复哭启动测试')
print('存货=0, 哭启动→喂→停→再启动')
print()

总帧 = 346
每帧tick = 250

def 阶段_of(tick):
    if tick <= 20000:
        return '基线'
    if tick <= 50000:
        return '板脸1'
    if tick <= 60000:
        return '团圆'
    return '板脸2'

for i in range(总帧):
    帧 = p.帧读器.get_data(i)
    光, 力, 皮肤 = 快速感知(帧)
    e.环境['光'] = 10 + 90 * 光
    e.环境['力'] = min(40, 10 + 200 * 力)
    e.环境['温'] = 280 + 30 * 皮肤

    for _ in range(每帧tick):
        e.推进()
        tick = int(e.时刻)
        round_id = tick // 10000

        # 仅在tick 50001-50010时喂一次
        if 50000 < tick <= 50010:
            if e.hunger >= 11.0 and not 喂事件列表:
                before = round(float(e.hunger), 1)
                e.hunger = max(0.0, e.hunger - 8.0)
                after = round(float(e.hunger), 1)
                喂事件列表.append({'tick': tick, '饿降前': before, '饿降后': after})
        # 之后不再主动喂, 让饿自然积累

    # 轮末记录
    if e.时刻 > 0 and e.时刻 % 10000 == 0:
        活筋 = [j for j in e.筋树.values() if j.get('位点', 0) > 0]
        饿 = round(float(e.hunger), 1)
        醒 = bool(e.awake)
        筋 = len(活筋)
        哭 = int(getattr(e, '哭合计', 0))
        最大位 = max((j['位点'] for j in 活筋), default=0) if 活筋 else 0
        轮记录.append({
            'tick': int(e.时刻), '饿': 饿, '醒': 醒, '筋': 筋,
            '哭合计': 哭, '存货': int(e.存货), '阶段': 阶段_of(int(e.时刻)),
            '最大位': 最大位,
        })
        print(f'[tick={int(e.时刻):6d}] 饿={饿:5.1f} 醒={"W" if 醒 else "S"} 筋={筋:5d} 哭={哭:6d} | {阶段_of(int(e.时刻))}')

# 分析
哭序列 = [r['哭合计'] for r in 轮记录]
增量序列 = [哭序列[j] - 哭序列[j-1] for j in range(1, len(哭序列))]

print(f'\n哭序列: {哭序列}')
print(f'哭增量: {增量序列}')

# 第一轮哭增量
创1_idx = next((i for i, r in enumerate(轮记录) if r['阶段'] == '板脸1' and r['哭合计'] > 0), None)
# 第二轮哭增量
再启_idx = None
for j in range(len(轮记录) - 1, -1, -1):
    if 轮记录[j]['阶段'] == '板脸2' and 轮记录[j]['哭合计'] > 轮记录[j-1]['哭合计']:
        再启_idx = j
        break

if 创1_idx is not None and 再启_idx is not None:
    第一轮增量 = 轮记录[创1_idx]['哭合计']
    第二轮增量 = 轮记录[再启_idx]['哭合计'] - 轮记录[再启_idx - 1]['哭合计']
    ratio = 第二轮增量 / max(第一轮增量, 1)
    print(f'\n第一轮哭增量: {第一轮增量}')
    print(f'第二轮哭增量: {第二轮增量}')
    print(f'比值: {ratio:.3f}')
    if 0.8 <= ratio <= 1.2:
        print('判决: 两轮一致 → 开关机制排他, 学习假说不支持')
    elif ratio > 1.5:
        print('判决: 第二轮增强 → 敏感化学习证据')
    elif ratio < 0.5:
        print('判决: 第二轮减弱 → 习惯化学习证据')
    else:
        print('判决: 差异中等 → 需更多循环或更大样本')
elif 创1_idx is None:
    print('未检测到第一轮哭启动')
elif 再启_idx is None:
    print('\n未检测到第二轮哭启动')
    if 轮记录 and 轮记录[-1]['饿'] < 12.0:
        print(f'原因: 最终饿={轮记录[-1]["饿"]} < HUNG_CRY=12, 饿未再升过门槛')
    else:
        print('原因: 未知, 需查完整轮内数据')

out = {
    'version': 'v3_repeat_cry',
    'feed_events': 喂事件列表,
    'round_records': 轮记录,
    'cry_sequence': 哭序列,
    'inc_sequence': 增量序列,
}
with open('/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v3_repeat.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print('落盘: _abai/logs/stillface_v3_repeat.json')
