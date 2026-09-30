"""静脸范式 v2.1 — 轮内时间戳日志（无 double increment）"""
import sys
sys.path.insert(0, '.')
import numpy as np
import json

from 身体稳态 import 身体
from _abai.perception import PerceptionProcessor

e = 身体(节点数=96, 种子=7)
e.存货 = 0
p = PerceptionProcessor()

轮内日志 = []
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


print('[静脸v2.1] 轮内时间戳日志')
print('存货=0, 轮6-7区域采样, 引擎单tick推进 + 外层喂')
print()

def 记轮内(tick, 饿, 哭, 阶段):
    轮内日志.append({'tick': int(tick), '饿': round(float(饿), 2), '哭合计': int(哭), '阶段': 阶段})


总帧 = 346
每帧tick = 250

# 主循环: 引擎自己推进, 外层只做环境注入 + 喂 + 采样
for i in range(总帧):
    帧 = p.帧读器.get_data(i)
    光, 力, 皮肤 = 快速感知(帧)
    e.环境['光'] = 10 + 90 * 光
    e.环境['力'] = min(40, 10 + 200 * 力)
    e.环境['温'] = 280 + 30 * 皮肤

    for _ in range(每帧tick):
        e.推进()
        tick = int(e.时刻)
        cry = int(getattr(e, '哭合计', 0))
        round_id = tick // 10000

        # 喂事件 (在引擎轮末之后立即喂)
        # 轮6 (ticks 60001-65000): 如果有能力就喂
        if 60000 < tick <= 65000 and round_id == 6:
            if e.hunger >= 10.0 and not any(ev['原因'].startswith('轮6') for ev in 喂事件列表):
                before = round(float(e.hunger), 1)
                e.hunger = max(0.0, e.hunger - 8.0)
                after = round(float(e.hunger), 1)
                喂事件列表.append({'tick': tick, '饿降前': before, '饿降后': after, '原因': '轮6团圆'})

        # 轮7 (ticks 70001-75000): 如果有能力就喂
        if 70000 < tick <= 75000 and round_id == 7:
            if e.hunger >= 10.0 and not any(ev['原因'].startswith('轮7') for ev in 喂事件列表):
                before = round(float(e.hunger), 1)
                e.hunger = max(0.0, e.hunger - 8.0)
                after = round(float(e.hunger), 1)
                喂事件列表.append({'tick': tick, '饿降前': before, '饿降后': after, '原因': '轮7团圆'})

        # 采轮内 (每10 tick采一次以控制日志量)
        if 60000 < tick <= 85000 and tick % 10 == 0:
            记轮内(tick, float(e.hunger), cry, f'round_{round_id}')

    # 每 5 帧摘要输出
    if i % 5 == 0 and 60000 < int(e.时刻) <= 85000:
        print(f'[tick={int(e.时刻)}] 饿={e.hunger:.2f} 哭合计={int(getattr(e, "哭合计", 0))}')


# 找哭起始 / 哭停
cry_有序 = [r for r in 轮内日志 if r['tick'] > 60000]
start_tick = cry_有序[0]['tick'] if cry_有序 else None

# 哭起始: 第一次 cry > 0 的 tick
cry_positive = [r for r in cry_有序 if r['哭合计'] > 0]
first_cry_tick = cry_positive[0]['tick'] if cry_positive else None

# 哭停: 喂后 cry 不增加的 tick
stop_tick = None
for j in range(1, len(cry_有序)):
    if cry_有序[j]['哭合计'] <= cry_有序[j-1]['哭合计'] and cry_有序[j]['tick'] > 61000:
        stop_tick = cry_有序[j]['tick']
        break

# 因果差
feed_tick = 喂事件列表[0]['tick'] if 喂事件列表 else None
causal_gap = (stop_tick - feed_tick) if (stop_tick and feed_tick) else None

# 饿降曲线
if feed_tick:
    curve = [r for r in 轮内日志 if feed_tick < r['tick'] <= feed_tick + 30]
    print(f'\n饿降曲线 (喂后30tick, 喂@tick={feed_tick}):')
    for r in curve:
        print(f'  tick={r["tick"]} 饿={r["饿"]:.2f} 哭={r["哭合计"]}')

print(f'\n哭起始 tick: {first_cry_tick}')
print(f'哭停 tick: {stop_tick}')
print(f'喂 tick: {feed_tick} -> 因果差: {causal_gap} ticks')

# 判定
if causal_gap and causal_gap <= 5:
    print('判决: 喂后<=5tick哭停 → 支持直接因果链，物理门槛机制')
elif causal_gap and causal_gap <= 100:
    print('判决: 喂后<=100tick哭停 → 支持因果链，但需排查中间变量')
elif causal_gap and causal_gap > 1000:
    print('判决: 喂后>1000tick才停 → 可能非直接因果')
else:
    print('判决: 未测到明确因果差')

out = {
    'version': 'v2.1_intratick',
    'feed_events': 喂事件列表,
    'cry_start_tick': first_cry_tick,
    'cry_stop_tick': stop_tick,
    'causal_gap_tick': causal_gap,
    'log_count': len(轮内日志),
}
with open('/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v2_1.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
with open('/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs/stillface_v2_1_inner.log', 'w') as f:
    json.dump(轮内日志, f, ensure_ascii=False, indent=2)

print('落盘: _abai/logs/stillface_v2_1.json + stillface_v2_1_inner.log')
