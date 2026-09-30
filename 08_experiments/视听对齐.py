"""音视频对齐：把视觉和听觉按帧号摆一起。
不做因果。两路摆着。标出两路同时高亮/同时低暗的帧，和对不上的帧。
"""
import json, sys
import numpy as np

TRACE = 'luoshenfu_home/traces/video_sensory_chain.jsonl'
with open(TRACE, 'r', encoding='utf-8') as f:
    recs = [json.loads(line) for line in f]

n = len(recs)
vis = np.array([r['visual']['mean'] for r in recs])
aud = np.array([r['auditory']['mean'] for r in recs])
aud_max = np.array([r['auditory']['max'] for r in recs])

def norm(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else x * 0

vis_n = norm(vis)
aud_n = norm(aud)

mismatch = []
match = []

print(f'{"帧":>4} {"时间":>7} | {"vis_μ":>8} {"vis_n":>7} | {"aud_μ":>8} {"aud_max":>8} {"aud_n":>7} | 两路状态')
print('-' * 95)
for i, r in enumerate(recs):
    tag = ''
    if r['auditory']['max'] >= 119.0:
        if vis_n[i] > 0.5:
            tag = 'A+V+ 两路同高'
            match.append(i)
        elif vis_n[i] < 0.2:
            tag = 'A+ V- 有声画面暗'
            mismatch.append(i)
        else:
            tag = 'A+ V~'
    elif vis_n[i] > 0.5 and aud_n[i] < 0.2:
        tag = 'V+ A- 画面亮无声'
        mismatch.append(i)
    elif vis_n[i] < 0.2 and aud_n[i] > 0.5:
        tag = 'A+ V- 有声画面暗'
        mismatch.append(i)

    print(f'{r["frame_idx"]:>4} {r["timestamp"]:>7} | {r["visual"]["mean"]:>8.2f} {vis_n[i]:>7.4f} | {r["auditory"]["mean"]:>8.2f} {r["auditory"]["max"]:>8.2f} {aud_n[i]:>7.4f} | {tag}')

# 有声段总览
speech_bursts = []
in_burst = False
for i in range(n):
    if aud_max[i] >= 119.0 and not in_burst:
        burst_start = i
        in_burst = True
    elif aud_max[i] < 110.0 and in_burst:
        speech_bursts.append((burst_start, i-1))
        in_burst = False
if in_burst:
    speech_bursts.append((burst_start, n-1))

print(f'\n【有声段总览（aud_max >= 119 的连续帧）】')
print(f'{"段":>4} {"帧范围":>12} {"时间":>12} {"vis_μ范围":>20} {"aud_μ范围":>16} 视觉峰值帧')
print('-' * 90)
for idx, (s, e) in enumerate(speech_bursts):
    vis_range = vis[s:e+1]
    aud_range = aud[s:e+1]
    vis_peak = s + np.argmax(vis_range)
    # 找 vis 在该段内 > 0.7 的高亮帧
    high_v = [j for j in range(s, e+1) if vis_n[j] > 0.7]
    print(f'{idx:>4} f{s:>3}~f{e:>3}  {recs[s]["timestamp"]:>5}~{recs[e]["timestamp"]:>5} [{vis_range.min():>7.1f}, {vis_range.max():>7.1f}] [{aud_range.min():>6.2f}, {aud_range.max():>6.2f}]  高亮: {high_v}')

# 匹配vs不匹配
print(f'\n【两路同高亮（aud_max≥119 且 vis_n>0.5）】共 {len(match)} 帧')
if match:
    print(f'  帧: {match}')

print(f'\n【对不上的帧（一路高一路低）】共 {len(mismatch)} 帧')
if mismatch:
    for i in mismatch:
        print(f'  f{i:>3} {recs[i]["timestamp"]:>7} vis_n={vis_n[i]:.3f} aud_n={aud_n[i]:.3f}')

# 相关
print(f'\n【相关性】')
print(f'  vis_raw vs aud_raw     r = {np.corrcoef(vis, aud)[0,1]:+.4f}')
print(f'  vis_norm vs aud_norm   r = {np.corrcoef(vis_n, aud_n)[0,1]:+.4f}')
print(f'  vis_raw vs aud_max     r = {np.corrcoef(vis, aud_max)[0,1]:+.4f}')
