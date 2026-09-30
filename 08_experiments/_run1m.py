"""100万拍。上限999999，声槛0.015。分段×5，每段20万拍。写痕。"""
import os, sys, time, json, pickle
from collections import Counter

import 身体稳态

from 身体稳态 import 身体

WORK = "/mnt/data/catpaw/home/workspace/洛神赋1.1"
SNAP = os.path.join(WORK, "snapshots_v1")
os.makedirs(SNAP, exist_ok=True)
BS, NS, TS = 200000, 96, 5

start_seg = 0
引擎 = None
for s in range(TS - 1, -1, -1):
    fp = os.path.join(SNAP, f"seg_{s}.pkl")
    if os.path.exists(fp):
        with open(fp, 'rb') as f:
            data = pickle.load(f)
        引擎 = data["eng"]
        start_seg = s + 1
        print(f"从段{start_seg}续传", flush=True)
        break

if 引擎 is None:
    引擎 = 身体(种子=1, 上限满值=999999.0)
    引擎.末梢['声']['低'] = 0.015
    start_seg = 0
    # 跑完后清空日志防止内存膨胀
    def _清空日志():
        if hasattr(引擎, '日志'):
            引擎.日志.clear()

初筋 = len(引擎.筋树) if start_seg == 0 else "查不到"
总点火 = [0]*4
段点火 = []
秒 = 0

def _每百拍清日志():
    """每100拍清空日志防内存膨胀"""
    if hasattr(引擎, '日志') and len(引擎.日志) > 200:
        引擎.日志.clear()

for s in range(start_seg, TS):
    点火段 = [0]*4
    t0 = time.time()
    for i in range(BS):
        引擎.环境['光'] = 200.0 + 引擎.随机器.uniform(-1, 1)
        引擎.环境['声'] = 50.0 * 300 / 80 + 引擎.随机器.uniform(-1, 1)
        引擎.环境['力'] = 10.0
        引擎.环境['温'] = 295.0
        引擎.推进()
        if i % 100 == 0:
            _每百拍清日志()
        a = 引擎.激活
        for idx in range(NS):
            if abs(a[idx]) > 0.5:
                点火段[idx//24] += 1
    for k in range(4):
        总点火[k] += 点火段[k]
    dt = time.time() - t0
    秒 += dt
    段点火.append(点火段)
    with open(os.path.join(SNAP, f"seg_{s}.pkl"), 'wb') as f:
        pickle.dump({"eng": 引擎, "段点火": 段点火, "总点火": 总点火}, f)
    print(f"段{s+1}: {BS}拍 {dt:.1f}秒 {BS/dt:.0f}拍/秒 点火={点火段} 筋={len(引擎.筋树)}", flush=True)

序列 = getattr(引擎, '_点序列', [])
final = {
    "跑拍数": BS * TS,
    "耗时秒数": round(秒, 1),
    "均拍每秒": round(BS * TS / 秒, 1) if 秒 > 0 else 0,
    "初筋": 初筋,
    "末筋": len(引擎.筋树),
    "净增筋数": len(引擎.筋树) - (初筋 if isinstance(初筋, int) else 0),
    "光脑点火次数": 总点火[0],
    "声脑点火次数": 总点火[1],
    "力脑点火次数": 总点火[2],
    "温脑点火次数": 总点火[3],
}
for s in range(TS):
    if s < len(段点火):
        d = (s+1)*20
        final[f"第{d}万拍光脑点火"] = 段点火[s][0]
        final[f"第{d}万拍声脑点火"] = 段点火[s][1]
        final[f"第{d}万拍力脑点火"] = 段点火[s][2]
        final[f"第{d}万拍温脑点火"] = 段点火[s][3]
final["自己点总次数"] = len(序列)
final["自己点选过的不同节点数"] = len(set(序列))

with open(os.path.join(SNAP, "final.json"), 'w') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

for k, v in final.items():
    print(f"{k} = {v}")
