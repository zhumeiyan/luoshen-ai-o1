"""100万拍。上限999999，声增强，种子=1。分5段存盘。"""
import time, os, json, pickle
from collections import Counter
from 身体稳态 import 身体

WORK = "/mnt/data/catpaw/home/workspace/洛神赋1.1"
SNAP = os.path.join(WORK, "snapshots")
os.makedirs(SNAP, exist_ok=True)
MN = 999999.0

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1, 上限满值=MN)
引擎.末梢["声"]["低"] = 0.015
初筋 = len(引擎.筋树)

orig_自己点 = 引擎._自己点
自己点记录 = []
_prev点次 = 0

def hijacked_自己点(记录=None):
    global _prev点次
    orig_自己点(记录)
    curr = getattr(引擎, '_点次数', 0)
    if curr > _prev点次:
        tgt = getattr(引擎, '_上次点目标', None)
        if tgt is not None:
            自己点记录.append((引擎.时刻, tgt))
        _prev点次 = curr

引擎._自己点 = hijacked_自己点

BS, NS, TS = 200000, 96, 5
总点火 = [0] * 4
段点火 = []
秒 = 0

for s in range(TS):
    点火段 = [0] * 4
    t0 = time.time()
    for i in range(BS):
        引擎.环境["光"] = 200.0 + 引擎.随机器.uniform(-1, 1)
        引擎.环境["声"] = 50.0 * 300 / 80 + 引擎.随机器.uniform(-1, 1)
        引擎.环境["力"] = 10.0
        引擎.环境["温"] = 295.0
        引擎.推进()
        a = 引擎.激活
        for idx in range(NS):
            if abs(a[idx]) > 0.5:
                nb = idx // 24
                总点火[nb] += 1
                点火段[nb] += 1
    dt = time.time() - t0
    秒 += dt
    段点火.append(点火段)
    with open(os.path.join(SNAP, f"s{s}.pkl"), 'wb') as f:
        pickle.dump({"eng": 引擎, "self": list(自己点记录), "ign": 总点火, "seg": 点火段}, f)
    print(f"段{s+1}: {BS}拍 {dt:.1f}秒 {BS/dt:.0f}拍/秒 点火={点火段}", flush=True)

with open(os.path.join(SNAP, "final.json"), 'w') as f:
    节点列表 = [n for _, n in 自己点记录]
    json.dump({
        "bs": BS, "ts": TS,
        "total": 总点火, "seg": 段点火,
        "seconds": 秒,
        "初筋": 初筋, "末筋": len(引擎.筋树),
        "self_total": len(自己点记录),
        "self_nodes": list(sorted(set(节点列表))),
        "self_dist": dict(Counter(节点列表))
    }, f, ensure_ascii=False)

print(f"完成 总拍{BS*TS} 总时{秒:.0f}秒 均{BS*TS/秒:.0f}拍/秒")
print(f"总点火: 光{总点火[0]} 声{总点火[1]} 力{总点火[2]} 温{总点火[3]}")
for s in range(TS):
    print(f"段{s+1}: 光{段点火[s][0]} 声{段点火[s][1]} 力{段点火[s][2]} 温{段点火[s][3]}")
print(f"初筋{初筋} 末筋{len(引擎.筋树)} 净增{len(引擎.筋树)-初筋}")
print(f"自己点: 总{len(自己点记录)}次 不同节点{len(set(n for _, n in 自己点记录))}个")
节点列表 = [n for _, n in 自己点记录]
for n, c in sorted(Counter(节点列表).items()):
    print(f"节点{n}: {c}次")
