"""洛神赋最小多视角实验。
3角度(0=正面,1=左侧,2=右侧)×16×16画面。
种子=1，上限999999，声倍数300，声末梢低槛0.015。
每组8000拍。"""
import os, sys, time, json, random
from collections import Counter

# 恢复写痕
import 身体稳态
from 身体稳态 import 身体

WORK = "/mnt/data/catpaw/home/workspace/洛神赋1.1"
random.seed(1)

# 三个16×16灰度图，模拟杯子三视角
# 角度0：正面——中间高亮
图_正面 = [[0.0]*16 for _ in range(16)]
for r in range(4, 12):
    for c in range(6, 10):
        图_正面[r][c] = 0.9

# 角度1：左侧——左半亮
图_左侧 = [[0.0]*16 for _ in range(16)]
for r in range(3, 13):
    for c in range(2, 8):
        图_左侧[r][c] = 0.85

# 角度2：右侧——右半亮
图_右侧 = [[0.0]*16 for _ in range(16)]
for r in range(3, 13):
    for c in range(8, 14):
        图_右侧[r][c] = 0.85

图片列表 = [图_正面, 图_左侧, 图_右侧]
位置标记 = [0.0, 1.0, 2.0]

def 跑实验(喂位置: bool):
    """喂位置=True → 力通道=位置标记；False → 力通道=默认值10.0"""
    引擎 = 身体(种子=1, 上限满值=999999.0)
    引擎.末梢['声']['低'] = 0.015

    节点数 = 引擎.节点数  # 96
    脑大小 = 引擎.脑大小  # 24
    光脑节点 = list(range(0, 脑大小))  # 0..23

    初筋 = len(引擎.筋树)
    光脑激活最大 = 0.0
    光脑点火 = 0
    # 每个角度对应的亮节点集合
    角度亮节点 = [set(), set(), set()]
    # 记录每个节点是否在任一拍超过0.5
    节点角度标记 = [set() for _ in range(节点数)]

    for 拍 in range(8000):
        角度序号 = (拍 // 16) % 3
        行号 = 拍 % 16
        图 = 图片列表[角度序号]
        # 光通道 = 画面行均值
        行均值 = sum(图[行号]) / 16.0
        引擎.环境['光'] = 行均值
        引擎.环境['力'] = 位置标记[角度序号] if 喂位置 else 10.0
        引擎.环境['声'] = 50.0 * 300 / 80 + 引擎.随机器.uniform(-1, 1)
        引擎.环境['温'] = 295.0
        引擎.推进()

        a = 引擎.激活
        for n in 光脑节点:
            if abs(a[n]) > 光脑激活最大:
                光脑激活最大 = abs(a[n])
            if abs(a[n]) > 0.5:
                光脑点火 += 1
                角度亮节点[角度序号].add(n)
                节点角度标记[n].add(角度序号)

    return {
        "跑拍数": 8000,
        "光脑激活最大": round(光脑激活最大, 4),
        "光脑点火次数": 光脑点火,
        "净增筋数": len(引擎.筋树) - 初筋,
        "角度0亮节点": sorted(角度亮节点[0]),
        "角度1亮节点": sorted(角度亮节点[1]),
        "角度2亮节点": sorted(角度亮节点[2]),
        "重叠节点号": sorted(set.intersection(*角度亮节点) if all(角度亮节点) else set()),
        "重叠节点数": len(set.intersection(*角度亮节点) if all(角度亮节点) else set()),
    }

print("===== 实验组：喂位置 =====", flush=True)
实验 = 跑实验(True)
for k, v in 实验.items():
    print(f"{k} = {v}")
print(flush=True)

print("===== 对照组：不喂位置 =====", flush=True)
对照 = 跑实验(False)
for k, v in 对照.items():
    print(f"{k} = {v}")
