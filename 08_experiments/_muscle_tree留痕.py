"""全零输入。A组：拍100给回接节点+0.5。B组：不扰动。
跑8000拍。统计回接节点之间的筋树数据。
"""
import 身体稳态
import numpy as np
from 身体稳态 import 身体

def 跑(扰动):
    引擎 = 身体(种子=1, 上限满值=999999.0)
    引擎.末梢['声']['低'] = 0.015
    引擎._自己点 = lambda *a, **k: None
    回接节点 = set(引擎._回接节点)
    目标 = list(引擎._回接节点)[0]
    for 拍 in range(1, 8001):
        引擎.环境['光'] = 0.0
        引擎.环境['声'] = 0.0
        引擎.环境['力'] = 0.0
        引擎.环境['温'] = 0.0
        if 扰动 and 拍 == 100:
            引擎.激活[目标] += 0.5
        引擎.推进()
    # 统计回接节点之间的筋
    筋数 = 0
    位点列表 = []
    for (j, i), 筋 in 引擎.筋树.items():
        if j in 回接节点 and i in 回接节点:
            筋数 += 1
            位点列表.append(筋['位点'])
    if 位点列表:
        return 筋数, np.mean(位点列表), max(位点列表)
    else:
        return 0, 0.0, 0

print("===== A组：扰动 =====")
An, Aa, Am = 跑(True)
print(f"回接节点之间，筋数 = {An}")
print(f"回接节点之间，平均位点 = {Aa:.4f}")
print(f"回接节点之间，最大位点 = {Am}")

print()
print("===== B组：不扰动 =====")
Bn, Ba, Bm = 跑(False)
print(f"回接节点之间，筋数 = {Bn}")
print(f"回接节点之间，平均位点 = {Ba:.4f}")
print(f"回接节点之间，最大位点 = {Bm}")

print()
print("对照：")
print(f"两组筋数差 = {An - Bn}")
print(f"两组平均位点差 = {Aa - Ba:.4f}")
print(f"两组最大位点差 = {Am - Bm}")
