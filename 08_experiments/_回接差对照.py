"""只喂光，光值恒定=50.0。
A组：回接差禁用（ _回接节点 置空）+ 自己点禁用。
B组：回接差正常 + 自己点禁用。
各跑8000拍，种子=1，上限999999。每1000拍报最大激活。
"""
import 身体稳态
from 身体稳态 import 身体

脑大小 = 24
光脑节点 = list(range(脑大小))

def 跑(禁用回接差):
    引擎 = 身体(种子=1, 上限满值=999999.0)
    引擎.末梢['声']['低'] = 0.015
    引擎._自己点 = lambda *a, **k: None
    if 禁用回接差:
        引擎._回接节点 = []  # sum([])/20 = 0.0
    结果 = {}
    for 拍 in range(1, 8001):
        引擎.环境['光'] = 50.0
        引擎.环境['声'] = 45.0
        引擎.环境['力'] = 10.0
        引擎.环境['温'] = 295.0
        引擎.推进()
        if 拍 % 1000 == 0:
            结果[拍] = round(max(引擎.激活[n] for n in 光脑节点), 6)
    return 结果

print("===== A组：回接差禁用 + 自己点禁用 =====")
A = 跑(True)
print(f"跑拍数 = 8000")
print(f"回接差 = 禁用")
print(f"自己点 = 禁用")
for k in sorted(A):
    print(f"第{k}拍光脑节点激活最大 = {A[k]}")

print()
print("===== B组：回接差正常 + 自己点禁用 =====")
B = 跑(False)
print(f"跑拍数 = 8000")
print(f"回接差 = 正常")
print(f"自己点 = 禁用")
for k in sorted(B):
    print(f"第{k}拍光脑节点激活最大 = {B[k]}")

print()
maxA = max(A.values())
maxB = max(B.values())
峰值拍A = [str(k) for k, v in A.items() if v == maxA]
峰值拍B = [str(k) for k, v in B.items() if v == maxB]
print(f"对照组：")
print(f"A组峰值 = {maxA}")
print(f"B组峰值 = {maxB}")
print(f"A组峰值出现的拍数 = {','.join(峰值拍A)}")
print(f"B组峰值出现的拍数 = {','.join(峰值拍B)}")
print(f"关掉回接差后，还有没有峰值 = {'是' if maxA > 0.3 else '否'}")
