"""只喂光通道，光值恒定=50.0。
跑8000拍，种子=1，上限999999。
每1000拍报一次最大激活。
A组：禁用自己点。
B组：开启自己点（正常）。
"""
import 身体稳态
from 身体稳态 import 身体

脑大小 = 24
光脑节点 = list(range(脑大小))
报告拍 = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000}

def 跑(禁用自己点):
    引擎 = 身体(种子=1, 上限满值=999999.0)
    引擎.末梢['声']['低'] = 0.015
    if 禁用自己点:
        引擎._自己点 = lambda *a, **k: None
    峰值记录 = {}
    for 拍 in range(1, 8001):
        引擎.环境['光'] = 50.0
        引擎.环境['声'] = 45.0
        引擎.环境['力'] = 10.0
        引擎.环境['温'] = 295.0
        引擎.推进()
        if 拍 in 报告拍:
            最大 = max(引擎.激活[n] for n in 光脑节点)
            峰值记录[拍] = round(最大, 6)
    return 峰值记录

print("===== A组：禁用自己点 =====")
峰值A = 跑(True)
print(f"跑拍数 = 8000")
print(f"自己点 = 禁用")
for 拍 in sorted(峰值A):
    print(f"第{拍}拍光脑节点激活最大 = {峰值A[拍]}")

print()
print("===== B组：开启自己点 =====")
峰值B = 跑(False)
print(f"跑拍数 = 8000")
print(f"自己点 = 开启")
for 拍 in sorted(峰值B):
    print(f"第{拍}拍光脑节点激活最大 = {峰值B[拍]}")

print()
print("===== 对照 =====")
print(f"A组峰值 = {max(峰值A.values())}")
print(f"B组峰值 = {max(峰值B.values())}")
# 峰值=最大值的拍数
峰值拍A = [str(k) for k, v in 峰值A.items() if v == max(峰值A.values())]
峰值拍B = [str(k) for k, v in 峰值B.items() if v == max(峰值B.values())]
print(f"	A组峰值出现的拍数 = {','.join(峰值拍A)}")
print(f"	B组峰值出现的拍数 = {','.join(峰值拍B)}")
有峰值A = max(峰值A.values()) > 0.3
print(f"关掉自己点后，还有没有峰值 = {'是' if 有峰值A else '否'}")
