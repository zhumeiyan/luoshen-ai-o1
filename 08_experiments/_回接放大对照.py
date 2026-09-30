"""全零输入，回接差放大对照。
A组差/20（原版），B组差/5，C组差/1。
自己点禁用，上限999999，种子=1。
跑8000拍。每1000拍报回接节点平均激活。
"""
import importlib
import sys
import os

身体稳态_path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py"

def 加载引擎(除数):
    """读源码，替换除数，动态加载"""
    with open(身体稳态_path, 'r', encoding='utf-8') as f:
        源码 = f.read()
    # 把 "差 / 20.0" 替换
    新源码 = 源码.replace('差 / 20.0', f'差 / {float(除数)}')
    # 写到临时文件
    tmp_path = f"/tmp/_身体_除{除数}.py"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(新源码)
    # 加载
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"身体_除{除数}", tmp_path)
    模块 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(模块)
    return 模块.身体

def 跑(除数):
    身体类 = 加载引擎(除数)
    引擎 = 身体类(种子=1, 上限满值=999999.0)
    引擎.末梢['声']['低'] = 0.015
    引擎._自己点 = lambda *a, **k: None
    回接节点 = list(引擎._回接节点)
    记录 = {}
    for 拍 in range(1, 8001):
        引擎.环境['光'] = 0.0
        引擎.环境['声'] = 0.0
        引擎.环境['力'] = 0.0
        引擎.环境['温'] = 0.0
        引擎.推进()
        if 拍 % 1000 == 0:
            记录[拍] = sum(引擎.激活[n] for n in 回接节点) / len(回接节点)
    return 记录

三组 = {}
for 除数, 名称 in [(20, 'A'), (5, 'B'), (1, 'C')]:
    print(f"===== {名称}组：差/{除数} =====")
    结果 = 跑(除数)
    三组[名称] = 结果
    print(f"跑拍数 = 8000")
    for 拍 in [1000,2000,3000,4000,5000,6000,7000,8000]:
        print(f"第{拍}拍回接节点平均激活 = {结果[拍]:.6f}")
    print()

print("对照：")
for 拍 in [1000,2000,3000,4000,5000,6000,7000,8000]:
    a = 三组['A'].get(拍, 0)
    b = 三组['B'].get(拍, 0)
    c = 三组['C'].get(拍, 0)
    print(f"  拍{拍}: A={a:.4f} B={b:.4f} C={c:.4f}")

import numpy as np
a_avg = np.mean(list(三组['A'].values()))
b_avg = np.mean(list(三组['B'].values()))
c_avg = np.mean(list(三组['C'].values()))
print(f"三组回接节点平均激活对比 = A={a_avg:.4f} B={b_avg:.4f} C={c_avg:.4f}")
