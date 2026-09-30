"""查光脑和声脑的感受器增益分布"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

from 身体稳态 import 身体

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
脑大小 = 24

# 各脑感受器增益统计
for nb, name in [(0,"光"), (1,"声")]:
    gains = []
    for idx in range(nb*脑大小, (nb+1)*脑大小):
        for k in range(5):
            增益 = 引擎.感受器[idx][k]
            gains.append(增益)
    gains = np.array(gains)
    print(f"{name}脑节点范围 = {nb*脑大小}到{(nb+1)*脑大小-1}")
    print(f"{name}脑感受器增益最大值 = {float(np.max(gains)):.6f}")
    print(f"{name}脑感受器增益最小值 = {float(np.min(gains)):.6f}")
    print(f"{name}脑感受器增益平均值 = {float(np.mean(gains)):.6f}")
    print(f"{name}脑感受器增益绝对值平均 = {float(np.mean(np.abs(gains))):.6f}")
    print()

# 逐个节点看 声脑24-47 的光通道增益
print("声脑节点24-47的光通道(k=0)增益:")
for idx in range(24, 48):
    print(f"节点{idx} k0={引擎.感受器[idx][0]:.4f} k1={引擎.感受器[idx][1]:.4f} k2={引擎.感受器[idx][2]:.4f} k3={引擎.感受器[idx][3]:.4f} k4={引擎.感受器[idx][4]:.4f}")

print()
print("光脑节点0-23的光通道(k=0)增益:")
for idx in range(0, 24):
    print(f"节点{idx} k0={引擎.感受器[idx][0]:.4f} k1={引擎.感受器[idx][1]:.4f} k2={引擎.感受器[idx][2]:.4f} k3={引擎.感受器[idx][3]:.4f} k4={引擎.感受器[idx][4]:.4f}")
