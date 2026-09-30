#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交叉验证：引擎(AR1) → 归档(bias*对) → 读数器(α)
=================================================
v2.0 2D 版。同 bias₀ + 常量 drive → 稳态 → 取 dim0 → α = a/b → 比较 drive₁/drive₂

三环：
  环1 引擎：AR(1) x[dim0] = leak*x + drive_const
  环2 归档：从 2D bias* 取 dim0（受驱动维度）
  环3 读数器：α = bias*_A[0] / bias*_B[0]，与 drive₁/drive₂ 比较

判据：|α − drive₁/drive₂| < 1e-8 → [事实]
"""
import sys, math
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
from 洛神 import Difference

LEAK = 0.99
STEPS = 5000        # 0.99^5000 ≈ 1.4e-22 → 充分收敛
TOL = 1e-8

def run_pair(drive_a, drive_b, bias0):
    a = Difference("a", list(bias0), LEAK)
    b = Difference("b", list(bias0), LEAK)   # 完全同初始化
    for _ in range(STEPS):
        a.step([drive_a, 0.0])
        b.step([drive_b, 0.0])
    return a, b

def alpha_scalar(a, b):
    return a.last[0] / b.last[0]   # dim0 = 受驱动维度

def main():
    bias0 = [1.0, 0.5]

    cases = [
        ("Case1", 0.3, 0.7),
        ("Case2", 0.1, 0.9),
        ("Case3", -0.5, 0.5),
    ]

    for name, d1, d2 in cases:
        a, b = run_pair(d1, d2, bias0)
        bias_star_a = a.last[0]
        bias_star_b = b.last[0]
        a_alpha = alpha_scalar(a, b)
        expected = d1 / d2
        diff = abs(a_alpha - expected)
        judge = "✓ [事实]" if diff < TOL else "✗ [判读]"
        print(f"{name} | drive₁={d1:+} drive₂={d2:+} | "
              f"bias*₁={bias_star_a:+.8f} bias*₂={bias_star_b:+.8f} | "
              f"α={a_alpha:.8f} | exp={expected:.8f} | "
              f"Δ={diff:.2e} | {judge}")

    # 增益验：bias* / drive ≈ 100 (= 1/(1-λ))
    a, b = run_pair(0.3, 0.7, bias0)
    gain_a = a.last[0] / 0.3
    gain_b = b.last[0] / 0.7
    print(f"\n增益验 | gain₁={gain_a:.4f} gain₂={gain_b:.4f} | 理论=100.0")

if __name__ == "__main__":
    main()
