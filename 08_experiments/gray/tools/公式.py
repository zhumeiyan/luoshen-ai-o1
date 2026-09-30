#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公式.py — α = (s-d)/(s+d)
只算，不定义。
"""
import math


def alpha(s, d):
    """α = (s - d) / (s + d)"""
    denom = s + d
    if denom == 0:
        return None
    return (s - d) / denom


def alpha_min_max(s_range, d_range):
    """闭式 min/max"""
    s_lo, s_hi = s_range
    d_lo, d_hi = d_range
    corners = [alpha(s_lo, d_lo), alpha(s_lo, d_hi),
               alpha(s_hi, d_lo), alpha(s_hi, d_hi)]
    corners = [c for c in corners if c is not None]
    if not corners:
        return None, None
    return min(corners), max(corners)


def print_table(s_vals, d_vals):
    header = "s\\d | " + " ".join(f"{d:+.1f}" for d in d_vals)
    print(header)
    print("-" * len(header))
    for s in s_vals:
        row = [f"{s:+.1f} |"]
        for d in d_vals:
            a = alpha(s, d)
            row.append(" NaN " if a is None else f"{a:+.3f}")
        print(" ".join(row))


if __name__ == "__main__":
    print("=== alpha = (s-d)/(s+d) ===")
    print(f"alpha(3, 1) = {alpha(3, 1):+.4f}")
    print(f"alpha(1, 3) = {alpha(1, 3):+.4f}")
    print(f"alpha(5, 5) = {alpha(5, 5):+.4f}")
    print()
    print("=== s in [0,5], d in [0,5] ===")
    print_table([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
