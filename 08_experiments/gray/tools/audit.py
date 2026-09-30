#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计.py — 先存 r 再算 α
A × B = C → r_C = r_A × r_B
只算，不定义。
"""


def compute_r(a, b):
    """r = a × b 的乘积"""
    return a * b


def audit_product(a_val, b_val, c_claimed):
    """
    验证 A × B = C。
    先算 r_A、r_B，再算 r_C = r_A × r_B，与 claimed C 比对。
    """
    r_a = a_val
    r_b = b_val
    r_c = r_a * r_b
    match = abs(r_c - c_claimed) < 1e-12
    return {
        "r_A": r_a,
        "r_B": r_b,
        "r_C_computed": r_c,
        "r_C_claimed": c_claimed,
        "match": match,
    }


def audit_chain(pairs):
    """审计一连串乘法的累积信度"""
    results = []
    cumulative_r = 1.0
    for i, (a, b, claimed_c) in enumerate(pairs):
        r = audit_product(a, b, claimed_c)
        cumulative_r *= r["r_A"] * r["r_B"]
        r["step"] = i
        r["cumulative_r"] = cumulative_r
        results.append(r)
    return results


if __name__ == "__main__":
    # 测试: 2 × 3 = 6
    r = audit_product(2.0, 3.0, 6.0)
    print(f"A×B=C: 2×3=6 → {r}")
    # 测试 chain
    pairs = [(1.0, 2.0, 2.0), (2.0, 3.0, 6.0), (6.0, 0.5, 3.0)]
    chain = audit_chain(pairs)
    for c in chain:
        print(f"step {c['step']}: r_A={c['r_A']}, r_B={c['r_B']}, r_C={c['r_C_computed']}, match={c['match']}, cum_r={c['cumulative_r']}")
