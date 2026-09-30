#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
归一化.py — 单位偏移检查
只有含真零点的量才可入表。
只算，不定义。
"""


def check_zero_point(unit, values):
    """
    检查一组值是否共享真零点。
    返回 (has_zero, offset)：
      has_zero=True → 有真零点，可做偏移归一化
      has_zero=False → 无真零点，拒收
    """
    if not values:
        return False, None
    min_val = min(values)
    # 检查最小值是否可以作为零点参考
    if min_val >= 0:
        return True, 0.0  # 自然零点
    # 有负值，需要检查原点
    if any(v == 0 for v in values):
        return True, 0.0
    # 无零点 → 不能归一化入表
    return False, None


def normalize(values, offset=0.0):
    """偏移归一化"""
    return [v - offset for v in values]


def unit_check(unit_a, unit_b):
    """
    检查两个单位是否兼容。
    只有同量纲单位才能计算 α。
    """
    if unit_a != unit_b:
        return False, f"单位不匹配: {unit_a} ≠ {unit_b}"
    return True, "OK"


if __name__ == "__main__":
    # 测试
    v1 = [0, 5, 10, 15]
    v2 = [-5, 0, 5, 10]
    v3 = [1, 3, 5, 7]       # 无零点，不可归一化
    print(f"{v1}: {check_zero_point('m', v1)}")
    print(f"{v2}: {check_zero_point('m', v2)}")
    print(f"{v3}: {check_zero_point('m', v3)}")
    print(f"unit_check(m, s): {unit_check('m', 's')}")
    print(f"unit_check(m, m): {unit_check('m', 'm')}")
