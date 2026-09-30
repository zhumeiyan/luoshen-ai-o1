#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
果因准入.py — 未来值拒收
只收已发生/已观测的值。
未来值、推演值、预测值 → 拒收。
只算，不定义。
"""

from datetime import datetime


def check_causality(observed_time, claimed_effect_time, value):
    """
    检查因果关系是否成立（果不能先于因）。
    返回 (admitted, reason)
    """
    now = datetime.now().isoformat()
    
    if claimed_effect_time > now:
        return False, f"果 ({claimed_effect_time}) 在未来 → 拒收"
    
    if claimed_effect_time < observed_time:
        return False, f"果 ({claimed_effect_time}) 在因 ({observed_time}) 之前 → 拒收"
    
    return True, "OK: 因果时序合法"


def admit_measurement(value, unit, when_observed):
    """
    准入一条观测值。
    未来值拒收，已观测值准入。
    """
    admitted, reason = check_causality(when_observed, when_observed, value)
    return {
        "value": value,
        "unit": unit,
        "when": when_observed,
        "admitted": admitted,
        "reason": reason,
    }


if __name__ == "__main__":
    # 测试：合法观测
    print(admit_measurement(3.0, "m", "2024-01-01T00:00:00"))
    # 测试：未来值
    print(admit_measurement(999.0, "m", "2099-01-01T00:00:00"))
    # 测试：果在因之前
    print(check_causality("2024-06-01T00:00:00", "2024-01-01T00:00:00", None))
