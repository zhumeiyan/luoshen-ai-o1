#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读表.py — 读灰度区数据，自动算 α
扫描 gray/tests/by_domain/ 下所有 `.md` 文件，
解析符合格式的行：... | a | b | ... → 算 α。
只算，不定义。
"""

import os
import re
from pathlib import Path

TOOLS_DIR = Path(__file__).parent
GRAY_DIR = TOOLS_DIR.parent


def parse_data_line(line):
    """解析一行：数据 | ... | a | b | α | ..."""
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 8:
        return None
    try:
        data = parts[0]
        a = float(parts[5])
        b = float(parts[6])
        alpha_str = parts[7].strip()
        alpha_val = None if alpha_str in ("", "—", "NaN") else float(alpha_str)
        return {"data": data, "a": a, "b": b, "alpha_claimed": alpha_val}
    except (ValueError, IndexError):
        return None


def read_domain_file(filepath):
    """读一个领域文件，解析有效数据行"""
    results = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("数据"):
                continue
            parsed = parse_data_line(line)
            if parsed:
                results.append(parsed)
    return results


def auto_alpha(a, b):
    """自动计算 α = (a-b)/(a+b)"""
    if a + b == 0:
        return None
    return (a - b) / (a + b)


def scan_all():
    domain_dir = GRAY_DIR / "tests" / "by_domain"
    if not domain_dir.exists():
        print(f"目录不存在: {domain_dir}")
        return
    total = 0
    for fpath in sorted(domain_dir.glob("*.md")):
        items = read_domain_file(fpath)
        if items:
            print(f"\n=== {fpath.name} ({len(items)}行) ===")
            for item in items:
                computed = auto_alpha(item["a"], item["b"])
                claimed = item["alpha_claimed"]
                if claimed is not None and computed is not None:
                    match = abs(computed - claimed) < 0.01
                    flag = "OK" if match else "MISMATCH"
                else:
                    flag = "NA"
                print(f"  {item['data'][:20]:20s} a={item['a']:6.2f} b={item['b']:6.2f} α_算={computed:+.4f} α_称={claimed} [{flag}]")
                total += 1
    print(f"\n总计 {total} 行")


if __name__ == "__main__":
    scan_all()
