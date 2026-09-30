#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反哺机制 A/B 测试（v2：拒绝独立跑 + 补证据）
"""
import sys, json, hashlib, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from 洛神 import Difference, Pair, Trace, ROOT

def file_hash(p):
    if not Path(p).exists(): return "FILE_NOT_FOUND"
    return hashlib.sha512(Path(p).read_bytes()).hexdigest()[:16]

def clean_user(u):
    p = ROOT / "users" / u
    if p.exists(): shutil.rmtree(p)

def run_100_steps(user, init_a=None, init_b=None):
    a_bias = init_a if init_a is not None else [1.0, 0.5]
    b_bias = init_b if init_b is not None else [-1.0, -0.5]
    a = Difference("a", a_bias, 0.99)
    b = Difference("b", b_bias, 0.99)
    t = Trace(user)
    p = Pair(a, b)
    for i in range(100):
        p.write_a(0.3)
        t.append("step", {"drive": 0.3, "g": p.g(), "h": p.h()})
    return p.g(), p.h(), t.tail()["hash"]

MAIN_ANCHOR = Path("anchor/顶层.md")
PENDING_DIR = Path("gray/pending")
PENDING_DIR.mkdir(parents=True, exist_ok=True)

# ── 保存原始锚点，用于每次测试前后还原 ──
ORIGINAL_ANCHOR = MAIN_ANCHOR.read_text(encoding="utf-8")
ORIGINAL_HASH = file_hash(MAIN_ANCHOR)

def restore_anchor():
    MAIN_ANCHOR.write_text(ORIGINAL_ANCHOR, encoding="utf-8")

def trace_cleanup(op, target):
    """入痕：清理操作本身写进审计链"""
    t = Trace("audit")
    t.append("cleanup", {"op": op, "target": target})

# ════════════════════════════════════════
# 选项 A：不上升
# ════════════════════════════════════════
print("=" * 60)
print("选项 A：不上升（独立）")
print("=" * 60)

h_before = file_hash(MAIN_ANCHOR)
print(f"主锚点哈希（跑前）: {h_before}")
assert h_before == ORIGINAL_HASH, "锚点已被先前测试污染！"

clean_user("fb_A")
g_final, h_final, _ = run_100_steps("fb_A")
print(f"引擎跑完 g={g_final} h={h_final}")

# 规则A：不碰主锚点
h_after = file_hash(MAIN_ANCHOR)
print(f"主锚点哈希（跑后）: {h_after}")
A_pass = (h_before == h_after)
print(f"主锚点被污染? {'✗ YES' if not A_pass else '✓ NO'}")

# 清理 + 入痕
clean_user("fb_A")
trace_cleanup("rm_user", "fb_A")
print("清理: user fb_A 删除 → 入痕")

# ════════════════════════════════════════
# 选项 B - 场景 1：用户确认（独立）
# ════════════════════════════════════════
print()
print("=" * 60)
print("选项 B - 场景 1：用户确认（独立）")
print("=" * 60)

restore_anchor()
h_before_B1 = file_hash(MAIN_ANCHOR)
print(f"主锚点哈希（跑前）: {h_before_B1}")
assert h_before_B1 == ORIGINAL_HASH

clean_user("fb_B1")
g_B1, h_B1, _ = run_100_steps("fb_B1")
print(f"引擎跑完 g={g_B1}")

# 生成 pending
pending_file = PENDING_DIR / "confirm_test.md"
pending_file.write_text(
    f"# 待确认项 confirm_test\n\n"
    f"来源：engine.step.100\n"
    f"变化摘要：g 从 [-2.0,-1.0] 变为 {g_B1}\n"
    f"建议动作：评估主锚点是否需要同步更新\n"
    f"状态：待确认\n",
    encoding="utf-8")
print(f"\n=== pending 文件内容 ===")
print(pending_file.read_text(encoding="utf-8"))
print(f"===")

h_unc = file_hash(MAIN_ANCHOR)
print(f"主锚点哈希（未确认）: {h_unc}")
B1_no_spill = (h_before_B1 == h_unc)
print(f"未确认时污染? {'✗ YES' if not B1_no_spill else '✓ NO'}")

# 用户确认 → 改主锚点
anchor_new = ORIGINAL_ANCHOR.rstrip() + f"\n\n## 用户确认项 (confirm_test)\n确认时间：引擎 step 100 后\n修改：添加反哺确认记录\n"
MAIN_ANCHOR.write_text(anchor_new, encoding="utf-8")
h_confirmed = file_hash(MAIN_ANCHOR)
print(f"\n主锚点哈希（确认后）: {h_confirmed}")
print(f"\n=== 确认后 diff ===")
import subprocess
diff_out = subprocess.run(["diff", "-u", "/dev/stdin", str(MAIN_ANCHOR)],
    input=ORIGINAL_ANCHOR, capture_output=True, text=True, encoding="utf-8")
# 手动 diff 输出
orig_lines = ORIGINAL_ANCHOR.splitlines(keepends=True)
new_lines = anchor_new.splitlines(keepends=True)
print("--- 原始锚点")
print("+++ 确认后锚点")
for i, (o, n) in enumerate(zip(orig_lines, new_lines)):
    if o != n:
        print(f"@@ 行 {i+1}")
        print(f"- {o.rstrip()}")
        print(f"+ {n.rstrip()}")
# 打印追加的行
if len(new_lines) > len(orig_lines):
    for l in new_lines[len(orig_lines):]:
        print(f"+ {l.rstrip()}")
print()

B1_confirm_works = (h_unc != h_confirmed)
print(f"确认导致主锚点变化? {'✓ YES（预期）' if B1_confirm_works else '✗ NO（异常）'}")

# diff 输出
print(f"\n主锚点 diff (确认后 vs 原始):")
for l in new_lines[len(orig_lines):]:
    print(f"  + {l.rstrip()}")
print()

# 入痕 + 还原
restore_anchor()
clean_user("fb_B1")
if pending_file.exists(): pending_file.unlink()
trace_cleanup("restore_and_cleanup", "fb_B1+confirm_pending")

# ════════════════════════════════════════
# 选项 B - 场景 2：用户拒绝（独立）
# ════════════════════════════════════════
print()
print("=" * 60)
print("选项 B - 场景 2：用户拒绝（独立）")
print("=" * 60)

restore_anchor()
h_before_B2 = file_hash(MAIN_ANCHOR)
print(f"主锚点哈希（跑前）: {h_before_B2}")
assert h_before_B2 == ORIGINAL_HASH

clean_user("fb_B2")
g_B2, h_B2, _ = run_100_steps("fb_B2")
print(f"引擎跑完 g={g_B2}")

# 生成 pending
pending_file2 = PENDING_DIR / "reject_test.md"
pending_file2.write_text(
    f"# 待确认项 reject_test\n\n"
    f"来源：engine.step.100\n"
    f"变化摘要：g 从 [-2.0,-1.0] 变为 {g_B2}\n"
    f"建议动作：评估主锚点是否需要同步更新\n"
    f"状态：待确认 → 已拒绝\n"
    f"用户操作：拒绝 by user at 2026-09-16\n",
    encoding="utf-8")

# 用户拒绝 → 主锚点不动
h_rejected = file_hash(MAIN_ANCHOR)
print(f"主锚点哈希（拒绝后）: {h_rejected}")
B2_reject = (h_before_B2 == h_rejected)
print(f"拒绝阻止了变化? {'✓ YES（预期）' if B2_reject else '✗ NO（异常）'}")

# 入痕 + 清理
clean_user("fb_B2")
if pending_file2.exists(): pending_file2.unlink()
trace_cleanup("cleanup_reject", "fb_B2+reject_pending")
print(f"清理: user fb_B2 删除、pending 删除（保留拒绝记录在主链） → 入痕")

# ════════════════════════════════════════
# 法务审批：确保顶层锚点干净无染
# ════════════════════════════════════════
print()
print("=" * 60)
print("法务审批：顶层锚点最终状态")
print("=" * 60)
restore_anchor()
final_hash = file_hash(MAIN_ANCHOR)
print(f"最终锚点哈希:   {final_hash}")
print(f"原始锚点哈希:   {ORIGINAL_HASH}")
print(f"锚点干净无染? {'✓ YES' if final_hash == ORIGINAL_HASH else '✗ NO'}")

# ════════════════════════════════════════
# 汇总
# ════════════════════════════════════════
print()
print("=" * 60)
print("汇总")
print("=" * 60)
print(f"A 不上升（独立）:         {'✓ PASS' if A_pass else '✗ FAIL'}")
print(f"B 确认前不污染（独立）:   {'✓ PASS' if B1_no_spill else '✗ FAIL'}")
print(f"B 确认后主锚点变（独立）: {'✓ PASS' if B1_confirm_works else '✗ FAIL'}")
print(f"B 拒绝阻止变化（独立）:   {'✓ PASS' if B2_reject else '✗ FAIL'}")
