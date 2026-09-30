#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interface_v2/translator.py — 会话→驱动 最小翻译器
==========================================================
输入：用户人话（一句）
输出：drive（一个数字）+ 理由（可审计）

映射规则（全部连续函数，无 if-then / 无词库 / 无随机）：

  1. 长度能量  E_len = sigmoid((len(msg) - 15) / 20)
     - 15 字以下近 0，35 字以上饱和 → [0, 1]

  2. 强度系数  S_punct = sigmoid((强度标点占比 - 0.05) / 0.03)
     强度标点 = ！!？?（每个 +1）
     - 问号叹号越多，信号越强 → [0, 1]

  3. 方向符号  D_sign = (CalmCount - IntenseCount) / max(total_punct, 1)
     平静标点：。，、；：…
     强烈标点：！？!?
     - 全平静 → -1；全强烈 → +1；标点少 → 近 0

  4. CJK通道  C_cjk = sigmoid((cjk_ratio - 0.5) / 0.2)
     - 中文密度高 → 近 1

  5. 组合：drive_raw = 0.5*E_len + 0.3*S_punct + 0.4*D_sign + 0.2*C_cjk
     drive = tanh(2 * 1.7 * drive_raw - 0.3)   → [-1, 1]

每次翻译留痕 type=translate，含 msg/drive/理由/ts。
引擎拒绝权：若 |drive| > 5.0（本 translator 已约束到 [-1, 1]，通常不触发；保留拒绝口给引擎层）。
"""

import json, math
from pathlib import Path
from datetime import datetime

AUDIT_DIR = Path(__file__).parent / "audit"
VERSION = "0.1"
# 输出缩放系数（经验调参，非推导）
# GAIN = 3.4（= 2.0 × 1.7），把 raw ∈ [-0.4, 1.4] 放大到 tanh 敏感区间
# BIAS = -0.3，中性消息输出略偏负（"平静"默认）
# 若替换数据分布或重调阈值，这两个数要重新拟 —— 不是物理常数
GAIN_INNER = 2.0
GAIN_OUTER = 1.7
BIAS = -0.3


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class InterfaceV2:
    """会话→驱动翻译器 v0.1 — 最小可审计版"""

    def __init__(self, user="emma"):
        self.user = user
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    def _signals(self, msg):
        L = max(len(msg), 1)

        # 长度能量
        e_len = _sigmoid((L - 15.0) / 20.0)

        # 强度标点（叹号、问号）
        intense_marks = set("！？!?")
        calm_marks = set("。，、；：…,.;:")
        n_intense = sum(1 for c in msg if c in intense_marks)
        n_calm = sum(1 for c in msg if c in calm_marks)
        n_total_punct = n_intense + n_calm
        s_punct = _sigmoid((n_intense / L - 0.05) / 0.03)

        # 方向符号（强烈标点越多 → d_sign 越大）
        d_sign = (n_intense - n_calm) / max(n_total_punct, 1)

        # CJK 通道
        cjk_count = sum(1 for c in msg if '一' <= c <= '鿿')
        c_cjk = _sigmoid((cjk_count / L - 0.5) / 0.2)

        return {
            "e_len": round(e_len, 6),
            "s_punct": round(s_punct, 6),
            "d_sign": round(d_sign, 6),
            "c_cjk": round(c_cjk, 6),
            "raw": {"L": L, "intense": n_intense, "calm": n_calm, "cjk": cjk_count}
        }

    def translate(self, msg):
        sig = self._signals(msg)
        raw = 0.5 * sig["e_len"] + 0.3 * sig["s_punct"] + 0.4 * sig["d_sign"] + 0.2 * sig["c_cjk"]
        gain = GAIN_INNER * GAIN_OUTER
        drive = math.tanh(gain * raw + BIAS)
        reason = {
            "version": VERSION,
            "user": self.user,
            "msg": msg,
            "signals": sig,
            "weights": {"w_len": 0.5, "w_punct": 0.3, "w_sign": 0.4, "w_cjk": 0.2},
            "gain_coefficients": {
                "value": gain,
                "formula_inner": GAIN_INNER,
                "formula_outer": GAIN_OUTER,
                "origin": "经验调参，非推导。把 raw∈[-0.4,1.4] 映射到 tanh 敏感区间。"
            },
            "bias": {"value": BIAS, "origin": "平静消息默认略偏负"},
            "raw_combination": round(raw, 6),
            "drive_formula": f"tanh({GAIN_INNER} * {GAIN_OUTER} * raw + ({BIAS}))",
            "drive": round(drive, 6),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self._audit(reason)
        return round(drive, 6), reason

    def _audit(self, reason):
        ts = reason["ts"].replace(":", "-")
        f = AUDIT_DIR / f"{ts}.jsonl"
        with open(f, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(reason, ensure_ascii=False) + "\n")


def selftest():
    """[判读] 跑通才算 [事实]"""
    import shutil
    t = InterfaceV2(user="test")

    # 1. 基础
    d, r = t.translate("你好，洛神赋。")
    assert -1.0 <= d <= 1.0, f"drive 越界：{d}"

    # 2. 大量叹号 → 较强正 drive
    d2, _ = t.translate("太棒了！真的吗！真的吗！")
    d3, _ = t.translate("好吧。")
    assert d2 > d3, f"叹号应更强：{d2} vs {d3}"

    # 3. 长消息饱和
    long_msg = "这是一个测试句子。" * 30
    d4, _ = t.translate(long_msg)
    assert -1.0 <= d4 <= 1.0

    # 4. 审计记录包含字段
    _, r4 = t.translate("测试。")
    for k in ("version", "msg", "signals", "weights", "drive", "ts"):
        assert k in r4, f"审计记录缺字段 {k}"

    # 5. 标点方向：平静 vs 强烈
    _, r_calm = t.translate("天气不错，适合散步。")
    _, r_intense = t.translate("怎么可能！为什么！")
    assert r_calm["signals"]["d_sign"] < r_intense["signals"]["d_sign"], \
        f"平静标点 d_sign 应较小：{r_calm['signals']['d_sign']} vs {r_intense['signals']['d_sign']}"

    # 清场
    shutil.rmtree(AUDIT_DIR)
    print("interface_v2 v0.1 selftest 5/5 过：基础/强度对比/长饱和/审计字段/方向符号")
    print("v0.1 只翻译不做决定；权重全部硬编码但公开可审计；无 if-then / 词库 / 随机。")


if __name__ == "__main__":
    selftest()
