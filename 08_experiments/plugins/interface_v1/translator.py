#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interface_v1/translator.py — 人话→drive 翻译器
==========================================================
人话 → 连续信号 → tanh 归一化 → drive ∈ [-1, 1]

设计约束：
  - 无 if-then、无词库、无 LLM、无 random
  - 所有信号通过连续数学函数组合
  - 输出归一化到 [-1, 1]（引擎在外部决定是否采用）
  - 每次转换写痕到 audit/ 目录

信号源：
  msg_len      : 消息长度 → sigmoid 能量信号（短消息近0，长消息饱和）
  punct_ratio  : 标点密度 → 情绪强度信号
  cjk_ratio    : 中文字符比例 → 语言通道信号
  digit_ratio  : 数字比例 → 事实性信号
  g_magnitude  : 当前 |g| 锚点量级 → 惯性阻尼信号
  desire_pri   : 当前最高激活欲望优先级比例 → 方向性信号

组合：drive = tanh(Σ w_i · signal_i)
  权重 w_i 本身也是锚点的连续函数（非硬编码 if-then）
"""
import json, math
from pathlib import Path
from datetime import datetime

AUDIT_DIR = Path(__file__).parent / "audit"


def _sigmoid(x):
    """logistic sigmoid，中心0，饱和 ±1"""
    return 2.0 / (1.0 + math.exp(-x)) - 1.0


def _normalize(x, scale):
    """sigmoid 归一化到 [-1, 1]，scale 控制中心斜率"""
    if scale == 0:
        return 0.0
    return _sigmoid(x / scale)


VERSION = "1.3"   # w_damp 公式由 sigmoid(|g|-3) 改为 sigmoid(|g|)，去符号翻转

class InterfaceV1:
    """人话 → drive 翻译器 v1.3"""

    def __init__(self):
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    def _extract_signals(self, message, anchor_state, desire_state):
        """从输入中提取数值信号。无词库。"""
        L = max(len(message), 1)

        punct_count = sum(1 for c in message if c in "。！？，、；：…—！？.,;:!?")
        cjk_count = sum(1 for c in message if '一' <= c <= '鿿')
        digit_count = sum(1 for c in message if c.isdigit())

        msg_len_energy = _normalize(L - 20, 40.0)
        punct_energy = _normalize(punct_count / L - 0.1, 0.05)
        cjk_channel = _normalize(cjk_count / L - 0.5, 0.2)
        factuality = _normalize(digit_count / L - 0.05, 0.03)

        g = anchor_state.get("g", 0.0)
        h = anchor_state.get("h", 0.0)
        g_mag = math.sqrt(g * g + 1e-20)
        inertia_damp = _normalize(g_mag, 5.0)

        desires = desire_state.get("desires", {})
        active_priorities = [d.get("优先级", 0) for d in desires.values()
                             if d.get("当前状态") == "激活"]
        if active_priorities:
            max_pri = max(active_priorities)
            desire_signal = _normalize(max_pri - 1, 3.0)
        else:
            desire_signal = 0.0

        return {
            "msg_len_energy": round(msg_len_energy, 6),
            "punct_energy": round(punct_energy, 6),
            "cjk_channel": round(cjk_channel, 6),
            "factuality": round(factuality, 6),
            "inertia_damp": round(inertia_damp, 6),
            "desire_signal": round(desire_signal, 6),
            "raw": {"L": L, "punct": punct_count, "cjk": cjk_count,
                    "digit": digit_count, "g": g, "h": h,
                    "active_prios": active_priorities}
        }

    def _compute_weights(self, anchor_state):
        """权重由锚点连续函数生成，非硬编码。
        v1.3：damp_factor = sigmoid(|g|)，去掉了 -3 偏移。
          - |g|=0 → damp=0 → w_damp=0（近平衡点无阻尼）
          - |g|→∞ → damp→1 → w_damp→-0.3（饱和阻尼）
        旧公式（v1.0-v1.2）在 |g|<3 时 w_damp 为正值（推动），行为不连续。"""
        g = anchor_state.get("g", 0.0)
        damp_factor = _sigmoid(abs(g))
        return {
            "w_len": round(0.4 * (1.0 - 0.5 * damp_factor), 4),
            "w_punct": 0.25,
            "w_cjk": 0.15,
            "w_fact": 0.2,
            "w_damp": round(-0.3 * damp_factor, 4),
            "w_desire": 0.35,
        }

    def translate(self, message, anchor_state, desire_state, user=None):
        """
        主入口：人话 → drive

        返回：
            drive:  float, ∈ [-1, 1]（引擎在外部决定是否采用）
            reason: dict,   完整信号与权重记录（可审计）
        """
        sig = self._extract_signals(message, anchor_state, desire_state)
        w = self._compute_weights(anchor_state)

        raw_drive = (
            w["w_len"]    * sig["msg_len_energy"] +
            w["w_punct"]  * sig["punct_energy"] +
            w["w_cjk"]    * sig["cjk_channel"] +
            w["w_fact"]    * sig["factuality"] +
            w["w_damp"]    * sig["inertia_damp"] +
            w["w_desire"]  * sig["desire_signal"]
        )

        drive = math.tanh(raw_drive)

        reason = {
            "signals": sig,
            "weights": w,
            "raw_drive": round(raw_drive, 6),
            "drive": round(drive, 6),
            "version": VERSION,
            "user": user,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }

        self._audit(reason)
        return round(drive, 6), reason

    def _audit(self, reason):
        """每次转换写一条审计记录"""
        ts = reason["ts"].replace(":", "-")
        f = AUDIT_DIR / f"{ts}.jsonl"
        with open(f, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(reason, ensure_ascii=False) + "\n")


def selftest():
    """[判读]：跑通才算 [事实]"""
    import shutil
    t = InterfaceV1()

    # 1. 基础翻译
    anchor = {"g": -5.0, "h": -5.0}
    desires = {"001": {"欲望名": "做AI", "当前状态": "激活", "优先级": 1}}
    d, r = t.translate("你好，洛神赋。", anchor, desires, user="test")
    assert -1.0 <= d <= 1.0, f"drive 越界：{d}"

    # 2. 极端长度
    long_msg = "测试" * 200
    d2, _ = t.translate(long_msg, anchor, desires, user="test")
    assert -1.0 <= d2 <= 1.0

    # 3. 高 g 阻尼
    high_g_anchor = {"g": 25.0, "h": 25.0}
    d3, r3 = t.translate("请继续", high_g_anchor, desires, user="test")
    assert -1.0 <= d3 <= 1.0
    assert r3["weights"]["w_damp"] < 0, "高 g 时阻尼权重应为负"

    # 4. 零消息
    d4, _ = t.translate("", {"g": 0, "h": 0}, desires, user="test")
    assert -1.0 <= d4 <= 1.0

    # 5. 权重随锚点变化
    w_low = t._compute_weights({"g": 0.0})["w_len"]
    w_high = t._compute_weights({"g": 20.0})["w_len"]
    assert w_low != w_high, "权重应随 g 变化"

    # 6. v1.3：w_damp 符号一致性 — |g|<3 时也必须为 0 或负，不能有正阻尼
    for g_val in [0.0, 0.5, 1.0, 2.0, 2.9]:
        w = t._compute_weights({"g": g_val})
        assert w["w_damp"] <= 0, f"|g|<3 时 w_damp 应为非正，got {w['w_damp']} @g={g_val}"
    # |g|=0 → w_damp 应为 0
    assert t._compute_weights({"g": 0.0})["w_damp"] == 0.0, "g=0 时 w_damp 必须为 0"
    # 审计记录包含版本号
    _, r6 = t.translate("测", {"g": 1.0}, desires, user="test")
    assert "version" in r6, "审计记录必须包含 version 字段"

    # 审计文件已写出
    audit_files = list(AUDIT_DIR.glob("*.jsonl"))
    assert len(audit_files) > 0, "审计报告未生成"

    # 清场
    shutil.rmtree(AUDIT_DIR)

    print("interface_v1 v1.3 selftest 6/6 过：基础/长度/阻尼/零消息/锚点权重/damp符号一致性")
    print("w_damp 公式改为 sigmoid(|g|)，去掉了 -3 偏移：近平衡无阻尼，远平衡饱和阻尼。")
    print("此刻起，本机上的实现升级[事实]；其余仍是[判读]。")


if __name__ == "__main__":
    selftest()
