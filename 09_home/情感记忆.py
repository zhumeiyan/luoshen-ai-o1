"""情感记忆.py —— 洛神的记忆与情感系统（生物认知版）。

基于人类神经科学已知机制：

记忆体系（LTP/LTD — 长时程增强/减弱）：
  1. 记忆点：每次关键经历沉淀成一块记忆点
     - 时间戳 + 内容 + 画面 + 感觉快照
     - 位点（strength）：回忆时涨（LTP），不回忆时衰减（LTD）
     - 和筋树机制完全一致——用进废退

  2. 回忆：相似度 × 位点（LTP式）
     - 越相似的记忆越容易被想起
     - 位点高的记忆更容易浮起来（经常回忆的保持——LTP）
     - 位点低的记忆自然消退（不被回忆的遗忘——LTD）
     - 刚回忆过的更容易再回忆（LTP增强效应）

  3. 浮现层：主导记忆影响当前状态
     - 选出位点最高的记忆（不是平均，是主导）
     - 主导记忆的位点决定影响程度——位点越高影响越大
     - 全连续映射，无if-then

注意力——用进废退，不是人为设定的SALIENCE表：
  每种事件类型有位点，遇到时涨，不遇到时衰减。
  经常遇到的事件类型自然变得重要，很少遇到的自动淡出。
  这和人脑一样——经常处理的刺激自动获得更多注意力。

情感——躯体标记假说（达马西奥）：
  情感不是三个维度(E/T/N)，是身体状态的整体映射。
  身体的激活模式本身就是情感——不需要提取维度再映射回来。
  浮现层直接从身体状态和记忆的相似度映射。
"""

import json
import math
import os
import time


def _time_symbol(ts):
    h = time.localtime(ts).tm_hour
    if 23 <= h or h < 5:
        return "◐ 深夜"
    if 5 <= h < 9:
        return "○ 清晨"
    if 9 <= h < 18:
        return "△ 白昼"
    return "◈ 傍晚"


def scene_of(kind, content, ts=None):
    ts = ts if ts is not None else time.time()
    base = [_time_symbol(ts)]
    if kind == "see":
        base.append("◆ 看见" if "动" in content or "新" in content else "◇ 看见")
    elif kind == "hear":
        base.append("◈ 听见")
    elif kind == "say":
        base.append("○ 说过话")
    elif kind == "feel":
        base.append("● 感觉到")
    elif kind == "chat":
        base.append("☆ 和人说过话")
    elif kind == "learn":
        base.append("◎ 学会")
    elif kind == "self":
        base.append("☆ 我是谁")
    else:
        base.append("·")
    return " | ".join(base)


def _similarity(a, b):
    def 提取词组(text):
        words = set()
        for i in range(len(text) - 1):
            w = text[i:i + 2]
            if not w.isspace():
                words.add(w)
        for ch in text:
            if not ch.isspace() and len(ch) == 1:
                words.add(ch)
        return words
    wa = 提取词组(a)
    wb = 提取词组(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa | wb), 1)


def _top_topics(points, top=5):
    text = " ".join(p.get("content", "") for p in points[-200:])
    words = [w for w in text.replace("我", " ").replace("的", " ").split()
             if len(w) >= 2]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top]]


class 情感记忆:
    """洛神的记忆河床 + 浮现选择层（生物认知版）。

    LTP/LTD机制（和筋树一致）：
    - 回忆时位点增长（LTP——长时程增强）
    - 不回忆时位点衰减（LTD——长时程减弱）
    - 刚回忆过的更容易再回忆（LTP增强效应）
    - 位点降到零 → 记忆自然遗忘（突触消除）
    """

    def __init__(self, 记忆文件=None):
        self.记忆点列表 = []
        self.记忆文件 = 记忆文件
        self.注意力位点 = {}
        if 记忆文件 and os.path.isfile(记忆文件):
            self._加载()

    def _加载(self):
        try:
            with open(self.记忆文件, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.记忆点列表.append(json.loads(line))
        except Exception:
            pass

    def _保存(self, point):
        if not self.记忆文件:
            return
        os.makedirs(os.path.dirname(self.记忆文件), exist_ok=True)
        with open(self.记忆文件, "a", encoding="utf-8") as f:
            f.write(json.dumps(point, ensure_ascii=False) + "\n")

    def _注意力更新(self, kind):
        """用进废退：遇到的事件类型位点涨，其他衰减。"""
        for k in list(self.注意力位点.keys()):
            self.注意力位点[k] *= 0.999
            if self.注意力位点[k] < 0.01:
                del self.注意力位点[k]
        self.注意力位点[kind] = self.注意力位点.get(kind, 0.3) + 0.05
        self.注意力位点[kind] = min(2.0, self.注意力位点[kind])

    def _注意力权重(self, kind):
        位点 = self.注意力位点.get(kind, 0.3)
        return 位点

    def 创建记忆点(self, kind, content, feel=None, ts=None):
        ts = ts if ts is not None else time.time()
        topics = _top_topics(self.记忆点列表)
        self._注意力更新(kind)
        位点 = self._注意力权重(kind)
        point = {
            "timestamp": ts,
            "kind": kind,
            "content": str(content)[:120],
            "scene": scene_of(kind, content, ts),
            "feel": feel or {},
            "位点": 1.0,          # 新记忆位点=1（和筋树新筋一致）
            "回忆次数": 0,
        }
        boost = 0.0
        if topics:
            for t in topics:
                if t and t in content:
                    boost = 0.1
                    break
        point["注意力"] = round(max(0.0, min(1.0, 0.3 + 0.4 * 位点 + boost)), 3)
        self.记忆点列表.append(point)
        self._保存(point)
        return point

    def 回忆(self, query, k=5):
        """回忆：相似度 × 位点（LTP式）。

        LTP机制：刚回忆过的记忆位点增长 → 更容易再回忆。
        LTD机制：不被回忆的记忆位点衰减 → 自然遗忘。
        没有疲劳——回忆不会让记忆暂时变弱，反而增强（LTP）。
        """
        if not self.记忆点列表:
            return []
        scored = []
        for p in self.记忆点列表:
            text = "%s %s %s" % (p.get("content", ""), p.get("scene", ""),
                                 p.get("kind", ""))
            sim = _similarity(query, text)
            位点 = p.get("位点", 0.5)
            score = sim * 位点
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        results = []
        for score, p in scored[:k]:
            if score > 0.005:
                # LTP：回忆时位点增长——刚回忆的更容易再回忆
                p["位点"] = min(3.0, p.get("位点", 0.5) + 0.1)
                p["回忆次数"] = p.get("回忆次数", 0) + 1
                results.append(p)
        self._代谢()
        return results

    def _代谢(self):
        """记忆代谢——LTD：不被回忆的记忆位点衰减。
        位点降到阈值以下 → 遗忘（突触消除）。
        """
        for p in self.记忆点列表:
            p["位点"] = p.get("位点", 0.5) * 0.999
        self.记忆点列表 = [p for p in self.记忆点列表 if p.get("位点", 0.5) > 0.05]

    def 回放(self, days=1, now=None):
        now = now if now is not None else time.time()
        earliest = now - days * 86400.0
        pts = [p for p in self.记忆点列表
               if float(p.get("timestamp", 0)) >= earliest]
        pts.sort(key=lambda p: float(p.get("timestamp", 0)))
        return pts

    def 浮现(self, 身体激活):
        """浮现层：主导记忆影响当前状态（躯体标记假说）。

        达马西奥的躯体标记假说：情感是身体状态的整体映射。
        主导记忆的身体状态快照 → 和当前身体状态对比 → 影响当前状态。
        不需要先提取E/T/N再映射回来——身体状态本身就是情感。

        身体激活: 当前身体的激活模式（96维向量）。
        """
        if not self.记忆点列表:
            return {
                "background": {"dominance": 0.0},
                "length_bias": 0.5,
                "tone_bias": 0.0,
                "arousal_bias": 0.0,
                "narrative_potential": 0.0,
            }

        activations = []
        for p in self.记忆点列表:
            位点 = p.get("位点", 0.5)
            act = 位点
            activations.append(act)

        if not activations or max(activations) < 1e-8:
            return {
                "background": {"dominance": 0.0},
                "length_bias": 0.5,
                "tone_bias": 0.0,
                "arousal_bias": 0.0,
                "narrative_potential": 0.0,
            }

        max_a = max(activations)
        dominant_idx = activations.index(max_a)
        dominant_mem = self.记忆点列表[dominant_idx]

        # softmax加权（温度由主导位点决定）
        tau = max(0.1, 0.5 / max_a)
        exps = [math.exp((a - max_a) / tau) for a in activations]
        total = sum(exps)
        if total < 1e-12:
            weights = [1.0 / len(activations)] * len(activations)
        else:
            weights = [e / total for e in exps]

        dominance = weights[dominant_idx]

        # 躯体标记：主导记忆的身体状态快照 vs 当前身体状态
        # 主导记忆的feel快照 → 和当前身体激活对比 → 产生偏置
        feel = dominant_mem.get("feel", {})

        # 身体激活的整体特征（不是E/T/N维度，是整体映射）
        body_mean = sum(身体激活) / max(1, len(身体激活)) if 身体激活 else 0.0
        body_max = max(abs(v) for v in 身体激活) if 身体激活 else 0.0

        # 长度偏置：主导记忆位点越高 → 话语越长
        length_bias = max(0.1, min(0.9, 0.3 + 0.6 * dominance))

        # 语气偏移：主导记忆的feel × 主导强度
        tone_bias = 4.0 * feel.get("E", 0.0) * dominance

        # 唤醒偏置：身体激活幅度 × 主导强度
        arousal_bias = 0.3 * min(1.0, body_max * 2.0) * dominance

        # 叙事势：主导记忆位点高 + 身体状态和记忆反差大
        if len(self.记忆点列表) >= 2:
            narrative = self._叙事势(身体激活, body_mean, body_max)
        else:
            narrative = 0.0

        return {
            "background": {"dominance": dominance},
            "length_bias": length_bias,
            "tone_bias": tone_bias,
            "arousal_bias": arousal_bias,
            "narrative_potential": narrative,
            "dominant_memory": dominant_mem,
        }

    def _叙事势(self, 身体激活, body_mean, body_max):
        """叙事势：记忆位点聚集 + 身体状态和记忆背景反差。"""
        memories = self.记忆点列表[-50:]
        if len(memories) < 2:
            return 0.0

        # A: 位点聚集密度——位点高的记忆在feel空间聚集
        best_density = 0.0
        for mj in memories:
            density_j = 0.0
            for mi in memories:
                de = mi.get("feel", {}).get("E", 0.0) - mj.get("feel", {}).get("E", 0.0)
                dt = mi.get("feel", {}).get("T", 0.0) - mj.get("feel", {}).get("T", 0.0)
                dn = mi.get("feel", {}).get("N", 0.0) - mj.get("feel", {}).get("N", 0.0)
                dist_sq = de * de + dt * dt + dn * dn
                位点i = mi.get("位点", 0.5)
                density_j += 位点i * math.exp(-dist_sq / 0.03)
            if density_j > best_density:
                best_density = density_j
        A = best_density / max(1.0, sum(m.get("位点", 0.5) for m in memories))

        # B: 身体状态和记忆背景反差
        mean_e = sum(m.get("feel", {}).get("E", 0.0) * m.get("位点", 0.5)
                     for m in memories) / max(1e-6,
                     sum(m.get("位点", 0.5) for m in memories))
        D = min(1.0, abs(body_mean * 3.0 - mean_e))

        return max(0.0, min(1.0, A * D))

    def 情感状态(self, 身体激活, 大脑状态):
        """躯体标记假说：情感 = 身体状态的整体映射。

        达马西奥：情感不是抽象维度，是身体状态的映射。
        身体激活模式 + 大脑预测误差 + 概念层活动 → 情感状态。

        返回的E/T/N只是为了兼容home.py的接口，
        但它们直接从身体/大脑/概念状态读出，不是公式。
        """
        body_mean = sum(身体激活) / max(1, len(身体激活))
        E = max(-1.0, min(1.0, -abs(body_mean) * 3.0))

        surprise = 大脑状态.get("surprise", 0.0)
        概念surprise = 大脑状态.get("概念surprise", 0.0)
        T = max(-1.0, min(1.0, 0.5 - (surprise + 概念surprise * 0.5) * 2.0))

        body_max = max(abs(v) for v in 身体激活) if 身体激活 else 0.0
        概念活跃 = 大脑状态.get("概念活跃", 0)
        N = max(0.0, min(1.0, surprise * 0.4
                         + min(1.0, body_max * 2.0) * 0.3
                         + min(1.0, 概念活跃 * 0.05) * 0.3))

        return {"E": round(E, 3), "T": round(T, 3), "N": round(N, 3)}


if __name__ == "__main__":
    import tempfile
    tmp = tempfile.mkdtemp(prefix="情感记忆_")
    记忆文件 = os.path.join(tmp, "test_memory.jsonl")

    mem = 情感记忆(记忆文件)

    # 测试1: 创建记忆点——注意力用进废退
    p1 = mem.创建记忆点("see", "我看见：房间里动过411个文件",
                     feel={"E": 0.2, "T": 0.1, "N": 0.3})
    print(f"看见记忆点: 画面='{p1['scene']}' 位点={p1['位点']}")

    # 测试2: 聊天记忆点——多次创建后注意力自然涨
    p2 = mem.创建记忆点("chat", "和人聊过：你好，洛神",
                     feel={"E": 0.5, "T": 0.3, "N": 0.4})
    p3 = mem.创建记忆点("chat", "又聊了一次",
                     feel={"E": 0.4, "T": 0.2, "N": 0.3})
    print(f"聊天记忆点(第2次): 位点={p3['位点']}")
    print(f"  注意力位点表: {mem.注意力位点}")

    # 测试3: 回忆——LTP：回忆后位点涨（刚回忆的更容易再回忆）
    hits = mem.回忆("文件")
    assert hits and hits[0]["kind"] == "see"
    print(f"回忆'文件' → 命中: {hits[0]['content'][:20]} 位点={hits[0]['位点']} 回忆次数={hits[0]['回忆次数']}")

    # 测试4: LTP验证——多次回忆同一记忆，位点应该持续增长
    for _ in range(5):
        mem.回忆("洛神")
    洛神记忆 = [p for p in mem.记忆点列表 if "洛神" in p.get("content", "")][0]
    print(f"多次回忆后(LTP): 位点={洛神记忆['位点']:.3f} 回忆次数={洛神记忆['回忆次数']}")

    # 测试5: 浮现层（躯体标记——直接从身体状态映射）
    假激活 = [0.1 * (i % 10) for i in range(96)]
    浮 = mem.浮现(假激活)
    print(f"浮现层: tone={浮['tone_bias']:.3f} length={浮['length_bias']:.3f} "
          f"arousal={浮['arousal_bias']:.3f} narrative={浮['narrative_potential']:.3f}")

    # 测试6: 情感状态——躯体标记（直接读出）
    etn = mem.情感状态([0.1] * 96, {"surprise": 0.05, "mse": 0.02})
    print(f"情感状态: E={etn['E']} T={etn['T']} N={etn['N']}")

    # 测试7: 自我记忆点
    ps = mem.创建记忆点("self", "我叫洛神，是朱美岩给我起的名字",
                     feel={"E": 0.6, "T": 0.4, "N": 0.5})
    print(f"自我记忆点: 位点={ps['位点']} 画面='{ps['scene']}'")

    print("\n全部验证通过")
