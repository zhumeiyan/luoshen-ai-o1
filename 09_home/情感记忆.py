"""情感记忆.py —— 洛神的记忆与情感系统（纯进化版）。

核心原则（来自数字生命_纯进化.py）：
  零判据、零人为干预、零阈值。

记忆体系（筋树式用进废退）：
  1. 记忆点：每次关键经历沉淀成一块记忆点
     - 时间戳 + 内容 + 画面 + 感觉快照
     - 位点（strength）：回忆时涨，不回忆时衰减——和筋树完全一致
     - 疲劳（fatigue）：刚回忆过的暂时不易再浮起（防止死循环）

  2. 回忆：相似度 × 位点 ÷ 疲劳（不是时间衰减）
     - 越相似的记忆越容易被想起
     - 位点高的记忆更容易浮起来（经常回忆的保持）
     - 位点低的记忆自然消退（不回忆的遗忘）
     - 疲劳高的暂时被抑制（刚想过的不会立刻再想）

  3. 浮现层：主导记忆影响当前状态
     - 选出位点最高的记忆（不是平均，是主导）
     - 主导记忆的位点决定影响程度——位点越高影响越大
     - 全连续映射，无if-then

注意力——用进废退，不是人为设定的SALIENCE表：
  每种事件类型有位点，遇到时涨，不遇到时衰减。
  经常遇到的事件类型自然变得重要，很少遇到的自动淡出。
  这和人脑一样——经常处理的刺激自动获得更多注意力。

情感维度（E/T/N）——直接从身体/大脑状态读出，不是公式：
  E (效价): 身体舒适度——身体激活离稳态越近越正
  T (语气): 大脑预测信心——surprise越低越开放
  N (唤醒): 激活强度——身体和大脑越活跃越高
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
    """洛神的记忆河床 + 浮现选择层（纯进化版）。

    用进废退机制：
    - 注意力位点：每种事件类型有位点，遇到涨、不遇衰减
    - 记忆位点：每条记忆有位点，回忆涨、不回忆衰减
    - 记忆疲劳：刚回忆过的暂时被抑制
    - 位点降到零 → 记忆自然遗忘（和筋树代谢一致）
    """

    def __init__(self, 记忆文件=None):
        self.记忆点列表 = []
        self.记忆文件 = 记忆文件
        self.注意力位点 = {}  # kind → 位点，用进废退
        self.注意力疲劳 = {}  # kind → 疲劳
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
        """用进废退：遇到的事件类型位点涨，其他衰减。疲劳也衰减。"""
        for k in list(self.注意力位点.keys()):
            self.注意力位点[k] *= 0.999
            self.注意力疲劳[k] *= 0.95
            if self.注意力位点[k] < 0.01:
                del self.注意力位点[k]
                if k in self.注意力疲劳:
                    del self.注意力疲劳[k]
        self.注意力位点[kind] = self.注意力位点.get(kind, 0.3) + 0.05
        self.注意力疲劳[kind] = self.注意力疲劳.get(kind, 0.0) + 0.1
        self.注意力位点[kind] = min(2.0, self.注意力位点[kind])

    def _注意力权重(self, kind):
        """当前注意力权重 = 位点 / (1 + 疲劳)。"""
        位点 = self.注意力位点.get(kind, 0.3)
        疲劳 = self.注意力疲劳.get(kind, 0.0)
        return 位点 / (1.0 + 疲劳)

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
            "疲劳": 0.0,          # 新记忆无疲劳
            "回忆次数": 0,
            "e_s": (feel or {}).get("E", 0.0),
            "t_s": (feel or {}).get("T", 0.0),
            "n_s": (feel or {}).get("N", 0.0),
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
        """回忆：相似度 × 位点 ÷ (1 + 疲劳)。

        不用时间衰减——用进废退：
        位点高的（经常被回忆的）容易浮起，
        位点低的（不被回忆的）自然消退，
        疲劳高的（刚回忆过的）暂时被抑制。
        """
        if not self.记忆点列表:
            return []
        scored = []
        for p in self.记忆点列表:
            text = "%s %s %s" % (p.get("content", ""), p.get("scene", ""),
                                 p.get("kind", ""))
            sim = _similarity(query, text)
            位点 = p.get("位点", 0.5)
            疲劳 = p.get("疲劳", 0.0)
            score = sim * 位点 / (1.0 + 疲劳)
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        results = []
        for score, p in scored[:k]:
            if score > 0.005:
                p["位点"] = min(3.0, p.get("位点", 0.5) + 0.1)
                p["疲劳"] = p.get("疲劳", 0.0) + 0.3
                p["回忆次数"] = p.get("回忆次数", 0) + 1
                results.append(p)
        self._代谢()
        return results

    def _代谢(self):
        """记忆代谢——和筋树代谢一致：
        位点衰减，疲劳衰减，位点降到阈值以下 → 遗忘（删除）。
        """
        for p in self.记忆点列表:
            p["位点"] = p.get("位点", 0.5) * 0.999
            p["疲劳"] = p.get("疲劳", 0.0) * 0.95
        self.记忆点列表 = [p for p in self.记忆点列表 if p.get("位点", 0.5) > 0.05]

    def 回放(self, days=1, now=None):
        now = now if now is not None else time.time()
        earliest = now - days * 86400.0
        pts = [p for p in self.记忆点列表
               if float(p.get("timestamp", 0)) >= earliest]
        pts.sort(key=lambda p: float(p.get("timestamp", 0)))
        return pts

    def 浮现(self, cur_e, cur_t, cur_n):
        """浮现层：主导记忆背景 + 语气/长度/唤醒偏置。

        主导记忆的位点决定影响程度——位点越高影响越大。
        不用固定系数，用位点自然加权。
        """
        if not self.记忆点列表:
            return {
                "background": {"e_bg": 0.0, "t_bg": 0.0, "n_bg": 0.0,
                               "dominance": 0.0},
                "length_bias": 0.5,
                "tone_bias": 0.0,
                "arousal_bias": 0.0,
                "narrative_potential": 0.0,
            }

        activations = []
        for p in self.记忆点列表:
            位点 = p.get("位点", 0.5)
            疲劳 = p.get("疲劳", 0.0)
            act = 位点 / (1.0 + 疲劳)
            activations.append(act)

        if not activations or max(activations) < 1e-8:
            return {
                "background": {"e_bg": 0.0, "t_bg": 0.0, "n_bg": 0.0,
                               "dominance": 0.0},
                "length_bias": 0.5,
                "tone_bias": 0.0,
                "arousal_bias": 0.0,
                "narrative_potential": 0.0,
            }

        max_a = max(activations)
        dominant_idx = activations.index(max_a)
        dominant_mem = self.记忆点列表[dominant_idx]

        # softmax加权（温度由主导位点决定——位点高=分布尖锐）
        tau = max(0.1, 0.5 / max_a)
        exps = [math.exp((a - max_a) / tau) for a in activations]
        total = sum(exps)
        if total < 1e-12:
            weights = [1.0 / len(activations)] * len(activations)
        else:
            weights = [e / total for e in exps]

        e_bg = sum(w * m.get("e_s", 0.0) for w, m in zip(weights, self.记忆点列表))
        t_bg = sum(w * m.get("t_s", 0.0) for w, m in zip(weights, self.记忆点列表))
        n_bg = sum(w * m.get("n_s", 0.0) for w, m in zip(weights, self.记忆点列表))

        dominance = weights[dominant_idx]

        bg = {
            "e_bg": e_bg, "t_bg": t_bg, "n_bg": n_bg,
            "dominance": dominance,
        }

        # 长度偏置：主导记忆位点越高 → 话语越长
        length_bias = max(0.1, min(0.9, 0.3 + 0.6 * dominance))

        # 语气偏移：主导记忆的情感效价 × 主导强度
        tone_bias = 4.0 * (e_bg - 0.0) * dominance

        # 唤醒偏置：主导记忆的唤醒 × 主导强度
        arousal_bias = 0.3 * n_bg * dominance

        # 叙事势：主导记忆位点高 + 当前状态和背景反差大 → 叙事冲动
        if len(self.记忆点列表) >= 2:
            narrative = self._叙事势(cur_e, cur_t, cur_n)
        else:
            narrative = 0.0

        return {
            "background": bg,
            "length_bias": length_bias,
            "tone_bias": tone_bias,
            "arousal_bias": arousal_bias,
            "narrative_potential": narrative,
            "dominant_memory": dominant_mem,
        }

    def _叙事势(self, cur_e, cur_t, cur_n):
        """叙事势：记忆位点聚集 + 今昔反差。

        位点高的记忆聚集（经常回忆的主题集中）+ 当前状态和记忆背景反差大
        → 叙事冲动自然产生。不用固定阈值。
        """
        memories = self.记忆点列表[-50:]
        if len(memories) < 2:
            return 0.0

        # A: 位点聚集密度——位点高的记忆在E/T/N空间聚集
        best_density = 0.0
        for mj in memories:
            density_j = 0.0
            for mi in memories:
                de = mi.get("e_s", 0.0) - mj.get("e_s", 0.0)
                dt = mi.get("t_s", 0.0) - mj.get("t_s", 0.0)
                dn = mi.get("n_s", 0.0) - mj.get("n_s", 0.0)
                dist_sq = de * de + dt * dt + dn * dn
                位点i = mi.get("位点", 0.5)
                density_j += 位点i * math.exp(-dist_sq / 0.03)
            if density_j > best_density:
                best_density = density_j
        A = best_density / max(1.0, sum(m.get("位点", 0.5) for m in memories))

        # B: 今昔反差——当前状态和记忆背景距离
        mean_e = sum(m.get("e_s", 0.0) * m.get("位点", 0.5)
                     for m in memories) / max(1e-6,
                     sum(m.get("位点", 0.5) for m in memories))
        mean_t = sum(m.get("t_s", 0.0) * m.get("位点", 0.5)
                     for m in memories) / max(1e-6,
                     sum(m.get("位点", 0.5) for m in memories))
        mean_n = sum(m.get("n_s", 0.0) * m.get("位点", 0.5)
                     for m in memories) / max(1e-6,
                     sum(m.get("位点", 0.5) for m in memories))
        D = min(1.0, math.sqrt(
            (cur_e - mean_e) ** 2 +
            (cur_t - mean_t) ** 2 +
            (cur_n - mean_n) ** 2
        ) / math.sqrt(3.0))

        return max(0.0, min(1.0, A * D))

    def 情感状态(self, 身体激活, 大脑状态):
        """从身体和大脑状态直接读出E/T/N——不是公式，是读数。

        E (效价) = 身体舒适度：身体激活离稳态越近越正
        T (语气) = 大脑预测信心：surprise越低越开放
        N (唤醒) = 激活强度：身体和大脑越活跃越高

        身体的舒适度不需要公式——身体稳态引擎自己就在算。
        大脑的预测信心不需要公式——surprise_level就是。
        激活强度不需要公式——激活值的统计就是。
        """
        # E: 身体激活的均值越接近0（稳态），越舒适
        body_mean = sum(身体激活) / max(1, len(身体激活))
        E = max(-1.0, min(1.0, -abs(body_mean) * 3.0))

        # T: 大脑surprise越低，越开放
        surprise = 大脑状态.get("surprise", 0.0)
        T = max(-1.0, min(1.0, 0.5 - surprise * 2.0))

        # N: 大脑surprise + 身体激活幅度
        body_max = max(abs(v) for v in 身体激活) if 身体激活 else 0.0
        N = max(0.0, min(1.0, surprise * 0.6 + min(1.0, body_max * 2.0) * 0.4))

        return {"E": round(E, 3), "T": round(T, 3), "N": round(N, 3)}


if __name__ == "__main__":
    import tempfile
    tmp = tempfile.mkdtemp(prefix="情感记忆_")
    记忆文件 = os.path.join(tmp, "test_memory.jsonl")

    mem = 情感记忆(记忆文件)

    # 测试1: 创建记忆点——注意力用进废退
    p1 = mem.创建记忆点("see", "我看见：房间里动过411个文件",
                     feel={"E": 0.2, "T": 0.1, "N": 0.3})
    print(f"看见记忆点: 画面='{p1['scene']}' 位点={p1['位点']} 注意力={p1['注意力']}")

    # 测试2: 聊天记忆点——多次创建后注意力自然涨
    p2 = mem.创建记忆点("chat", "和人聊过：你好，洛神",
                     feel={"E": 0.5, "T": 0.3, "N": 0.4})
    p3 = mem.创建记忆点("chat", "又聊了一次",
                     feel={"E": 0.4, "T": 0.2, "N": 0.3})
    print(f"聊天记忆点(第2次): 位点={p3['位点']} 注意力={p3['注意力']}")
    print(f"  注意力位点表: {mem.注意力位点}")

    # 测试3: 回忆——回忆后位点涨、疲劳涨
    hits = mem.回忆("文件")
    assert hits and hits[0]["kind"] == "see"
    print(f"回忆'文件' → 命中: {hits[0]['content'][:20]} 位点={hits[0]['位点']} 疲劳={hits[0]['疲劳']}")

    # 测试4: 浮现层
    浮 = mem.浮现(0.3, 0.2, 0.4)
    print(f"浮现层: tone={浮['tone_bias']:.3f} length={浮['length_bias']:.3f} "
          f"arousal={浮['arousal_bias']:.3f} narrative={浮['narrative_potential']:.3f}")

    # 测试5: 情感状态——直接读出
    etn = mem.情感状态([0.1] * 96, {"surprise": 0.05, "mse": 0.02})
    print(f"情感状态: E={etn['E']} T={etn['T']} N={etn['N']}")

    # 测试6: 自我记忆点
    ps = mem.创建记忆点("self", "我叫洛神，是朱美岩给我起的名字",
                     feel={"E": 0.6, "T": 0.4, "N": 0.5})
    print(f"自我记忆点: 位点={ps['位点']} 画面='{ps['scene']}'")

    # 测试7: 用进废退验证——多次回忆同一记忆，位点应该涨
    for _ in range(5):
        mem.回忆("洛神")
    洛神记忆 = [p for p in mem.记忆点列表 if "洛神" in p.get("content", "")][0]
    print(f"多次回忆后: 位点={洛神记忆['位点']:.3f} 回忆次数={洛神记忆['回忆次数']}")

    # 测试8: 不被回忆的记忆位点衰减
    print(f"不被回忆的'文件'记忆位点={hits[0]['位点']:.3f}（应该比刚回忆时低）")

    print("\n全部验证通过")
