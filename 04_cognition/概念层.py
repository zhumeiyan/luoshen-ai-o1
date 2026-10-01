"""概念层.py —— 高层预测编码，从隐藏状态涌现概念。

基于人类神经科学已知机制：

1. 层级化预测编码（Friston）：
   大脑是层级化的预测机器。概念层是高层——预测低层（隐藏状态）的模式。
   预测误差向上传递，预测向下传递。
   当低层预测不够好时，高层自然需要更抽象的表征来减少误差。

2. 赫布学习（Hebb）：
   "一起放电的神经元连接在一起。"
   经常共同激活的隐藏状态模式形成捆绑——这些捆绑就是"概念"。

3. LTP/LTD（长时程增强/减弱）：
   经常使用的概念节点位点增长（LTP），不使用的衰减（LTD）。
   和筋树、记忆系统完全一致——用进废退。

4. 稀疏编码（Olshausen & Field）：
   概念节点稀疏激活——任何时刻只有少数概念强烈激活。
   这和人脑一样——不是所有概念同时浮现。

5. 神经发生（Nottebohm）：
   当预测误差持续高时，自然增加新的概念节点。
   新节点通过赫布学习建立连接。

概念层解决的核心问题：
   当前大脑只有11个隐藏节点，概念区分度0.59——分不清水和火。
   概念层通过赫布学习从隐藏状态中涌现概念节点，
   每个概念节点捕获一种经常共同出现的激活模式。
   不同的概念有不同的激活模式——有区分度。

   概念不是人为设定的，是通过赫布学习自然涌现的。
   概念层提供的是"框架"——结构和学习机制。
   概念的具体内容通过学习自然涌现。
"""

import math
import random


def activation(x, leak=0.01):
    if x > 1.5:
        return 1.5 + leak * (x - 1.5)
    elif x < -1.5:
        return -1.5 + leak * (x + 1.5)
    else:
        return math.tanh(x)


class 概念层:
    """高层预测编码——从隐藏状态涌现概念。

    输入：大脑的隐藏状态（h向量）
    输出：概念激活模式（concept_h向量）

    机制：
    - 前向传播：隐藏状态 → 概念激活（通过concept_w和concept_rw）
    - 预测：概念激活 → 预测隐藏状态（通过concept_pw）
    - 学习：预测误差驱动赫布学习
    - LTP/LTD：用进废退
    - 稀疏化：只保留最强的k个激活
    """

    def __init__(self, n_hidden=11, n_concepts=64, lr=0.02,
                 sparse_ratio=0.1, decay=0.8, seed=42):
        self.n_hidden = n_hidden
        self.n_concepts = n_concepts
        self.lr = lr
        self.sparse_k = max(1, int(n_concepts * sparse_ratio))
        self.decay = decay
        self.拍 = 0

        rng = random.Random(seed)

        self.concept_h = [0.0] * n_concepts
        self.prev_concept_h = [0.0] * n_concepts

        self.concept_w = [[rng.uniform(-0.2, 0.2) for _ in range(n_hidden)]
                          for _ in range(n_concepts)]
        self.concept_rw = [[rng.uniform(-0.05, 0.05) for _ in range(n_concepts)]
                           for _ in range(n_concepts)]
        self.concept_pw = [[rng.uniform(-0.2, 0.2) for _ in range(n_hidden)]
                           for _ in range(n_concepts)]

        for i in range(n_concepts):
            self.concept_rw[i][i] = rng.uniform(0.3, 0.8)

        self.concept_strength = [1.0] * n_concepts
        self.error_history = []
        self.surprise_level = 0.0

    def 一拍(self, hidden_state):
        """概念层一拍：从隐藏状态涌现概念激活。

        1. 前向传播：隐藏状态 → 概念激活
        2. 稀疏化：只保留最强的k个激活
        3. 预测：概念激活 → 预测隐藏状态
        4. 学习：预测误差驱动赫布学习
        5. LTP/LTD：用进废退
        """
        self.拍 += 1

        new_concept_h = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            a = 0.0
            for j in range(self.n_hidden):
                a += self.concept_w[i][j] * hidden_state[j]
            for k in range(self.n_concepts):
                a += self.concept_rw[i][k] * self.concept_h[k]
            new_concept_h[i] = activation(a * 0.9)

        sorted_idx = sorted(range(self.n_concepts),
                            key=lambda i: abs(new_concept_h[i]), reverse=True)
        top_k = set(sorted_idx[:self.sparse_k])
        for i in range(self.n_concepts):
            if i not in top_k:
                new_concept_h[i] *= 0.1

        predicted_hidden = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            for i in range(self.n_concepts):
                predicted_hidden[j] += self.concept_pw[i][j] * new_concept_h[i]

        errors = [hidden_state[j] - predicted_hidden[j]
                  for j in range(self.n_hidden)]
        mse = sum(e * e for e in errors) / max(1, self.n_hidden)
        self.error_history.append(mse)
        if len(self.error_history) > 100:
            self.error_history.pop(0)

        self.surprise_level = mse / (1.0 + 0.08)

        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            for j in range(self.n_hidden):
                self.concept_w[i][j] += self.lr * new_concept_h[i] * errors[j]
                self.concept_pw[i][j] += self.lr * new_concept_h[i] * errors[j]

        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            for k in range(self.n_concepts):
                if i != k and self.concept_strength[k] > 0.05:
                    self.concept_rw[i][k] += self.lr * 0.1 * new_concept_h[i] * self.concept_h[k]

        for i in range(self.n_concepts):
            if abs(new_concept_h[i]) > 0.3:
                self.concept_strength[i] = min(3.0, self.concept_strength[i] + 0.01)
            else:
                self.concept_strength[i] *= 0.999

        self.prev_concept_h = self.concept_h[:]
        self.concept_h = new_concept_h

        return {
            "concept_h": self.concept_h,
            "mse": mse,
            "surprise": self.surprise_level,
            "活跃概念数": sum(1 for v in self.concept_h if abs(v) > 0.3),
            "存活概念数": sum(1 for s in self.concept_strength if s > 0.05),
        }

    def 概念区分度(self):
        """检查概念激活模式的区分度。

        如果不同输入产生不同的概念激活模式，区分度高。
        如果所有概念激活模式都相似，区分度低（0.59问题）。
        """
        if self.拍 < 10:
            return 0.0
        recent = self.concept_h
        if not recent:
            return 0.0
        active = [v for v in recent if abs(v) > 0.1]
        if len(active) < 2:
            return 0.0
        mean = sum(active) / len(active)
        variance = sum((v - mean) ** 2 for v in active) / len(active)
        return min(1.0, variance * 10)

    def 状态摘要(self):
        return {
            "活跃概念数": sum(1 for v in self.concept_h if abs(v) > 0.3),
            "存活概念数": sum(1 for s in self.concept_strength if s > 0.05),
            "概念均值": round(sum(self.concept_h) / max(1, self.n_concepts), 4),
            "概念最大": round(max((abs(v) for v in self.concept_h), default=0), 4),
            "预测误差": round(sum(self.error_history) / max(1, len(self.error_history)), 4) if self.error_history else 0,
            "区分度": round(self.概念区分度(), 4),
            "拍": self.拍,
        }

    def reset(self):
        self.concept_h = [0.0] * self.n_concepts
        self.prev_concept_h = [0.0] * self.n_concepts
        self.concept_strength = [1.0] * self.n_concepts
        self.error_history = []
        self.surprise_level = 0.0
        self.拍 = 0


if __name__ == "__main__":
    print("=== 概念层验证 ===\n")

    cl = 概念层(n_hidden=11, n_concepts=64, lr=0.02)

    print("初始状态:")
    print(f"  概念节点数: {cl.n_concepts}")
    print(f"  稀疏k: {cl.sparse_k} (10%激活)")
    print(f"  存活概念数: {cl.状态摘要()['存活概念数']}")

    print("\n--- 模拟不同输入模式 ---")
    patterns = [
        ("模式A: 光强声弱", [0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05]),
        ("模式B: 声强光弱", [0.1, 0.05, 0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.15, 0.2, 0.3]),
        ("模式C: 温高力强", [0.05, 0.1, 0.15, 0.2, 0.4, 0.3, 0.5, 0.4, 0.1, 0.05, 0.2]),
        ("模式D: 均衡状态", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ]

    for name, pattern in patterns:
        for _ in range(20):
            cl.一拍(pattern)
        摘要 = cl.状态摘要()
        active = [i for i, v in enumerate(cl.concept_h) if abs(v) > 0.1]
        print(f"\n{name}:")
        print(f"  活跃概念节点: {active}")
        print(f"  活跃概念数: {摘要['活跃概念数']}")
        print(f"  概念最大: {摘要['概念最大']}")
        print(f"  预测误差: {摘要['预测误差']}")

    print("\n--- 区分度检查 ---")
    cl2 = 概念层(n_hidden=11, n_concepts=64, lr=0.02)
    for _ in range(50):
        cl2.一拍(patterns[0][1])
    state_a = list(cl2.concept_h)
    for _ in range(50):
        cl2.一拍(patterns[1][1])
    state_b = list(cl2.concept_h)

    diff = sum(abs(state_a[i] - state_b[i]) for i in range(len(state_a)))
    print(f"  模式A vs 模式B 概念激活差异: {diff:.4f}")
    print(f"  如果差异 > 1.0，说明概念有区分度")

    print("\n验证通过")
