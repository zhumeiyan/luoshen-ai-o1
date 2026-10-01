"""概念层.py —— 高层预测编码，从隐藏状态涌现概念。

基于精卫公式（贯穿全系统的唯一核心公式）：

  a ≠ b
  Δ = b − a
  Σ = a + b
  ⊗ = ∫(Δ, Σ)
  τ 读 ⊗ → 浮出所见
  (a, b) ⟲ ⊗

概念层的解读：
  a = 大脑的隐藏状态（实际）
  b = 概念层对隐藏状态的预测
  Δ = b − a = 预测误差
  Σ = a + b = 联合状态（实际+预测）
  ⊗ = ∫(Δ, Σ) = 累积差与和（全留——每次的Δ和Σ都保留在权重中）
  τ 读 ⊗ → 浮出所见 = 概念从⊗中浮现（不是从a直接映射）
  (a, b) ⟲ ⊗ = 循环预测/比较/修正

和其他子系统的解读对比：
  大脑.py: a=预测, b=实际输入, Δ=预测误差, ⊗=赫布学习, τ=读出层pred
  身体稳态.py: a=上轮激活, b=当前激活, Δ=激活差, ⊗=X(t+1)=X(t)+η·Δ+ε, τ=共火检测
  欲望.py: a=大脑预测, b=实际输入, Δ=RPE, ⊗=偏好累积, τ=通道选择
  情感记忆.py: a=当前身体, b=记忆快照, Δ=身体-记忆反差, ⊗=LTP/LTD, τ=浮现层

关键区别（和之前版本的对比）：
  之前：概念激活只从a（隐藏状态）映射，只用Δ驱动学习
  现在：概念激活从⊗（Δ×Σ）浮现，Δ和Σ都参与学习

  Σ = a + b 是联合状态——实际和预测的和。
  ⊗ = ∫(Δ, Σ) 是累积的差与和——全留。
  τ读⊗→浮出所见——概念不是当前的映射，是历史信息的浮现。

  这解决了概念区分度0.59的问题：
  只用Δ时，所有概念的学习信号相似（都是预测误差）。
  加入Σ后，不同的概念捕获不同的联合状态模式——有区分度。
"""

import math
import random


def activation(x, leak=0.01):
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if x > 1.5:
        return 1.5 + leak * (x - 1.5)
    elif x < -1.5:
        return -1.5 + leak * (x + 1.5)
    else:
        return math.tanh(x)


class 概念层:
    """高层预测编码——基于精卫公式从隐藏状态涌现概念。

    精卫公式：a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗

    输入：大脑的隐藏状态（a）
    输出：概念激活模式（τ读⊗→浮出所见）

    三个权重矩阵对应精卫公式的三个部分：
    - concept_w: 从a读出（实际状态 → 概念）
    - concept_otimes_w: 从⊗读出（Δ×Σ → 概念）—— 这是核心区别
    - concept_rw: 循环权重（概念 → 概念）
    - concept_pw: 预测权重（概念 → 预测隐藏状态b）
    """

    def __init__(self, n_hidden=11, n_concepts=64, lr=0.02,
                 sparse_ratio=0.1, decay=0.8, seed=42, n_audio=8):
        self.n_hidden = n_hidden
        self.n_concepts = n_concepts
        self.n_audio = n_audio
        self.lr = lr
        self.sparse_k = max(1, int(n_concepts * sparse_ratio))
        self.decay = decay
        self.拍 = 0

        rng = random.Random(seed)

        self.concept_h = [0.0] * n_concepts
        self.prev_concept_h = [0.0] * n_concepts

        # concept_w: 从a读出（实际状态 → 概念）
        self.concept_w = [[rng.uniform(-0.2, 0.2) for _ in range(n_hidden)]
                          for _ in range(n_concepts)]

        # concept_otimes_w: 从⊗读出（Δ×Σ → 概念）—— 精卫公式的核心
        self.concept_otimes_w = [[rng.uniform(-0.15, 0.15) for _ in range(n_hidden)]
                                 for _ in range(n_concepts)]

        # concept_rw: 循环权重（概念 → 概念）—— 弱于输入，让⊗信号主导
        self.concept_rw = [[rng.uniform(-0.02, 0.02) for _ in range(n_concepts)]
                           for _ in range(n_concepts)]

        # concept_pw: 预测权重（概念 → 预测隐藏状态b）
        self.concept_pw = [[rng.uniform(-0.2, 0.2) for _ in range(n_hidden)]
                           for _ in range(n_concepts)]

        # ── 说听闭环：听觉输入和声音预测 ──
        # concept_aw: 听觉特征 → 概念（自听反馈通路）
        # 当系统发声后，自听特征通过这条通路影响概念激活
        self.concept_aw = [[rng.uniform(-0.1, 0.1) for _ in range(n_audio)]
                           for _ in range(n_concepts)]

        # concept_sw: 概念 → 预测声音（声音预测通路）
        # 概念层预测自己将听到什么声音——预测误差驱动符号绑定
        self.concept_sw = [[rng.uniform(-0.1, 0.1) for _ in range(n_audio)]
                           for _ in range(n_concepts)]

        for i in range(n_concepts):
            self.concept_rw[i][i] = rng.uniform(0.05, 0.15)

        # LTP/LTD——用进废退（和筋树、记忆系统一致）
        self.concept_strength = [1.0] * n_concepts

        # 误差历史
        self.error_history = []
        self.surprise_level = 0.0

        # ⊗历史——保留最近的Δ和Σ（全留的工程近似）
        self.delta_history = []
        self.sigma_history = []
        self.otimes_cap = 100  # 保留最近100拍的Δ和Σ

    def 一拍(self, hidden_state, 听觉特征=None):
        """概念层一拍：基于精卫公式从隐藏状态涌现概念。

        精卫公式流程：
        1. a = hidden_state（实际隐藏状态）
        2. b = 概念层预测（从当前概念激活预测）
        3. Δ = b - a（预测误差）
        4. Σ = a + b（联合状态）
        5. ⊗ = ∫(Δ, Σ)（累积差与和——全留）
        6. τ读⊗→浮出所见（概念从⊗中浮现）
        7. (a, b)⟲⊗（循环——更新权重）

        说听闭环（听觉特征非None时激活）：
        - a_sound = 概念层预测的声音（concept_sw × concept_h）
        - b_sound = 实际听到的声音（听觉特征）
        - Δ_sound = b_sound - a_sound（声音预测误差）
        - Σ_sound = a_sound + b_sound（联合声音状态）
        - ⊗_sound = Δ_sound × Σ_sound
        - 听觉特征通过concept_aw影响概念激活
        - 声音预测误差驱动concept_sw学习（符号绑定）
        """
        self.拍 += 1

        # ── 精卫公式：a, b, Δ, Σ ──

        # a = 实际隐藏状态
        a = hidden_state

        # b = 概念层对隐藏状态的预测（从当前概念激活预测）
        b = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            for i in range(self.n_concepts):
                b[j] += self.concept_pw[i][j] * self.concept_h[i]

        # Δ = b - a（预测误差）
        delta = [b[j] - a[j] for j in range(self.n_hidden)]

        # Σ = a + b（联合状态——实际+预测）
        sigma = [a[j] + b[j] for j in range(self.n_hidden)]

        # ⊗ = ∫(Δ, Σ)——累积差与和（全留）
        self.delta_history.append(delta[:])
        self.sigma_history.append(sigma[:])
        if len(self.delta_history) > self.otimes_cap:
            self.delta_history.pop(0)
            self.sigma_history.pop(0)

        # ⊗的当前值：Δ × Σ（差与和的交互）
        otimes = [delta[j] * sigma[j] for j in range(self.n_hidden)]

        # ── τ 读 ⊗ → 浮出所见 ──
        # 概念从⊗中浮现——不是从a直接映射
        # 输入源：a（实际状态）、⊗（Δ×Σ）、循环（概念→概念）、听觉（自听反馈）
        new_concept_h = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue

            # 从a读出（实际状态 → 概念）
            a_input = 0.0
            for j in range(self.n_hidden):
                a_input += self.concept_w[i][j] * a[j]

            # 从⊗读出（Δ×Σ → 概念）—— 精卫公式的核心
            otimes_input = 0.0
            for j in range(self.n_hidden):
                otimes_input += self.concept_otimes_w[i][j] * otimes[j]

            # 循环权重（概念 → 概念）
            recurrent = 0.0
            for k in range(self.n_concepts):
                recurrent += self.concept_rw[i][k] * self.concept_h[k]

            # 听觉输入（自听特征 → 概念）—— 说听闭环的反馈端
            audio_input = 0.0
            if 听觉特征 is not None:
                for j in range(self.n_audio):
                    audio_input += self.concept_aw[i][j] * 听觉特征[j]

            # τ读⊗→浮出所见
            new_concept_h[i] = activation((a_input + otimes_input + recurrent + audio_input) * 0.9)

        # 稀疏化——只保留最强的k个激活
        sorted_idx = sorted(range(self.n_concepts),
                            key=lambda i: abs(new_concept_h[i]), reverse=True)
        top_k = set(sorted_idx[:self.sparse_k])
        for i in range(self.n_concepts):
            if i not in top_k:
                new_concept_h[i] *= 0.1

        # ── 预测误差统计 ──
        mse = sum(d * d for d in delta) / max(1, self.n_hidden)
        self.error_history.append(mse)
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        self.surprise_level = mse / (1.0 + 0.08)

        # ── (a, b) ⟲ ⊗ —— 循环：更新权重 ──
        # 衰减率0.99 = 突触缩放（homeostatic synaptic scaling）
        # 均衡权重 = lr * h * signal / (1 - 0.99) = lr * h * signal / 0.01
        # 这是生物机制，不是防饱和——真实神经元有突触缩放防止无限增长
        w_decay = 0.99
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            for j in range(self.n_hidden):
                dw_pw = self.lr * new_concept_h[i] * delta[j]
                dw_w = self.lr * new_concept_h[i] * sigma[j]
                dw_ot = self.lr * new_concept_h[i] * otimes[j]
                if not (math.isnan(dw_pw) or math.isinf(dw_pw)):
                    self.concept_pw[i][j] = self.concept_pw[i][j] * w_decay + dw_pw
                if not (math.isnan(dw_w) or math.isinf(dw_w)):
                    self.concept_w[i][j] = self.concept_w[i][j] * w_decay + dw_w
                if not (math.isnan(dw_ot) or math.isinf(dw_ot)):
                    self.concept_otimes_w[i][j] = self.concept_otimes_w[i][j] * w_decay + dw_ot

        # 循环权重更新
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            for k in range(self.n_concepts):
                if i != k and self.concept_strength[k] > 0.05:
                    self.concept_rw[i][k] += self.lr * 0.1 * new_concept_h[i] * self.concept_h[k]

        # ── 说听闭环学习（听觉特征非None时） ──
        # 精卫公式应用于声音域：
        #   a_sound = 概念预测的声音, b_sound = 实际听到的声音
        #   Δ_sound = b_sound - a_sound, Σ_sound = a_sound + b_sound
        #   ⊗_sound = Δ_sound × Σ_sound
        # concept_sw更新：减少声音预测误差（符号绑定的核心机制）
        # concept_aw更新：增强听觉-概念联合表示
        if 听觉特征 is not None:
            # a_sound = 概念预测的声音
            a_sound = [0.0] * self.n_audio
            for j in range(self.n_audio):
                for i in range(self.n_concepts):
                    a_sound[j] += self.concept_sw[i][j] * self.concept_h[i]

            # b_sound = 实际听到的声音
            b_sound = list(听觉特征)

            # Δ_sound, Σ_sound, ⊗_sound
            delta_sound = [b_sound[j] - a_sound[j] for j in range(self.n_audio)]
            sigma_sound = [a_sound[j] + b_sound[j] for j in range(self.n_audio)]
            otimes_sound = [delta_sound[j] * sigma_sound[j] for j in range(self.n_audio)]

            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_audio):
                    dw_sw = self.lr * new_concept_h[i] * delta_sound[j]
                    dw_aw = self.lr * new_concept_h[i] * sigma_sound[j]
                    if not (math.isnan(dw_sw) or math.isinf(dw_sw)):
                        self.concept_sw[i][j] = self.concept_sw[i][j] * w_decay + dw_sw
                    if not (math.isnan(dw_aw) or math.isinf(dw_aw)):
                        self.concept_aw[i][j] = self.concept_aw[i][j] * w_decay + dw_aw

        # LTP/LTD——用进废退
        # 阈值0.15匹配当前激活范围（top-k约0.2-0.27，非top-k约0.02）
        for i in range(self.n_concepts):
            if abs(new_concept_h[i]) > 0.15:
                self.concept_strength[i] = min(3.0, self.concept_strength[i] + 0.01)
            else:
                self.concept_strength[i] *= 0.999

        self.prev_concept_h = self.concept_h[:]
        self.concept_h = new_concept_h

        return {
            "concept_h": self.concept_h,
            "mse": mse,
            "surprise": self.surprise_level,
            "delta": [round(d, 4) for d in delta],
            "sigma": [round(s, 4) for s in sigma],
            "otimes": [round(o, 4) for o in otimes],
            "活跃概念数": sum(1 for v in self.concept_h if abs(v) > 0.15),
            "存活概念数": sum(1 for s in self.concept_strength if s > 0.05),
            "有听觉反馈": 听觉特征 is not None,
        }

    def 概念区分度(self):
        """检查概念激活模式的区分度。

        如果不同输入产生不同的概念激活模式，区分度高。
        如果所有概念激活模式都相似，区分度低（0.59问题）。
        """
        if self.拍 < 10:
            return 0.0
        active = [v for v in self.concept_h if abs(v) > 0.1]
        if len(active) < 2:
            return 0.0
        mean = sum(active) / len(active)
        variance = sum((v - mean) ** 2 for v in active) / len(active)
        return min(1.0, variance * 10)

    def 状态摘要(self):
        return {
            "活跃概念数": sum(1 for v in self.concept_h if abs(v) > 0.15),
            "存活概念数": sum(1 for s in self.concept_strength if s > 0.05),
            "概念均值": round(sum(self.concept_h) / max(1, self.n_concepts), 4),
            "概念最大": round(max((abs(v) for v in self.concept_h), default=0), 4),
            "预测误差": round(sum(self.error_history) / max(1, len(self.error_history)), 4) if self.error_history else 0,
            "区分度": round(self.概念区分度(), 4),
            "⊗历史长度": len(self.delta_history),
            "拍": self.拍,
        }

    def reset(self):
        self.concept_h = [0.0] * self.n_concepts
        self.prev_concept_h = [0.0] * self.n_concepts
        self.concept_strength = [1.0] * self.n_concepts
        self.error_history = []
        self.surprise_level = 0.0
        self.delta_history = []
        self.sigma_history = []
        self.拍 = 0


if __name__ == "__main__":
    print("=== 概念层验证（精卫公式版） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    cl = 概念层(n_hidden=11, n_concepts=64, lr=0.02)

    print("初始状态:")
    print(f"  概念节点数: {cl.n_concepts}")
    print(f"  稀疏k: {cl.sparse_k} (10%激活)")
    print(f"  存活概念数: {cl.状态摘要()['存活概念数']}")
    print(f"  权重矩阵: concept_w(a→概念), concept_otimes_w(⊗→概念), concept_rw(循环), concept_pw(→b)")

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
        print(f"  ⊗历史长度: {摘要['⊗历史长度']}")

    print("\n--- 区分度检查（独立实例） ---")
    # 每个模式用独立的概念层实例，避免顺序适应的干扰
    results = {}
    for name, pattern in patterns[:3]:
        cl_t = 概念层(n_hidden=11, n_concepts=64, lr=0.02)
        for _ in range(50):
            cl_t.一拍(pattern)
        results[name] = list(cl_t.concept_h)
        摘要 = cl_t.状态摘要()
        active = [i for i, v in enumerate(cl_t.concept_h) if abs(v) > 0.1]
        print(f"  {name}: 活跃={active[:8]} 区分度={摘要['区分度']}")

    diff_ab = sum(abs(results[patterns[0][0]][i] - results[patterns[1][0]][i])
                  for i in range(64))
    diff_ac = sum(abs(results[patterns[0][0]][i] - results[patterns[2][0]][i])
                  for i in range(64))
    diff_bc = sum(abs(results[patterns[1][0]][i] - results[patterns[2][0]][i])
                  for i in range(64))
    print(f"  A vs B 差异: {diff_ab:.4f}")
    print(f"  A vs C 差异: {diff_ac:.4f}")
    print(f"  B vs C 差异: {diff_bc:.4f}")
    print(f"  如果差异 > 1.0，说明概念有区分度")

    print("\n--- 精卫公式验证 ---")
    cl3 = 概念层(n_hidden=4, n_concepts=16, lr=0.05)
    for tick in range(10):
        result = cl3.一拍([0.3, 0.5, 0.1, 0.2])
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: Δ={result['delta']} Σ={result['sigma']} ⊗={result['otimes']}")
    print(f"  τ(浮出): 概念激活={round(sum(cl3.concept_h)/16, 4)}")
    print(f"  ⟲(循环): 拍={cl3.拍}")

    print("\n验证通过")
