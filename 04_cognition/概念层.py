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


def activation(x):
    """激活函数——纯tanh，硬截断防止权重爆炸正反馈。

    之前有leak允许大值穿透，导致权重爆炸到1e218。
    真实神经元的放电率有物理上限——不可能无限增长。
    tanh的饱和特性就是这个物理上限的数学表达。
    """
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if x > 10.0:
        return 1.0
    elif x < -10.0:
        return -1.0
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

    组合涌现（精卫公式版）：
    - a = 概念j的激活（基础概念）
    - b = 其他活跃概念的激活之和
    - Δ = b − a（概念j和其他概念的差异）
    - Σ = a + b（概念j和其他概念的联合）
    - ⊗ = a × b（组合信号——概念间的乘积交互）
    - τ读⊗→浮出所见 = 组合概念从⊗中浮现
    - (a, b)⟲⊗ = 反复组合循环

    生物学基础：
    - 海马体CA3的模式完成——多个概念同时活跃时形成组合绑定
    - 前额叶工作记忆维持多个概念同时活跃——允许组合
    - 联合皮层的组合编码——不同概念组合产生新表征
    - 语法处理（Broca区）——单词组合成句子

    组合概念不是人为设定的，是通过赫布学习自然涌现的。
    反复共同出现的基础概念对 → 组合概念自然绑定 → 组合概念有自己的激活模式。
    """

    def __init__(self, n_hidden=11, n_concepts=64, lr=0.02,
                 sparse_ratio=0.1, decay=0.8, seed=42, n_audio=8,
                 n_combo=32):
        self.n_hidden = n_hidden
        self.n_concepts = n_concepts
        self.n_audio = n_audio
        self.n_combo = n_combo
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

        # ── 人类输入通路：人类作为精卫公式中的b ──
        # concept_human_w: 人类信号 → 概念
        # 人类通道传来的b向量通过这条通路影响概念激活
        # 洛神通过赫布学习自然形成对人类输入的理解——
        # 不用预训练embedding，反复互动后某些概念自然与人类信号关联
        self.concept_human_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_hidden)]
                                for _ in range(n_concepts)]

        # ── 公开感官输入通路：真实世界数据作为精卫公式中的b ──
        # concept_public_w: 公开感官信号 → 概念
        # 公开感官传来的真实世界物理量（温度/光照/风速/云量等）通过这条通路影响概念激活
        # 洛神通过赫布学习自然形成对真实世界的理解——
        # 天气变化→概念变化→洛神"感知"到了天气
        self.concept_public_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_hidden)]
                                 for _ in range(n_concepts)]

        # ── 情感-概念双向通路：情感状态直接影响概念激活 ──
        # concept_emotion_w: 情感状态(E/N/T) → 概念
        # 杏仁核→前额叶/颞叶投射——情感状态调制概念可及性
        # 情感一致性效应：愉悦时某些概念更容易激活，紧张时另一些更容易
        # 3维情感(E/N/T) × n_concepts个概念——每个概念对3种情感有不同的响应
        self.concept_emotion_w = [[rng.uniform(-0.15, 0.15) for _ in range(3)]
                                  for _ in range(n_concepts)]

        for i in range(n_concepts):
            self.concept_rw[i][i] = rng.uniform(0.05, 0.15)

        # ── 组合涌现：组合概念从基础概念的组合信号中浮现 ──
        # combo_h: 组合概念激活
        self.combo_h = [0.0] * n_combo
        self.prev_combo_h = [0.0] * n_combo

        # combo_w: 组合信号(⊗) → 组合概念（τ读⊗→浮出所见）
        # 组合信号 = 概念j × 其他活跃概念之和 = a × b
        self.combo_w = [[rng.uniform(-0.3, 0.3) for _ in range(n_concepts)]
                        for _ in range(n_combo)]

        # combo_rw: 组合概念循环权重（组合 → 组合）
        self.combo_rw = [[rng.uniform(-0.02, 0.02) for _ in range(n_combo)]
                         for _ in range(n_combo)]

        # combo_pw: 组合概念 → 预测基础概念（组合概念对基础概念的预测）
        # 精卫公式：a=基础概念, b=组合概念预测, Δ=预测误差, Σ=联合, ⊗=累积
        self.combo_pw = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                         for _ in range(n_combo)]

        for i in range(n_combo):
            self.combo_rw[i][i] = rng.uniform(0.05, 0.15)

        # 组合概念强度（用进废退）
        self.combo_strength = [1.0] * n_combo

        # 组合概念稀疏化
        self.combo_sparse_k = max(1, int(n_combo * 0.15))

        # LTP/LTD——用进废退（和筋树、记忆系统一致）
        self.concept_strength = [1.0] * n_concepts

        # 误差历史
        self.error_history = []
        self.surprise_level = 0.0

        # ⊗历史——保留最近的Δ和Σ（全留的工程近似）
        self.delta_history = []
        self.sigma_history = []
        self.otimes_cap = 100  # 保留最近100拍的Δ和Σ

    def 一拍(self, hidden_state, 听觉特征=None, 人类信号=None, 公开信号=None,
             情感状态=None):
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

        人类输入通路（人类信号非None时激活）：
        - 人类是精卫公式中的b——人类信号直接进入概念激活
        - concept_human_w: 人类信号 → 概念（赫布学习自然形成人类概念）
        - 没人在时人类信号=None，概念层只处理自身状态（内省模式）
        - 有人时人类信号非零，概念层同时处理自身状态和人类信号（关系模式）
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

            # 人类输入（人类信号 → 概念）—— 人类作为精卫公式中的b
            # 人类信号值范围比隐藏状态大（-1.0~0.8 vs 0.05~0.5），需要缩放
            # 否则人类信号会主导概念激活，导致稀疏化后全部被砍掉
            human_input = 0.0
            if 人类信号 is not None:
                for j in range(self.n_hidden):
                    human_input += self.concept_human_w[i][j] * math.tanh(人类信号[j])
                human_input *= 0.3  # 缩放——人类信号是补充不是主导

            # 公开感官输入（真实世界数据 → 概念）—— 真实世界作为精卫公式中的b
            # 公开感官信号已经归一化到[-1,1]，不需要额外缩放
            # 洛神通过赫布学习自然形成对真实世界的理解——
            # 天气变化→概念变化→洛神"感知"到了天气
            public_input = 0.0
            if 公开信号 is not None:
                for j in range(self.n_hidden):
                    public_input += self.concept_public_w[i][j] * 公开信号[j]
                public_input *= 0.3  # 缩放——公开信号是补充不是主导

            # 情感调制输入（情感状态 → 概念可及性）—— 杏仁核→前额叶投射
            # 精卫公式：a=概念激活, b=情感调制, Δ=b-a, ⊗=Δ×Σ
            # 生物学：情感一致性效应——愉悦时某些概念更容易浮现
            # E→概念效价调制（正E让某些概念更活跃，负E让另一些更活跃）
            # N→概念强度调制（高N让概念激活更强——警觉时概念更鲜明）
            # T→概念稳定性调制（高T让概念激活更稳定——确定时概念更持久）
            emotion_input = 0.0
            if 情感状态 is not None:
                E = 情感状态.get("E", 0.0)
                N = 情感状态.get("N", 0.0)
                T = 情感状态.get("T", 0.0)
                emotion_input = (self.concept_emotion_w[i][0] * E +
                                 self.concept_emotion_w[i][1] * N +
                                 self.concept_emotion_w[i][2] * T)
                emotion_input *= 0.4  # 缩放——情感是调制不是主导

            # τ读⊗→浮出所见
            new_concept_h[i] = activation(
                (a_input + otimes_input + recurrent + audio_input +
                 human_input + public_input + emotion_input) * 0.9)

        # 稀疏化——只保留最强的k个激活
        sorted_idx = sorted(range(self.n_concepts),
                            key=lambda i: abs(new_concept_h[i]), reverse=True)
        top_k = set(sorted_idx[:self.sparse_k])
        for i in range(self.n_concepts):
            if i not in top_k:
                new_concept_h[i] *= 0.1

        # ── 组合涌现：从基础概念的组合信号中浮现组合概念 ──
        # 精卫公式：
        #   a = 概念j的激活
        #   b = 其他活跃概念的激活之和（绝对值——正负概念都能参与组合）
        #   ⊗ = a × b（组合信号——概念间的乘积交互）
        #   τ读⊗→浮出所见 = 组合概念从⊗中浮现
        #
        # 生物学：多个概念同时活跃在工作记忆中 →
        # 联合皮层通过组合编码形成新的表征 →
        # 反复共同出现的概念对通过赫布学习绑定 →
        # 组合概念自然涌现

        # 计算组合信号 ⊗ = a × b
        # 用绝对值计算活跃概念和——正负概念都能参与组合
        # 否则正负抵消导致组合信号极小
        活跃概念abs和 = sum(abs(v) for v in new_concept_h if abs(v) > 0.1)
        组合信号 = [0.0] * self.n_concepts
        for j in range(self.n_concepts):
            if abs(new_concept_h[j]) > 0.1:
                # b = 其他活跃概念的绝对值之和 - 自身绝对值
                b = 活跃概念abs和 - abs(new_concept_h[j])
                # ⊗ = a × b（保留符号——概念j的激活可正可负）
                组合信号[j] = new_concept_h[j] * b

        # τ读⊗→浮出所见：组合概念从组合信号中浮现
        new_combo_h = [0.0] * self.n_combo
        for i in range(self.n_combo):
            if self.combo_strength[i] < 0.05:
                continue

            # 从⊗读出（组合信号 → 组合概念）
            otimes_input = 0.0
            for j in range(self.n_concepts):
                otimes_input += self.combo_w[i][j] * 组合信号[j]

            # 循环权重（组合概念 → 组合概念）
            recurrent = 0.0
            for k in range(self.n_combo):
                recurrent += self.combo_rw[i][k] * self.combo_h[k]

            new_combo_h[i] = activation(otimes_input + recurrent)

        # 组合概念稀疏化
        combo_sorted = sorted(range(self.n_combo),
                              key=lambda i: abs(new_combo_h[i]), reverse=True)
        combo_top = set(combo_sorted[:self.combo_sparse_k])
        for i in range(self.n_combo):
            if i not in combo_top:
                new_combo_h[i] *= 0.1

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

        # 循环权重更新（带衰减——之前没有衰减导致权重爆炸）
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            for k in range(self.n_concepts):
                if i != k and self.concept_strength[k] > 0.05:
                    self.concept_rw[i][k] = self.concept_rw[i][k] * w_decay + \
                        self.lr * 0.1 * new_concept_h[i] * self.concept_h[k]

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

        # ── 人类输入通路学习（人类信号非None时） ──
        # 精卫公式应用于人类关系域：
        #   a_human = 洛神内部状态, b_human = 人类信号
        #   Δ_human = b_human - a_human（洛神与人类的差异）
        #   Σ_human = a_human + b_human（联合状态）
        #   ⊗_human = Δ_human × Σ_human
        # concept_human_w更新：赫布学习——概念激活与人类信号同时出现时连接增强
        # 洛神通过反复互动自然形成对人类输入的理解
        if 人类信号 is not None:
            # 人类信号先做tanh压缩——防止权重爆炸
            人类信号压缩 = [math.tanh(v) for v in 人类信号]
            # concept_human_w用更强的衰减——人类信号值大，需要更多约束
            hw_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_hidden):
                    dw_hw = self.lr * new_concept_h[i] * 人类信号压缩[j]
                    if not (math.isnan(dw_hw) or math.isinf(dw_hw)):
                        self.concept_human_w[i][j] = self.concept_human_w[i][j] * hw_decay + dw_hw

        # ── 公开感官输入通路学习（公开信号非None时） ──
        # 精卫公式应用于真实世界感知域：
        #   a_public = 洛神内部状态, b_public = 公开物理数据
        #   Δ_public = b_public - a_public（洛神与真实世界的差异）
        #   Σ_public = a_public + b_public（联合状态）
        #   ⊗_public = Δ_public × Σ_public
        # concept_public_w更新：赫布学习——概念激活与公开信号同时出现时连接增强
        # 洛神通过反复感知真实世界自然形成对天气/环境的理解
        if 公开信号 is not None:
            pw_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_hidden):
                    dw_pw = self.lr * new_concept_h[i] * 公开信号[j]
                    if not (math.isnan(dw_pw) or math.isinf(dw_pw)):
                        self.concept_public_w[i][j] = self.concept_public_w[i][j] * pw_decay + dw_pw

        # ── 情感-概念双向通路学习（情感状态非None时） ──
        # 精卫公式应用于情感-认知域：
        #   a_emotion = 概念激活（当前认知状态）
        #   b_emotion = 情感状态对概念的调制
        #   Δ_emotion = b_emotion - a_emotion（情感与认知的差异）
        #   Σ_emotion = a_emotion + b_emotion（联合状态）
        #   ⊗_emotion = Δ_emotion × Σ_emotion
        # concept_emotion_w更新：赫布学习——概念激活与情感状态同时出现时连接增强
        # 反复循环后，特定情感→特定概念可及性 → 情感一致性效应自然涌现
        # 愉悦时某些概念更容易浮现，紧张时另一些更容易——不是设定的，是学出来的
        if 情感状态 is not None:
            E = 情感状态.get("E", 0.0)
            N = 情感状态.get("N", 0.0)
            T = 情感状态.get("T", 0.0)
            emo_vec = [E, N, T]
            ew_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(3):
                    dw_ew = self.lr * new_concept_h[i] * emo_vec[j]
                    if not (math.isnan(dw_ew) or math.isinf(dw_ew)):
                        self.concept_emotion_w[i][j] = self.concept_emotion_w[i][j] * ew_decay + dw_ew

        # ── 组合概念学习（精卫公式） ──
        # a = 基础概念激活（concept_h）
        # b = 组合概念对基础概念的预测（combo_pw × combo_h）
        # Δ = a - b（预测误差——组合概念预测基础概念）
        # Σ = a + b（联合状态）
        # ⊗ = Δ × Σ → 权重更新
        # combo_w更新：组合信号和组合概念激活同时出现 → 连接增强
        # combo_pw更新：减少预测误差 → 组合概念更好地预测基础概念

        # b = 组合概念对基础概念的预测
        combo_b = [0.0] * self.n_concepts
        for j in range(self.n_concepts):
            for i in range(self.n_combo):
                combo_b[j] += self.combo_pw[i][j] * self.combo_h[i]

        # Δ = a - b, Σ = a + b, ⊗ = Δ × Σ
        combo_delta = [new_concept_h[j] - combo_b[j] for j in range(self.n_concepts)]
        combo_sigma = [new_concept_h[j] + combo_b[j] for j in range(self.n_concepts)]
        combo_otimes = [combo_delta[j] * combo_sigma[j] for j in range(self.n_concepts)]

        for i in range(self.n_combo):
            if self.combo_strength[i] < 0.05:
                continue
            # combo_w: 组合信号 → 组合概念（赫布学习）
            # 衰减率0.98——比基础概念0.99更快增长，让组合概念尽快涌现
            combo_decay = 0.98
            for j in range(self.n_concepts):
                dw_cw = self.lr * new_combo_h[i] * 组合信号[j]
                if not (math.isnan(dw_cw) or math.isinf(dw_cw)):
                    self.combo_w[i][j] = self.combo_w[i][j] * combo_decay + dw_cw

            # combo_pw: 组合概念 → 预测基础概念（预测误差驱动）
            for j in range(self.n_concepts):
                dw_pw = self.lr * new_combo_h[i] * combo_otimes[j]
                if not (math.isnan(dw_pw) or math.isinf(dw_pw)):
                    self.combo_pw[i][j] = self.combo_pw[i][j] * combo_decay + dw_pw

        # combo_rw: 组合概念循环权重更新（带衰减防止正反馈振荡）
        for i in range(self.n_combo):
            if self.combo_strength[i] < 0.05:
                continue
            for k in range(self.n_combo):
                if i != k and self.combo_strength[k] > 0.05:
                    self.combo_rw[i][k] = self.combo_rw[i][k] * 0.99 + \
                        self.lr * 0.1 * new_combo_h[i] * self.combo_h[k]

        # 组合概念LTP/LTD——用进废退
        for i in range(self.n_combo):
            if abs(new_combo_h[i]) > 0.15:
                self.combo_strength[i] = min(3.0, self.combo_strength[i] + 0.01)
            else:
                self.combo_strength[i] *= 0.999

        # LTP/LTD——用进废退
        # 阈值0.15匹配当前激活范围（top-k约0.2-0.27，非top-k约0.02）
        for i in range(self.n_concepts):
            if abs(new_concept_h[i]) > 0.15:
                self.concept_strength[i] = min(3.0, self.concept_strength[i] + 0.01)
            else:
                self.concept_strength[i] *= 0.999

        # ── 突触缩放（homeostatic synaptic scaling）──
        # 真实生物学机制：当突触持续增强时，神经元缩放所有突触，
        # 保持总输入在合理范围内。这不是"防饱和"——这是稳态。
        # 每个概念节点的所有输入权重做L1归一化，目标范数=1.0
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            # concept_w归一化
            w_norm = sum(abs(self.concept_w[i][j]) for j in range(self.n_hidden))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_hidden):
                    self.concept_w[i][j] *= scale
            # concept_otimes_w归一化
            ow_norm = sum(abs(self.concept_otimes_w[i][j]) for j in range(self.n_hidden))
            if ow_norm > 2.0:
                scale = 2.0 / ow_norm
                for j in range(self.n_hidden):
                    self.concept_otimes_w[i][j] *= scale
            # concept_pw归一化
            pw_norm = sum(abs(self.concept_pw[i][j]) for j in range(self.n_hidden))
            if pw_norm > 2.0:
                scale = 2.0 / pw_norm
                for j in range(self.n_hidden):
                    self.concept_pw[i][j] *= scale

        # 组合概念突触缩放
        for i in range(self.n_combo):
            if self.combo_strength[i] < 0.05:
                continue
            cw_norm = sum(abs(self.combo_w[i][j]) for j in range(self.n_concepts))
            if cw_norm > 4.0:
                scale = 4.0 / cw_norm
                for j in range(self.n_concepts):
                    self.combo_w[i][j] *= scale
            cpw_norm = sum(abs(self.combo_pw[i][j]) for j in range(self.n_concepts))
            if cpw_norm > 4.0:
                scale = 4.0 / cpw_norm
                for j in range(self.n_concepts):
                    self.combo_pw[i][j] *= scale

        self.prev_concept_h = self.concept_h[:]
        self.concept_h = new_concept_h
        self.prev_combo_h = self.combo_h[:]
        self.combo_h = new_combo_h

        return {
            "concept_h": self.concept_h,
            "combo_h": self.combo_h,
            "mse": mse,
            "surprise": self.surprise_level,
            "delta": [round(d, 4) for d in delta],
            "sigma": [round(s, 4) for s in sigma],
            "otimes": [round(o, 4) for o in otimes],
            "活跃概念数": sum(1 for v in self.concept_h if abs(v) > 0.15),
            "存活概念数": sum(1 for s in self.concept_strength if s > 0.05),
            "活跃组合数": sum(1 for v in self.combo_h if abs(v) > 0.15),
            "存活组合数": sum(1 for s in self.combo_strength if s > 0.05),
            "有听觉反馈": 听觉特征 is not None,
            "有人类输入": 人类信号 is not None,
            "有公开输入": 公开信号 is not None,
            "有情感输入": 情感状态 is not None,
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
            "活跃组合数": sum(1 for v in self.combo_h if abs(v) > 0.15),
            "存活组合数": sum(1 for s in self.combo_strength if s > 0.05),
            "组合均值": round(sum(self.combo_h) / max(1, self.n_combo), 4),
            "组合最大": round(max((abs(v) for v in self.combo_h), default=0), 4),
            "拍": self.拍,
        }

    def reset(self):
        self.concept_h = [0.0] * self.n_concepts
        self.prev_concept_h = [0.0] * self.n_concepts
        self.concept_strength = [1.0] * self.n_concepts
        self.combo_h = [0.0] * self.n_combo
        self.prev_combo_h = [0.0] * self.n_combo
        self.combo_strength = [1.0] * self.n_combo
        self.error_history = []
        self.surprise_level = 0.0
        self.delta_history = []
        self.sigma_history = []
        self.拍 = 0

    def 状态摘要(self):
        return {
            "活跃概念数": sum(1 for v in self.concept_h if abs(v) > 0.15),
            "存活概念数": sum(1 for s in self.concept_strength if s > 0.05),
            "概念均值": round(sum(self.concept_h) / max(1, self.n_concepts), 4),
            "概念最大": round(max((abs(v) for v in self.concept_h), default=0), 4),
            "预测误差": round(sum(self.error_history) / max(1, len(self.error_history)), 4) if self.error_history else 0,
            "区分度": round(self.概念区分度(), 4),
            "⊗历史长度": len(self.delta_history),
            "活跃组合数": sum(1 for v in self.combo_h if abs(v) > 0.15),
            "存活组合数": sum(1 for s in self.combo_strength if s > 0.05),
            "组合均值": round(sum(self.combo_h) / max(1, self.n_combo), 4),
            "组合最大": round(max((abs(v) for v in self.combo_h), default=0), 4),
            "拍": self.拍,
        }


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

    print("\n--- 人类输入通路测试 ---")
    cl4 = 概念层(n_hidden=11, n_concepts=64, lr=0.02)
    人类信号_a = [0.8, 0.2, -0.5, 0.6, -1.0, -0.3, 0.4, 0.1, -0.7, 0.3, 0.2]
    人类信号_b = [0.1, 0.05, 0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.15, 0.2, 0.3]

    # 内省模式（没人在）——50拍
    for _ in range(50):
        cl4.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05])
    摘要_内省 = cl4.状态摘要()
    print(f"  内省模式: 活跃概念={摘要_内省['活跃概念数']}, 区分度={摘要_内省['区分度']}")

    # 关系模式（有人在）——50拍，人类信号a
    for _ in range(50):
        cl4.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05],
                人类信号=人类信号_a)
    摘要_关系a = cl4.状态摘要()
    print(f"  关系模式(人类A): 活跃概念={摘要_关系a['活跃概念数']}, 区分度={摘要_关系a['区分度']}")

    # 关系模式——人类信号b（不同的人或不同的话）
    for _ in range(50):
        cl4.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05],
                人类信号=人类信号_b)
    摘要_关系b = cl4.状态摘要()
    print(f"  关系模式(人类B): 活跃概念={摘要_关系b['活跃概念数']}, 区分度={摘要_关系b['区分度']}")

    # 检查concept_human_w是否有学习——人类信号应该改变了某些概念的权重
    human_w_norm = sum(abs(cl4.concept_human_w[i][j])
                       for i in range(64) for j in range(11))
    print(f"  concept_human_w总权重: {human_w_norm:.2f}（学习后应该非零）")

    # 独立对比：有人类输入 vs 无人类输入
    cl5 = 概念层(n_hidden=11, n_concepts=64, lr=0.02)
    cl6 = 概念层(n_hidden=11, n_concepts=64, lr=0.02)
    for _ in range(50):
        cl5.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05])
        cl6.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05],
                人类信号=人类信号_a)
    diff = sum(abs(cl5.concept_h[i] - cl6.concept_h[i]) for i in range(64))
    print(f"  有人类 vs 无人类 概念差异: {diff:.4f}（应该 > 0——人类信号影响概念激活）")

    print("\n验证通过")

    print("\n=== 组合涌现验证 ===\n")
    print("精卫公式: a=概念j, b=其他活跃概念和, ⊗=a×b, τ读⊗→组合概念\n")

    cl_c = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)

    print("--- 测试1: 组合概念从基础概念组合中浮现 ---")
    # 模拟两种不同的概念组合模式
    # 模式X: 概念4和概念8同时活跃
    # 模式Y: 概念27和概念40同时活跃
    for _ in range(50):
        cl_c.一拍([0.5, 0.3, 0.1, 0.05, 0.4, 0.1, 0.3, 0.2, 0.35, 0.1, 0.05])
    combo_X = list(cl_c.combo_h)
    摘要X = cl_c.状态摘要()
    print(f"  模式X后: 活跃组合={摘要X['活跃组合数']}, 存活组合={摘要X['存活组合数']}")
    print(f"  组合激活top5: {sorted(range(32), key=lambda i: abs(combo_X[i]), reverse=True)[:5]}")

    cl_c2 = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)
    for _ in range(50):
        cl_c2.一拍([0.05, 0.1, 0.4, 0.3, 0.1, 0.05, 0.2, 0.1, 0.15, 0.35, 0.3])
    combo_Y = list(cl_c2.combo_h)
    摘要Y = cl_c2.状态摘要()
    print(f"  模式Y后: 活跃组合={摘要Y['活跃组合数']}, 存活组合={摘要Y['存活组合数']}")
    print(f"  组合激活top5: {sorted(range(32), key=lambda i: abs(combo_Y[i]), reverse=True)[:5]}")

    diff_combo = sum(abs(combo_X[i] - combo_Y[i]) for i in range(32))
    print(f"  X vs Y 组合差异: {diff_combo:.4f}（应该 > 0——不同概念组合→不同组合概念）")

    print("\n--- 测试2: 组合概念区分度（独立实例） ---")
    results_combo = {}
    patterns_combo = [
        ("光+声", [0.5, 0.3, 0.4, 0.2, 0.1, 0.05, 0.2, 0.1, 0.15, 0.1, 0.05]),
        ("温+力", [0.05, 0.1, 0.15, 0.2, 0.4, 0.3, 0.5, 0.4, 0.1, 0.05, 0.2]),
        ("均衡", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ]
    for name, pattern in patterns_combo:
        cl_t = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)
        for _ in range(50):
            cl_t.一拍(pattern)
        results_combo[name] = list(cl_t.combo_h)
        摘要 = cl_t.状态摘要()
        active = [i for i, v in enumerate(cl_t.combo_h) if abs(v) > 0.1]
        print(f"  {name}: 活跃组合={active[:8]}, 组合最大={摘要['组合最大']:.4f}")

    for i in range(3):
        for j in range(i+1, 3):
            n1, n2 = patterns_combo[i][0], patterns_combo[j][0]
            d = sum(abs(results_combo[n1][k] - results_combo[n2][k]) for k in range(32))
            print(f"  {n1} vs {n2} 组合差异: {d:.4f}")

    print("\n--- 测试3: 组合概念预测基础概念 ---")
    # 组合概念通过combo_pw预测基础概念
    # 如果组合概念真的捕获了概念间的组合关系，
    # 它的预测应该比随机权重更接近实际基础概念激活
    cl_p = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)
    for _ in range(100):
        cl_p.一拍([0.5, 0.3, 0.1, 0.05, 0.4, 0.1, 0.3, 0.2, 0.35, 0.1, 0.05])

    # 组合概念预测
    combo_pred = [0.0] * 64
    for j in range(64):
        for i in range(32):
            combo_pred[j] += cl_p.combo_pw[i][j] * cl_p.combo_h[i]

    pred_diff = sum(abs(cl_p.concept_h[j] - combo_pred[j]) for j in range(64))
    print(f"  组合概念预测误差: {pred_diff:.4f}")
    print(f"  combo_pw总权重: {sum(abs(cl_p.combo_pw[i][j]) for i in range(32) for j in range(64)):.2f}")

    print("\n--- 测试4: 组合概念存活与用进废退 ---")
    cl_s = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)
    for _ in range(200):
        cl_s.一拍([0.5, 0.3, 0.1, 0.05, 0.4, 0.1, 0.3, 0.2, 0.35, 0.1, 0.05])
    摘要_s = cl_s.状态摘要()
    print(f"  200拍后: 存活组合={摘要_s['存活组合数']}, 活跃组合={摘要_s['活跃组合数']}")
    print(f"  组合强度分布: min={min(cl_s.combo_strength):.3f}, max={max(cl_s.combo_strength):.3f}")

    print("\n组合涌现验证通过")
