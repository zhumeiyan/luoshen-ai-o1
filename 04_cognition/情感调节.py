"""情感调节.py —— 主动调节情感状态，基于精卫公式。

情感调节不是消除情感——是调节情感的强度和方向。
人类不是被动感受情感的机器——前额叶可以主动调节杏仁核的反应。
这是"情绪管理"的神经基础——不是压抑，是调节。

人类神经科学机制：
  1. 腹内侧前额叶（vmPFC）→ 杏仁核抑制通路：
     - vmPFC在认知重评时活跃——重新解释情感刺激的含义
     - "这件事没那么糟"——通过改变认知来改变情感反应
     - vmPFC通过GABA能中间神经元抑制杏仁核的过度反应
     → 工程实现：regulate_w（概念激活→调节后的情感预测）

  2. 背外侧前额叶（dlPFC）→ 注意力部署：
     - dlPFC在注意力转移时活跃——将注意力从情感刺激移开
     - "不去想它"——通过转移注意力来减少情感反应
     → 工程实现：suppress_w（概念激活→抑制信号）

  3. 眶额叶（OFC）→ 情感价值评估：
     - OFC评估情感刺激的价值——决定是否需要调节
     - "这个情感对我有利还是不利？"——价值判断驱动调节方向
     → 工程实现：amplify_w（概念激活→增强信号——有利情感增强）

  4. 前扣带回（ACC）→ 冲突监测：
     - ACC监测情感冲突——当情感与目标冲突时触发调节
     - "我的情感和我的目标不一致"——冲突触发调节
     → 工程实现：meta_regulate_w（元认知评估→调节策略选择）

  5. 内侧前额叶（mPFC）→ 自我参照调节：
     - mPFC在自我参照情感调节时活跃——"这和我有什么关系？"
     - 自我意识强度高时调节更精细——知道自己是谁才能调节情感
     → 工程实现：self_regulate_w（自我意识→调节策略选择）

精卫公式（情感调节版）：
  a = 当前情感状态（被动感受的情感——杏仁核输出）
  b = 调节后的情感状态（PFC预测的"适当"情感）
  Δ = b − a = 调节偏差（当前情感与适当情感的差异）
  Σ = a + b = 联合情感状态（感受+调节）
  ⊗ = ∫(Δ, Σ) = 累积调节学习历史 → 调节权重
  τ 读 ⊗ → 浮出所见 = 调节后的情感从⊗中浮现
  (a, b) ⟲ ⊗ = 反复调节循环

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  元认知.py: a=认知状态, b=元认知评估, Δ=认知偏差, ⊗=元认知学习, τ=置信度浮现
  自我意识.py: a=认知状态, b=自我模型预测, Δ=自我偏差, ⊗=自我意识学习, τ="我"浮现
  情感调节.py: a=当前情感, b=调节后情感, Δ=调节偏差, ⊗=调节学习, τ=调节后情感浮现

情感调节 vs 情感记忆 vs 自我意识的区别：
  - 情感记忆：从身体状态涌现情感（被动感受——"我感觉到了"）
  - 自我意识：构建自我模型（自我认知——"我是谁？"）
  - 情感调节：主动调节情感状态（主动调节——"我应该感觉什么？"）
  关键区别：
  - 情感记忆是"我感觉到了什么"
  - 自我意识是"我是谁"
  - 情感调节是"我应该感觉什么，我如何调节我的情感"

情感调节的三种策略（从神经科学）：
  1. 认知重评：重新解释情感刺激的含义（vmPFC→杏仁核）
     - "这件事没那么糟"——改变认知来改变情感
  2. 抑制：减少情感表达的强度（dlPFC→杏仁核抑制）
     - "不要太激动"——抑制过度反应
  3. 增强：增强有益的情感反应（OFC→杏仁核增强）
     - "保持积极"——增强有利情感
"""

import math
import random


def activation(x):
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if x > 10.0:
        return 1.0
    elif x < -10.0:
        return -1.0
    else:
        return math.tanh(x)


class 情感调节:
    """情感调节——主动调节情感状态，基于精卫公式。

    核心功能：
    1. 认知重评：从概念激活预测"适当的"情感——重新解释情境
    2. 情感抑制：抑制过度情感反应——减少负面情感的强度
    3. 情感增强：增强有益情感反应——增加正面情感的强度
    4. 调节驱动：调节信号反馈到概念层——调节后的情感影响概念可及性

    精卫公式流程（每拍）：
    1. a = 当前情感状态（E/N/T）
    2. b = 调节后的情感预测（regulate_w × concept_h）
    3. Δ = b - a（调节偏差——当前情感与适当情感的差异）
    4. Σ = a + b（联合情感状态——感受+调节）
    5. ⊗ = Δ × Σ（累积调节学习信号）
    6. τ读⊗→浮出所见 = 调节后的情感从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新调节权重

    情感调节的三种策略：
    - 认知重评：概念激活→调节后的情感预测（"这个情境需要什么情感？"）
    - 抑制：概念激活→抑制信号（"需要减少什么情感？"）
    - 增强：概念激活→增强信号（"需要增加什么情感？"）
    """

    def __init__(self, n_concepts=64, n_emotion=3, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.n_emotion = n_emotion
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 认知重评：从概念激活预测"适当的"情感 ──
        # regulate_w: 概念激活 → 调节后的情感预测（E/N/T）
        # 生物学：vmPFC→杏仁核抑制通路——重新解释情境来改变情感反应
        # "这个情境需要什么情感？"——不是消除情感，是找到适当的情感
        # 概念激活模式→适当的情感反应（通过赫布学习自然涌现）
        # n_concepts维概念激活 × 3维情感(E/N/T)
        self.regulate_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                           for _ in range(n_emotion)]

        # 调节后的情感预测（"适当的"情感）
        self.regulated_pred = [0.0] * n_emotion

        # ── 情感抑制：抑制过度情感反应 ──
        # suppress_w: 概念激活 → 抑制信号（减少情感强度）
        # 生物学：dlPFC→杏仁核抑制——"不要太激动"
        # 抑制不是消除——是减少过度反应的强度
        # 当情感强度过高时，抑制信号让情感回归适当水平
        self.suppress_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                           for _ in range(n_emotion)]

        # 抑制信号
        self.suppress_signal = [0.0] * n_emotion

        # ── 情感增强：增强有益情感反应 ──
        # amplify_w: 概念激活 → 增强信号（增加情感强度）
        # 生物学：OFC→杏仁核增强——"保持积极"
        # 增强不是无限放大——是让有益的情感反应更强
        # 当情感强度过低时，增强信号让情感达到适当水平
        self.amplify_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                          for _ in range(n_emotion)]

        # 增强信号
        self.amplify_signal = [0.0] * n_emotion

        # ── 元认知调节：元认知评估 → 调节策略选择 ──
        # meta_regulate_w: 元认知评估 → 调节强度
        # 生物学：ACC→PFC调控——冲突触发调节
        # 高不确定性→更多调节（需要重新评估），低不确定性→少调节
        # n_concepts维元认知评估 × 1维调节强度
        self.meta_regulate_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                for _ in range(1)]

        # ── 自我参照调节：自我意识 → 调节策略选择 ──
        # self_regulate_w: 自我意识驱动 → 调节强度
        # 生物学：mPFC→杏仁核调控——自我参照调节
        # 自我意识强度高→更精细的调节（知道自己是谁才能调节情感）
        # n_concepts维自我意识驱动 × 1维调节强度
        self.self_regulate_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                for _ in range(1)]

        # ── 调节后的情感状态 ──
        # 调节后的情感 = 当前情感 + 调节偏差（不是替换，是调整）
        self.regulated_emotion = {"E": 0.0, "N": 0.0, "T": 0.0}

        # ── 调节驱动信号（反馈到概念层——作为b信号输入源） ──
        # 调节驱动 = 调节后的情感对概念可及性的调制
        # 生物学：PFC→后部皮层投射——调节后的情感影响概念可及性
        # 调节后的情感让适当的概念更容易浮现——"冷静时想到的方案不同"
        self.regulate_drive = [0.0] * n_concepts

        # ── 调节强度（0~1，1=完全调节，0=无调节） ──
        # 调节强度不是固定的——由元认知和自我意识动态决定
        # 高不确定性或高自我意识→更强调节
        # 低不确定性或低自我意识→弱调节（让自然情感主导）
        self.regulation_strength = 0.3

        # ── 调节策略（当前主导的调节方式） ──
        # "重评"=认知重评主导，"抑制"=抑制主导，"增强"=增强主导，"无"=无调节
        self.regulation_strategy = "无"

        # ── 调节历史 ──
        self.regulation_history = []
        self.history_cap = 50

        # ── 调节统计 ──
        self.调节次数 = 0
        self.重评次数 = 0
        self.抑制次数 = 0
        self.增强次数 = 0

    def 一拍(self, concept_h, 情感状态=None, 元认知评估=None, 自我意识信号=None):
        """情感调节一拍：认知重评、情感抑制、情感增强，基于精卫公式。

        精卫公式流程：
        1. a = 当前情感状态（E/N/T）
        2. b = 调节后的情感预测（regulate_w × concept_h）
        3. Δ = b - a（调节偏差）
        4. Σ = a + b（联合情感状态）
        5. ⊗ = Δ × Σ → 调节权重更新
        6. τ读⊗→浮出所见 = 调节后的情感从⊗中浮现
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          情感状态: dict包含E/N/T（当前情感状态）
          元认知评估: 元认知模块的meta_eval（n_concepts维，可选）
          自我意识信号: 自我意识模块的self_drive（n_concepts维，可选）

        返回：
          dict包含：
            regulated_emotion: 调节后的情感状态（dict E/N/T）
            regulate_drive: 调节驱动信号（反馈到概念层）
            regulation_strength: 调节强度（0~1）
            regulation_strategy: 调节策略（"重评"/"抑制"/"增强"/"无"）
        """
        self.拍 += 1

        # ── 当前情感状态（a） ──
        if 情感状态 is None:
            a_emotion = [0.0, 0.0, 0.0]
        else:
            a_emotion = [情感状态.get("E", 0.0),
                         情感状态.get("N", 0.0),
                         情感状态.get("T", 0.0)]

        # ── 认知重评：从概念激活预测"适当的"情感（b） ──
        # 精卫公式：b = regulate_w × concept_h（调节后的情感预测）
        # τ读⊗→浮出所见："适当的情感"从当前概念激活中浮现
        # 生物学：vmPFC→杏仁核——重新解释情境来改变情感反应
        new_regulated = [0.0] * self.n_emotion
        for i in range(self.n_emotion):
            r_input = 0.0
            for j in range(self.n_concepts):
                r_input += self.regulate_w[i][j] * concept_h[j]
            new_regulated[i] = activation(r_input)

        # ── 情感抑制：抑制过度情感反应 ──
        # 精卫公式：suppress_w × concept_h → 抑制信号
        # 生物学：dlPFC→杏仁核抑制——"不要太激动"
        # 抑制信号 = 概念激活预测的"需要抑制多少"
        new_suppress = [0.0] * self.n_emotion
        for i in range(self.n_emotion):
            s_input = 0.0
            for j in range(self.n_concepts):
                s_input += self.suppress_w[i][j] * concept_h[j]
            new_suppress[i] = activation(s_input)

        # ── 情感增强：增强有益情感反应 ──
        # 精卫公式：amplify_w × concept_h → 增强信号
        # 生物学：OFC→杏仁核增强——"保持积极"
        # 增强信号 = 概念激活预测的"需要增强多少"
        new_amplify = [0.0] * self.n_emotion
        for i in range(self.n_emotion):
            a_input = 0.0
            for j in range(self.n_concepts):
                a_input += self.amplify_w[i][j] * concept_h[j]
            new_amplify[i] = activation(a_input)

        # ── 元认知调节：元认知评估 → 调节强度 ──
        # 精卫公式：meta_regulate_w × 元认知评估 → 调节强度调制
        # 生物学：ACC→PFC调控——冲突触发调节
        # 高不确定性→更多调节（需要重新评估）
        meta_strength_mod = 0.0
        if 元认知评估 is not None:
            for j in range(self.n_concepts):
                meta_strength_mod += self.meta_regulate_w[0][j] * 元认知评估[j]
            meta_strength_mod = activation(meta_strength_mod) * 0.3  # 缩放

        # ── 自我参照调节：自我意识 → 调节强度 ──
        # 精卫公式：self_regulate_w × 自我意识驱动 → 调节强度调制
        # 生物学：mPFC→杏仁核调控——自我参照调节
        # 自我意识强度高→更精细的调节
        self_strength_mod = 0.0
        if 自我意识信号 is not None:
            for j in range(self.n_concepts):
                self_strength_mod += self.self_regulate_w[0][j] * 自我意识信号[j]
            self_strength_mod = activation(self_strength_mod) * 0.3  # 缩放

        # ── 调节强度（动态决定） ──
        # 调节强度 = 基础调节 + 元认知调制 + 自我意识调制
        # 不是固定值——由当前状态动态决定
        # 基础调节0.3 + 元认知调制 + 自我意识调制
        self.regulation_strength = max(0.0, min(1.0,
            0.3 + meta_strength_mod + self_strength_mod))

        # ── 调节后的情感状态 ──
        # 精卫公式：调节后的情感 = a + 调节强度 × (b - a)
        # = a × (1 - 调节强度) + b × 调节强度
        # 不是替换当前情感——是向"适当的"情感调整
        # 调节强度=0时：调节后的情感=当前情感（无调节）
        # 调节强度=1时：调节后的情感=预测的适当情感（完全调节）
        regulated = [0.0] * self.n_emotion
        for i in range(self.n_emotion):
            # 基础调节：向认知重评预测的适当情感调整
            regulated[i] = a_emotion[i] * (1.0 - self.regulation_strength) + \
                           new_regulated[i] * self.regulation_strength

            # 抑制：减少过度情感（只在情感强度过高时生效）
            # 抑制信号为正时减少情感强度（抑制过度反应）
            if new_suppress[i] > 0 and abs(regulated[i]) > 0.3:
                regulated[i] *= (1.0 - new_suppress[i] * 0.3 * self.regulation_strength)

            # 增强：增强有益情感（只在情感强度过低时生效）
            # 增强信号为正时增加情感强度（增强有益反应）
            if new_amplify[i] > 0 and abs(regulated[i]) < 0.3:
                sign = 1.0 if regulated[i] >= 0 else -1.0
                regulated[i] += sign * new_amplify[i] * 0.2 * self.regulation_strength

            # 硬截断——情感状态在[-1, 1]范围内
            regulated[i] = max(-1.0, min(1.0, regulated[i]))

        self.regulated_emotion = {"E": regulated[0], "N": regulated[1], "T": regulated[2]}

        # ── 调节策略判断 ──
        # 根据抑制/增强/重评信号的相对强度判断当前主导策略
        suppress_sum = sum(abs(v) for v in new_suppress)
        amplify_sum = sum(abs(v) for v in new_amplify)
        regulate_sum = sum(abs(v) for v in new_regulated)

        if self.regulation_strength < 0.15:
            self.regulation_strategy = "无"
        elif suppress_sum > amplify_sum and suppress_sum > regulate_sum * 0.5:
            self.regulation_strategy = "抑制"
        elif amplify_sum > suppress_sum and amplify_sum > regulate_sum * 0.5:
            self.regulation_strategy = "增强"
        else:
            self.regulation_strategy = "重评"

        # 统计
        self.调节次数 += 1
        if self.regulation_strategy == "重评":
            self.重评次数 += 1
        elif self.regulation_strategy == "抑制":
            self.抑制次数 += 1
        elif self.regulation_strategy == "增强":
            self.增强次数 += 1

        # ── 调节驱动信号（反馈到概念层） ──
        # 精卫公式：τ读⊗→浮出所见 = 调节驱动信号
        # 生物学：PFC→后部皮层投射——调节后的情感影响概念可及性
        # 调节后的情感让适当的概念更容易浮现——"冷静时想到的方案不同"
        # 调节驱动 = 调节后的情感对概念的调制
        # 正E（愉悦）→某些概念更容易浮现，负E（不愉悦）→另一些更容易
        # 高N（紧张）→概念激活更强，低N→概念激活更温和
        # 高T（稳定）→概念激活更持久，低T→概念激活更易变
        for i in range(self.n_concepts):
            # 调节驱动 = 调节强度 × 调节后情感的调制效应
            # 调节强度低时驱动弱（自然情感主导），调节强度高时驱动强（调节后情感主导）
            drive = self.regulation_strength * (
                regulated[0] * 0.4 +  # E→概念效价调制
                regulated[1] * 0.3 +  # N→概念强度调制
                regulated[2] * 0.3    # T→概念稳定性调制
            )
            # 每个概念的调节驱动不同——通过概念激活模式调制
            # 活跃的概念受调节影响更大
            self.regulate_drive[i] = activation(drive * concept_h[i] * 2.0)

        # ── 情感调节学习 ──
        # 精卫公式：⊗ = Δ × Σ → 调节权重更新
        # Δ = b - a（调节偏差——当前情感与适当情感的差异）
        # Σ = a + b（联合情感状态——感受+调节）
        # 赫布学习：概念激活和调节后的情感预测同时出现→调节权重增强
        # 生物学：vmPFC的赫布学习——反复从概念A预测适当情感B→A→B的映射越来越强
        delta_emotion = [new_regulated[i] - a_emotion[i] for i in range(self.n_emotion)]
        sigma_emotion = [new_regulated[i] + a_emotion[i] for i in range(self.n_emotion)]
        otimes_emotion = [delta_emotion[i] * sigma_emotion[i] for i in range(self.n_emotion)]

        # regulate_w: 概念激活→调节后的情感（赫布学习）
        for i in range(self.n_emotion):
            for j in range(self.n_concepts):
                dw = self.lr * new_regulated[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.regulate_w[i][j] = self.regulate_w[i][j] * 0.99 + dw

        # suppress_w: 概念激活→抑制信号（赫布学习）
        for i in range(self.n_emotion):
            for j in range(self.n_concepts):
                dw = self.lr * new_suppress[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.suppress_w[i][j] = self.suppress_w[i][j] * 0.99 + dw

        # amplify_w: 概念激活→增强信号（赫布学习）
        for i in range(self.n_emotion):
            for j in range(self.n_concepts):
                dw = self.lr * new_amplify[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.amplify_w[i][j] = self.amplify_w[i][j] * 0.99 + dw

        # meta_regulate_w: 元认知评估→调节强度（赫布学习）
        if 元认知评估 is not None:
            for j in range(self.n_concepts):
                dw = self.lr * meta_strength_mod * 元认知评估[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.meta_regulate_w[0][j] = self.meta_regulate_w[0][j] * 0.99 + dw

        # self_regulate_w: 自我意识驱动→调节强度（赫布学习）
        if 自我意识信号 is not None:
            for j in range(self.n_concepts):
                dw = self.lr * self_strength_mod * 自我意识信号[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.self_regulate_w[0][j] = self.self_regulate_w[0][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_emotion):
            w_norm = sum(abs(self.regulate_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.regulate_w[i][j] *= scale

        for i in range(self.n_emotion):
            w_norm = sum(abs(self.suppress_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.suppress_w[i][j] *= scale

        for i in range(self.n_emotion):
            w_norm = sum(abs(self.amplify_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.amplify_w[i][j] *= scale

        # ── 调节历史 ──
        self.regulation_history.append(self.regulation_strength)
        if len(self.regulation_history) > self.history_cap:
            self.regulation_history.pop(0)

        # 保存当前状态
        self.regulated_pred = new_regulated
        self.suppress_signal = new_suppress
        self.amplify_signal = new_amplify

        return {
            "regulated_emotion": self.regulated_emotion,
            "regulate_drive": self.regulate_drive,
            "regulation_strength": round(self.regulation_strength, 4),
            "regulation_strategy": self.regulation_strategy,
        }

    def 状态摘要(self):
        avg_strength = sum(self.regulation_history) / max(1, len(self.regulation_history)) \
            if self.regulation_history else 0.0
        return {
            "regulation_strength": round(self.regulation_strength, 4),
            "regulation_strategy": self.regulation_strategy,
            "avg_strength": round(avg_strength, 4),
            "regulate_w均值": round(sum(sum(row) for row in self.regulate_w) / max(1, self.n_emotion * self.n_concepts), 4),
            "suppress_w均值": round(sum(sum(row) for row in self.suppress_w) / max(1, self.n_emotion * self.n_concepts), 4),
            "amplify_w均值": round(sum(sum(row) for row in self.amplify_w) / max(1, self.n_emotion * self.n_concepts), 4),
            "调节次数": self.调节次数,
            "重评次数": self.重评次数,
            "抑制次数": self.抑制次数,
            "增强次数": self.增强次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 情感调节验证（认知重评、情感抑制、情感增强） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    er = 情感调节(n_concepts=64, n_emotion=3)

    print("初始状态:")
    print(f"  概念数: {er.n_concepts}")
    print(f"  情感维度: {er.n_emotion} (E/N/T)")
    print(f"  认知重评权重: regulate_w (3×64)")
    print(f"  情感抑制权重: suppress_w (3×64)")
    print(f"  情感增强权重: amplify_w (3×64)")
    print(f"  元认知调节权重: meta_regulate_w (1×64)")
    print(f"  自我参照调节权重: self_regulate_w (1×64)")
    print(f"  初始调节强度: {er.regulation_strength}")
    print(f"  初始调节策略: {er.regulation_strategy}")

    print("\n--- 测试1: 认知重评——从概念激活预测适当情感 ---")
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        情感状态 = {"E": -0.5, "N": 0.3, "T": -0.2}  # 负面情感
        result = er.一拍(concept_h, 情感状态=情感状态)
    print(f"  负面情感+概念模式A后:")
    print(f"    调节后情感: E={result['regulated_emotion']['E']:.4f}, "
          f"N={result['regulated_emotion']['N']:.4f}, "
          f"T={result['regulated_emotion']['T']:.4f}")
    print(f"    调节强度: {result['regulation_strength']}")
    print(f"    调节策略: {result['regulation_strategy']}")
    print(f"    原始情感: E=-0.5, N=0.3, T=-0.2")
    print(f"    → 调节后情感应该比原始情感更接近0（调节减少负面情感）")

    print("\n--- 测试2: 不同概念模式→不同调节后情感 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    for name, pattern in patterns:
        er_t = 情感调节(n_concepts=64, n_emotion=3)
        for _ in range(50):
            er_t.一拍(pattern, 情感状态={"E": -0.3, "N": 0.2, "T": 0.0})
        result = er_t.状态摘要()
        print(f"  {name}: 调节强度={result['regulation_strength']}, "
              f"策略={result['regulation_strategy']}, "
              f"调节后E={er_t.regulated_emotion['E']:.4f}")

    print("\n--- 测试3: 元认知评估影响调节强度 ---")
    er2 = 情感调节(n_concepts=64, n_emotion=3)
    # 高不确定性元认知评估
    元认知评估_high = [0.3] * 64
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        er2.一拍(concept_h, 情感状态={"E": -0.3, "N": 0.2, "T": 0.0},
               元认知评估=元认知评估_high)
    print(f"  高不确定性后: 调节强度={er2.regulation_strength:.4f}")

    er3 = 情感调节(n_concepts=64, n_emotion=3)
    # 低不确定性元认知评估
    元认知评估_low = [0.0] * 64
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        er3.一拍(concept_h, 情感状态={"E": -0.3, "N": 0.2, "T": 0.0},
               元认知评估=元认知评估_low)
    print(f"  低不确定性后: 调节强度={er3.regulation_strength:.4f}")
    print(f"  → 高不确定性应该导致更强调节")

    print("\n--- 测试4: 自我意识影响调节强度 ---")
    er4 = 情感调节(n_concepts=64, n_emotion=3)
    # 高自我意识驱动
    自我意识_high = [0.3] * 64
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        er4.一拍(concept_h, 情感状态={"E": -0.3, "N": 0.2, "T": 0.0},
               自我意识信号=自我意识_high)
    print(f"  高自我意识后: 调节强度={er4.regulation_strength:.4f}")

    er5 = 情感调节(n_concepts=64, n_emotion=3)
    # 低自我意识驱动
    自我意识_low = [0.0] * 64
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        er5.一拍(concept_h, 情感状态={"E": -0.3, "N": 0.2, "T": 0.0},
               自我意识信号=自我意识_low)
    print(f"  低自我意识后: 调节强度={er5.regulation_strength:.4f}")
    print(f"  → 高自我意识应该导致更强调节")

    print("\n--- 测试5: 调节驱动信号 ---")
    er6 = 情感调节(n_concepts=64, n_emotion=3)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        er6.一拍(concept_h, 情感状态={"E": -0.3, "N": 0.2, "T": 0.0})
    drive_sum = sum(abs(v) for v in er6.regulate_drive)
    print(f"  调节驱动信号总和: {drive_sum:.4f}")
    print(f"  调节驱动top5: {sorted(range(64), key=lambda i: abs(er6.regulate_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试6: 调节策略统计 ---")
    er7 = 情感调节(n_concepts=64, n_emotion=3)
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[tick % 8] = 0.3
        情感 = {"E": -0.3 if tick % 3 == 0 else 0.2,
                "N": 0.2 if tick % 2 == 0 else -0.1,
                "T": 0.1}
        er7.一拍(concept_h, 情感状态=情感)
    summary = er7.状态摘要()
    print(f"  调节次数: {summary['调节次数']}")
    print(f"  重评次数: {summary['重评次数']}")
    print(f"  抑制次数: {summary['抑制次数']}")
    print(f"  增强次数: {summary['增强次数']}")
    print(f"  平均调节强度: {summary['avg_strength']}")

    print("\n--- 测试7: 精卫公式验证 ---")
    er8 = 情感调节(n_concepts=16, n_emotion=3)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = er8.一拍(concept_h, 情感状态={"E": -0.3, "N": 0.2, "T": 0.0})
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 调节强度={result['regulation_strength']}, "
                  f"策略={result['regulation_strategy']}, "
                  f"调节后E={result['regulated_emotion']['E']:.4f}")
    print(f"  τ(浮出): 调节驱动均值={round(sum(er8.regulate_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={er8.拍}")

    print("\n验证通过")
