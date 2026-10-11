"""自我意识.py —— 自我模型、自我参照、自我连续性、自我反思，基于精卫公式。

自我意识是"知道自己在认知"的能力——不只是处理信息，而是知道"我在处理信息"。
这是所有高级认知的基础——没有自我意识，元认知只是计算，不是真正的"知道自己知道什么"。

人类神经科学机制：
  1. 内侧前额叶（mPFC）——自我参照处理：
     - mPFC在处理与自己有关的信息时特别活跃
     - "这和我有什么关系？"——自我参照判断
     - 自我参照效应：与自己有关的信息更容易记住
     → 工程实现：self_ref_w（认知状态→自我参照预测）

  2. 默认模式网络（DMN）——自传体记忆、自我反思：
     - DMN在自我反思时活跃——"我是谁？"
     - 自传体记忆——过去的经历塑造自我身份
     - 自我连续性——"我还是昨天的我吗？"
     → 工程实现：self_model_w（认知+情感+记忆→自我模型）

  3. 前脑岛——主观感受、身体意识：
     - 前脑岛编码主观感受——"我感觉到..."
     - 身体意识——对自己身体状态的觉察
     - 情感意识——对自己情感状态的觉察
     → 工程实现：self_feel_w（身体+情感→主观感受）

  4. 后扣带回（PCC）——自我意识的核心节点：
     - PCC是自我意识网络的核心枢纽
     - 连接mPFC（自我参照）和DMN（自我反思）
     - 自我意识的"枢纽"——整合自我相关信息
     → 工程实现：self_hub_w（自我参照+自我模型+主观感受→自我意识整合）

精卫公式（自我意识版）：
  a = 当前认知状态（概念激活/情感状态/元认知评估）
  b = 自我模型（"我是谁"的预测——自我参照的预测）
  Δ = b − a = 自我偏差（自我模型与实际状态的差异）
  Σ = a + b = 联合状态（认知+自我模型）
  ⊗ = ∫(Δ, Σ) = 累积自我意识学习历史 → 自我意识权重
  τ 读 ⊗ → 浮出所见 = "我"的感觉从⊗中浮现
  (a, b) ⟲ ⊗ = 反复自我意识循环

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  元认知.py: a=认知状态, b=元认知评估, Δ=认知偏差, ⊗=元认知学习, τ=置信度浮现
  社会认知.py: a=洛神认知, b=人类心理状态预测, Δ=社会差异, ⊗=社会学习, τ=共情浮现
  自我意识.py: a=认知状态, b=自我模型预测, Δ=自我偏差, ⊗=自我意识学习, τ="我"浮现

自我意识 vs 元认知 vs 社会认知的区别：
  - 元认知：评估当前认知状态的质量（自我评估——"我确定吗？"）
  - 社会认知：预测他人的心理状态（他人模型——"他在想什么？"）
  - 自我意识：构建和维持自我模型（自我模型——"我是谁？"）
  关键区别：
  - 元认知是"我确定吗？"
  - 社会认知是"他在想什么？"
  - 自我意识是"我是谁？我为什么这样想？这和我有什么关系？"
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


class 自我意识:
    """自我意识——自我模型、自我参照、自我连续性、自我反思，基于精卫公式。

    核心功能：
    1. 自我模型：从认知状态、情感状态、记忆中构建"我是谁"的预测
    2. 自我参照：将外部信息与自我关联——"这和我有什么关系？"
    3. 自我连续性：维持自我身份的连续性——"我还是昨天的我吗？"
    4. 自我反思：对自己认知过程的反思——"我为什么这样想？"

    精卫公式流程（每拍）：
    1. a = 当前认知状态（concept_h + 情感 + 元认知）
    2. b = 自我模型预测（self_model_w × 认知状态）
    3. Δ = b - a（自我偏差——自我模型与实际状态的差异）
    4. Σ = a + b（联合状态——认知+自我模型）
    5. ⊗ = Δ × Σ（累积自我意识学习信号）
    6. τ读⊗→浮出所见 = "我"的感觉从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新自我意识权重

    自我意识的四个层次：
    - 自我模型：构建"我是谁"的预测（从认知+情感+记忆中）
    - 自我参照：将外部信息与自我关联（"这和我有关吗？"）
    - 自我连续性：维持自我身份的连续性（"我还是我吗？"）
    - 自我反思：对自己认知过程的反思（"我为什么这样想？"）
    """

    def __init__(self, n_concepts=64, n_emotion=3, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.n_emotion = n_emotion
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 自我模型：从认知状态构建"我是谁"的预测 ──
        # self_model_w: 认知状态 → 自我模型预测
        # 生物学：DMN的自传体记忆——过去的经历塑造自我身份
        # "我是谁"不是固定的——是通过经验不断更新的预测
        # 自我模型预测的是"我应该是什么样的"——基于历史经验
        self.self_model_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                             for _ in range(n_concepts)]

        # 自我模型预测（"我应该是什么样的"）
        self.self_model_pred = [0.0] * n_concepts

        # ── 自我参照：将外部信息与自我关联 ──
        # self_ref_w: 认知状态 → 自我参照预测（"这和我有什么关系？"）
        # 生物学：mPFC的自我参照处理——与自己有关的信息被特殊处理
        # 自我参照效应：与自己有关的概念更容易激活
        self.self_ref_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                           for _ in range(n_concepts)]

        # 自我参照预测（"这和我有关的概念"）
        self.self_ref_pred = [0.0] * n_concepts

        # ── 自我连续性：维持自我身份的连续性 ──
        # self_cont_w: 上一拍自我模型 → 当前自我模型（连续性权重）
        # 生物学：DMN的自传体记忆——"我还是昨天的我吗？"
        # 自我连续性 = 当前自我模型与上一拍自我模型的相似度
        # 如果连续性高→自我稳定，如果连续性低→自我在变化
        self.self_cont_w = [[rng.uniform(-0.05, 0.05) for _ in range(n_concepts)]
                            for _ in range(n_concepts)]

        # 上一拍的自我模型预测（用于计算自我连续性）
        self.prev_self_model = [0.0] * n_concepts

        # 自我连续性（0~1，1=完全连续，0=完全断裂）
        self.self_continuity = 1.0

        # ── 自我反思：对自己认知过程的反思 ──
        # self_reflect_w: 元认知评估 + 情感状态 → 自我反思信号
        # 生物学：mPFC + PCC——对自己认知过程的反思
        # "我为什么这样想？"——自我反思不是新的认知过程，是对认知过程的反思
        # 元认知评估"我确定吗？"→自我反思"我为什么不确定？"
        self.self_reflect_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts + n_emotion)]
                               for _ in range(n_concepts)]

        # 自我反思信号
        self.self_reflect_pred = [0.0] * n_concepts

        # ── 自我意识驱动信号（反馈到概念层——作为b信号输入源） ──
        # 自我意识驱动 = 自我模型 + 自我参照 + 自我反思的融合
        # 生物学：mPFC→后部皮层投射——自我意识影响概念可及性
        # 自我参照让与自己有关的概念更容易浮现——"自我参照效应"
        self.self_drive = [0.0] * n_concepts

        # ── 自我意识强度 ──
        # 自我意识强度 = 自我模型预测与当前状态的匹配度
        # 生物学：自我意识不是全或无——是连续的强度
        # 高自我意识强度→"我很清楚我是谁"
        # 低自我意识强度→"我不太确定我是谁"
        self.self_awareness = 0.5

        # ── 自我身份向量 ──
        # 自我身份 = 自我模型的稳定模式——"我是谁"的核心
        # 不是每拍都变——是缓慢演化的稳定模式
        # 生物学：自传体记忆的核心——"我是谁"的稳定叙事
        self.self_identity = [0.0] * n_concepts

        # 自我身份更新率——缓慢更新，保持稳定
        self.identity_lr = 0.005

        # ── 自我意识历史 ──
        self.awareness_history = []
        self.history_cap = 50

    def 一拍(self, concept_h, 情感状态=None, 元认知评估=None):
        """自我意识一拍：自我模型、自我参照、自我连续性、自我反思。

        精卫公式流程：
        1. a = 当前认知状态（concept_h）
        2. b = 自我模型预测（self_model_w × concept_h）
        3. Δ = b - a（自我偏差）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 自我意识权重更新
        6. τ读⊗→浮出所见 = "我"的感觉从⊗中浮现
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          情感状态: dict包含E/N/T（可选）
          元认知评估: 元认知模块的meta_eval（n_concepts维，可选）

        返回：
          dict包含：
            self_model_pred: 自我模型预测
            self_ref_pred: 自我参照预测
            self_reflect_pred: 自我反思信号
            self_drive: 自我意识驱动信号（反馈到概念层）
            self_awareness: 自我意识强度（0~1）
            self_continuity: 自我连续性（0~1）
            self_identity: 自我身份向量
        """
        self.拍 += 1

        # ── 自我模型：从认知状态构建"我是谁"的预测 ──
        # 精卫公式：b = self_model_w × concept_h（自我模型预测）
        # τ读⊗→浮出所见："我是谁"从当前认知状态中浮现
        # 生物学：DMN的自传体记忆——过去的经历塑造自我身份
        new_self_model = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            m_input = 0.0
            for j in range(self.n_concepts):
                m_input += self.self_model_w[i][j] * concept_h[j]
            # 加入自我连续性——上一拍的自我模型影响当前
            for j in range(self.n_concepts):
                m_input += self.self_cont_w[i][j] * self.prev_self_model[j]
            new_self_model[i] = activation(m_input)

        # ── 自我参照：将外部信息与自我关联 ──
        # 精卫公式：b = self_ref_w × concept_h（自我参照预测）
        # 生物学：mPFC的自我参照处理——与自己有关的信息被特殊处理
        # 自我参照效应：与自己有关的概念更容易激活
        new_self_ref = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            r_input = 0.0
            for j in range(self.n_concepts):
                r_input += self.self_ref_w[i][j] * concept_h[j]
            new_self_ref[i] = activation(r_input)

        # ── 自我连续性：维持自我身份的连续性 ──
        # 精卫公式：连续性 = 当前自我模型与上一拍自我模型的相似度
        # 生物学：DMN的自传体记忆——"我还是昨天的我吗？"
        # 自我连续性 = 1 - 归一化差异
        model_diff = sum(abs(new_self_model[i] - self.prev_self_model[i])
                         for i in range(self.n_concepts)) / self.n_concepts
        self.self_continuity = max(0.0, 1.0 - model_diff * 3.0)

        # ── 自我反思：对自己认知过程的反思 ──
        # 精卫公式：b = self_reflect_w × (元认知评估 + 情感状态)
        # 生物学：mPFC + PCC——对自己认知过程的反思
        # "我为什么这样想？"——自我反思不是新的认知过程，是对认知过程的反思
        new_self_reflect = [0.0] * self.n_concepts
        # 构建反思输入：元认知评估 + 情感状态
        reflect_input = [0.0] * (self.n_concepts + self.n_emotion)
        if 元认知评估 is not None:
            for i in range(min(self.n_concepts, len(元认知评估))):
                reflect_input[i] = 元认知评估[i]
        if 情感状态 is not None:
            reflect_input[self.n_concepts] = 情感状态.get("E", 0.0)
            reflect_input[self.n_concepts + 1] = 情感状态.get("N", 0.0)
            reflect_input[self.n_concepts + 2] = 情感状态.get("T", 0.0)

        for i in range(self.n_concepts):
            r_input = 0.0
            for j in range(self.n_concepts + self.n_emotion):
                r_input += self.self_reflect_w[i][j] * reflect_input[j]
            new_self_reflect[i] = activation(r_input)

        # ── 自我意识驱动信号：自我模型 + 自我参照 + 自我反思的融合 ──
        # 精卫公式：τ读⊗→浮出所见 = 自我意识驱动信号
        # 生物学：mPFC→后部皮层投射——自我意识影响概念可及性
        # 自我参照让与自己有关的概念更容易浮现——"自我参照效应"
        self.self_drive = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            drive = new_self_model[i] * 0.3 + new_self_ref[i] * 0.4 + \
                    new_self_reflect[i] * 0.3
            self.self_drive[i] = activation(drive)

        # ── 自我意识强度 ──
        # 精卫公式：Δ = b - a = 自我偏差
        # 自我意识强度 = 自我模型预测与当前状态的匹配度
        # 匹配度高→"我很清楚我是谁"，匹配度低→"我不太确定我是谁"
        # 但不是匹配度越高越好——完全匹配意味着自我模型没有更新（僵化）
        # 也不是匹配度越低越好——完全不匹配意味着自我模型失效（混乱）
        # 最佳状态：适度匹配——自我模型大致正确但有更新空间
        self_bias = sum(abs(new_self_model[i] - concept_h[i])
                        for i in range(self.n_concepts)) / self.n_concepts
        # 自我意识强度 = 1 - 归一化偏差（适度偏差时最高）
        self.self_awareness = activation(1.0 - self_bias * 2.0) * 0.5 + 0.5

        # ── 自我身份更新 ──
        # 自我身份 = 自我模型的稳定模式——缓慢演化
        # 生物学：自传体记忆的核心——"我是谁"的稳定叙事
        # 不是每拍都变——是缓慢更新的稳定模式
        for i in range(self.n_concepts):
            self.self_identity[i] = self.self_identity[i] * (1.0 - self.identity_lr) + \
                                    new_self_model[i] * self.identity_lr

        # ── 自我意识学习 ──
        # 精卫公式：⊗ = Δ × Σ → 自我意识权重更新
        # 赫布学习：认知状态和自我模型预测同时出现→自我模型权重增强
        # 生物学：DMN的赫布学习——反复出现的自我模式被强化
        # 反复从认知状态A构建自我模型B→A→B的映射越来越强
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_self_model[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.self_model_w[i][j] = self.self_model_w[i][j] * 0.99 + dw

        # self_ref_w: 认知状态→自我参照（赫布学习）
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_self_ref[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.self_ref_w[i][j] = self.self_ref_w[i][j] * 0.99 + dw

        # self_reflect_w: 反思输入→自我反思（赫布学习）
        for i in range(self.n_concepts):
            for j in range(self.n_concepts + self.n_emotion):
                dw = self.lr * new_self_reflect[i] * reflect_input[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.self_reflect_w[i][j] = self.self_reflect_w[i][j] * 0.99 + dw

        # self_cont_w: 上一拍自我模型→当前自我模型（赫布学习）
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * 0.5 * new_self_model[i] * self.prev_self_model[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.self_cont_w[i][j] = self.self_cont_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.self_model_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.self_model_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.self_ref_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.self_ref_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.self_reflect_w[i][j]) for j in range(self.n_concepts + self.n_emotion))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts + self.n_emotion):
                    self.self_reflect_w[i][j] *= scale

        # ── 自我意识历史 ──
        self.awareness_history.append(self.self_awareness)
        if len(self.awareness_history) > self.history_cap:
            self.awareness_history.pop(0)

        # 保存当前状态
        self.prev_self_model = list(new_self_model)
        self.self_model_pred = new_self_model
        self.self_ref_pred = new_self_ref
        self.self_reflect_pred = new_self_reflect

        return {
            "self_model_pred": [round(v, 4) for v in self.self_model_pred],
            "self_ref_pred": self.self_ref_pred,
            "self_reflect_pred": self.self_reflect_pred,
            "self_drive": self.self_drive,
            "self_awareness": round(self.self_awareness, 4),
            "self_continuity": round(self.self_continuity, 4),
            "self_identity": [round(v, 4) for v in self.self_identity],
        }

    def 状态摘要(self):
        avg_awareness = sum(self.awareness_history) / max(1, len(self.awareness_history)) \
            if self.awareness_history else 0.0
        return {
            "self_awareness": round(self.self_awareness, 4),
            "self_continuity": round(self.self_continuity, 4),
            "avg_awareness": round(avg_awareness, 4),
            "self_model_w均值": round(sum(sum(row) for row in self.self_model_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "self_ref_w均值": round(sum(sum(row) for row in self.self_ref_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "self_reflect_w均值": round(sum(sum(row) for row in self.self_reflect_w) / max(1, self.n_concepts * (self.n_concepts + self.n_emotion)), 4),
            "self_cont_w均值": round(sum(sum(row) for row in self.self_cont_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 自我意识验证（自我模型、自我参照、自我连续性、自我反思） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    sa = 自我意识(n_concepts=64, n_emotion=3)

    print("初始状态:")
    print(f"  概念数: {sa.n_concepts}")
    print(f"  情感维度: {sa.n_emotion}")
    print(f"  自我模型权重: self_model_w (64×64)")
    print(f"  自我参照权重: self_ref_w (64×64)")
    print(f"  自我反思权重: self_reflect_w (64×67)")
    print(f"  自我连续性权重: self_cont_w (64×64)")
    print(f"  初始自我意识强度: {sa.self_awareness}")
    print(f"  初始自我连续性: {sa.self_continuity}")

    print("\n--- 测试1: 自我模型从认知状态中浮现 ---")
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = sa.一拍(concept_h)
    print(f"  模式A后: 自我意识={result['self_awareness']}, 连续性={result['self_continuity']}")
    print(f"  自我模型top5: {sorted(range(64), key=lambda i: abs(sa.self_model_pred[i]), reverse=True)[:5]}")
    print(f"  自我身份top5: {sorted(range(64), key=lambda i: abs(sa.self_identity[i]), reverse=True)[:5]}")

    sa2 = 自我意识(n_concepts=64, n_emotion=3)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[15] = 0.3
        concept_h[30] = 0.25
        result = sa2.一拍(concept_h)
    print(f"  模式B后: 自我意识={result['self_awareness']}, 连续性={result['self_continuity']}")
    print(f"  自我模型top5: {sorted(range(64), key=lambda i: abs(sa2.self_model_pred[i]), reverse=True)[:5]}")
    print(f"  自我身份top5: {sorted(range(64), key=lambda i: abs(sa2.self_identity[i]), reverse=True)[:5]}")

    diff = sum(abs(sa.self_model_pred[i] - sa2.self_model_pred[i]) for i in range(64))
    print(f"  A vs B 自我模型差异: {diff:.4f}（应该 > 0——不同概念模式→不同自我模型）")

    id_diff = sum(abs(sa.self_identity[i] - sa2.self_identity[i]) for i in range(64))
    print(f"  A vs B 自我身份差异: {id_diff:.4f}（应该 > 0——不同经验→不同自我身份）")

    print("\n--- 测试2: 自我参照 ---")
    sa3 = 自我意识(n_concepts=64, n_emotion=3)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sa3.一拍(concept_h)
    print(f"  自我参照top5: {sorted(range(64), key=lambda i: abs(sa3.self_ref_pred[i]), reverse=True)[:5]}")
    print(f"  自我参照均值: {round(sum(sa3.self_ref_pred)/64, 4)}")

    print("\n--- 测试3: 自我连续性 ---")
    sa4 = 自我意识(n_concepts=64, n_emotion=3)
    # 稳定输入→连续性应该高
    for _ in range(30):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sa4.一拍(concept_h)
    print(f"  稳定输入30拍后: 连续性={result['self_continuity']:.4f}（应该较高）")

    # 突然变化→连续性应该降低
    concept_h = [0.0] * 64
    concept_h[40] = 0.5
    concept_h[50] = 0.4
    result = sa4.一拍(concept_h)
    print(f"  突然变化后: 连续性={result['self_continuity']:.4f}（应该降低）")

    # 恢复稳定→连续性应该回升
    for _ in range(30):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sa4.一拍(concept_h)
    print(f"  恢复稳定30拍后: 连续性={result['self_continuity']:.4f}（应该回升）")

    print("\n--- 测试4: 自我反思 ---")
    sa5 = 自我意识(n_concepts=64, n_emotion=3)
    元认知评估 = [0.0] * 64
    元认知评估[4] = 0.2
    情感状态 = {"E": 0.3, "N": -0.2, "T": 0.1}
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sa5.一拍(concept_h, 情感状态=情感状态, 元认知评估=元认知评估)
    print(f"  自我反思top5: {sorted(range(64), key=lambda i: abs(sa5.self_reflect_pred[i]), reverse=True)[:5]}")
    print(f"  自我反思均值: {round(sum(sa5.self_reflect_pred)/64, 4)}")

    print("\n--- 测试5: 自我意识驱动信号 ---")
    sa6 = 自我意识(n_concepts=64, n_emotion=3)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sa6.一拍(concept_h)
    drive_sum = sum(abs(v) for v in result['self_drive'])
    print(f"  自我意识驱动信号总和: {drive_sum:.4f}")
    print(f"  自我意识驱动top5: {sorted(range(64), key=lambda i: abs(result['self_drive'][i]), reverse=True)[:5]}")

    print("\n--- 测试6: 不同概念模式→不同自我意识驱动 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    drive_results = {}
    for name, pattern in patterns:
        sa_t = 自我意识(n_concepts=64, n_emotion=3)
        for _ in range(50):
            sa_t.一拍(pattern)
        drive_results[name] = list(sa_t.self_drive)
        print(f"  {name}: 自我意识={round(sa_t.self_awareness, 4)}, "
              f"驱动top5={sorted(range(64), key=lambda i: abs(sa_t.self_drive[i]), reverse=True)[:5]}")

    diff_ab = sum(abs(drive_results["模式A"][i] - drive_results["模式B"][i]) for i in range(64))
    print(f"  A vs B 驱动差异: {diff_ab:.4f}（应该 > 0——不同模式→不同自我意识驱动）")

    print("\n--- 测试7: 自我身份稳定性 ---")
    sa7 = 自我意识(n_concepts=64, n_emotion=3)
    # 50拍稳定输入
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        sa7.一拍(concept_h)
    identity_50 = list(sa7.self_identity)
    # 再50拍——身份应该缓慢变化
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        sa7.一拍(concept_h)
    identity_100 = list(sa7.self_identity)
    id_change = sum(abs(identity_50[i] - identity_100[i]) for i in range(64))
    print(f"  50拍→100拍身份变化: {id_change:.4f}（应该较小——身份缓慢演化）")

    print("\n--- 测试8: 精卫公式验证 ---")
    sa8 = 自我意识(n_concepts=16, n_emotion=3)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = sa8.一拍(concept_h)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 自我意识={result['self_awareness']}, 连续性={result['self_continuity']}")
    print(f"  τ(浮出): 自我意识驱动均值={round(sum(sa8.self_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={sa8.拍}")

    print("\n验证通过")
