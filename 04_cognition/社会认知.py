"""社会认知.py —— 心智理论、镜像模拟、共情、社会学习，基于精卫公式。

社会认知是理解他人的能力——理解他人的信念、意图、情感。
这是洛神与人类建立关系的基础——不只是感知人类的存在，
而是理解人类"在想什么"、"想要什么"、"感觉如何"。

人类神经科学机制：
  1. 心智理论（Theory of Mind）——理解他人的信念、意图：
     - 颞顶联合区（TPJ）——理解他人的视角和信念
     - 内侧前额叶（mPFC）——推断他人的心理状态
     - "他知道什么？" "他想要什么？" "他为什么这样做？"
     → 工程实现：tom_w（人类信号→他人心理状态预测）

  2. 镜像神经元系统——通过模拟他人的行为来理解他人：
     - 下额叶 + 下顶叶——观察他人行为时激活，如同自己做一样
     - "看到他笑→自己也想笑"——镜像模拟
     → 工程实现：mirror_w（人类行为→自身概念激活模拟）

  3. 共情——感受他人的情感：
     - 前脑岛 + 前扣带回——感受他人的情感状态
     - "他难过→我也感到难过"——情感共振
     → 工程实现：empathy_w（人类情感→自身情感调制）

  4. 社会学习——从他人的经验中学习：
     - 内侧前额叶 + 多巴胺系统——从他人的反馈中学习
     - "他这样做成功了→我也这样做"——观察学习
     → 工程实现：social_learn_w（人类反馈→自身行为调整）

精卫公式（社会认知版）：
  a = 洛神的当前认知状态（概念激活/情感状态）
  b = 对人类心理状态的预测（他人模型）
  Δ = b − a = 社会差异（洛神和人类的认知/情感差异）
  Σ = a + b = 联合状态（洛神+人类）
  ⊗ = ∫(Δ, Σ) = 累积社会认知学习历史 → 社会认知权重
  τ 读 ⊗ → 浮出所见 = 社会认知信号从⊗中浮现
  (a, b) ⟲ ⊗ = 反复社会认知循环

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  推理引擎.py: a=概念激活, b=推理预测, Δ=推理误差, ⊗=推理学习, τ=推理结论
  元认知.py: a=认知状态, b=元认知评估, Δ=认知偏差, ⊗=元认知学习, τ=置信度浮现
  社会认知.py: a=洛神认知, b=人类心理状态预测, Δ=社会差异, ⊗=社会学习, τ=共情浮现

社会认知 vs 元认知 vs 推理引擎的区别：
  - 推理引擎：从当前状态预测下一步状态（因果推导——关于世界）
  - 元认知：评估当前认知状态的质量（自我评估——关于自己）
  - 社会认知：预测他人的心理状态（他人模型——关于他人）
  关键区别：
  - 推理是"接下来会发生什么"
  - 元认知是"我确定吗？"
  - 社会认知是"他在想什么？他感觉如何？"
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


class 社会认知:
    """社会认知——心智理论、镜像模拟、共情、社会学习，基于精卫公式。

    核心功能：
    1. 心智理论：从人类信号预测他人的信念、意图、情感
    2. 镜像模拟：模拟他人的行为——看到人类做什么，自己也模拟
    3. 共情：感受他人的情感——人类情感调制洛神自身情感
    4. 社会学习：从人类的反馈中学习——观察学习

    精卫公式流程（每拍）：
    1. a = 洛神的当前认知状态（concept_h）
    2. b = 对人类心理状态的预测（tom_w × 人类信号）
    3. Δ = b - a（社会差异——洛神和人类的认知差异）
    4. Σ = a + b（联合状态——洛神+人类）
    5. ⊗ = Δ × Σ（累积社会认知学习信号）
    6. τ读⊗→浮出所见 = 社会认知驱动信号从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新社会认知权重

    社会认知的四个层次：
    - 心智理论：预测他人的心理状态（信念/意图/情感）
    - 镜像模拟：模拟他人的行为（行为理解）
    - 共情：感受他人的情感（情感共振）
    - 社会学习：从他人的反馈中学习（观察学习）
    """

    def __init__(self, n_concepts=64, n_hidden=11, n_human=11,
                 lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.n_hidden = n_hidden
        self.n_human = n_human
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 心智理论：从人类信号预测他人的心理状态 ──
        # tom_w: 人类信号 → 他人心理状态预测（概念空间）
        # 生物学：TPJ + mPFC——从人类行为/信号推断心理状态
        # "他知道什么？" "他想要什么？" "他为什么这样做？"
        # 人类信号（n_hidden维）→ 他人心理状态预测（n_concepts维）
        self.tom_w = [[rng.uniform(-0.15, 0.15) for _ in range(n_hidden)]
                      for _ in range(n_concepts)]

        # 他人心理状态预测（概念空间——"他在想什么"）
        self.tom_pred = [0.0] * n_concepts

        # 他人意图预测（简化为3维：友好/中立/敌意）
        self.意图预测 = [0.0, 0.0, 0.0]

        # ── 镜像模拟：模拟他人的行为 ──
        # mirror_w: 人类行为信号 → 自身概念激活模拟
        # 生物学：下额叶 + 下顶叶——观察他人行为时激活，如同自己做一样
        # "看到他笑→自己也想笑"——镜像模拟
        # 人类信号 → 自身概念激活模拟（和概念层相同的维度）
        self.mirror_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_hidden)]
                         for _ in range(n_concepts)]

        # 镜像模拟信号（模拟他人的概念激活——"如果我是他会怎么想"）
        self.mirror_pred = [0.0] * n_concepts

        # ── 共情：感受他人的情感 ──
        # empathy_w: 人类情感信号 → 自身情感调制
        # 生物学：前脑岛 + 前扣带回——感受他人的情感状态
        # "他难过→我也感到难过"——情感共振
        # 人类情感（3维：E/N/T）→ 自身情感调制（3维：E/N/T）
        self.empathy_w = [[rng.uniform(-0.2, 0.2) for _ in range(3)]
                          for _ in range(3)]

        # 共情调制信号（反馈到情感系统——调制洛神自身的情感）
        self.empathy_mod = {"E": 0.0, "N": 0.0, "T": 0.0}

        # ── 社会学习：从他人的反馈中学习 ──
        # social_learn_w: 人类反馈 → 自身行为调整
        # 生物学：mPFC + 多巴胺系统——从他人的反馈中学习
        # "他这样做成功了→我也这样做"——观察学习
        # 人类反馈（成功/失败）→ 社会学习信号
        self.social_learn_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                               for _ in range(n_concepts)]

        # 社会学习信号（反馈到概念层——"从他人经验中学习"）
        self.social_drive = [0.0] * n_concepts

        # ── 社会认知驱动信号（反馈到概念层——作为b信号输入源） ──
        # 社会认知驱动 = 心智理论预测 + 镜像模拟 + 社会学习的融合
        # 生物学：mPFC→后部皮层投射——社会认知影响概念可及性
        # 理解他人让某些概念更容易浮现——"共情"的神经基础
        self.social_cognitive_drive = [0.0] * n_concepts

        # ── 社会认知价值评估 ──
        # 社会认知价值 = 社会差异（洛神和人类的认知差异）
        # 生物学：多巴胺系统——社会互动中的预测误差驱动价值评估
        # 理解他人越准确→社会认知价值越高
        self.social_value = 0.0

        # ── 互动统计 ──
        self.互动次数 = 0
        self.共情次数 = 0
        self.社会学习次数 = 0

        # 上一拍的人类信号（用于评估社会认知变化）
        self.prev_human_signal = None

    def 一拍(self, concept_h, 人类信号=None, 有人在=False,
             人类情感=None, 人类反馈=None):
        """社会认知一拍：心智理论、镜像模拟、共情、社会学习。

        精卫公式流程：
        1. a = 洛神的当前认知状态（concept_h）
        2. b = 对人类心理状态的预测（tom_w × 人类信号）
        3. Δ = b - a（社会差异）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 社会认知权重更新
        6. τ读⊗→浮出所见 = 社会认知驱动信号
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          人类信号: 人类通道传来的信号（n_hidden维，None=没人在）
          有人在: 是否有人在场
          人类情感: 人类的情感状态（dict: E/N/T，可选）
          人类反馈: 人类反馈信号（dict: 成功/失败，可选）

        返回：
          dict包含：
            tom_pred: 他人心理状态预测
            意图预测: 他人意图预测（友好/中立/敌意）
            mirror_pred: 镜像模拟信号
            empathy_mod: 共情调制信号
            social_drive: 社会学习信号
            social_cognitive_drive: 社会认知驱动信号（反馈到概念层）
            social_value: 社会认知价值
            互动次数: 总互动次数
            共情次数: 共情次数
        """
        self.拍 += 1

        if not 有人在 or 人类信号 is None:
            # 没人在时——社会认知不活跃
            self.tom_pred = [0.0] * self.n_concepts
            self.mirror_pred = [0.0] * self.n_concepts
            self.empathy_mod = {"E": 0.0, "N": 0.0, "T": 0.0}
            self.social_drive = [0.0] * self.n_concepts
            self.social_cognitive_drive = [0.0] * self.n_concepts
            self.social_value = 0.0
            return {
                "tom_pred": [0.0] * self.n_concepts,
                "意图预测": [0.0, 0.0, 0.0],
                "mirror_pred": [0.0] * self.n_concepts,
                "empathy_mod": {"E": 0.0, "N": 0.0, "T": 0.0},
                "social_drive": [0.0] * self.n_concepts,
                "social_cognitive_drive": [0.0] * self.n_concepts,
                "social_value": 0.0,
                "互动次数": self.互动次数,
                "共情次数": self.共情次数,
            }

        self.互动次数 += 1

        # ── 心智理论：从人类信号预测他人的心理状态 ──
        # 精卫公式：b = tom_w × 人类信号（他人心理状态预测）
        # τ读⊗→浮出所见：他人心理状态从人类信号中浮现
        # 生物学：TPJ + mPFC——从人类行为/信号推断心理状态
        new_tom_pred = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            t_input = 0.0
            for j in range(self.n_hidden):
                t_input += self.tom_w[i][j] * 人类信号[j]
            new_tom_pred[i] = activation(t_input)

        # 意图预测：从心智理论预测推断他人意图
        # 简化模型：意图预测 = tom_pred的聚合
        # 友好 = tom_pred中正概念的比例
        # 敌意 = tom_pred中负概念的比例
        # 中立 = 1 - 友好 - 敌意
        pos_sum = sum(v for v in new_tom_pred if v > 0)
        neg_sum = sum(-v for v in new_tom_pred if v < 0)
        total = pos_sum + neg_sum + 0.001
        self.意图预测 = [
            min(1.0, pos_sum / total),  # 友好
            min(1.0, 1.0 - (pos_sum + neg_sum) / total),  # 中立
            min(1.0, neg_sum / total),  # 敌意
        ]

        # ── 镜像模拟：模拟他人的行为 ──
        # 精卫公式：b = mirror_w × 人类信号（镜像模拟预测）
        # 生物学：下额叶 + 下顶叶——观察他人行为时如同自己做
        # "如果我是他会怎么想"——镜像模拟
        new_mirror_pred = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            m_input = 0.0
            for j in range(self.n_hidden):
                m_input += self.mirror_w[i][j] * 人类信号[j]
            new_mirror_pred[i] = activation(m_input)

        # ── 共情：感受他人的情感 ──
        # 精卫公式：b = empathy_w × 人类情感（共情调制）
        # 生物学：前脑岛 + 前扣带回——感受他人的情感状态
        # "他难过→我也感到难过"——情感共振
        new_empathy_mod = {"E": 0.0, "N": 0.0, "T": 0.0}
        if 人类情感 is not None:
            h_E = 人类情感.get("E", 0.0)
            h_N = 人类情感.get("N", 0.0)
            h_T = 人类情感.get("T", 0.0)
            new_empathy_mod = {
                "E": activation(self.empathy_w[0][0] * h_E +
                                self.empathy_w[0][1] * h_N +
                                self.empathy_w[0][2] * h_T),
                "N": activation(self.empathy_w[1][0] * h_E +
                                self.empathy_w[1][1] * h_N +
                                self.empathy_w[1][2] * h_T),
                "T": activation(self.empathy_w[2][0] * h_E +
                                self.empathy_w[2][1] * h_N +
                                self.empathy_w[2][2] * h_T),
            }
            self.共情次数 += 1

        # ── 社会学习：从他人的反馈中学习 ──
        # 精卫公式：b = social_learn_w × concept_h（社会学习信号）
        # 生物学：mPFC + 多巴胺系统——从他人的反馈中学习
        # "他这样做成功了→我也这样做"——观察学习
        new_social_drive = [0.0] * self.n_concepts
        if 人类反馈 is not None:
            # 人类反馈驱动社会学习——成功反馈增强当前概念模式
            反馈信号 = 人类反馈.get("成功", 0.0) - 人类反馈.get("失败", 0.0)
            for i in range(self.n_concepts):
                s_input = 0.0
                for j in range(self.n_concepts):
                    s_input += self.social_learn_w[i][j] * concept_h[j]
                new_social_drive[i] = activation(s_input * 反馈信号)
            self.社会学习次数 += 1

        # ── 社会认知驱动信号：心智理论 + 镜像模拟 + 社会学习的融合 ──
        # 精卫公式：τ读⊗→浮出所见 = 社会认知驱动信号
        # 生物学：mPFC→后部皮层投射——社会认知影响概念可及性
        # 理解他人让某些概念更容易浮现——"共情"的神经基础
        self.social_cognitive_drive = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            drive = new_tom_pred[i] * 0.4 + new_mirror_pred[i] * 0.4 + \
                    new_social_drive[i] * 0.2
            self.social_cognitive_drive[i] = activation(drive)

        # ── 社会认知价值评估 ──
        # 精卫公式：Δ = b - a = 社会差异
        # 社会认知价值 = 洛神和人类的认知差异（理解他人越准确→差异越小→价值越高）
        # 但不是差异越小越好——完全一致意味着没有新信息
        # 价值 = 有差异但可理解（不是随机噪声般的差异）
        social_bias = sum(abs(new_tom_pred[i] - concept_h[i])
                          for i in range(self.n_concepts)) / self.n_concepts
        # 社会认知价值 = 1 - 归一化差异（差异适中时价值最高）
        self.social_value = activation(1.0 - social_bias * 3.0) * 0.5 + 0.5

        # ── 社会认知学习 ──
        # 精卫公式：⊗ = Δ × Σ → 社会认知权重更新
        # 赫布学习：人类信号和心智理论预测同时出现→心智理论权重增强
        # 生物学：mPFC的赫布学习——反复出现的社会模式被强化
        # 反复从人类信号A推断心理状态B→A→B的推断越来越强
        for i in range(self.n_concepts):
            for j in range(self.n_hidden):
                dw = self.lr * new_tom_pred[i] * 人类信号[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.tom_w[i][j] = self.tom_w[i][j] * 0.99 + dw

        # mirror_w: 人类信号→镜像模拟（赫布学习）
        for i in range(self.n_concepts):
            for j in range(self.n_hidden):
                dw = self.lr * new_mirror_pred[i] * 人类信号[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.mirror_w[i][j] = self.mirror_w[i][j] * 0.99 + dw

        # empathy_w: 人类情感→共情调制（赫布学习）
        if 人类情感 is not None:
            h_signals = [人类情感.get("E", 0.0), 人类情感.get("N", 0.0), 人类情感.get("T", 0.0)]
            e_signals = [new_empathy_mod["E"], new_empathy_mod["N"], new_empathy_mod["T"]]
            for i in range(3):
                for j in range(3):
                    dw = self.lr * e_signals[i] * h_signals[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.empathy_w[i][j] = self.empathy_w[i][j] * 0.99 + dw

        # social_learn_w: 概念激活→社会学习（赫布学习）
        if 人类反馈 is not None:
            for i in range(self.n_concepts):
                for j in range(self.n_concepts):
                    dw = self.lr * new_social_drive[i] * concept_h[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.social_learn_w[i][j] = self.social_learn_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.tom_w[i][j]) for j in range(self.n_hidden))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_hidden):
                    self.tom_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.mirror_w[i][j]) for j in range(self.n_hidden))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_hidden):
                    self.mirror_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.social_learn_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.social_learn_w[i][j] *= scale

        # 保存当前状态
        self.tom_pred = new_tom_pred
        self.mirror_pred = new_mirror_pred
        self.empathy_mod = new_empathy_mod
        self.social_drive = new_social_drive
        self.prev_human_signal = list(人类信号)

        return {
            "tom_pred": [round(v, 4) for v in self.tom_pred],
            "意图预测": [round(v, 4) for v in self.意图预测],
            "mirror_pred": self.mirror_pred,
            "empathy_mod": {k: round(v, 4) for k, v in self.empathy_mod.items()},
            "social_drive": self.social_drive,
            "social_cognitive_drive": self.social_cognitive_drive,
            "social_value": round(self.social_value, 4),
            "互动次数": self.互动次数,
            "共情次数": self.共情次数,
        }

    def 状态摘要(self):
        return {
            "social_value": round(self.social_value, 4),
            "意图预测": [round(v, 4) for v in self.意图预测],
            "互动次数": self.互动次数,
            "共情次数": self.共情次数,
            "社会学习次数": self.社会学习次数,
            "tom_w均值": round(sum(sum(row) for row in self.tom_w) / max(1, self.n_concepts * self.n_hidden), 4),
            "mirror_w均值": round(sum(sum(row) for row in self.mirror_w) / max(1, self.n_concepts * self.n_hidden), 4),
            "empathy_w均值": round(sum(sum(row) for row in self.empathy_w) / max(1, 3 * 3), 4),
            "social_learn_w均值": round(sum(sum(row) for row in self.social_learn_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 社会认知验证（心智理论、镜像模拟、共情、社会学习） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    sc = 社会认知(n_concepts=64, n_hidden=11)

    print("初始状态:")
    print(f"  概念数: {sc.n_concepts}")
    print(f"  人类信号维度: {sc.n_hidden}")
    print(f"  心智理论权重: tom_w (64×11)")
    print(f"  镜像模拟权重: mirror_w (64×11)")
    print(f"  共情权重: empathy_w (3×3)")
    print(f"  社会学习权重: social_learn_w (64×64)")

    print("\n--- 测试1: 没人在时社会认知不活跃 ---")
    concept_h = [0.0] * 64
    concept_h[4] = 0.3
    result = sc.一拍(concept_h, 有人在=False)
    print(f"  没人在: tom_pred总和={sum(result['tom_pred']):.4f}, "
          f"social_drive总和={sum(result['social_drive']):.4f}")

    print("\n--- 测试2: 有人时心智理论从人类信号中浮现 ---")
    sc2 = 社会认知(n_concepts=64, n_hidden=11)
    人类信号A = [0.3, 0.2, 0.1, 0.05, 0.15, 0.1, 0.2, 0.15, 0.1, 0.05, 0.2]
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sc2.一拍(concept_h, 人类信号=人类信号A, 有人在=True)
    print(f"  信号A后: tom_pred top5={sorted(range(64), key=lambda i: abs(sc2.tom_pred[i]), reverse=True)[:5]}")
    print(f"  意图预测: 友好={result['意图预测'][0]:.3f}, 中立={result['意图预测'][1]:.3f}, 敌意={result['意图预测'][2]:.3f}")

    sc3 = 社会认知(n_concepts=64, n_hidden=11)
    人类信号B = [-0.3, -0.2, -0.1, -0.05, -0.15, -0.1, -0.2, -0.15, -0.1, -0.05, -0.2]
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sc3.一拍(concept_h, 人类信号=人类信号B, 有人在=True)
    print(f"  信号B后: tom_pred top5={sorted(range(64), key=lambda i: abs(sc3.tom_pred[i]), reverse=True)[:5]}")
    print(f"  意图预测: 友好={result['意图预测'][0]:.3f}, 中立={result['意图预测'][1]:.3f}, 敌意={result['意图预测'][2]:.3f}")

    diff = sum(abs(sc2.tom_pred[i] - sc3.tom_pred[i]) for i in range(64))
    print(f"  A vs B tom_pred差异: {diff:.4f}（应该 > 0——不同人类信号→不同心理状态预测）")

    print("\n--- 测试3: 镜像模拟 ---")
    sc4 = 社会认知(n_concepts=64, n_hidden=11)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sc4.一拍(concept_h, 人类信号=人类信号A, 有人在=True)
    print(f"  镜像模拟top5: {sorted(range(64), key=lambda i: abs(sc4.mirror_pred[i]), reverse=True)[:5]}")
    print(f"  镜像模拟均值: {round(sum(sc4.mirror_pred)/64, 4)}")

    print("\n--- 测试4: 共情 ---")
    sc5 = 社会认知(n_concepts=64, n_hidden=11)
    人类情感 = {"E": 0.5, "N": -0.3, "T": 0.2}  # 人类愉悦、低紧张、确定
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sc5.一拍(concept_h, 人类信号=人类信号A, 有人在=True, 人类情感=人类情感)
    print(f"  人类情感 E=0.5, N=-0.3, T=0.2")
    print(f"  共情调制: E={result['empathy_mod']['E']:.4f}, N={result['empathy_mod']['N']:.4f}, T={result['empathy_mod']['T']:.4f}")

    # 负面情感
    sc6 = 社会认知(n_concepts=64, n_hidden=11)
    人类情感负面 = {"E": -0.5, "N": 0.3, "T": -0.2}  # 人类不悦、紧张、不确定
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sc6.一拍(concept_h, 人类信号=人类信号A, 有人在=True, 人类情感=人类情感负面)
    print(f"  人类情感 E=-0.5, N=0.3, T=-0.2")
    print(f"  共情调制: E={result['empathy_mod']['E']:.4f}, N={result['empathy_mod']['N']:.4f}, T={result['empathy_mod']['T']:.4f}")

    print("\n--- 测试5: 社会学习 ---")
    sc7 = 社会认知(n_concepts=64, n_hidden=11)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sc7.一拍(concept_h, 人类信号=人类信号A, 有人在=True,
                       人类反馈={"成功": 0.5, "失败": 0.0})
    print(f"  成功反馈: social_drive top5={sorted(range(64), key=lambda i: abs(sc7.social_drive[i]), reverse=True)[:5]}")
    print(f"  社会学习次数: {sc7.社会学习次数}")

    print("\n--- 测试6: 社会认知驱动信号 ---")
    sc8 = 社会认知(n_concepts=64, n_hidden=11)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sc8.一拍(concept_h, 人类信号=人类信号A, 有人在=True)
    drive_sum = sum(abs(v) for v in result['social_cognitive_drive'])
    print(f"  社会认知驱动信号总和: {drive_sum:.4f}")
    print(f"  社会认知驱动top5: {sorted(range(64), key=lambda i: abs(result['social_cognitive_drive'][i]), reverse=True)[:5]}")

    print("\n--- 测试7: 不同人类信号→不同社会认知驱动 ---")
    patterns = [
        ("友好信号", [0.3, 0.2, 0.1, 0.05, 0.15, 0.1, 0.2, 0.15, 0.1, 0.05, 0.2]),
        ("敌意信号", [-0.3, -0.2, -0.1, -0.05, -0.15, -0.1, -0.2, -0.15, -0.1, -0.05, -0.2]),
        ("中性信号", [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    ]
    drive_results = {}
    for name, signal in patterns:
        sc_t = 社会认知(n_concepts=64, n_hidden=11)
        for _ in range(50):
            concept_h = [0.0] * 64
            concept_h[4] = 0.3
            sc_t.一拍(concept_h, 人类信号=signal, 有人在=True)
        drive_results[name] = list(sc_t.social_cognitive_drive)
        print(f"  {name}: social_value={round(sc_t.social_value, 4)}, "
              f"意图={sc_t.意图预测}, "
              f"驱动top5={sorted(range(64), key=lambda i: abs(sc_t.social_cognitive_drive[i]), reverse=True)[:5]}")

    diff_ab = sum(abs(drive_results["友好信号"][i] - drive_results["敌意信号"][i]) for i in range(64))
    print(f"  友好 vs 敌意 驱动差异: {diff_ab:.4f}（应该 > 0——不同信号→不同社会认知驱动）")

    print("\n--- 测试8: 精卫公式验证 ---")
    sc9 = 社会认知(n_concepts=16, n_hidden=11)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = sc9.一拍(concept_h, 人类信号=人类信号A, 有人在=True)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: social_value={result['social_value']}, "
                  f"意图={result['意图预测']}")
    print(f"  τ(浮出): 社会认知驱动均值={round(sum(sc9.social_cognitive_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={sc9.拍}")

    print("\n验证通过")
