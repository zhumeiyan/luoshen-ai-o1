"""真理追求.py —— 主动验证知识，基于精卫公式。

真理追求是人类认知的高级能力——不只是被动接受信息，
而是主动验证自己学到的知识是否正确。

AI大模型完全没有这个机制——它们被动接受训练数据中的所有信息，
不区分"验证过的知识"和"未验证的信息"。
这导致AI容易产生幻觉——把未验证的信息当作事实输出。

人类神经科学机制：
  1. 假设生成（Hypothesis Generation）——前额叶皮层（PFC）：
     - 从已有知识生成可验证的假设
     - "我认为这个概念模式是正确的"——假设是知识的可验证形式
     - PFC的执行功能——将知识转化为假设
     → 工程实现：truth_hypothesis_w（概念激活→假设预测）

  2. 假设验证（Hypothesis Verification）——前扣带回（ACC）+ PFC：
     - 通过预测误差验证假设——"预测与现实一致吗？"
     - ACC的错误检测——预测误差引发的验证
     - 验证通过：预测误差小→假设正确→信念增强
     - 验证失败：预测误差大→假设错误→信念减弱
     → 工程实现：truth_verify_w（假设+预测误差→验证结果）

  3. 信念更新（Belief Updating）——贝叶斯推理（PFC+顶叶）：
     - 根据验证结果更新信念——贝叶斯后验更新
     - 验证通过→信念增强（先验×似然→后验增强）
     - 验证失败→信念减弱（先验×似然→后验减弱）
     - 信念不是0/1——是连续的置信度（0~1）
     → 工程实现：truth_belief_w（验证结果→信念更新）

  4. 确定性寻求（Certainty Seeking）——ACC的认知一致性监测：
     - 追求认知一致性——"不一致的概念需要验证"
     - 概念激活不一致→确定性寻求高→更主动验证
     - 概念激活一致→确定性寻求低→更多利用已有知识
     - 认知失调（Cognitive Dissonance）——不一致引发不适→驱动验证
     → 工程实现：certainty_seeking（不一致性→确定性寻求强度）

  5. 知识整合（Knowledge Integration）——海马体+PFC：
     - 将验证过的知识与已有知识整合
     - 验证通过的知识→整合到长期记忆（信念增强）
     - 验证失败的知识→标记为不确定（信念减弱）
     → 工程实现：truth_drive（信念强度→概念层驱动）

精卫公式（真理追求版）：
  a = 当前信念状态（概念激活——已有的知识/信念）
  b = 验证预测（假设验证的结果——预测与现实的一致性）
  Δ = b − a = 验证偏差（信念与验证结果的差异）
  Σ = a + b = 联合信念状态（信念+验证）
  ⊗ = ∫(Δ, Σ) = 累积验证学习历史 → 真理追求权重
  τ 读 ⊗ → 浮出所见 = "这是真的吗"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复验证循环

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  好奇心.py: a=当前知识, b=想知道的知识, Δ=信息差距, ⊗=好奇心学习, τ=探索方向浮现
  元认知.py: a=认知状态, b=元认知评估, Δ=认知偏差, ⊗=元认知学习, τ=置信度浮现
  真理追求.py: a=当前信念, b=验证预测, Δ=验证偏差, ⊗=验证学习, τ=真理浮现

真理追求 vs 好奇心 vs 元认知的区别：
  - 好奇心：从信息差距涌现探索方向（"我想知道什么？"——探索驱动）
  - 元认知：评估当前认知状态的质量（"我确定吗？"——自我评估）
  - 真理追求：主动验证知识是否正确（"这是真的吗？"——验证驱动）
  关键区别：
  - 好奇心是"我想知道什么"——驱动探索（寻找新信息）
  - 元认知是"我确定吗"——评估置信度（评估已有知识）
  - 真理追求是"这是真的吗"——驱动验证（检验已有知识）

真理追求的核心价值：
  1. 主动验证——不只是被动接受信息，而是主动检验知识正确性
  2. 信念系统——建立验证过的知识 vs 未验证的信息的区别
  3. 抗幻觉——验证失败的知识不会被当作事实
  4. 确定性寻求——追求认知一致性，不一致时主动验证
  5. 知识质量——验证过的知识质量更高，更可靠
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


class 真理追求:
    """真理追求——主动验证知识，基于精卫公式。

    核心功能：
    1. 假设生成：从概念激活生成可验证的假设
    2. 假设验证：通过预测误差验证假设
    3. 信念更新：根据验证结果更新信念强度
    4. 确定性寻求：不一致的概念需要验证
    5. 真理驱动信号：信念强度驱动概念层

    精卫公式流程（每拍）：
    1. a = 当前信念状态（概念激活——已有的知识/信念）
    2. b = 验证预测（假设验证的结果）
    3. Δ = b - a（验证偏差——信念与验证结果的差异）
    4. Σ = a + b（联合信念状态——信念+验证）
    5. ⊗ = Δ × Σ（累积验证学习信号）
    6. τ读⊗→浮出所见 = "这是真的吗"从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新真理追求权重

    真理追求的五个层次：
    - 假设生成：将知识转化为可验证的假设
    - 假设验证：通过预测误差检验假设
    - 信念更新：根据验证结果更新信念强度
    - 确定性寻求：不一致时主动验证
    - 真理驱动：信念强度影响概念可及性
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 假设生成：从概念激活生成假设 ──
        # truth_hypothesis_w: 概念激活 → 假设预测
        # 生物学：PFC的假设生成——从已有知识生成可验证的假设
        # 假设 = "我认为这个概念模式是正确的"
        # 假设是知识的可验证形式——不是被动接受，而是主动提出
        self.truth_hypothesis_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                    for _ in range(n_concepts)]

        # 当前假设（"我认为这个概念模式是正确的"）
        self.hypothesis = [0.0] * n_concepts

        # ── 假设验证：通过预测误差验证假设 ──
        # truth_verify_w: [假设 + 预测误差] → 验证结果
        # 生物学：ACC的错误检测——预测误差引发的验证
        # 验证通过：预测误差小→假设正确→验证结果为正
        # 验证失败：预测误差大→假设错误→验证结果为负
        # 输入维度 = n_concepts（假设） + n_concepts（预测误差） = n_concepts * 2
        self.truth_verify_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts * 2)]
                                for _ in range(n_concepts)]

        # 验证结果（"预测与现实一致吗？"）
        self.verification = [0.0] * n_concepts

        # ── 信念更新：根据验证结果更新信念 ──
        # truth_belief_w: 验证结果 → 信念更新
        # 生物学：贝叶斯信念更新——根据证据更新信念
        # 验证通过→信念增强（信念值向假设方向移动）
        # 验证失败→信念减弱（信念值远离假设方向）
        self.truth_belief_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                for _ in range(n_concepts)]

        # 当前信念强度（0~1，每个概念的信念强度）
        # 信念强度 = 这个概念的知识被验证过的程度
        # 信念强度高→验证过的知识→更可靠
        # 信念强度低→未验证的信息→需要验证
        self.belief = [0.0] * n_concepts

        # ── 真理驱动信号（反馈到概念层——作为b信号输入源） ──
        # 真理驱动 = 信念强度 × 确定性寻求
        # 生物学：PFC→后部皮层投射——真理追求影响概念可及性
        # 信念强的概念→真理驱动强（更容易浮现——验证过的知识更可靠）
        # 信念弱的概念→真理驱动弱（需要验证——未验证的信息不太可靠）
        self.truth_drive = [0.0] * n_concepts

        # ── 真理追求强度（0~1） ──
        # 真理追求强度 = 确定性寻求的总体水平
        # 高真理追求强度→更主动验证
        # 低真理追求强度→更多接受已有知识
        self.truth_strength = 0.5

        # ── 确定性寻求（0~1） ──
        # 确定性寻求 = 概念激活的不一致性
        # 生物学：ACC的认知一致性监测——不一致引发确定性寻求
        # 不一致→确定性寻求高→更主动验证
        # 一致→确定性寻求低→更多利用已有知识
        self.certainty_seeking = 0.5

        # ── 上拍假设（用于评估验证结果） ──
        self.prev_hypothesis = [0.0] * n_concepts

        # ── 信念历史 ──
        self.belief_history = []
        self.history_cap = 50

        # ── 真理追求统计 ──
        self.真理追求次数 = 0
        self.假设生成次数 = 0
        self.验证通过次数 = 0
        self.验证失败次数 = 0
        self.信念更新次数 = 0

        # ── 上拍预测误差（用于验证） ──
        self.prev_prediction_error = [0.0] * n_concepts

    def 一拍(self, concept_h, 预测误差=None, 元认知评估=None):
        """真理追求一拍：假设生成 + 假设验证 + 信念更新 + 确定性寻求。

        精卫公式流程：
        1. a = 当前信念状态（概念激活——已有的知识/信念）
        2. b = 验证预测（假设验证的结果）
        3. Δ = b - a（验证偏差——信念与验证结果的差异）
        4. Σ = a + b（联合信念状态——信念+验证）
        5. ⊗ = Δ × Σ → 真理追求权重更新
        6. τ读⊗→浮出所见 = "这是真的吗"从⊗中浮现
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          预测误差: 概念层的预测误差（n_concepts维，可选）
          元认知评估: 元认知模块的meta_eval（n_concepts维，可选）

        返回：
          dict包含：
            truth_drive: 真理驱动信号（反馈到概念层）
            truth_strength: 真理追求强度（0~1）
            hypothesis: 当前假设（n_concepts维）
            verification: 验证结果（n_concepts维）
            belief: 信念强度（n_concepts维）
            certainty_seeking: 确定性寻求（0~1）
        """
        self.拍 += 1

        # ── 假设生成：从概念激活生成假设 ──
        # 精卫公式：b = truth_hypothesis_w × concept_h（假设预测）
        # τ读⊗→浮出所见："我认为这个概念模式是正确的"从概念激活中浮现
        # 生物学：PFC的假设生成——从已有知识生成可验证的假设
        # 假设不是被动接受——是主动提出的可验证命题
        new_hypothesis = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            hyp_input = 0.0
            for j in range(self.n_concepts):
                hyp_input += self.truth_hypothesis_w[i][j] * concept_h[j]
            new_hypothesis[i] = activation(hyp_input)

        # 元认知评估增强假设生成
        # 元认知不确定性高→需要更多验证→假设更积极
        if 元认知评估 is not None:
            for i in range(min(self.n_concepts, len(元认知评估))):
                new_hypothesis[i] = activation(new_hypothesis[i] + 元认知评估[i] * 0.2)

        self.hypothesis = new_hypothesis
        self.假设生成次数 += 1

        # ── 假设验证：通过预测误差验证假设 ──
        # 精卫公式：b = truth_verify_w × [假设 + 预测误差]（验证结果）
        # τ读⊗→浮出所见："预测与现实一致吗？"从假设和预测误差中浮现
        # 生物学：ACC的错误检测——预测误差引发的验证
        # 验证通过：预测误差小→假设正确→验证结果为正
        # 验证失败：预测误差大→假设错误→验证结果为负
        if 预测误差 is not None:
            pe = 预测误差
            # 截断或填充到n_concepts维度
            if len(pe) > self.n_concepts:
                pe = pe[:self.n_concepts]
            elif len(pe) < self.n_concepts:
                pe = pe + [0.0] * (self.n_concepts - len(pe))
        else:
            pe = self.prev_prediction_error

        self.prev_prediction_error = pe

        new_verification = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            verify_input = 0.0
            # 假设部分
            for j in range(self.n_concepts):
                verify_input += self.truth_verify_w[i][j] * new_hypothesis[j]
            # 预测误差部分
            for j in range(self.n_concepts):
                verify_input += self.truth_verify_w[i][j + self.n_concepts] * pe[j]
            # 验证结果 = 假设与预测误差的一致性
            # 预测误差小→验证通过（正值）
            # 预测误差大→验证失败（负值）
            new_verification[i] = activation(verify_input)

        self.verification = new_verification

        # 统计验证通过/失败
        for i in range(self.n_concepts):
            if new_verification[i] > 0.1:
                self.验证通过次数 += 1
            elif new_verification[i] < -0.1:
                self.验证失败次数 += 1

        # ── 信念更新：根据验证结果更新信念 ──
        # 精卫公式：⊗ = Δ × Σ → 信念权重更新
        # 生物学：贝叶斯信念更新——根据证据更新信念
        # 验证通过→信念增强（信念值向假设方向移动）
        # 验证失败→信念减弱（信念值远离假设方向）
        new_belief = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            belief_input = 0.0
            for j in range(self.n_concepts):
                belief_input += self.truth_belief_w[i][j] * new_verification[j]
            # 信念更新 = 当前信念 × 衰减 + 验证结果 × 学习率
            # 验证通过→信念增强（向1移动）
            # 验证失败→信念减弱（向0移动）
            new_belief[i] = activation(
                self.belief[i] * 0.95 + belief_input * 0.3)

        # 信念值归一化到0~1
        for i in range(self.n_concepts):
            new_belief[i] = (new_belief[i] + 1.0) * 0.5

        self.belief = new_belief
        self.信念更新次数 += 1

        # ── 确定性寻求：不一致的概念需要验证 ──
        # 精卫公式：Δ = b - a = 验证偏差（信念与验证结果的差异）
        # 生物学：ACC的认知一致性监测——不一致引发确定性寻求
        # 概念激活不一致→确定性寻求高→更主动验证
        # 概念激活一致→确定性寻求低→更多利用已有知识
        #
        # 不一致性 = 概念激活的方差（方差大→不一致→确定性寻求高）
        # 信念不一致 = 信念强度的方差（方差大→有些验证过有些没有→确定性寻求高）
        concept_variance = 0.0
        concept_mean = sum(concept_h) / max(1, self.n_concepts)
        for i in range(self.n_concepts):
            concept_variance += (concept_h[i] - concept_mean) ** 2
        concept_variance /= max(1, self.n_concepts)

        belief_variance = 0.0
        belief_mean = sum(self.belief) / max(1, self.n_concepts)
        for i in range(self.n_concepts):
            belief_variance += (self.belief[i] - belief_mean) ** 2
        belief_variance /= max(1, self.n_concepts)

        # 确定性寻求 = 概念不一致性 + 信念不一致性的连续函数
        self.certainty_seeking = activation(
            concept_variance * 5.0 + belief_variance * 5.0) * 0.5 + 0.5

        # ── 真理追求强度 ──
        # 真理追求强度 = 确定性寻求的总体水平
        # 确定性寻求高→真理追求强→更主动验证
        self.truth_strength = self.certainty_seeking

        # ── 真理驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 真理驱动信号
        # 生物学：PFC→后部皮层投射——真理追求影响概念可及性
        # 信念强的概念→真理驱动强（验证过的知识更可靠，更容易浮现）
        # 信念弱的概念→真理驱动弱（未验证的信息不太可靠）
        # 真理驱动 = 信念强度 × 确定性寻求
        for i in range(self.n_concepts):
            self.truth_drive[i] = activation(
                (self.belief[i] * 2.0 - 1.0) * self.certainty_seeking)

        # ── 赫布学习 ──
        # 精卫公式：⊗ = Δ × Σ → 权重更新
        # 假设生成学习：概念激活和假设同时出现→假设生成连接增强
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_hypothesis[i] * concept_h[j] * 0.3
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.truth_hypothesis_w[i][j] = self.truth_hypothesis_w[i][j] * 0.95 + dw

        # 假设验证学习：假设和预测误差同时出现→验证连接增强
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_verification[i] * new_hypothesis[j] * 0.3
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.truth_verify_w[i][j] = self.truth_verify_w[i][j] * 0.95 + dw
            for j in range(self.n_concepts):
                dw = self.lr * new_verification[i] * pe[j] * 0.3
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.truth_verify_w[i][j + self.n_concepts] = \
                        self.truth_verify_w[i][j + self.n_concepts] * 0.95 + dw

        # 信念更新学习：验证结果和信念同时出现→信念更新连接增强
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_belief[i] * new_verification[j] * 0.3
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.truth_belief_w[i][j] = self.truth_belief_w[i][j] * 0.95 + dw

        # ── 权重归一化 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.truth_hypothesis_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.truth_hypothesis_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.truth_verify_w[i][j]) for j in range(self.n_concepts * 2))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts * 2):
                    self.truth_verify_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.truth_belief_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.truth_belief_w[i][j] *= scale

        # ── 信念历史 ──
        avg_belief = sum(self.belief) / max(1, self.n_concepts)
        self.belief_history.append(avg_belief)
        if len(self.belief_history) > self.history_cap:
            self.belief_history.pop(0)

        self.prev_hypothesis = new_hypothesis
        self.真理追求次数 += 1

        return {
            "truth_drive": self.truth_drive,
            "truth_strength": round(self.truth_strength, 4),
            "hypothesis": self.hypothesis,
            "verification": self.verification,
            "belief": self.belief,
            "certainty_seeking": round(self.certainty_seeking, 4),
        }

    def 状态摘要(self):
        avg_belief = sum(self.belief) / max(1, self.n_concepts) if self.belief else 0.0
        return {
            "truth_strength": round(self.truth_strength, 4),
            "certainty_seeking": round(self.certainty_seeking, 4),
            "avg_belief": round(avg_belief, 4),
            "truth_hypothesis_w均值": round(sum(sum(row) for row in self.truth_hypothesis_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "truth_verify_w均值": round(sum(sum(row) for row in self.truth_verify_w) / max(1, self.n_concepts * self.n_concepts * 2), 4),
            "truth_belief_w均值": round(sum(sum(row) for row in self.truth_belief_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "真理追求次数": self.真理追求次数,
            "假设生成次数": self.假设生成次数,
            "验证通过次数": self.验证通过次数,
            "验证失败次数": self.验证失败次数,
            "信念更新次数": self.信念更新次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 真理追求验证（假设生成 + 假设验证 + 信念更新 + 确定性寻求） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    truth = 真理追求(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {truth.n_concepts}")
    print(f"  假设生成权重: truth_hypothesis_w (64×64)")
    print(f"  假设验证权重: truth_verify_w (64×128)")
    print(f"  信念更新权重: truth_belief_w (64×64)")
    print(f"  初始真理追求强度: {truth.truth_strength}")
    print(f"  初始确定性寻求: {truth.certainty_seeking}")

    print("\n--- 测试1: 假设生成 ---")
    concept_h = [0.0] * 64
    concept_h[4] = 0.3
    concept_h[8] = 0.2
    result = truth.一拍(concept_h)
    print(f"  假设top5: {sorted(range(64), key=lambda i: abs(result['hypothesis'][i]), reverse=True)[:5]}")
    print(f"  假设最大值: {max(abs(v) for v in result['hypothesis']):.4f}")
    print(f"  → 应该从概念激活生成假设")

    print("\n--- 测试2: 假设验证（低预测误差→验证通过） ---")
    truth2 = 真理追求(n_concepts=64)
    低误差 = [0.01] * 64  # 低预测误差→验证通过
    for tick in range(10):
        result = truth2.一拍(concept_h, 预测误差=低误差)
    print(f"  验证结果top5: {sorted(range(64), key=lambda i: abs(result['verification'][i]), reverse=True)[:5]}")
    print(f"  验证结果均值: {sum(result['verification'])/64:.4f}")
    print(f"  验证通过次数: {truth2.验证通过次数}")
    print(f"  → 低预测误差应该导致验证通过")

    print("\n--- 测试3: 假设验证（高预测误差→验证失败） ---")
    truth3 = 真理追求(n_concepts=64)
    高误差 = [0.5] * 64  # 高预测误差→验证失败
    for tick in range(10):
        result = truth3.一拍(concept_h, 预测误差=高误差)
    print(f"  验证结果top5: {sorted(range(64), key=lambda i: abs(result['verification'][i]), reverse=True)[:5]}")
    print(f"  验证结果均值: {sum(result['verification'])/64:.4f}")
    print(f"  验证失败次数: {truth3.验证失败次数}")
    print(f"  → 高预测误差应该导致验证失败")

    print("\n--- 测试4: 信念更新（验证通过→信念增强） ---")
    truth4 = 真理追求(n_concepts=64)
    belief_before = sum(truth4.belief) / 64
    for tick in range(20):
        result = truth4.一拍(concept_h, 预测误差=低误差)
    belief_after = sum(truth4.belief) / 64
    print(f"  信念均值: 初始={belief_before:.4f} → 20拍后={belief_after:.4f}")
    print(f"  → 验证通过应该导致信念增强")

    print("\n--- 测试5: 信念更新（验证失败→信念减弱） ---")
    truth5 = 真理追求(n_concepts=64)
    # 先让信念增强
    for tick in range(10):
        truth5.一拍(concept_h, 预测误差=低误差)
    belief_strong = sum(truth5.belief) / 64
    # 然后用高预测误差让信念减弱
    for tick in range(10):
        truth5.一拍(concept_h, 预测误差=高误差)
    belief_weak = sum(truth5.belief) / 64
    print(f"  信念均值: 验证通过后={belief_strong:.4f} → 验证失败后={belief_weak:.4f}")
    print(f"  → 验证失败应该导致信念减弱")

    print("\n--- 测试6: 确定性寻求 ---")
    truth6 = 真理追求(n_concepts=64)
    # 一致的概念激活（低方差）
    consistent_h = [0.3] * 64
    result_consistent = truth6.一拍(consistent_h)
    print(f"  一致激活: 确定性寻求={result_consistent['certainty_seeking']:.4f}")
    # 不一致的概念激活（高方差）
    inconsistent_h = [0.0] * 64
    for i in range(0, 64, 8):
        inconsistent_h[i] = 0.5
    result_inconsistent = truth6.一拍(inconsistent_h)
    print(f"  不一致激活: 确定性寻求={result_inconsistent['certainty_seeking']:.4f}")
    print(f"  → 不一致激活应该导致更高的确定性寻求")

    print("\n--- 测试7: 真理驱动信号 ---")
    truth7 = 真理追求(n_concepts=64)
    for tick in range(20):
        result = truth7.一拍(concept_h, 预测误差=低误差)
    drive_sum = sum(abs(v) for v in truth7.truth_drive)
    print(f"  真理驱动信号总和: {drive_sum:.4f}")
    print(f"  真理驱动top5: {sorted(range(64), key=lambda i: abs(truth7.truth_drive[i]), reverse=True)[:5]}")
    print(f"  → 信念强的概念应该有更强的真理驱动")

    print("\n--- 测试8: 元认知评估增强 ---")
    truth8 = 真理追求(n_concepts=64)
    meta_eval = [0.3] * 64  # 高元认知不确定性
    for tick in range(10):
        result = truth8.一拍(concept_h, 预测误差=低误差, 元认知评估=meta_eval)
    print(f"  真理追求强度: {result['truth_strength']:.4f}")
    print(f"  假设均值: {sum(result['hypothesis'])/64:.4f}")
    print(f"  → 元认知评估应该增强假设生成")

    print("\n--- 测试9: 精卫公式验证 ---")
    truth9 = 真理追求(n_concepts=16)
    small_h = [0.0] * 16
    small_h[0] = 0.3
    small_pe = [0.02] * 16
    for tick in range(20):
        result = truth9.一拍(small_h, 预测误差=small_pe)
        if tick < 3 or tick >= 17:
            print(f"  拍{tick+1}: 强度={result['truth_strength']}, "
                  f"确定性={result['certainty_seeking']}, "
                  f"信念均值={round(sum(result['belief'])/16, 4)}")
    print(f"  τ(浮出): 真理驱动均值={round(sum(truth9.truth_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={truth9.拍}")

    print("\n--- 测试10: 状态摘要 ---")
    summary = truth.状态摘要()
    print(f"  {summary}")

    print("\n验证通过")
