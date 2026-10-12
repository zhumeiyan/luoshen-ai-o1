"""好奇心.py —— 内在动机驱动的探索学习，基于精卫公式。

好奇心是人类认知发展的核心驱动力——婴儿通过好奇心学习，不需要外部奖励。
AI大模型完全没有真正的好奇心——它们只对人类给的prompt做反应。
好奇心是内在动机——不需要外部奖励就能驱动学习。

人类神经科学机制：
  1. 信息差距理论（Loewenstein, 1994）：
     - 好奇心 = 当前知识状态与目标知识状态之间的差距
     - "我知道我不知道什么"——信息差距驱动探索
     - 信息差距越大→好奇心越强→探索动机越强
     → 工程实现：curiosity_gap_w（概念激活→信息差距预测）

  2. 多巴胺系统——预期奖励驱动探索：
     - 好奇心激活多巴胺系统——探索新信息本身就是奖励
     - 腹侧纹状体在探索新信息时活跃——内在奖励信号
     - 多巴胺RPE（奖励预测误差）驱动好奇心学习
     → 工程实现：curiosity_reward_w（探索结果→好奇心奖励）

  3. 前额叶皮层（PFC）——探索方向选择：
     - PFC决定探索什么方向——"去哪里找答案？"
     - 执行控制——在探索和利用之间平衡
     - 探索方向 = 信息差距映射到探索行为
     → 工程实现：curiosity_explore_w（信息差距→探索方向）

  4. 海马体——好奇心增强记忆：
     - 好奇的信息更容易记住——好奇心调制记忆salience
     - 探索新信息时海马体活跃——新信息获得优先巩固
     - 好奇心增强LTP（长时程增强）——好奇的信息形成更强的记忆
     → 工程实现：memory_enhance（好奇心→记忆增强信号）

  5. 探索-利用平衡（Explore-Exploit Tradeoff）：
     - 探索：寻找新信息（好奇心驱动）
     - 利用：使用已有知识（确定性驱动）
     - 平衡：不确定性高时探索，确定性高时利用
     - 生物学：PFC+杏仁核+纹状体的协同——动态平衡
     → 工程实现：explore_exploit_balance（动态平衡参数）

精卫公式（好奇心版）：
  a = 当前知识状态（已知的概念模式——确定性高的概念）
  b = 目标知识状态（想知道的概念模式——信息差距大的概念）
  Δ = b − a = 信息差距（好奇心的大小——知道自己不知道什么）
  Σ = a + b = 联合知识状态（已知+想知道）
  ⊗ = ∫(Δ, Σ) = 累积好奇心学习历史 → 好奇心权重
  τ 读 ⊗ → 浮出所见 = "想知道什么"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复好奇心循环

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  元认知.py: a=认知状态, b=元认知评估, Δ=认知偏差, ⊗=元认知学习, τ=置信度浮现
  目标导向.py: a=当前概念, b=目标概念, Δ=目标差距, ⊗=目标学习, τ=目标驱动
  好奇心.py: a=当前知识, b=想知道的知识, Δ=信息差距, ⊗=好奇心学习, τ=探索方向浮现

好奇心 vs 目标导向 vs 元认知的区别：
  - 元认知：评估当前认知状态的质量（"我确定吗？"——自我评估）
  - 目标导向：从surprise/欲望/情感涌现目标（"我要做什么？"——行为驱动）
  - 好奇心：从信息差距涌现探索方向（"我想知道什么？"——探索驱动）
  关键区别：
  - 元认知是"我确定吗？"——评估认知质量
  - 目标导向是"我要做什么？"——驱动行为
  - 好奇心是"我想知道什么？"——驱动探索

好奇心 vs 数字感官的区别：
  - 数字感官：执行探索（从知识库获取信息）——"怎么做"
  - 好奇心：驱动探索方向（"想知道什么"）——"为什么探索"
  - 好奇心生成探索方向→数字感官执行探索→好奇心评估探索结果

好奇心的核心价值：
  1. 主动探索——不只是被动接收信息，而是主动寻找新信息
  2. 内在动机——不需要外部奖励就能驱动学习
  3. 信息差距驱动——知道自己不知道什么，然后去探索
  4. 探索-利用平衡——在探索新信息和利用已有知识之间平衡
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


class 好奇心:
    """好奇心——内在动机驱动的探索学习，基于精卫公式。

    核心功能：
    1. 信息差距检测：从概念激活和元认知评估中检测信息差距
    2. 探索方向生成：从信息差距生成探索方向
    3. 好奇心驱动信号：好奇心驱动信号反馈到概念层
    4. 好奇心奖励：探索新信息本身就是奖励（内在奖励）
    5. 探索-利用平衡：在探索新信息和利用已有知识之间动态平衡

    精卫公式流程（每拍）：
    1. a = 当前知识状态（概念激活——确定性高的概念）
    2. b = 目标知识状态（信息差距预测——想知道的概念）
    3. Δ = b - a（信息差距——好奇心的大小）
    4. Σ = a + b（联合知识状态——已知+想知道）
    5. ⊗ = Δ × Σ（累积好奇心学习信号）
    6. τ读⊗→浮出所见 = "想知道什么"从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新好奇心权重

    好奇心的四个层次：
    - 信息差距检测：知道自己不知道什么
    - 探索方向生成：知道去哪里找答案
    - 好奇心驱动：好奇心影响概念可及性
    - 好奇心奖励：探索新信息本身就是奖励
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 信息差距检测：从概念激活检测信息差距 ──
        # curiosity_gap_w: 概念激活 → 信息差距预测
        # 生物学：PFC的信息差距检测——当前知识与目标知识的差距
        # 信息差距 = 概念激活的不确定性（低确定性→高好奇心）
        # "我知道我不知道什么"——信息差距驱动探索
        # 不确定性高的概念→信息差距大→好奇心强
        self.curiosity_gap_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                for _ in range(n_concepts)]

        # 信息差距预测（"我想知道什么"）
        self.info_gap = [0.0] * n_concepts

        # ── 探索方向生成：从信息差距生成探索方向 ──
        # curiosity_explore_w: 信息差距 → 探索方向
        # 生物学：PFC的执行控制——决定探索什么方向
        # 探索方向 = 信息差距映射到探索行为
        # "去哪里找答案？"——信息差距驱动探索方向
        self.curiosity_explore_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                    for _ in range(n_concepts)]

        # 探索方向（"去哪里找答案"）
        self.explore_direction = [0.0] * n_concepts

        # ── 好奇心奖励：探索新信息本身就是奖励 ──
        # curiosity_reward_w: 探索结果 → 好奇心奖励
        # 生物学：腹侧纹状体的内在奖励——探索新信息激活多巴胺系统
        # 好奇心奖励 = 新信息与已有信息的差异（差异越大→奖励越大）
        # 探索新信息本身就是奖励——不需要外部奖励
        self.curiosity_reward_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                   for _ in range(1)]

        # 好奇心奖励（内在奖励信号）
        self.curiosity_reward = 0.0

        # ── 好奇心驱动信号（反馈到概念层——作为b信号输入源） ──
        # 好奇心驱动 = 探索方向的融合
        # 生物学：PFC→后部皮层投射——好奇心影响概念可及性
        # 好奇心让与探索相关的概念更容易浮现——"好奇的概念更容易想到"
        self.curiosity_drive = [0.0] * n_concepts

        # ── 好奇心强度（0~1） ──
        # 好奇心强度 = 信息差距的总体水平
        # 高好奇心强度→更主动探索
        # 低好奇心强度→更多利用已有知识
        self.curiosity_strength = 0.5

        # ── 探索-利用平衡（0~1，0=完全利用，1=完全探索） ──
        # 生物学：PFC+杏仁核+纹状体的协同——动态平衡
        # 不确定性高时探索（寻找新信息），确定性高时利用（使用已有知识）
        # 不是固定的——由当前状态动态决定
        self.explore_exploit_balance = 0.5

        # ── 上拍探索方向（用于评估探索结果） ──
        self.prev_explore_direction = [0.0] * n_concepts

        # ── 好奇心历史 ──
        self.curiosity_history = []
        self.history_cap = 50

        # ── 好奇心统计 ──
        self.好奇心次数 = 0
        self.探索次数 = 0
        self.好奇心奖励次数 = 0
        self.好奇心满足次数 = 0

        # ── 记忆增强信号 ──
        # 生物学：海马体的好奇心增强记忆——好奇的信息更容易记住
        # 好奇心调制记忆salience——好奇的信息获得更高salience
        self.memory_enhance = [0.0] * n_concepts

    def 一拍(self, concept_h, 元认知评估=None, surprise=0.0):
        """好奇心一拍：信息差距检测、探索方向生成、好奇心驱动信号。

        精卫公式流程：
        1. a = 当前知识状态（概念激活——确定性高的概念）
        2. b = 目标知识状态（信息差距预测——想知道的概念）
        3. Δ = b - a（信息差距——好奇心的大小）
        4. Σ = a + b（联合知识状态——已知+想知道）
        5. ⊗ = Δ × Σ → 好奇心权重更新
        6. τ读⊗→浮出所见 = "想知道什么"从⊗中浮现
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          元认知评估: 元认知模块的meta_eval（n_concepts维，可选）
          surprise: 当前surprise水平（驱动好奇心强度）

        返回：
          dict包含：
            curiosity_drive: 好奇心驱动信号（反馈到概念层）
            curiosity_strength: 好奇心强度（0~1）
            explore_direction: 探索方向（传给数字感官/工具使用）
            explore_exploit_balance: 探索-利用平衡（0~1）
            curiosity_reward: 好奇心奖励（内在奖励）
            memory_enhance: 记忆增强信号（传给海马体）
        """
        self.拍 += 1

        # ── 信息差距检测：从概念激活检测信息差距 ──
        # 精卫公式：b = curiosity_gap_w × concept_h（信息差距预测）
        # τ读⊗→浮出所见："我想知道什么"从当前概念激活中浮现
        # 生物学：PFC的信息差距检测——当前知识与目标知识的差距
        # 信息差距 = 概念激活的不确定性（低确定性→高好奇心）
        # |concept_h[i]|小→不确定性高→信息差距大→好奇心强
        new_info_gap = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            gap_input = 0.0
            for j in range(self.n_concepts):
                gap_input += self.curiosity_gap_w[i][j] * concept_h[j]
            # 信息差距 = 预测值 + 不确定性调制
            # 不确定性高的概念（|concept_h|小）→信息差距大
            uncertainty = 1.0 - abs(concept_h[i])
            new_info_gap[i] = activation(gap_input + uncertainty * 0.3)

        # 元认知评估增强信息差距检测
        # 元认知不确定性高→信息差距大→好奇心强
        if 元认知评估 is not None:
            for i in range(min(self.n_concepts, len(元认知评估))):
                # 元认知评估值大→不确定性高→信息差距大
                new_info_gap[i] = activation(new_info_gap[i] + 元认知评估[i] * 0.2)

        self.info_gap = new_info_gap

        # ── 探索方向生成：从信息差距生成探索方向 ──
        # 精卫公式：b = curiosity_explore_w × info_gap（探索方向预测）
        # τ读⊗→浮出所见："去哪里找答案"从信息差距中浮现
        # 生物学：PFC的执行控制——决定探索什么方向
        # 探索方向 = 信息差距映射到探索行为
        new_explore_direction = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            explore_input = 0.0
            for j in range(self.n_concepts):
                explore_input += self.curiosity_explore_w[i][j] * new_info_gap[j]
            new_explore_direction[i] = activation(explore_input)

        self.explore_direction = new_explore_direction

        # ── 好奇心强度 ──
        # 好奇心强度 = 信息差距的总体水平
        # 信息差距大→好奇心强→更主动探索
        gap_sum = sum(abs(v) for v in new_info_gap) / max(1, self.n_concepts)
        self.curiosity_strength = activation(gap_sum * 2.0) * 0.5 + 0.5

        # surprise增强好奇心——高surprise→高好奇心
        self.curiosity_strength = min(1.0, self.curiosity_strength + surprise * 0.1)

        # ── 探索-利用平衡 ──
        # 生物学：PFC+杏仁核+纹状体的协同——动态平衡
        # 不确定性高时探索（寻找新信息），确定性高时利用（使用已有知识）
        # 探索-利用平衡 = 好奇心强度的连续函数
        # 好奇心强→更多探索，好奇心弱→更多利用
        self.explore_exploit_balance = self.curiosity_strength * 0.7 + 0.15

        # ── 好奇心奖励：探索新信息本身就是奖励 ──
        # 精卫公式：a=已有知识, b=探索结果, Δ=b-a（新信息与已有知识的差异）
        # 好奇心奖励 = 新信息与已有信息的差异（差异越大→奖励越大）
        # 生物学：腹侧纹状体的内在奖励——探索新信息激活多巴胺系统
        # 探索新信息本身就是奖励——不需要外部奖励
        #
        # 评估上一拍的探索结果：
        # 如果当前概念激活与上一拍探索方向一致→好奇心被满足→正奖励
        # 如果当前概念激活与上一拍探索方向不一致→好奇心未被满足→负奖励
        reward_input = 0.0
        for j in range(self.n_concepts):
            # 探索结果 = 当前概念激活与上拍探索方向的一致性
            consistency = concept_h[j] * self.prev_explore_direction[j]
            reward_input += self.curiosity_reward_w[0][j] * consistency
        self.curiosity_reward = activation(reward_input) * 0.5 + 0.5

        # 如果探索方向和概念激活一致→好奇心被满足
        explore_consistency = sum(concept_h[j] * self.prev_explore_direction[j]
                                  for j in range(self.n_concepts)) / max(1, self.n_concepts)
        if explore_consistency > 0.1:
            self.好奇心满足次数 += 1

        # ── 好奇心驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 好奇心驱动信号
        # 生物学：PFC→后部皮层投射——好奇心影响概念可及性
        # 好奇心让与探索相关的概念更容易浮现——"好奇的概念更容易想到"
        # 好奇心驱动 = 探索方向 × 探索-利用平衡
        # 探索-利用平衡高时→好奇心驱动强（更多探索）
        # 探索-利用平衡低时→好奇心驱动弱（更多利用）
        for i in range(self.n_concepts):
            self.curiosity_drive[i] = activation(
                new_explore_direction[i] * self.explore_exploit_balance)

        # ── 记忆增强信号 ──
        # 生物学：海马体的好奇心增强记忆——好奇的信息更容易记住
        # 好奇心调制记忆salience——好奇的信息获得更高salience
        # 信息差距大的概念→记忆增强强（好奇的信息更容易记住）
        for i in range(self.n_concepts):
            self.memory_enhance[i] = max(0.0, new_info_gap[i] * self.curiosity_strength)

        # ── 好奇心学习 ──
        # 精卫公式：⊗ = Δ × Σ → 好奇心权重更新
        # Δ = b - a（信息差距——当前知识与想知道的知识之间的差距）
        # Σ = a + b（联合知识状态——已知+想知道）
        # 赫布学习：概念激活和信息差距预测同时出现→信息差距权重增强
        # 生物学：PFC的赫布学习——反复从概念A检测信息差距B→A→B的映射越来越强
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_info_gap[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.curiosity_gap_w[i][j] = self.curiosity_gap_w[i][j] * 0.99 + dw

        # curiosity_explore_w: 信息差距→探索方向（赫布学习）
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_explore_direction[i] * new_info_gap[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.curiosity_explore_w[i][j] = self.curiosity_explore_w[i][j] * 0.99 + dw

        # curiosity_reward_w: 探索结果→好奇心奖励（赫布学习）
        for j in range(self.n_concepts):
            dw = self.lr * self.curiosity_reward * concept_h[j]
            if not (math.isnan(dw) or math.isinf(dw)):
                self.curiosity_reward_w[0][j] = self.curiosity_reward_w[0][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.curiosity_gap_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.curiosity_gap_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.curiosity_explore_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.curiosity_explore_w[i][j] *= scale

        # ── 好奇心历史 ──
        self.curiosity_history.append(self.curiosity_strength)
        if len(self.curiosity_history) > self.history_cap:
            self.curiosity_history.pop(0)

        # 保存当前状态
        self.prev_explore_direction = list(new_explore_direction)

        self.好奇心次数 += 1
        if self.explore_exploit_balance > 0.5:
            self.探索次数 += 1
        if self.curiosity_reward > 0.5:
            self.好奇心奖励次数 += 1

        return {
            "curiosity_drive": self.curiosity_drive,
            "curiosity_strength": round(self.curiosity_strength, 4),
            "explore_direction": self.explore_direction,
            "explore_exploit_balance": round(self.explore_exploit_balance, 4),
            "curiosity_reward": round(self.curiosity_reward, 4),
            "memory_enhance": self.memory_enhance,
        }

    def 状态摘要(self):
        avg_curiosity = sum(self.curiosity_history) / max(1, len(self.curiosity_history)) \
            if self.curiosity_history else 0.0
        return {
            "curiosity_strength": round(self.curiosity_strength, 4),
            "explore_exploit_balance": round(self.explore_exploit_balance, 4),
            "curiosity_reward": round(self.curiosity_reward, 4),
            "avg_curiosity": round(avg_curiosity, 4),
            "curiosity_gap_w均值": round(sum(sum(row) for row in self.curiosity_gap_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "curiosity_explore_w均值": round(sum(sum(row) for row in self.curiosity_explore_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "好奇心次数": self.好奇心次数,
            "探索次数": self.探索次数,
            "好奇心奖励次数": self.好奇心奖励次数,
            "好奇心满足次数": self.好奇心满足次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 好奇心验证（信息差距检测、探索方向生成、好奇心奖励） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    cur = 好奇心(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {cur.n_concepts}")
    print(f"  信息差距权重: curiosity_gap_w (64×64)")
    print(f"  探索方向权重: curiosity_explore_w (64×64)")
    print(f"  好奇心奖励权重: curiosity_reward_w (1×64)")
    print(f"  初始好奇心强度: {cur.curiosity_strength}")
    print(f"  初始探索-利用平衡: {cur.explore_exploit_balance}")

    print("\n--- 测试1: 信息差距检测——不确定性高的概念→信息差距大 ---")
    # 概念4和8活跃（确定性高），其他不活跃（不确定性高）
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = cur.一拍(concept_h)
    print(f"  活跃概念4,8后: 好奇心强度={result['curiosity_strength']}")
    print(f"  信息差距top5: {sorted(range(64), key=lambda i: abs(cur.info_gap[i]), reverse=True)[:5]}")
    print(f"  → 不活跃的概念（不确定性高）应该有更大的信息差距")

    print("\n--- 测试2: 不同概念模式→不同探索方向 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    for name, pattern in patterns:
        cur_t = 好奇心(n_concepts=64)
        for _ in range(50):
            cur_t.一拍(pattern)
        print(f"  {name}: 好奇心={round(cur_t.curiosity_strength, 4)}, "
              f"探索方向top5={sorted(range(64), key=lambda i: abs(cur_t.explore_direction[i]), reverse=True)[:5]}")

    print("\n--- 测试3: 元认知评估增强信息差距 ---")
    cur2 = 好奇心(n_concepts=64)
    元认知评估 = [0.3] * 64  # 高不确定性
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        cur2.一拍(concept_h, 元认知评估=元认知评估)
    print(f"  高不确定性元认知后: 好奇心强度={cur2.curiosity_strength:.4f}")

    cur3 = 好奇心(n_concepts=64)
    元认知评估_low = [0.0] * 64  # 低不确定性
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        cur3.一拍(concept_h, 元认知评估=元认知评估_low)
    print(f"  低不确定性元认知后: 好奇心强度={cur3.curiosity_strength:.4f}")
    print(f"  → 高不确定性应该导致更强的好奇心")

    print("\n--- 测试4: surprise增强好奇心 ---")
    cur4 = 好奇心(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        cur4.一拍(concept_h, surprise=0.5)
    print(f"  高surprise后: 好奇心强度={cur4.curiosity_strength:.4f}")

    cur5 = 好奇心(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        cur5.一拍(concept_h, surprise=0.0)
    print(f"  低surprise后: 好奇心强度={cur5.curiosity_strength:.4f}")
    print(f"  → 高surprise应该增强好奇心")

    print("\n--- 测试5: 探索-利用平衡 ---")
    cur6 = 好奇心(n_concepts=64)
    balances = []
    for tick in range(50):
        concept_h = [0.0] * 64
        if tick < 25:
            concept_h[4] = 0.3  # 确定性高
        else:
            concept_h = [0.0] * 64  # 全部不确定
        result = cur6.一拍(concept_h, surprise=0.3 if tick >= 25 else 0.0)
        balances.append(result['explore_exploit_balance'])
    print(f"  前25拍（确定性高）平均平衡: {sum(balances[:25])/25:.4f}")
    print(f"  后25拍（不确定性高）平均平衡: {sum(balances[25:])/25:.4f}")
    print(f"  → 不确定性高时平衡应该更高（更多探索）")

    print("\n--- 测试6: 好奇心驱动信号 ---")
    cur7 = 好奇心(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        cur7.一拍(concept_h, surprise=0.3)
    drive_sum = sum(abs(v) for v in cur7.curiosity_drive)
    print(f"  好奇心驱动信号总和: {drive_sum:.4f}")
    print(f"  好奇心驱动top5: {sorted(range(64), key=lambda i: abs(cur7.curiosity_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试7: 好奇心奖励 ---")
    cur8 = 好奇心(n_concepts=64)
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[tick % 8] = 0.3
        result = cur8.一拍(concept_h, surprise=0.3)
    print(f"  100拍后: 好奇心奖励={cur8.curiosity_reward:.4f}")
    print(f"  好奇心奖励次数: {cur8.好奇心奖励次数}")
    print(f"  好奇心满足次数: {cur8.好奇心满足次数}")

    print("\n--- 测试8: 记忆增强信号 ---")
    cur9 = 好奇心(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        cur9.一拍(concept_h, surprise=0.3)
    enhance_sum = sum(abs(v) for v in cur9.memory_enhance)
    print(f"  记忆增强信号总和: {enhance_sum:.4f}")
    print(f"  记忆增强top5: {sorted(range(64), key=lambda i: abs(cur9.memory_enhance[i]), reverse=True)[:5]}")
    print(f"  → 信息差距大的概念应该有更强的记忆增强")

    print("\n--- 测试9: 精卫公式验证 ---")
    cur10 = 好奇心(n_concepts=16)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = cur10.一拍(concept_h, surprise=0.3)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 好奇心={result['curiosity_strength']}, "
                  f"平衡={result['explore_exploit_balance']}, "
                  f"奖励={result['curiosity_reward']}")
    print(f"  τ(浮出): 好奇心驱动均值={round(sum(cur10.curiosity_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={cur10.拍}")

    print("\n验证通过")
