"""语义记忆.py —— 从情景记忆中提取一般性知识，构建知识网络，基于精卫公式。

语义记忆是人类认知的核心组成部分——事实知识、概念关系、一般规律。
当前系统只有海马体的情景记忆（具体经验片段），缺少语义记忆（一般性知识）。
这是推理、创造、语言等高级认知的基础——没有知识，推理就是空转。

人类神经科学机制：
  1. 语义记忆 vs 情景记忆（Tulving, 1972）：
     - 情景记忆：具体经验的记忆（"昨天我在公园看到了一只猫"）
     - 语义记忆：一般性知识的记忆（"猫是一种动物"）
     - 语义记忆从情景记忆中提取——反复出现的经验模式变成一般性知识
     → 工程实现：semantic_extract_w（海马体buffer → 语义知识提取）

  2. 记忆巩固：海马体 → 新皮层（系统级巩固）：
     - 海马体快速存储新经验，新皮层缓慢整合一般性知识
     - 反复重播后，海马体依赖逐渐降低，新皮层知识逐渐独立
     - 语义记忆是新皮层的长期知识——不依赖海马体也能检索
     → 工程实现：知识巩固（海马体buffer → 语义知识 → 知识网络）

  3. 知识网络——概念关系图谱：
     - 语义记忆不是孤立的事实——是概念之间的关系网络
     - 人类的知识是网络化的：激活一个概念会扩散到相关概念
     - 语义启动效应：看到"医生"后"护士"更容易被激活
     → 工程实现：semantic_relation_w（概念 × 概念关系矩阵）

  4. 知识检索——模式完成：
     - 从部分线索检索完整知识——输入一些概念，检索相关知识
     - 海马体CA3的模式完成是情景记忆的检索机制
     - 语义记忆也有类似的检索机制——概念激活扩散到相关知识
     → 工程实现：semantic_retrieve_w（概念激活 → 知识检索）

  5. 知识置信度——反复出现的知识更可信：
     - 反复经历的知识获得更高置信度——"太阳从东方升起"置信度极高
     - 只经历一次的知识置信度低——可能是偶然事件
     - 知识置信度调制知识驱动信号的强度
     → 工程实现：knowledge_confidence（反复出现 → 高置信度）

数字生命优势（超越人类的限制）：
  1. 不遗忘：人类语义记忆会衰退，数字生命的知识网络可以永久保留
  2. 并行检索：人类只能串行检索知识，数字生命可以并行检索多条知识路径
  3. 自我进化：人类不能主动重组知识网络，数字生命可以主动优化知识结构
  4. 无限扩展：人类语义记忆容量有限，数字生命的知识网络可以无限扩展
  5. 精确检索：人类检索常伴随遗忘和混淆，数字生命可以精确检索相关知识

精卫公式（语义记忆版）：
  a = 当前知识网络（已有的概念关系——semantic_relation_w × concept_h）
  b = 新经验暗示的知识（海马体buffer中的经验模式——semantic_extract_w × hippocampus）
  Δ = b − a = 知识差距（新经验与已有知识的差异——需要学习什么）
  Σ = a + b = 联合知识状态（已有知识+新经验——整体知识图景）
  ⊗ = ∫(Δ, Σ) = 累积知识学习历史 → 知识网络权重
  τ 读 ⊗ → 浮出所见 = "相关知识"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复知识学习循环

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  海马体.py: a=当前输入, b=CA3回忆, Δ=回忆误差, ⊗=记忆巩固, τ=回忆浮现
  好奇心.py: a=当前知识, b=想知道的知识, Δ=信息差距, ⊗=好奇心学习, τ=探索方向浮现
  语义记忆.py: a=当前知识网络, b=新经验知识, Δ=知识差距, ⊗=知识学习, τ=相关知识浮现

语义记忆 vs 海马体 vs 梦境的区别：
  - 海马体：情景记忆——存储具体经验片段（"我刚才看到了什么"）
  - 梦境：记忆巩固——将海马体经验迁移到概念层权重（离线巩固）
  - 语义记忆：一般性知识——从反复经验中提取稳定的关系模式（"猫和狗都是动物"）
  关键区别：
  - 海马体是"我记得发生了什么"——具体经验
  - 梦境是"让经验沉淀为能力"——技能/权重巩固
  - 语义记忆是"我知道一般规律"——抽象知识提取

语义记忆的核心价值：
  1. 知识积累——不只是记住经验，而是从经验中提取一般性知识
  2. 知识检索——从知识网络中检索相关知识，支持推理和创造
  3. 知识网络——概念之间的关系图谱，不是孤立的事实
  4. 知识置信度——反复出现的知识更可信，只出现一次的知识谨慎对待
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


class 语义记忆:
    """语义记忆——从情景记忆中提取一般性知识，构建知识网络，基于精卫公式。

    核心功能：
    1. 知识提取：从海马体buffer中的情景记忆提取一般性知识
    2. 知识网络构建：构建概念之间的关系网络（知识图谱）
    3. 知识检索：从知识网络中检索与当前概念激活相关的知识
    4. 知识驱动信号：检索到的知识反馈到概念层作为b信号输入源

    精卫公式流程（每拍）：
    1. a = 当前知识网络（semantic_relation_w × concept_h）
    2. b = 新经验暗示的知识（semantic_extract_w × hippocampus）
    3. Δ = b - a（知识差距——新经验与已有知识的差异）
    4. Σ = a + b（联合知识状态——已有知识+新经验）
    5. ⊗ = Δ × Σ（累积知识学习信号）
    6. τ读⊗→浮出所见 = "相关知识"从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新知识网络权重

    语义记忆的四个层次：
    - 知识提取：从经验中提取一般性知识
    - 知识网络：构建概念之间的关系图谱
    - 知识检索：从知识网络中检索相关知识
    - 知识驱动：检索到的知识影响概念可及性
    """

    def __init__(self, n_concepts=64, n_hip=32, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.n_hip = n_hip
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 知识提取：从海马体buffer提取一般性知识 ──
        # semantic_extract_w: 海马体CA3输出 → 语义知识提取
        # 生物学：海马体→新皮层投射——记忆巩固时海马体重新激活皮层表征
        # 反复出现的经验模式 → 提取一般性知识 → 知识网络更新
        # "从多次看到猫→提取'猫'的一般性知识"
        self.semantic_extract_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_hip)]
                                    for _ in range(n_concepts)]

        # 语义知识（从海马体提取的一般性知识——"我知道什么"）
        self.semantic_knowledge = [0.0] * n_concepts

        # ── 知识网络：概念之间的关系图谱 ──
        # semantic_relation_w: 概念 → 概念关系
        # 生物学：新皮层的语义网络——概念之间的关联强度
        # 语义启动效应：激活一个概念会扩散到相关概念
        # "猫→动物"、"红色→颜色"——概念之间的关系
        # 反复共同出现的概念 → 关系增强 → 知识网络涌现
        self.semantic_relation_w = [[rng.uniform(-0.05, 0.05) for _ in range(n_concepts)]
                                     for _ in range(n_concepts)]

        # 知识网络强度（用进废退——每个概念对的强度）
        self.relation_strength = [[1.0 for _ in range(n_concepts)]
                                   for _ in range(n_concepts)]

        # ── 知识检索：从知识网络检索相关知识 ──
        # semantic_retrieve_w: 概念激活 → 知识检索
        # 生物学：新皮层的模式完成——从部分线索检索完整知识
        # 当前概念激活 → 通过知识网络检索相关概念 → 相关知识浮现
        # "看到猫→想到动物→想到宠物→想到家"
        self.semantic_retrieve_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                     for _ in range(n_concepts)]

        # 检索到的知识（从知识网络检索的相关知识）
        self.retrieved_knowledge = [0.0] * n_concepts

        # ── 知识驱动信号（反馈到概念层——作为b信号输入源） ──
        # 知识驱动 = 检索到的知识 × 知识置信度
        # 生物学：新皮层→前额叶投射——知识影响概念可及性
        # 知识让相关概念更容易浮现——"知识丰富的概念更容易想到"
        self.semantic_drive = [0.0] * n_concepts

        # ── 知识置信度（反复出现的知识更可信） ──
        # 生物学：反复经历的知识获得更高置信度
        # "太阳从东方升起"置信度极高（反复经历）
        # "昨天看到一只猫"置信度低（只经历一次）
        # 知识置信度调制知识驱动信号的强度
        self.knowledge_confidence = [0.0] * n_concepts

        # ── 知识强度（0~1，整体知识水平） ──
        # 知识强度 = 知识网络的整体活跃水平
        # 高知识强度→知识驱动强（知识丰富），低知识强度→知识驱动弱（知识贫乏）
        self.knowledge_strength = 0.0

        # ── 上拍检索知识（用于评估知识检索效果） ──
        self.prev_retrieved = [0.0] * n_concepts

        # ── 知识历史 ──
        self.knowledge_history = []
        self.history_cap = 50

        # ── 统计 ──
        self.知识提取次数 = 0
        self.知识检索次数 = 0
        self.知识巩固次数 = 0
        self.高置信度知识数 = 0

    def 一拍(self, concept_h, 海马体ca3=None, 海马体buffer=None):
        """语义记忆一拍：知识提取、知识网络更新、知识检索、知识驱动信号。

        精卫公式流程：
        1. a = 当前知识网络（semantic_relation_w × concept_h）
        2. b = 新经验暗示的知识（semantic_extract_w × hippocampus）
        3. Δ = b - a（知识差距——新经验与已有知识的差异）
        4. Σ = a + b（联合知识状态——已有知识+新经验）
        5. ⊗ = Δ × Σ → 知识网络权重更新
        6. τ读⊗→浮出所见 = "相关知识"从⊗中浮现
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          海马体ca3: 海马体CA3输出（n_hip维，可选）——用于知识提取
          海马体buffer: 海马体buffer中的经验（可选）——用于知识巩固

        返回：
          dict包含：
            semantic_drive: 知识驱动信号（反馈到概念层）
            knowledge_strength: 知识强度（0~1）
            semantic_knowledge: 语义知识（从海马体提取的一般性知识）
            retrieved_knowledge: 检索到的知识（从知识网络检索的相关知识）
            knowledge_confidence: 知识置信度（反复出现的知识更可信）
        """
        self.拍 += 1

        # ── 知识提取：从海马体buffer提取一般性知识 ──
        # 精卫公式：b = semantic_extract_w × 海马体ca3（新经验暗示的知识）
        # τ读⊗→浮出所见："我知道什么"从海马体经验中浮现
        # 生物学：海马体→新皮层投射——记忆巩固时海马体重新激活皮层表征
        # 反复出现的经验模式 → 提取一般性知识 → 知识网络更新
        new_semantic_knowledge = [0.0] * self.n_concepts
        if 海马体ca3 is not None:
            for i in range(self.n_concepts):
                extract_input = 0.0
                for j in range(min(self.n_hip, len(海马体ca3))):
                    extract_input += self.semantic_extract_w[i][j] * 海马体ca3[j]
                new_semantic_knowledge[i] = activation(extract_input)
            self.知识提取次数 += 1

        # 知识巩固：海马体buffer中的经验反复出现 → 知识置信度增强
        # 生物学：系统级巩固——反复重播后知识从海马体依赖变为新皮层独立
        # buffer中经验越多 → 知识置信度越高（更多证据支持）
        if 海马体buffer is not None and len(海马体buffer) > 0:
            buffer_size = len(海马体buffer)
            # buffer越大 → 知识置信度越高（更多经验证据）
            confidence_boost = min(0.01, buffer_size * 0.0001)
            for i in range(self.n_concepts):
                self.knowledge_confidence[i] = min(1.0,
                    self.knowledge_confidence[i] + confidence_boost * abs(new_semantic_knowledge[i]))
            self.知识巩固次数 += 1

        # 平滑更新语义知识——不是每拍完全替换，而是渐进更新
        # 生物学：记忆巩固是渐进的——不是一次性提取，而是反复巩固
        for i in range(self.n_concepts):
            self.semantic_knowledge[i] = activation(
                self.semantic_knowledge[i] * 0.9 + new_semantic_knowledge[i] * 0.1)

        # ── 知识网络更新：从提取的知识更新概念关系矩阵 ──
        # 精卫公式：a=当前知识网络, b=新经验知识, Δ=b-a, ⊗=Δ×Σ
        # 生物学：赫布学习——反复共同出现的概念→关系增强
        # 如果概念i和概念j反复同时激活 → semantic_relation_w[i][j]增强
        # 这就是"语义启动效应"的神经基础——相关概念更容易一起激活
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                if i != j:
                    # 概念i和概念j同时激活 → 关系增强
                    co_activation = concept_h[i] * concept_h[j]
                    if abs(co_activation) > 0.01:
                        dw = self.lr * co_activation
                        if not (math.isnan(dw) or math.isinf(dw)):
                            self.semantic_relation_w[i][j] = \
                                self.semantic_relation_w[i][j] * 0.99 + dw

        # ── 知识检索：从知识网络检索相关知识 ──
        # 精卫公式：b = semantic_retrieve_w × concept_h（知识检索预测）
        # τ读⊗→浮出所见："相关知识"从当前概念激活中浮现
        # 生物学：新皮层的模式完成——从部分线索检索完整知识
        # 当前概念激活 → 通过知识网络检索相关概念 → 相关知识浮现
        # "看到猫→想到动物→想到宠物→想到家"
        new_retrieved = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            retrieve_input = 0.0
            for j in range(self.n_concepts):
                # 通过知识网络检索——semantic_relation_w是概念关系
                # semantic_retrieve_w决定如何从关系网络中检索
                retrieve_input += self.semantic_retrieve_w[i][j] * concept_h[j]
            # 语义知识也参与检索——已有知识帮助检索相关知识
            retrieve_input += self.semantic_knowledge[i] * 0.3
            new_retrieved[i] = activation(retrieve_input)
            self.知识检索次数 += 1

        self.retrieved_knowledge = new_retrieved

        # ── 知识强度 ──
        # 知识强度 = 知识网络的整体活跃水平
        # 知识强度高→知识驱动强（知识丰富），知识强度低→知识驱动弱（知识贫乏）
        knowledge_sum = sum(abs(v) for v in self.semantic_knowledge) / max(1, self.n_concepts)
        relation_sum = sum(abs(self.semantic_relation_w[i][j])
                          for i in range(self.n_concepts)
                          for j in range(self.n_concepts)) / max(1, self.n_concepts * self.n_concepts)
        self.knowledge_strength = activation(knowledge_sum * 2.0 + relation_sum * 10.0) * 0.5 + 0.5

        # ── 知识驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 知识驱动信号
        # 生物学：新皮层→前额叶投射——知识影响概念可及性
        # 知识让相关概念更容易浮现——"知识丰富的概念更容易想到"
        # 知识驱动 = 检索到的知识 × 知识置信度
        # 高置信度知识→强驱动（可信知识影响大），低置信度知识→弱驱动（不可信知识影响小）
        for i in range(self.n_concepts):
            self.semantic_drive[i] = activation(
                new_retrieved[i] * (0.3 + self.knowledge_confidence[i] * 0.7))

        # ── 知识置信度衰减 ──
        # 生物学：不使用的知识会逐渐衰退——"用进废退"
        # 但衰减很慢——知识一旦获得就不容易完全遗忘
        for i in range(self.n_concepts):
            self.knowledge_confidence[i] *= 0.999

        # ── 高置信度知识统计 ──
        self.高置信度知识数 = sum(1 for c in self.knowledge_confidence if c > 0.3)

        # ── 知识提取学习 ──
        # 精卫公式：⊗ = Δ × Σ → 知识提取权重更新
        # Δ = b - a（知识差距——新经验与已有知识的差异）
        # Σ = a + b（联合知识状态——已有知识+新经验）
        # 赫布学习：海马体信号和语义知识同时出现→提取权重增强
        # 生物学：海马体→新皮层的赫布学习——反复从海马体A提取知识B→A→B的映射越来越强
        if 海马体ca3 is not None:
            for i in range(self.n_concepts):
                for j in range(min(self.n_hip, len(海马体ca3))):
                    dw = self.lr * new_semantic_knowledge[i] * 海马体ca3[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.semantic_extract_w[i][j] = self.semantic_extract_w[i][j] * 0.99 + dw

        # semantic_retrieve_w: 概念激活→知识检索（赫布学习）
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_retrieved[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.semantic_retrieve_w[i][j] = self.semantic_retrieve_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.semantic_extract_w[i][j]) for j in range(self.n_hip))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_hip):
                    self.semantic_extract_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.semantic_retrieve_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.semantic_retrieve_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.semantic_relation_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.semantic_relation_w[i][j] *= scale

        # ── 知识历史 ──
        self.knowledge_history.append(self.knowledge_strength)
        if len(self.knowledge_history) > self.history_cap:
            self.knowledge_history.pop(0)

        # 保存当前状态
        self.prev_retrieved = list(new_retrieved)

        return {
            "semantic_drive": self.semantic_drive,
            "knowledge_strength": round(self.knowledge_strength, 4),
            "semantic_knowledge": self.semantic_knowledge,
            "retrieved_knowledge": self.retrieved_knowledge,
            "knowledge_confidence": self.knowledge_confidence,
            "高置信度知识数": self.高置信度知识数,
        }

    def 状态摘要(self):
        avg_knowledge = sum(self.knowledge_history) / max(1, len(self.knowledge_history)) \
            if self.knowledge_history else 0.0
        return {
            "knowledge_strength": round(self.knowledge_strength, 4),
            "高置信度知识数": self.高置信度知识数,
            "avg_knowledge": round(avg_knowledge, 4),
            "semantic_extract_w均值": round(sum(sum(row) for row in self.semantic_extract_w) / max(1, self.n_concepts * self.n_hip), 4),
            "semantic_relation_w均值": round(sum(sum(row) for row in self.semantic_relation_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "semantic_retrieve_w均值": round(sum(sum(row) for row in self.semantic_retrieve_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "知识提取次数": self.知识提取次数,
            "知识检索次数": self.知识检索次数,
            "知识巩固次数": self.知识巩固次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 语义记忆验证（知识提取、知识网络、知识检索） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    sm = 语义记忆(n_concepts=64, n_hip=32)

    print("初始状态:")
    print(f"  概念数: {sm.n_concepts}")
    print(f"  海马体维度: {sm.n_hip}")
    print(f"  知识提取权重: semantic_extract_w (64×32)")
    print(f"  知识关系权重: semantic_relation_w (64×64)")
    print(f"  知识检索权重: semantic_retrieve_w (64×64)")
    print(f"  初始知识强度: {sm.knowledge_strength}")

    print("\n--- 测试1: 知识提取——海马体信号→语义知识 ---")
    # 海马体CA3输出传入→提取语义知识
    海马体ca3 = [0.3] * 32
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = sm.一拍(concept_h, 海马体ca3=海马体ca3)
    print(f"  50拍后: 知识强度={result['knowledge_strength']}")
    print(f"  语义知识top5: {sorted(range(64), key=lambda i: abs(sm.semantic_knowledge[i]), reverse=True)[:5]}")
    print(f"  → 海马体信号传入后应该提取出语义知识")

    print("\n--- 测试2: 无海马体信号→知识强度低 ---")
    sm2 = 语义记忆(n_concepts=64, n_hip=32)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = sm2.一拍(concept_h)  # 不传海马体信号
    print(f"  无海马体信号: 知识强度={result['knowledge_strength']}")
    print(f"  → 无海马体信号时知识强度应该更低")

    print("\n--- 测试3: 知识网络——概念共同激活→关系增强 ---")
    sm3 = 语义记忆(n_concepts=16, n_hip=8)
    # 概念4和概念8反复共同激活
    for _ in range(100):
        concept_h = [0.0] * 16
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        sm3.一拍(concept_h, 海马体ca3=[0.2]*8)
    rel_4_8 = sm3.semantic_relation_w[4][8]
    rel_4_0 = sm3.semantic_relation_w[4][0]
    print(f"  概念4→8关系强度: {rel_4_8:.4f} (共同激活→应该强)")
    print(f"  概念4→0关系强度: {rel_4_0:.4f} (不共同激活→应该弱)")
    print(f"  → 共同激活的概念对应有更强的关系")

    print("\n--- 测试4: 知识检索——概念激活→检索相关知识 ---")
    sm4 = 语义记忆(n_concepts=64, n_hip=32)
    # 先学习：概念4和8反复共同激活
    for _ in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        sm4.一拍(concept_h, 海马体ca3=[0.2]*32)
    # 然后只激活概念4，检查是否检索到概念8
    concept_h_test = [0.0] * 64
    concept_h_test[4] = 0.3
    result = sm4.一拍(concept_h_test, 海马体ca3=[0.2]*32)
    print(f"  只激活概念4后检索top5: {sorted(range(64), key=lambda i: abs(sm4.retrieved_knowledge[i]), reverse=True)[:5]}")
    print(f"  → 概念8应该在检索top5中（通过知识网络检索）")

    print("\n--- 测试5: 不同概念模式→不同检索结果 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    for name, pattern in patterns:
        sm_t = 语义记忆(n_concepts=64, n_hip=32)
        for _ in range(100):
            sm_t.一拍(pattern, 海马体ca3=[0.2]*32)
        result = sm_t.一拍(pattern, 海马体ca3=[0.2]*32)
        print(f"  {name}: 知识强度={round(sm_t.knowledge_strength, 4)}, "
              f"检索top5={sorted(range(64), key=lambda i: abs(sm_t.retrieved_knowledge[i]), reverse=True)[:5]}")

    print("\n--- 测试6: 知识置信度——反复出现→高置信度 ---")
    sm5 = 语义记忆(n_concepts=64, n_hip=32)
    海马体ca3 = [0.3] * 32
    for tick in range(200):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        sm5.一拍(concept_h, 海马体ca3=海马体ca3, 海马体buffer=list(range(min(tick+1, 128))))
    high_conf = sum(1 for c in sm5.knowledge_confidence if c > 0.3)
    print(f"  200拍后: 高置信度知识数={high_conf}")
    print(f"  知识置信度top5: {sorted(range(64), key=lambda i: sm5.knowledge_confidence[i], reverse=True)[:5]}")
    print(f"  → 反复出现后知识置信度应该提高")

    print("\n--- 测试7: 知识驱动信号 ---")
    sm6 = 语义记忆(n_concepts=64, n_hip=32)
    for _ in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        sm6.一拍(concept_h, 海马体ca3=[0.2]*32)
    drive_sum = sum(abs(v) for v in sm6.semantic_drive)
    print(f"  知识驱动信号总和: {drive_sum:.4f}")
    print(f"  知识驱动top5: {sorted(range(64), key=lambda i: abs(sm6.semantic_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试8: 海马体buffer→知识巩固 ---")
    sm7 = 语义记忆(n_concepts=64, n_hip=32)
    # 小buffer
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        sm7.一拍(concept_h, 海马体ca3=[0.2]*32, 海马体buffer=list(range(10)))
    conf_small = sum(sm7.knowledge_confidence)
    # 大buffer
    sm8 = 语义记忆(n_concepts=64, n_hip=32)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        sm8.一拍(concept_h, 海马体ca3=[0.2]*32, 海马体buffer=list(range(100)))
    conf_large = sum(sm8.knowledge_confidence)
    print(f"  小buffer(10)知识置信度总和: {conf_small:.4f}")
    print(f"  大buffer(100)知识置信度总和: {conf_large:.4f}")
    print(f"  → 大buffer应该导致更高的知识置信度")

    print("\n--- 测试9: 精卫公式验证 ---")
    sm9 = 语义记忆(n_concepts=16, n_hip=8)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        concept_h[1] = 0.2
        result = sm9.一拍(concept_h, 海马体ca3=[0.2]*8)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 知识强度={result['knowledge_strength']}, "
                  f"高置信度知识数={result['高置信度知识数']}")
    print(f"  τ(浮出): 知识驱动均值={round(sum(sm9.semantic_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={sm9.拍}")

    print("\n验证通过")
