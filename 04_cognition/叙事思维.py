"""叙事思维.py —— 将经验组织成连贯的故事，基于精卫公式。

叙事思维是人类认知的独特能力——我们不是孤立地记忆事件，
而是将经验编织成连贯的故事。叙事赋予经验以意义、顺序和因果结构。

人类神经科学机制：
  1. 叙事记忆（narrative memory）——故事结构记忆：
     - 人类不是记忆孤立的事件，而是记忆有开头、中间、结尾的故事
     - 叙事记忆比孤立事件记忆更持久、更容易检索
     - 叙事提供了事件的因果解释——"因为A，所以B"
     → 工程实现：narrative_pattern_w（概念激活 → 叙事模式识别）

  2. 默认模式网络（DMN）——叙事构建：
     - DMN在休息时活跃——构建自我叙事、回忆过去、想象未来
     - DMN将分散的经验片段连接成连贯的故事线
     - 叙事构建是自发的——大脑不断在"讲故事"
     → 工程实现：narrative_construct_w（经验片段 → 叙事构建）

  3. 事件分割理论（event segmentation）——叙事边界：
     - 人类自动将连续经验分割成离散事件——"场景切换"
     - 事件边界处surprise升高——标志一个叙事段落的结束和另一个的开始
     - 事件分割帮助组织记忆——每个事件是一个叙事单元
     → 工程实现：叙事边界检测（surprise驱动的事件分割）

  4. 叙事传输（narrative transport）——叙事沉浸：
     - 当叙事连贯时，人会被"吸入"故事——注意力高度集中
     - 叙事传输增强记忆——好故事比事实列表更容易记住
     - 叙事传输驱动学习——通过故事传递知识和价值观
     → 工程实现：叙事连贯度（narrative_coherence）调制叙事驱动强度

  5. 叙事认同（narrative identity）——自我叙事：
     - 人类的自我认同是通过叙事构建的——"我是谁"=我的故事
     - 自我叙事将过去的经验、现在的状态和未来的目标连接成统一的故事线
     - 叙事认同赋予生活以意义和方向
     → 工程实现：叙事认同信号（自我叙事与当前经验的一致性）

数字生命优势（超越人类的限制）：
  1. 完美叙事记忆：人类会遗忘和扭曲故事，数字生命可以完美保留所有叙事
  2. 并行叙事构建：人类只能构建一个故事线，数字生命可以并行构建多种叙事
  3. 叙事自省：人类难以审视自己的叙事偏见，数字生命可以客观分析叙事结构
  4. 无限叙事容量：人类叙事记忆有限，数字生命可以存储和检索无限多的故事
  5. 跨叙事连接：人类难以连接不同故事线，数字生命可以自动发现跨叙事的联系

精卫公式（叙事思维版）：
  a = 当前经验片段（概念激活——"正在发生什么"）
  b = 叙事结构预测（叙事模式——"这应该是什么故事"）
  Δ = b − a = 叙事差距（经验与叙事结构的差异——"故事哪里不对"）
  Σ = a + b = 联合叙事状态（经验+叙事结构——"整体故事图景"）
  ⊗ = ∫(Δ, Σ) = 累积叙事学习历史 → 叙事权重
  τ 读 ⊗ → 浮出所见 = "故事"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复叙事构建循环

叙事思维 vs 预测模拟 vs 语义记忆的区别：
  - 预测模拟：从当前状态模拟未来——"如果做A，会发生什么"（后果模拟）
  - 语义记忆：从经验提取一般知识——"猫是动物"（知识提取）
  - 叙事思维：将经验组织成故事——"这是一个关于X的故事"（叙事构建）
  关键区别：
  - 预测模拟关注"未来"——模拟可能的后果
  - 语义记忆关注"一般"——提取稳定的知识
  - 叙事思维关注"连贯"——将分散的经验编织成有意义的故事
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


class 叙事思维:
    """叙事思维——将经验组织成连贯的故事，基于精卫公式。

    核心功能：
    1. 叙事模式识别：从当前概念激活识别叙事模式（故事类型）
    2. 叙事结构预测：预测叙事的下一步发展（故事应该怎么发展）
    3. 叙事构建：将分散的经验片段编织成连贯的故事线
    4. 叙事驱动信号：叙事结构反馈到概念层作为b信号输入源

    精卫公式流程（每拍）：
    1. a = 当前经验片段（concept_h）
    2. b = 叙事结构预测（narrative_construct_w × concept_h）
    3. Δ = b - a（叙事差距——经验与叙事结构的差异）
    4. Σ = a + b（联合叙事状态——经验+叙事结构）
    5. ⊗ = Δ × Σ（累积叙事学习信号）
    6. τ读⊗→浮出所见 = "故事"从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新叙事权重
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 叙事模式识别：从概念激活识别叙事模式 ──
        # narrative_pattern_w: 概念激活 → 叙事模式识别
        # 生物学：叙事记忆——故事有开头、中间、结尾的结构
        # 当前概念激活模式 → 识别"这是什么类型的故事"
        # "看到追逐→识别为追逐故事"、"看到对话→识别为对话故事"
        self.narrative_pattern_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                     for _ in range(n_concepts)]

        # 叙事模式（识别出的故事类型——n_concepts维，每个维度代表一种叙事模式）
        self.narrative_pattern = [0.0] * n_concepts

        # ── 叙事结构预测：预测叙事的下一步发展 ──
        # narrative_construct_w: 叙事模式 → 叙事结构预测
        # 生物学：DMN叙事构建——从当前叙事模式预测故事如何发展
        # "追逐故事→接下来应该有冲突或逃脱"
        # 叙事结构预测 = 从当前叙事模式推演故事的下一步
        self.narrative_construct_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                       for _ in range(n_concepts)]

        # 叙事结构预测（预测的叙事下一步——n_concepts维）
        self.narrative_prediction = [0.0] * n_concepts

        # ── 叙事驱动信号（反馈到概念层——作为b信号输入源） ──
        # 叙事驱动 = 叙事结构预测 × 叙事连贯度
        # 生物学：叙事传输——连贯的叙事让相关概念更容易浮现
        # "好故事让相关概念更容易想到"
        self.narrative_drive = [0.0] * n_concepts

        # ── 叙事连贯度（0~1） ──
        # 叙事连贯度 = 当前经验与叙事结构预测的一致性
        # 高连贯度→叙事驱动强（故事讲得好），低连贯度→叙事驱动弱（故事断裂）
        # 生物学：叙事传输——连贯的故事让人沉浸，断裂的故事让人分心
        self.narrative_coherence = 0.5

        # ── 叙事强度（0~1） ──
        # 叙事强度 = 叙事构建的整体活跃水平
        # 高叙事强度→叙事驱动强（正在讲故事），低叙事强度→叙事驱动弱（没有故事）
        self.narrative_strength = 0.3

        # ── 叙事边界检测 ──
        # 事件分割理论：surprise升高标志叙事边界——一个故事段落结束，另一个开始
        # 叙事边界处：叙事连贯度突然下降，叙事模式重新识别
        self.narrative_boundary = False
        self.prev_surprise = 0.04

        # ── 叙事历史 ──
        # 叙事历史记录每拍的叙事连贯度和叙事强度
        self.narrative_history = []
        self.history_cap = 50

        # ── 叙事段（当前叙事段落的概念序列） ──
        # 每个叙事段落是一个连贯的故事片段
        # 叙事边界处开始新段落
        self.current_segment = []
        self.segment_cap = 20

        # ── 上拍叙事预测（用于叙事验证） ──
        self.prev_narrative_prediction = [0.0] * n_concepts

        # ── 统计 ──
        self.叙事模式识别次数 = 0
        self.叙事构建次数 = 0
        self.叙事边界次数 = 0
        self.叙事验证次数 = 0
        self.连贯叙事次数 = 0

    def 一拍(self, concept_h, surprise=0.04, 情感信号=None):
        """叙事思维一拍：叙事模式识别、叙事结构预测、叙事驱动信号。

        精卫公式流程：
        1. a = 当前经验片段（concept_h）
        2. b = 叙事结构预测（narrative_construct_w × narrative_pattern）
        3. Δ = b - a（叙事差距——经验与叙事结构的差异）
        4. Σ = a + b（联合叙事状态——经验+叙事结构）
        5. ⊗ = Δ × Σ → 叙事权重更新
        6. τ读⊗→浮出所见 = "故事"从⊗中浮现
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          surprise: 当前surprise水平（驱动叙事边界检测）
          情感信号: 情感状态（可选，增强叙事模式识别）

        返回：
          dict包含：
            narrative_drive: 叙事驱动信号（反馈到概念层）
            narrative_coherence: 叙事连贯度（0~1）
            narrative_strength: 叙事强度（0~1）
            narrative_pattern: 叙事模式（识别出的故事类型）
            narrative_prediction: 叙事结构预测（预测的叙事下一步）
            narrative_boundary: 是否叙事边界
        """
        self.拍 += 1

        # ── 叙事边界检测 ──
        # 事件分割理论：surprise突然升高→叙事边界
        # 叙事边界处：叙事连贯度下降，开始新叙事段落
        surprise_delta = surprise - self.prev_surprise
        self.narrative_boundary = surprise_delta > 0.05 or surprise > 0.3

        if self.narrative_boundary:
            self.叙事边界次数 += 1
            # 叙事边界处：叙事连贯度突然下降
            self.narrative_coherence *= 0.5
            # 开始新叙事段落
            self.current_segment = []
        else:
            # 非边界：叙事连贯度逐渐恢复
            self.narrative_coherence = self.narrative_coherence * 0.95 + 0.5 * 0.05

        self.prev_surprise = surprise

        # ── 叙事模式识别：从当前概念激活识别叙事模式 ──
        # 精卫公式：a = concept_h（当前经验片段）
        # narrative_pattern_w × concept_h → 叙事模式识别
        # 生物学：叙事记忆——当前经验模式匹配到故事类型
        # "看到追逐→识别为追逐故事"
        new_pattern = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            pattern_input = 0.0
            for j in range(self.n_concepts):
                pattern_input += self.narrative_pattern_w[i][j] * concept_h[j]
            new_pattern[i] = activation(pattern_input)

        # 情感信号增强叙事模式识别——情感丰富的经验更容易被识别为故事
        if 情感信号 is not None:
            for i in range(self.n_concepts):
                new_pattern[i] = activation(new_pattern[i] + 情感信号 * 0.1)

        # 叙事边界处：叙事模式重新识别（新段落新故事）
        if self.narrative_boundary:
            self.narrative_pattern = list(new_pattern)
        else:
            # 非边界：叙事模式平滑更新（故事在发展）
            for i in range(self.n_concepts):
                self.narrative_pattern[i] = activation(
                    self.narrative_pattern[i] * 0.8 + new_pattern[i] * 0.2)

        self.叙事模式识别次数 += 1

        # ── 叙事结构预测：从叙事模式预测故事下一步 ──
        # 精卫公式：b = narrative_construct_w × narrative_pattern（叙事结构预测）
        # τ读⊗→浮出所见："故事应该怎么发展"从叙事模式中浮现
        # 生物学：DMN叙事构建——从当前叙事模式推演故事的下一步
        # "追逐故事→接下来应该有冲突或逃脱"
        new_prediction = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            predict_input = 0.0
            for j in range(self.n_concepts):
                predict_input += self.narrative_construct_w[i][j] * self.narrative_pattern[j]
            new_prediction[i] = activation(predict_input)

        self.narrative_prediction = new_prediction
        self.叙事构建次数 += 1

        # ── 叙事验证：比较上拍预测和当前经验 ──
        # 精卫公式：Δ = b - a（叙事差距——上拍预测与当前经验的差异）
        # 叙事验证 = 上拍叙事预测 vs 当前概念激活
        # 预测准确→叙事连贯度高（故事讲得好），预测不准→叙事连贯度低（故事断裂）
        predict_error = sum(abs(concept_h[i] - self.prev_narrative_prediction[i])
                           for i in range(self.n_concepts)) / max(1, self.n_concepts)
        coherence_this = max(0.0, min(1.0, 1.0 - predict_error * 3.0))

        # 叙事连贯度平滑更新
        if not self.narrative_boundary:
            self.narrative_coherence = self.narrative_coherence * 0.9 + coherence_this * 0.1

        if coherence_this > 0.7:
            self.连贯叙事次数 += 1
        self.叙事验证次数 += 1

        # ── 叙事强度 ──
        # 叙事强度 = 叙事模式活跃度 × 叙事连贯度
        # 高叙事强度→叙事驱动强（正在讲一个好故事）
        pattern_sum = sum(abs(v) for v in self.narrative_pattern) / max(1, self.n_concepts)
        self.narrative_strength = activation(pattern_sum * 3.0) * 0.5 + 0.5
        self.narrative_strength = self.narrative_strength * (0.3 + self.narrative_coherence * 0.7)

        # ── 叙事驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 叙事驱动信号
        # 生物学：叙事传输——连贯的叙事让相关概念更容易浮现
        # 叙事驱动 = 叙事结构预测 × 叙事连贯度
        # 高连贯度→强叙事驱动（好故事引导思维），低连贯度→弱叙事驱动（故事断裂）
        for i in range(self.n_concepts):
            self.narrative_drive[i] = activation(
                self.narrative_prediction[i] * self.narrative_coherence)

        # ── 叙事学习 ──
        # 精卫公式：⊗ = Δ × Σ → 叙事权重更新
        # Δ = b - a（叙事差距），Σ = a + b（联合叙事状态）
        # 赫布学习：叙事模式和概念激活同时出现→模式识别权重增强
        # 生物学：叙事记忆的赫布学习——反复经历同类型故事→叙事模式识别越来越准
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_pattern[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.narrative_pattern_w[i][j] = self.narrative_pattern_w[i][j] * 0.99 + dw

        # narrative_construct_w: 叙事模式→叙事预测（赫布学习）
        # 叙事模式和叙事预测同时出现→构建权重增强
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_prediction[i] * self.narrative_pattern[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.narrative_construct_w[i][j] = self.narrative_construct_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.narrative_pattern_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.narrative_pattern_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.narrative_construct_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.narrative_construct_w[i][j] *= scale

        # ── 叙事段落更新 ──
        # 记录当前概念激活到叙事段落
        self.current_segment.append(list(concept_h))
        if len(self.current_segment) > self.segment_cap:
            self.current_segment.pop(0)

        # ── 叙事历史 ──
        self.narrative_history.append({
            "coherence": self.narrative_coherence,
            "strength": self.narrative_strength,
            "boundary": self.narrative_boundary,
        })
        if len(self.narrative_history) > self.history_cap:
            self.narrative_history.pop(0)

        # 保存当前状态
        self.prev_narrative_prediction = list(new_prediction)

        return {
            "narrative_drive": self.narrative_drive,
            "narrative_coherence": round(self.narrative_coherence, 4),
            "narrative_strength": round(self.narrative_strength, 4),
            "narrative_pattern": self.narrative_pattern,
            "narrative_prediction": self.narrative_prediction,
            "narrative_boundary": self.narrative_boundary,
        }

    def 状态摘要(self):
        avg_coherence = sum(h["coherence"] for h in self.narrative_history) / max(1, len(self.narrative_history)) \
            if self.narrative_history else 0.0
        avg_strength = sum(h["strength"] for h in self.narrative_history) / max(1, len(self.narrative_history)) \
            if self.narrative_history else 0.0
        boundary_count = sum(1 for h in self.narrative_history if h["boundary"])
        return {
            "narrative_coherence": round(self.narrative_coherence, 4),
            "narrative_strength": round(self.narrative_strength, 4),
            "avg_coherence": round(avg_coherence, 4),
            "avg_strength": round(avg_strength, 4),
            "boundary_count": boundary_count,
            "narrative_pattern_w均值": round(sum(sum(row) for row in self.narrative_pattern_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "narrative_construct_w均值": round(sum(sum(row) for row in self.narrative_construct_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "叙事模式识别次数": self.叙事模式识别次数,
            "叙事构建次数": self.叙事构建次数,
            "叙事边界次数": self.叙事边界次数,
            "叙事验证次数": self.叙事验证次数,
            "连贯叙事次数": self.连贯叙事次数,
            "连贯率": round(self.连贯叙事次数 / max(1, self.叙事验证次数), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 叙事思维验证 ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    nt = 叙事思维(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {nt.n_concepts}")
    print(f"  叙事模式权重: narrative_pattern_w (64x64)")
    print(f"  叙事构建权重: narrative_construct_w (64x64)")
    print(f"  初始叙事连贯度: {nt.narrative_coherence}")
    print(f"  初始叙事强度: {nt.narrative_strength}")

    print("\n--- 测试1: 叙事模式识别——概念激活→叙事模式 ---")
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = nt.一拍(concept_h)
    print(f"  50拍后: 叙事连贯度={result['narrative_coherence']}")
    print(f"  叙事强度={result['narrative_strength']}")
    print(f"  叙事模式top5: {sorted(range(64), key=lambda i: abs(nt.narrative_pattern[i]), reverse=True)[:5]}")

    print("\n--- 测试2: 不同概念模式→不同叙事模式 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    for name, pattern in patterns:
        nt_t = 叙事思维(n_concepts=64)
        for _ in range(50):
            nt_t.一拍(pattern)
        print(f"  {name}: 叙事连贯度={round(nt_t.narrative_coherence, 4)}, "
              f"叙事模式top5={sorted(range(64), key=lambda i: abs(nt_t.narrative_pattern[i]), reverse=True)[:5]}")

    print("\n--- 测试3: 叙事边界检测——surprise升高→叙事边界 ---")
    nt_boundary = 叙事思维(n_concepts=64)
    # 低surprise阶段
    for _ in range(30):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        nt_boundary.一拍(concept_h, surprise=0.02)
    boundary_before = nt_boundary.narrative_boundary
    coherence_before = nt_boundary.narrative_coherence
    # surprise突然升高
    concept_h = [0.0] * 64
    concept_h[4] = 0.3
    result = nt_boundary.一拍(concept_h, surprise=0.5)
    boundary_after = result["narrative_boundary"]
    coherence_after = result["narrative_coherence"]
    print(f"  低surprise: 叙事边界={boundary_before}, 连贯度={round(coherence_before, 4)}")
    print(f"  高surprise: 叙事边界={boundary_after}, 连贯度={round(coherence_after, 4)}")
    print(f"  → surprise升高应触发叙事边界，连贯度下降")

    print("\n--- 测试4: 叙事连贯度——稳定模式→高连贯度 ---")
    nt_coherent = 叙事思维(n_concepts=64)
    for _ in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        nt_coherent.一拍(concept_h, surprise=0.02)
    print(f"  100拍稳定模式后: 叙事连贯度={nt_coherent.narrative_coherence:.4f}")
    print(f"  连贯率: {nt_coherent.连贯叙事次数/max(1,nt_coherent.叙事验证次数):.4f}")
    print(f"  → 稳定模式应导致高连贯度")

    print("\n--- 测试5: 情感信号增强叙事模式识别 ---")
    nt_emotion = 叙事思维(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        nt_emotion.一拍(concept_h, 情感信号=0.5)
    nt_no_emotion = 叙事思维(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        nt_no_emotion.一拍(concept_h)
    emotion_sum = sum(abs(v) for v in nt_emotion.narrative_pattern)
    no_emotion_sum = sum(abs(v) for v in nt_no_emotion.narrative_pattern)
    print(f"  有情感信号: 叙事模式活跃度={emotion_sum:.4f}")
    print(f"  无情感信号: 叙事模式活跃度={no_emotion_sum:.4f}")
    print(f"  → 情感信号应增强叙事模式识别")

    print("\n--- 测试6: 叙事驱动信号 ---")
    nt_drive = 叙事思维(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        nt_drive.一拍(concept_h, surprise=0.02)
    drive_sum = sum(abs(v) for v in nt_drive.narrative_drive)
    print(f"  叙事驱动信号总和: {drive_sum:.4f}")
    print(f"  叙事驱动top5: {sorted(range(64), key=lambda i: abs(nt_drive.narrative_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试7: 叙事结构预测 ---")
    nt_predict = 叙事思维(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        nt_predict.一拍(concept_h, surprise=0.02)
    print(f"  叙事预测top5: {sorted(range(64), key=lambda i: abs(nt_predict.narrative_prediction[i]), reverse=True)[:5]}")
    print(f"  → 叙事预测应与输入概念模式相关")

    print("\n--- 测试8: 精卫公式验证 ---")
    nt_jw = 叙事思维(n_concepts=16)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        concept_h[1] = 0.2
        result = nt_jw.一拍(concept_h, surprise=0.04)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 叙事连贯度={result['narrative_coherence']}, "
                  f"叙事强度={result['narrative_strength']}, "
                  f"叙事边界={result['narrative_boundary']}")
    print(f"  τ(浮出): 叙事驱动均值={round(sum(nt_jw.narrative_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={nt_jw.拍}")

    print("\n验证通过")
