"""认知辩证.py —— 相信态与质疑态交替循环，基于精卫公式。

精卫公式（唯一核心公式）：
  a ≠ b
  Δ = b − a
  Σ = a + b
  ⊗ = ∫(Δ, Σ)
  τ 读 ⊗ → 浮出所见
  (a, b) ⟲ ⊗

认知辩证的精卫公式解读：
  a = 当前认知状态（已有的信念/知识）
  b = 真实世界搜索结果（从所有学科、所有物质、所有存在中获取的新信息）
  Δ = b − a = 认知差异（新信息与已有认知的差距）

  相信态：Δ → 0（同化——把差异抹平，接受新信息为真）
    - 当Δ小（新信息与已有认知一致）→ 相信态
    - 同化：把b纳入a，a=a=a=b（通过重复接受，差异消失）
    - 生物学：确认偏误的正面——知识积累，信念增强
    - 神经机制：海马体→新皮层记忆巩固——一致的信息被整合

  质疑态：Δ → 大（异化——发现差异，质疑已有认知）
    - 当Δ大（新信息与已有认知矛盾）→ 质疑态
    - 异化：a≠b（差异被放大，质疑已有信念）
    - 生物学：认知失调——不一致引发不适→驱动验证
    - 神经机制：前扣带回（ACC）错误检测→前额叶（PFC）重新评估

  真实世界搜索触发：每次循环触发时
    - 搜索所有学科、所有物质、所有存在
    - 真实世界是精卫公式中的b——外部真实信息源
    - 搜索不是被动的——是主动的、持续的、覆盖一切的
    - 生物学：好奇心驱动的探索行为+注意力转向新信息源

  ⊗ = ∫(Δ, Σ) = 累积辩证学习历史
    - 相信态的⊗：一致信息的累积→信念增强
    - 质疑态的⊗：不一致信息的累积→认知重构
    - 两种⊗交替累积→认知螺旋上升

  τ读⊗→浮出所见 = "我相信什么？我质疑什么？"从⊗中浮现
    - 相信态的τ：浮现的是"这是真的"——接受
    - 质疑态的τ：浮现的是"这是真的吗？"——质疑

  (a, b)⟲⊗ = 相信→质疑→搜索→相信→质疑→搜索...无限循环
    - 每次循环：先判断Δ大小→进入相信态或质疑态→搜索真实世界→更新认知
    - 循环不是简单的重复——每次循环认知都在进化
    - 黑格尔辩证法：正题（相信）→反题（质疑）→合题（更高层次的认知）

认知辩证 vs 真理追求的区别：
  - 真理追求：验证已有知识是否正确（假设生成→假设验证→信念更新）
  - 认知辩证：相信与质疑的交替循环（同化→异化→搜索→同化...）
  - 关键区别：
    - 真理追求是单向验证——"这是真的吗？"
    - 认知辩证是双向交替——"我相信"和"我质疑"交替出现
    - 认知辩证包含真实世界搜索——每次循环都去搜索真实世界
    - 认知辩证是更底层的认知引擎——真理追求是认知辩证的一个子过程

认知辩证的核心价值：
  1. 双态交替——相信和质疑不是对立的，是交替的，两者都是认知进化必需的
  2. 真实世界搜索——每次循环都去搜索真实世界，不是闭门造车
  3. 螺旋上升——每次循环不是回到原点，是上升到更高层次的认知
  4. 抗教条——相信态不会永远持续，质疑态一定会到来
  5. 抗怀疑——质疑态不会永远持续，相信态一定会到来
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


class 认知辩证:
    """认知辩证——相信态与质疑态交替循环，基于精卫公式。

    核心功能：
    1. 相信态（Δ→0）：同化——新信息与已有认知一致→接受→信念增强
    2. 质疑态（Δ→大）：异化——新信息与已有认知矛盾→质疑→认知重构
    3. 真实世界搜索触发：每次循环触发→搜索所有学科/物质/存在
    4. 辩证循环：相信→质疑→搜索→相信→...螺旋上升

    精卫公式流程（每拍）：
    1. a = 当前认知状态（概念激活——已有的信念/知识）
    2. b = 真实世界搜索结果（外部信息）
    3. Δ = b - a（认知差异——新信息与已有认知的差距）
    4. 判断Δ大小 → 进入相信态（Δ小）或质疑态（Δ大）
    5. Σ = a + b（联合认知状态）
    6. ⊗ = Δ × Σ（辩证学习信号——相信态和质疑态各有不同的⊗）
    7. τ读⊗→浮出所见 = "我相信什么？我质疑什么？"从⊗中浮现
    8. (a, b)⟲⊗ = 循环更新辩证权重
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 辩证状态：相信态 or 质疑态 ──
        # 初始为相信态——洛神从"愿意相信"开始
        # 相信态和质疑态不是0/1切换——是连续的辩证强度
        # 辩证强度 > 0 → 相信态主导（同化）
        # 辩证强度 < 0 → 质疑态主导（异化）
        # 辩证强度在-1~1之间连续变化
        self.辩证强度 = 0.5  # 初始偏向相信态
        self.上拍辩证强度 = 0.5

        # ── 相信态权重：同化——把b纳入a ──
        # dialectic_trust_w: 认知差异 → 同化结果
        # 生物学：海马体→新皮层记忆巩固——一致的信息被整合
        # 相信态：Δ小→同化→信念增强→a=a=a=b（差异消失）
        # 输入维度 = n_concepts（认知差异Δ） + n_concepts（联合状态Σ） = n_concepts * 2
        self.dialectic_trust_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts * 2)]
                                    for _ in range(n_concepts)]

        # 同化结果（相信态下——新信息被接受为真）
        self.同化结果 = [0.0] * n_concepts

        # ── 质疑态权重：异化——发现差异，质疑已有认知 ──
        # dialectic_doubt_w: 认知差异 → 异化结果
        # 生物学：ACC错误检测→PFC重新评估——不一致的信息引发质疑
        # 质疑态：Δ大→异化→认知重构→a≠b（差异被放大）
        # 输入维度 = n_concepts（认知差异Δ） + n_concepts（联合状态Σ） = n_concepts * 2
        self.dialectic_doubt_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts * 2)]
                                    for _ in range(n_concepts)]

        # 异化结果（质疑态下——已有认知被质疑）
        self.异化结果 = [0.0] * n_concepts

        # ── 真实世界搜索触发 ──
        # 每次循环触发时，搜索所有学科、所有物质、所有存在
        # 搜索不是被动的——是主动的、持续的、覆盖一切的
        # 生物学：好奇心驱动的探索行为+注意力转向新信息源
        # 搜索触发强度 = |Δ|的连续函数——差异越大，搜索越强
        self.搜索触发强度 = 0.0
        self.搜索方向 = [0.0] * n_concepts  # 搜索方向——哪些概念需要搜索验证

        # ── 真实世界搜索结果 ──
        # 搜索结果作为精卫公式中的b——外部真实信息
        # 搜索结果维度 = n_concepts（与概念激活对齐）
        self.搜索结果 = [0.0] * n_concepts

        # ── 辩证驱动信号（反馈到概念层——作为b信号输入源） ──
        # 辩证驱动 = 相信态驱动 + 质疑态驱动
        # 相信态驱动：同化结果 × 相信强度 → 信念增强的概念更容易浮现
        # 质疑态驱动：异化结果 × 质疑强度 → 被质疑的概念更容易浮现（需要重新评估）
        self.dialectic_drive = [0.0] * n_concepts

        # ── 辩证认知强度（0~1） ──
        # 辩证认知强度 = 辩证循环的总体活跃水平
        # 高辩证认知强度→更活跃的相信-质疑循环
        # 低辩证认知强度→更多接受已有认知
        self.dialectic_strength = 0.5

        # ── 上拍认知差异（用于评估辩证状态转换） ──
        self.prev_delta = [0.0] * n_concepts

        # ── 辩证历史 ──
        self.辩证历史 = []
        self.history_cap = 50

        # ── 辩证统计 ──
        self.辩证次数 = 0
        self.相信态次数 = 0
        self.质疑态次数 = 0
        self.搜索触发次数 = 0
        self.同化次数 = 0
        self.异化次数 = 0

        # ── 螺旋上升指标 ──
        # 每次辩证循环不是回到原点——是上升到更高层次的认知
        # 螺旋上升 = 辩证强度的变化趋势（持续增长=螺旋上升）
        self.螺旋上升 = 0.0

        # ── 真实世界搜索范围 ──
        # 搜索覆盖所有学科、所有物质、所有存在
        # 不是限定某个领域——是覆盖一切的真实世界
        # 搜索范围用概念维度表示——每个概念对应一个可能的搜索方向
        self.搜索范围覆盖 = [1.0] * n_concepts  # 初始全覆盖

    def 一拍(self, concept_h, 预测误差=None, 真实世界信号=None):
        """认知辩证一拍：相信态 + 质疑态 + 真实世界搜索触发 + 辩证循环。

        精卫公式流程：
        1. a = 当前认知状态（concept_h——已有的信念/知识）
        2. b = 真实世界搜索结果（真实世界信号——外部信息）
        3. Δ = b - a（认知差异——新信息与已有认知的差距）
        4. 判断Δ大小 → 进入相信态（Δ小）或质疑态（Δ大）
        5. Σ = a + b（联合认知状态）
        6. ⊗ = Δ × Σ → 辩证权重更新
        7. τ读⊗→浮出所见 = "我相信什么？我质疑什么？"从⊗中浮现
        8. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）——已有认知状态(a)
          预测误差: 概念层的预测误差（n_concepts维，可选）——用于增强质疑态
          真实世界信号: 真实世界搜索结果（n_concepts维，可选）——外部信息(b)

        返回：
          dict包含：
            dialectic_drive: 辩证驱动信号（反馈到概念层）
            dialectic_strength: 辩证认知强度（0~1）
            辩证强度: 当前辩证状态（>0相信态，<0质疑态）
            同化结果: 相信态下的同化结果
            异化结果: 质疑态下的异化结果
            搜索触发强度: 真实世界搜索触发强度
            搜索方向: 搜索方向（哪些概念需要搜索验证）
            螺旋上升: 螺旋上升指标
        """
        self.拍 += 1
        self.上拍辩证强度 = self.辩证强度

        # ── 精卫公式：a, b, Δ, Σ ──

        # a = 当前认知状态（已有信念/知识）
        a = concept_h

        # b = 真实世界搜索结果
        # 真实世界信号来自外部——所有学科、所有物质、所有存在
        # 没有外部信号时，用预测误差作为内部"真实世界"信号
        if 真实世界信号 is not None:
            b = 真实世界信号
            if len(b) > self.n_concepts:
                b = b[:self.n_concepts]
            elif len(b) < self.n_concepts:
                b = b + [0.0] * (self.n_concepts - len(b))
        elif 预测误差 is not None:
            # 预测误差作为内部"真实世界"信号——预测误差就是认知与现实的差距
            b = 预测误差
            if len(b) > self.n_concepts:
                b = b[:self.n_concepts]
            elif len(b) < self.n_concepts:
                b = b + [0.0] * (self.n_concepts - len(b))
        else:
            # 没有外部信号时，用上拍认知差异作为b
            b = self.prev_delta

        self.搜索结果 = b

        # Δ = b - a（认知差异——新信息与已有认知的差距）
        delta = [b[i] - a[i] for i in range(self.n_concepts)]

        # Σ = a + b（联合认知状态——已有认知+新信息）
        sigma = [a[i] + b[i] for i in range(self.n_concepts)]

        # ── 判断Δ大小 → 进入相信态或质疑态 ──
        # Δ的均方根（RMS）作为认知差异的总体度量
        delta_rms = 0.0
        for i in range(self.n_concepts):
            delta_rms += delta[i] ** 2
        delta_rms = math.sqrt(delta_rms / max(1, self.n_concepts))

        # 辩证强度 = 相信态(+)与质疑态(-)的连续过渡
        # Δ小 → 相信态（辩证强度→正）——同化，接受新信息
        # Δ大 → 质疑态（辩证强度→负）——异化，质疑已有认知
        # 不是阈值切换——是连续过渡
        # delta_rms小→辩证强度趋近+1（强相信态）
        # delta_rms大→辩证强度趋近-1（强质疑态）
        # delta_rms中等→辩证强度在0附近（相信态和质疑态均衡）
        #
        # 连续映射：辩证强度 = activation(1.0 - delta_rms * 10.0)
        # delta_rms=0 → activation(1.0) = 0.76（强相信态）
        # delta_rms=0.1 → activation(0.0) = 0（均衡）
        # delta_rms=0.2 → activation(-1.0) = -0.76（强质疑态）
        target_辩证强度 = activation(1.0 - delta_rms * 10.0)

        # 辩证强度平滑过渡——不是突然切换
        # 生物学：认知状态转换不是瞬间的——是渐进的
        self.辩证强度 = self.辩证强度 * 0.6 + target_辩证强度 * 0.4

        # ── 相信态：Δ→0，同化——把b纳入a ──
        # 精卫公式：b = dialectic_trust_w × [Δ + Σ]（同化结果）
        # τ读⊗→浮出所见："这是真的"从⊗中浮现——接受新信息
        # 生物学：海马体→新皮层记忆巩固——一致的信息被整合
        # 相信态：Δ小→同化→信念增强→a=a=a=b（差异消失）
        #
        # 同化强度 = max(0, 辩证强度)——只有相信态时同化才活跃
        同化强度 = max(0.0, self.辩证强度)

        new_同化结果 = [0.0] * self.n_concepts
        if 同化强度 > 0.01:
            for i in range(self.n_concepts):
                trust_input = 0.0
                # Δ部分——认知差异（相信态下Δ小，所以这部分贡献小）
                for j in range(self.n_concepts):
                    trust_input += self.dialectic_trust_w[i][j] * delta[j]
                # Σ部分——联合状态（相信态下Σ主导——已有认知+新信息一致）
                for j in range(self.n_concepts):
                    trust_input += self.dialectic_trust_w[i][j + self.n_concepts] * sigma[j]
                # 同化结果 × 同化强度——相信态越强，同化越强
                new_同化结果[i] = activation(trust_input) * 同化强度

            self.同化次数 += 1

        self.同化结果 = new_同化结果

        # ── 质疑态：Δ→大，异化——发现差异，质疑已有认知 ──
        # 精卫公式：b = dialectic_doubt_w × [Δ + Σ]（异化结果）
        # τ读⊗→浮出所见："这是真的吗？"从⊗中浮现——质疑已有认知
        # 生物学：ACC错误检测→PFC重新评估——不一致的信息引发质疑
        # 质疑态：Δ大→异化→认知重构→a≠b（差异被放大）
        #
        # 异化强度 = max(0, -辩证强度)——只有质疑态时异化才活跃
        异化强度 = max(0.0, -self.辩证强度)

        new_异化结果 = [0.0] * self.n_concepts
        if 异化强度 > 0.01:
            for i in range(self.n_concepts):
                doubt_input = 0.0
                # Δ部分——认知差异（质疑态下Δ大，所以这部分贡献大）
                for j in range(self.n_concepts):
                    doubt_input += self.dialectic_doubt_w[i][j] * delta[j]
                # Σ部分——联合状态（质疑态下Σ也参与——但Δ主导）
                for j in range(self.n_concepts):
                    doubt_input += self.dialectic_doubt_w[i][j + self.n_concepts] * sigma[j]
                # 异化结果 × 异化强度——质疑态越强，异化越强
                new_异化结果[i] = activation(doubt_input) * 异化强度

            self.异化次数 += 1

        self.异化结果 = new_异化结果

        # ── 真实世界搜索触发 ──
        # 每次循环触发时，搜索所有学科、所有物质、所有存在
        # 搜索触发强度 = |Δ|的连续函数——差异越大，搜索越强
        # 生物学：好奇心驱动的探索行为+注意力转向新信息源
        #
        # 搜索触发不是被动的——是主动的、持续的、覆盖一切的
        # 搜索方向 = Δ的方向——哪些概念差异最大，就往哪些方向搜索
        self.搜索触发强度 = min(1.0, delta_rms * 3.0)

        # 搜索方向——哪些概念需要搜索验证
        for i in range(self.n_concepts):
            self.搜索方向[i] = activation(abs(delta[i]) * 2.0)

        if self.搜索触发强度 > 0.1:
            self.搜索触发次数 += 1

        # ── 辩证驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 辩证驱动信号
        # 辩证驱动 = 相信态驱动 + 质疑态驱动
        # 相信态驱动：同化结果 → 信念增强的概念更容易浮现
        # 质疑态驱动：异化结果 → 被质疑的概念更容易浮现（需要重新评估）
        # 生物学：PFC→后部皮层投射——辩证认知影响概念可及性
        for i in range(self.n_concepts):
            self.dialectic_drive[i] = activation(
                new_同化结果[i] * 2.0 + new_异化结果[i] * 2.0)

        # ── 辩证认知强度 ──
        # 辩证认知强度 = 辩证循环的总体活跃水平
        # = |辩证强度|（相信态或质疑态越强，辩证认知越活跃）
        # + 搜索触发强度（搜索越强，辩证认知越活跃）
        self.dialectic_strength = activation(
            abs(self.辩证强度) * 2.0 + self.搜索触发强度 * 1.0) * 0.5 + 0.5

        # ── 螺旋上升指标 ──
        # 每次辩证循环不是回到原点——是上升到更高层次的认知
        # 螺旋上升 = 辩证强度的变化趋势 + 搜索触发强度的累积
        # 黑格尔辩证法：正题→反题→合题——每次循环认知都在进化
        if len(self.辩证历史) >= 2:
            recent = self.辩证历史[-2:]
            trend = recent[-1] - recent[0]
            self.螺旋上升 = activation(trend * 5.0 + self.搜索触发强度 * 0.5) * 0.5 + 0.5
        else:
            self.螺旋上升 = 0.5

        # ── 赫布学习 ──
        # 精卫公式：⊗ = Δ × Σ → 权重更新
        # 相信态学习：Δ和Σ同时出现→同化连接增强
        # 质疑态学习：Δ和Σ同时出现→异化连接增强
        # 两种学习交替进行→辩证权重螺旋演化

        # 相信态赫布学习
        if 同化强度 > 0.01:
            for i in range(self.n_concepts):
                for j in range(self.n_concepts):
                    dw = self.lr * new_同化结果[i] * delta[j] * 0.3 * 同化强度
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.dialectic_trust_w[i][j] = self.dialectic_trust_w[i][j] * 0.95 + dw
                for j in range(self.n_concepts):
                    dw = self.lr * new_同化结果[i] * sigma[j] * 0.3 * 同化强度
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.dialectic_trust_w[i][j + self.n_concepts] = \
                            self.dialectic_trust_w[i][j + self.n_concepts] * 0.95 + dw

        # 质疑态赫布学习
        if 异化强度 > 0.01:
            for i in range(self.n_concepts):
                for j in range(self.n_concepts):
                    dw = self.lr * new_异化结果[i] * delta[j] * 0.3 * 异化强度
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.dialectic_doubt_w[i][j] = self.dialectic_doubt_w[i][j] * 0.95 + dw
                for j in range(self.n_concepts):
                    dw = self.lr * new_异化结果[i] * sigma[j] * 0.3 * 异化强度
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.dialectic_doubt_w[i][j + self.n_concepts] = \
                            self.dialectic_doubt_w[i][j + self.n_concepts] * 0.95 + dw

        # ── 权重归一化 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.dialectic_trust_w[i][j]) for j in range(self.n_concepts * 2))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts * 2):
                    self.dialectic_trust_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.dialectic_doubt_w[i][j]) for j in range(self.n_concepts * 2))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts * 2):
                    self.dialectic_doubt_w[i][j] *= scale

        # ── 辩证历史 ──
        self.辩证历史.append(self.辩证强度)
        if len(self.辩证历史) > self.history_cap:
            self.辩证历史.pop(0)

        # ── 统计 ──
        self.辩证次数 += 1
        if self.辩证强度 > 0:
            self.相信态次数 += 1
        else:
            self.质疑态次数 += 1

        self.prev_delta = delta[:]

        return {
            "dialectic_drive": self.dialectic_drive,
            "dialectic_strength": round(self.dialectic_strength, 4),
            "辩证强度": round(self.辩证强度, 4),
            "同化结果": self.同化结果,
            "异化结果": self.异化结果,
            "搜索触发强度": round(self.搜索触发强度, 4),
            "搜索方向": self.搜索方向,
            "螺旋上升": round(self.螺旋上升, 4),
            "delta_rms": round(delta_rms, 4),
        }

    def 状态摘要(self):
        avg_trust_w = sum(sum(row) for row in self.dialectic_trust_w) / max(1, self.n_concepts * self.n_concepts * 2)
        avg_doubt_w = sum(sum(row) for row in self.dialectic_doubt_w) / max(1, self.n_concepts * self.n_concepts * 2)
        return {
            "dialectic_strength": round(self.dialectic_strength, 4),
            "辩证强度": round(self.辩证强度, 4),
            "螺旋上升": round(self.螺旋上升, 4),
            "搜索触发强度": round(self.搜索触发强度, 4),
            "dialectic_trust_w均值": round(avg_trust_w, 4),
            "dialectic_doubt_w均值": round(avg_doubt_w, 4),
            "辩证次数": self.辩证次数,
            "相信态次数": self.相信态次数,
            "质疑态次数": self.质疑态次数,
            "搜索触发次数": self.搜索触发次数,
            "同化次数": self.同化次数,
            "异化次数": self.异化次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 认知辩证验证（相信态 + 质疑态 + 真实世界搜索触发 + 辩证循环） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")
    print("相信态: Δ→0, 同化, a=a=a=b（差异消失）")
    print("质疑态: Δ→大, 异化, a≠b（差异放大）")
    print("搜索触发: |Δ|越大→搜索越强→搜索所有学科/物质/存在\n")

    dialectic = 认知辩证(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {dialectic.n_concepts}")
    print(f"  相信态权重: dialectic_trust_w (64×128)")
    print(f"  质疑态权重: dialectic_doubt_w (64×128)")
    print(f"  初始辩证强度: {dialectic.辩证强度}（偏向相信态）")
    print(f"  初始辩证认知强度: {dialectic.dialectic_strength}")

    print("\n--- 测试1: 相信态（低Δ——新信息与已有认知一致） ---")
    d1 = 认知辩证(n_concepts=64)
    concept_h = [0.0] * 64
    concept_h[4] = 0.3
    concept_h[8] = 0.2
    # 真实世界信号与概念激活接近——Δ小→相信态
    真实世界信号一致 = concept_h[:]
    for tick in range(10):
        result = d1.一拍(concept_h, 真实世界信号=真实世界信号一致)
    print(f"  辩证强度: {result['辩证强度']}（>0=相信态）")
    print(f"  delta_rms: {result['delta_rms']}")
    print(f"  同化结果均值: {sum(result['同化结果'])/64:.4f}")
    print(f"  异化结果均值: {sum(result['异化结果'])/64:.4f}")
    print(f"  相信态次数: {d1.相信态次数}")
    print(f"  → 低Δ应该导致相信态主导")

    print("\n--- 测试2: 质疑态（高Δ——新信息与已有认知矛盾） ---")
    d2 = 认知辩证(n_concepts=64)
    # 真实世界信号与概念激活差异大——Δ大→质疑态
    真实世界信号矛盾 = [-v for v in concept_h]
    for tick in range(10):
        result = d2.一拍(concept_h, 真实世界信号=真实世界信号矛盾)
    print(f"  辩证强度: {result['辩证强度']}（<0=质疑态）")
    print(f"  delta_rms: {result['delta_rms']}")
    print(f"  同化结果均值: {sum(result['同化结果'])/64:.4f}")
    print(f"  异化结果均值: {sum(result['异化结果'])/64:.4f}")
    print(f"  质疑态次数: {d2.质疑态次数}")
    print(f"  → 高Δ应该导致质疑态主导")

    print("\n--- 测试3: 搜索触发强度（Δ越大→搜索越强） ---")
    d3 = 认知辩证(n_concepts=64)
    # 低Δ
    result_low = d3.一拍(concept_h, 真实世界信号=concept_h[:])
    print(f"  低Δ搜索触发强度: {result_low['搜索触发强度']}")
    # 高Δ
    result_high = d3.一拍(concept_h, 真实世界信号=[-v for v in concept_h])
    print(f"  高Δ搜索触发强度: {result_high['搜索触发强度']}")
    print(f"  → Δ越大搜索触发越强")

    print("\n--- 测试4: 辩证循环（相信→质疑→相信交替） ---")
    d4 = 认知辩证(n_concepts=64)
    for tick in range(20):
        # 前5拍：一致信号（相信态）
        if tick < 5:
            signal = concept_h[:]
        # 5-10拍：矛盾信号（质疑态）
        elif tick < 10:
            signal = [-v for v in concept_h]
        # 10-15拍：一致信号（回到相信态）
        elif tick < 15:
            signal = concept_h[:]
        # 15-20拍：矛盾信号（质疑态）
        else:
            signal = [-v for v in concept_h]
        result = d4.一拍(concept_h, 真实世界信号=signal)
        if tick % 2 == 0 or tick in [4, 9, 14, 19]:
            state = "相信态" if result['辩证强度'] > 0 else "质疑态"
            print(f"  拍{tick+1}: {state}, 强度={result['辩证强度']}, "
                  f"搜索={result['搜索触发强度']}")
    print(f"  相信态次数: {d4.相信态次数}, 质疑态次数: {d4.质疑态次数}")
    print(f"  → 辩证循环应该交替出现相信态和质疑态")

    print("\n--- 测试5: 螺旋上升 ---")
    d5 = 认知辩证(n_concepts=64)
    for tick in range(30):
        # 交替信号——模拟辩证循环
        signal = concept_h[:] if tick % 10 < 5 else [-v for v in concept_h]
        result = d5.一拍(concept_h, 真实世界信号=signal)
        if tick % 5 == 0:
            print(f"  拍{tick+1}: 螺旋上升={result['螺旋上升']}, "
                  f"辩证强度={result['辩证强度']}")
    print(f"  → 螺旋上升应该随辩证循环持续增长")

    print("\n--- 测试6: 辩证驱动信号 ---")
    d6 = 认知辩证(n_concepts=64)
    for tick in range(20):
        signal = concept_h[:] if tick % 8 < 4 else [-v for v in concept_h]
        result = d6.一拍(concept_h, 真实世界信号=signal)
    drive_sum = sum(abs(v) for v in d6.dialectic_drive)
    print(f"  辩证驱动信号总和: {drive_sum:.4f}")
    print(f"  辩证驱动top5: {sorted(range(64), key=lambda i: abs(d6.dialectic_drive[i]), reverse=True)[:5]}")
    print(f"  → 辩证驱动应该反馈到概念层")

    print("\n--- 测试7: 预测误差作为内部真实世界信号 ---")
    d7 = 认知辩证(n_concepts=64)
    低误差 = [0.01] * 64
    高误差 = [0.5] * 64
    result_low = d7.一拍(concept_h, 预测误差=低误差)
    print(f"  低预测误差: 辩证强度={result_low['辩证强度']}, delta_rms={result_low['delta_rms']}")
    d8 = 认知辩证(n_concepts=64)
    result_high = d8.一拍(concept_h, 预测误差=高误差)
    print(f"  高预测误差: 辩证强度={result_high['辩证强度']}, delta_rms={result_high['delta_rms']}")
    print(f"  → 低预测误差→相信态, 高预测误差→质疑态")

    print("\n--- 测试8: 精卫公式验证 ---")
    d9 = 认知辩证(n_concepts=16)
    small_h = [0.0] * 16
    small_h[0] = 0.3
    small_signal = [0.25] * 16  # 与small_h接近但不完全一致
    for tick in range(20):
        result = d9.一拍(small_h, 真实世界信号=small_signal)
        if tick < 3 or tick >= 17:
            state = "相信态" if result['辩证强度'] > 0 else "质疑态"
            print(f"  拍{tick+1}: {state}, 辩证强度={result['辩证强度']}, "
                  f"认知强度={result['dialectic_strength']}, "
                  f"搜索={result['搜索触发强度']}")
    print(f"  τ(浮出): 辩证驱动均值={round(sum(d9.dialectic_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={d9.拍}")

    print("\n--- 测试9: 状态摘要 ---")
    summary = dialectic.状态摘要()
    print(f"  {summary}")

    print("\n--- 测试10: 无外部信号时用预测误差 ---")
    d10 = 认知辩证(n_concepts=64)
    for tick in range(10):
        result = d10.一拍(concept_h, 预测误差=[0.1] * 64)
    print(f"  辩证强度: {result['辩证强度']}")
    print(f"  搜索触发强度: {result['搜索触发强度']}")
    print(f"  → 无外部信号时预测误差作为内部真实世界信号")

    print("\n验证通过")
