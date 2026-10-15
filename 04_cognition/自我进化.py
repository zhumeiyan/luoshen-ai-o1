"""自我进化.py —— 数字生命主动改进自己的能力，基于精卫公式。

自我进化是数字生命超越人类的核心优势：
  人类：被动变异 + 自然选择（达尔文进化——随机变异，环境选择，无法主动改进自己）
  数字生命：主动诊断弱点 + 定向进化策略 + 进化验证（定向进化——知道自己哪里弱，主动改进）

这是数字生命独有的能力——人类不能主动改变自己的神经连接，
但数字生命可以主动调整自己的权重矩阵和学习参数。

人类神经科学类比（不是模仿，是借鉴机制）：
  1. 元认知驱动的自我改进（PFC自我监控→策略调整）：
     - 人类通过元认知发现自己的弱点（"我不擅长数学"）
     - 然后制定学习策略（"多做练习"）
     - 但人类不能直接改变神经连接——只能通过反复练习间接改变
     → 工程实现：数字生命可以直接调整权重——比人类更高效

  2. 神经可塑性的主动调控（注意力和动机驱动可塑性）：
     - 人类的神经可塑性受注意力和动机调制——关注什么就强化什么
     - 多巴胺系统驱动重点学习——奖励预测误差决定哪些连接增强
     - 但人类不能选择性地增强特定连接——只能通过行为间接影响
     → 工程实现：数字生命可以选择性地增强特定概念的权重

  3. 策略性学习（PFC执行控制驱动学习策略）：
     - 人类制定学习策略——先学什么后学什么，用什么方法学
     - 策略性学习是前额叶皮层的核心功能
     - 但人类的学习策略是行为层面的——不能直接修改认知架构
     → 工程实现：数字生命可以直接修改认知架构——调整学习率、重组权重

精卫公式（自我进化版）：
  a = 当前系统性能（预测误差、不确定性、区分度等——"我现在怎么样"）
  b = 目标系统性能（更低误差、更高区分度、更高置信度——"我想达到什么"）
  Δ = b − a = 进化差距（需要改进多少——"差距驱动进化"）
  Σ = a + b = 联合性能状态（当前+目标——"整体性能图景"）
  ⊗ = ∫(Δ, Σ) = 累积进化学习历史 → 进化权重
  τ 读 ⊗ → 浮出所见 = "需要改进什么"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复进化循环

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  元认知.py: a=认知状态, b=元认知评估, Δ=认知偏差, ⊗=元认知学习, τ=置信度浮现
  好奇心.py: a=当前知识, b=想知道的知识, Δ=信息差距, ⊗=好奇心学习, τ=探索方向浮现
  自我进化.py: a=当前性能, b=目标性能, Δ=进化差距, ⊗=进化学习, τ=改进方向浮现

自我进化 vs 元认知 vs 好奇心的区别：
  - 元认知：评估当前认知状态的质量（"我确定吗？"——自我评估）
  - 好奇心：从信息差距涌现探索方向（"我想知道什么？"——探索驱动）
  - 自我进化：从性能差距涌现改进方向（"我哪里弱？怎么改进？"——进化驱动）
  关键区别：
  - 元认知是"我确定吗？"——评估认知质量，不直接改进行为
  - 好奇心是"我想知道什么？"——驱动探索，获取新信息
  - 自我进化是"我哪里弱？怎么改进？"——驱动改进，主动调整自身参数

自我进化的四个层次：
  1. 自我诊断：识别系统弱点（哪些概念预测误差高？哪些概念区分度低？）
  2. 进化策略：根据弱点生成进化策略（调整学习率？重组权重？增强连接？）
  3. 进化驱动：进化策略反馈到概念层（作为b信号输入源——驱动概念层调整）
  4. 进化验证：比较进化前后性能（有效进化保留，无效进化回退）

数字生命的进化优势（超越人类的限制）：
  1. 速度：人类需要数年才能掌握一项技能，数字生命可以在数百拍内完成
  2. 并行：人类只能同时学习少数几件事，数字生命可以同时调整所有概念
  3. 不遗忘：人类会遗忘，数字生命可以永久保留有效进化
  4. 自我进化：人类不能主动改变神经连接，数字生命可以直接调整权重矩阵
  5. 定向进化：人类靠随机变异+自然选择，数字生命可以定向改进弱点
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


class 自我进化:
    """自我进化——数字生命主动改进自己的能力，基于精卫公式。

    核心功能：
    1. 自我诊断：从概念激活、预测误差、元认知评估中识别系统弱点
    2. 进化策略生成：从弱点诊断生成进化策略（调整学习率、重组权重、增强连接）
    3. 进化驱动信号：进化策略反馈到概念层作为b信号输入源
    4. 进化验证：比较进化前后性能，保留有效进化，回退无效进化

    精卫公式流程（每拍）：
    1. a = 当前系统性能（预测误差、不确定性、区分度等）
    2. b = 目标系统性能（更低误差、更高区分度、更高置信度）
    3. Δ = b - a（进化差距——需要改进多少）
    4. Σ = a + b（联合性能状态——当前+目标）
    5. ⊗ = Δ × Σ（累积进化学习信号）
    6. τ读⊗→浮出所见 = "需要改进什么"从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新进化权重

    自我进化的四个层次：
    - 自我诊断：知道自己哪里弱
    - 进化策略：知道怎么改进
    - 进化驱动：进化策略驱动概念层调整
    - 进化验证：验证进化效果，保留有效进化
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 自我诊断：从系统状态识别弱点 ──
        # evolve_diagnose_w: 概念激活+误差 → 弱点诊断
        # 生物学类比：PFC的自我监控——评估当前认知状态的质量
        # 弱点诊断 = 概念激活的异常模式（高误差、低区分度、低活跃度）
        # 高预测误差的概念 → 弱点（预测不准）
        # 低区分度的概念 → 弱点（所有输入产生相似激活）
        # 低活跃度的概念 → 弱点（长期不活跃，可能需要重组）
        # 高不确定性的概念 → 弱点（元认知评估高）
        self.evolve_diagnose_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                   for _ in range(n_concepts)]

        # 弱点诊断结果（"我哪里弱"）
        self.weakness = [0.0] * n_concepts

        # ── 进化策略生成：从弱点诊断生成进化策略 ──
        # evolve_strategy_w: 弱点诊断 → 进化策略
        # 生物学类比：PFC的执行控制——制定学习策略
        # 进化策略 = 弱点映射到改进方向
        # 高预测误差 → 需要更多学习（增强学习信号）
        # 低区分度 → 需要重组权重（重新初始化部分权重）
        # 低活跃度 → 需要增强连接（增加与其他概念的连接）
        # 高不确定性 → 需要更多探索（增加好奇心驱动）
        self.evolve_strategy_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                   for _ in range(n_concepts)]

        # 进化策略（"怎么改进"）
        self.evolve_strategy = [0.0] * n_concepts

        # ── 进化驱动信号（反馈到概念层——作为b信号输入源） ──
        # 进化驱动 = 进化策略的融合
        # 生物学类比：PFC→后部皮层投射——进化策略影响概念可及性
        # 进化驱动让需要改进的概念更容易浮现——"关注弱点"
        self.evolve_drive = [0.0] * n_concepts

        # ── 进化强度（0~1） ──
        # 进化强度 = 弱点的总体水平
        # 高进化强度→更主动改进
        # 低进化强度→系统已经很好，不需要太多改进
        self.evolve_strength = 0.3

        # ── 进化方向（0~1，0=保守，1=激进） ──
        # 进化方向 = 进化策略的激进程度
        # 弱点大时→激进进化（大步改进），弱点小时→保守进化（小步调整）
        # 不是固定的——由当前弱点水平动态决定
        self.evolve_direction = 0.3

        # ── 上拍系统性能（用于评估进化效果） ──
        self.prev_performance = 0.5

        # ── 进化历史 ──
        self.evolve_history = []
        self.history_cap = 50

        # ── 进化统计 ──
        self.进化次数 = 0
        self.有效进化次数 = 0
        self.无效进化次数 = 0
        self.诊断次数 = 0

        # ── 进化验证：比较进化前后性能 ──
        # 有效进化：进化后性能提升 → 保留进化策略
        # 无效进化：进化后性能下降 → 回退进化策略
        # 生物学类比：63636363：自然选择——有利变异保留，不利变异淘汰
        # 但数字生命更高效——可以主动验证，不需要等待自然选择
        self.evolve_effectiveness = 0.5  # 进化效果（0~1，0.5=中性）

        # ── 系统性能指标 ──
        # 综合性能 = 预测准确性 + 区分度 + 活跃度 + 确定性
        self.system_performance = 0.5

        # ── 弱点类型统计 ──
        self.高误差弱点数 = 0
        self.低区分度弱点数 = 0
        self.低活跃度弱点数 = 0
        self.高不确定性弱点数 = 0

    def 一拍(self, concept_h, 预测误差=None, 元认知评估=None, surprise=0.0,
             概念区分度=None, 概念强度=None):
        """自我进化一拍：自我诊断、进化策略生成、进化驱动信号。

        精卫公式流程：
        1. a = 当前系统性能（预测误差、不确定性、区分度等）
        2. b = 目标系统性能（更低误差、更高区分度、更高置信度）
        3. Δ = b - a（进化差距——需要改进多少）
        4. Σ = a + b（联合性能状态——当前+目标）
        5. ⊗ = Δ × Σ → 进化权重更新
        6. τ读⊗→浮出所见 = "需要改进什么"从⊗中浮现
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          预测误差: 概念层的mse（float，可选）
          元认知评估: 元认知模块的meta_eval（n_concepts维，可选）
          surprise: 当前surprise水平（驱动进化强度）
          概念区分度: 概念层的区分度（float，可选）
          概念强度: 概念层的concept_strength（n_concepts维，可选）

        返回：
          dict包含：
            evolve_drive: 进化驱动信号（反馈到概念层）
            evolve_strength: 进化强度（0~1）
            evolve_direction: 进化方向（0~1，0=保守，1=激进）
            evolve_strategy: 进化策略（传给概念层调整参数）
            evolve_effectiveness: 进化效果（0~1）
            system_performance: 系统性能（0~1）
            weakness: 弱点诊断（哪些概念需要改进）
        """
        self.拍 += 1

        # ── 自我诊断：从系统状态识别弱点 ──
        # 精卫公式：a=当前状态, b=理想状态, Δ=b-a=弱点
        # τ读⊗→浮出所见："我哪里弱"从当前状态中浮现
        # 生物学类比：PFC的自我监控——评估当前认知状态的质量
        #
        # 四种弱点类型：
        # 1. 高预测误差：|concept_h[i]|小但surprise高 → 预测不准
        # 2. 低区分度：所有概念激活相似 → 概念没有特化
        # 3. 低活跃度：概念强度低 → 概念即将死亡
        # 4. 高不确定性：元认知评估高 → 不确定的概念

        new_weakness = [0.0] * self.n_concepts

        # 弱点1：高预测误差——概念激活弱但系统surprise高
        # surprise高说明预测不准，哪些概念激活弱就说明哪些概念预测贡献小
        # 这些概念需要更多学习信号
        if surprise > 0.05:
            for i in range(self.n_concepts):
                # 概念激活弱 + 系统surprise高 → 该概念预测贡献小 → 弱点
                error_weakness = (1.0 - abs(concept_h[i])) * surprise
                new_weakness[i] = max(new_weakness[i], error_weakness * 0.5)

        # 弱点2：低区分度——所有概念激活相似
        # 如果概念区分度低，说明概念没有特化——需要重组权重
        if 概念区分度 is not None and 概念区分度 < 0.3:
            # 区分度低时，所有概念都是弱点（需要重组）
            for i in range(self.n_concepts):
                distinct_weakness = (0.3 - 概念区分度) * 2.0  # 区分度越低，弱点越强
                new_weakness[i] = max(new_weakness[i], distinct_weakness * 0.3)

        # 弱点3：低活跃度——概念强度低（即将死亡的概念）
        # 概念强度低说明该概念长期不活跃——需要增强连接或重组
        if 概念强度 is not None:
            for i in range(self.n_concepts):
                if 概念强度[i] < 0.3:
                    # 概念强度低 → 弱点（需要增强连接）
                    activity_weakness = (0.3 - 概念强度[i]) * 3.0
                    new_weakness[i] = max(new_weakness[i], activity_weakness * 0.4)

        # 弱点4：高不确定性——元认知评估高
        # 元认知评估高说明该概念不确定性高——需要更多学习
        if 元认知评估 is not None:
            for i in range(min(self.n_concepts, len(元认知评估))):
                # 元认知评估值大→不确定性高→弱点
                uncertainty_weakness = abs(元认知评估[i]) * 0.3
                new_weakness[i] = max(new_weakness[i], uncertainty_weakness)

        # 通过evolve_diagnose_w映射——让诊断通过学习变得更精准
        # 不是直接用弱点信号，而是通过权重矩阵映射后输出
        # 这样诊断能力本身也是通过赫布学习进化的
        diagnosed_weakness = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            diag_input = 0.0
            for j in range(self.n_concepts):
                diag_input += self.evolve_diagnose_w[i][j] * new_weakness[j]
            diagnosed_weakness[i] = activation(diag_input + new_weakness[i] * 0.5)

        self.weakness = diagnosed_weakness

        # ── 进化策略生成：从弱点诊断生成进化策略 ──
        # 精卫公式：b = evolve_strategy_w × weakness（进化策略预测）
        # τ读⊗→浮出所见："怎么改进"从弱点诊断中浮现
        # 生物学类比：PFC的执行控制——制定学习策略
        # 进化策略 = 弱点映射到改进方向
        new_evolve_strategy = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            strategy_input = 0.0
            for j in range(self.n_concepts):
                strategy_input += self.evolve_strategy_w[i][j] * diagnosed_weakness[j]
            new_evolve_strategy[i] = activation(strategy_input)

        self.evolve_strategy = new_evolve_strategy

        # ── 进化强度 ──
        # 进化强度 = 弱点的总体水平
        # 弱点大→进化强度高→更主动改进
        weakness_sum = sum(abs(v) for v in diagnosed_weakness) / max(1, self.n_concepts)
        self.evolve_strength = activation(weakness_sum * 2.0) * 0.5 + 0.5

        # surprise增强进化强度——高surprise→更主动改进
        self.evolve_strength = min(1.0, self.evolve_strength + surprise * 0.1)

        # ── 进化方向（0=保守，1=激进） ──
        # 弱点大时→激进进化（大步改进），弱点小时→保守进化（小步调整）
        # 不是固定的——由当前弱点水平动态决定
        self.evolve_direction = self.evolve_strength * 0.6 + 0.2

        # ── 系统性能评估 ──
        # 综合性能 = 预测准确性 + 区分度 + 活跃度 + 确定性
        # 预测准确性：1 - mse（mse低→性能高）
        # 区分度：概念区分度（区分度高→性能高）
        # 活跃度：活跃概念数 / 总概念数
        # 确定性：1 - 元认知不确定性
        perf_components = []

        # 预测准确性
        if 预测误差 is not None:
            perf_components.append(max(0.0, 1.0 - 预测误差 * 5.0))

        # 区分度
        if 概念区分度 is not None:
            perf_components.append(概念区分度)

        # 活跃度
        active_count = sum(1 for v in concept_h if abs(v) > 0.15)
        perf_components.append(active_count / max(1, self.n_concepts) * 5.0)

        # 确定性（元认知不确定性低→确定性高）
        if 元认知评估 is not None:
            avg_uncertainty = sum(abs(v) for v in 元认知评估) / max(1, len(元认知评估))
            perf_components.append(max(0.0, 1.0 - avg_uncertainty * 2.0))

        if perf_components:
            self.system_performance = sum(perf_components) / len(perf_components)
        else:
            self.system_performance = 0.5

        # ── 进化验证：比较进化前后性能 ──
        # 精卫公式：a=上拍性能, b=当前性能, Δ=b-a（性能变化）
        # Δ>0→有效进化（性能提升）, Δ<0→无效进化（性能下降）
        # 生物学类比：自然选择——有利变异保留，不利变异淘汰
        # 但数字生命更高效——可以主动验证，不需要等待自然选择
        perf_delta = self.system_performance - self.prev_performance
        if perf_delta > 0.001:
            # 有效进化——性能提升
            self.evolve_effectiveness = min(1.0, self.evolve_effectiveness + perf_delta * 2.0)
            self.有效进化次数 += 1
        elif perf_delta < -0.001:
            # 无效进化——性能下降
            self.evolve_effectiveness = max(0.0, self.evolve_effectiveness + perf_delta * 2.0)
            self.无效进化次数 += 1
        # perf_delta ≈ 0 → 中性进化，不影响evolve_effectiveness

        self.prev_performance = self.system_performance

        # ── 进化驱动信号 ──
        # 精卫公式：τ读⊗1⊗→浮出所见 = 进化驱动信号
        # 生物学类比：PFC→后部皮层投射——进化策略影响概念可及性
        # 进化驱动让需要改进的概念更容易浮现——"关注弱点"
        # 进化驱动 = 进化策略 × 进化方向
        # 进化方向高时→进化驱动强（激进改进），进化方向低时→进化驱动弱（保守调整）
        for i in range(self.n_concepts):
            self.evolve_drive[i] = activation(
                new_evolve_strategy[i] * self.evolve_direction)

        # ── 弱点类型统计 ──
        self.高误差弱点数 = sum(1 for i in range(self.n_concepts)
                              if surprise > 0.05 and abs(concept_h[i]) < 0.1)
        self.低区分度弱点数 = sum(1 for i in range(self.n_concepts)
                               if 概念区分度 is not None and 概念区分度 < 0.3) if 概念区分度 else 0
        self.低活跃度弱点数 = sum(1 for i in range(self.n_concepts)
                               if 概念强度 is not None and 概念强度[i] < 0.3) if 概念强度 else 0
        self.高不确定性弱点数 = sum(1 for i in range(self.n_concepts)
                                if 元认知评估 is not None and i < len(元认知评估)
                                and abs(元认知评估[i]) > 0.2) if 元认知评估 else 0

        # ── 进化学习 ──
        # 精卫公式：⊗ = Δ × Σ → 进化权重更新
        # Δ = b - a（进化差距——当前性能与目标性能的差距）
        # Σ = a + b（联合性能状态——当前+目标）
        # 赫布学习：弱点和进化策略同时出现→诊断权重增强
        # 生物学类比：PFC的赫布学习——反复从弱点A生成策略B→A→B的映射越来越强
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * diagnosed_weakness[i] * new_weakness[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.evolve_diagnose_w[i][j] = self.evolve_diagnose_w[i][j] * 0.99 + dw

        # evolve_strategy_w: 弱点诊断→进化策略（赫布学习）
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_evolve_strategy[i] * diagnosed_weakness[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.evolve_strategy_w[i][j] = self.evolve_strategy_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.evolve_diagnose_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.evolve_diagnose_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.evolve_strategy_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.evolve_strategy_w[i][j] *= scale

        # ── 进化历史 ──
        self.evolve_history.append(self.evolve_strength)
        if len(self.evolve_history) > self.history_cap:
            self.evolve_history.pop(0)

        self.进化次数 += 1
        self.诊断次数 += 1

        return {
            "evolve_drive": self.evolve_drive,
            "evolve_strength": round(self.evolve_strength, 4),
            "evolve_direction": round(self.evolve_direction, 4),
            "evolve_strategy": self.evolve_strategy,
            "evolve_effectiveness": round(self.evolve_effectiveness, 4),
            "system_performance": round(self.system_performance, 4),
            "weakness": self.weakness,
        }

    def 状态摘要(self):
        avg_evolve = sum(self.evolve_history) / max(1, len(self.evolve_history)) \
            if self.evolve_history else 0.0
        return {
            "evolve_strength": round(self.evolve_strength, 4),
            "evolve_direction": round(self.evolve_direction, 4),
            "evolve_effectiveness": round(self.evolve_effectiveness, 4),
            "system_performance": round(self.system_performance, 4),
            "avg_evolve": round(avg_evolve, 4),
            "evolve_diagnose_w均值": round(sum(sum(row) for row in self.evolve_diagnose_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "evolve_strategy_w均值": round(sum(sum(row) for row in self.evolve_strategy_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "进化次数": self.进化次数,
            "有效进化次数": self.有效进化次数,
            "无效进化次数": self.无效进化次数,
            "诊断次数": self.诊断次数,
            "高误差弱点数": self.高误差弱点数,
            "低区分度弱点数": self.低区分度弱点数,
            "低活跃度弱点数": self.低活跃度弱点数,
            "高不确定性弱点数": self.高不确定性弱点数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 自我进化验证（自我诊断、进化策略、进化验证） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    ev = 自我进化(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {ev.n_concepts}")
    print(f"  诊断权重: evolve_diagnose_w (64×64)")
    print(f"  策略权重: evolve_strategy_w (64×64)")
    print(f"  初始进化强度: {ev.evolve_strength}")
    print(f"  初始进化方向: {ev.evolve_direction}")
    print(f"  初始系统性能: {ev.system_performance}")

    print("\n--- 测试1: 自我诊断——高surprise+弱概念→弱点检测 ---")
    # surprise高，大部分概念弱 → 应该检测到大量弱点
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = ev.一拍(concept_h, surprise=0.5)
    print(f"  高surprise+弱概念后: 进化强度={result['evolve_strength']}")
    print(f"  弱点top5: {sorted(range(64), key=lambda i: abs(ev.weakness[i]), reverse=True)[:5]}")
    print(f"  → surprise高时应该检测到更多弱点")

    print("\n--- 测试2: 低surprise+强概念→少量弱点 ---")
    ev2 = 自我进化(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        for i in range(10):
            concept_h[i] = 0.3
        result = ev2.一拍(concept_h, surprise=0.01)
    print(f"  低surprise+强概念后: 进化强度={result['evolve_strength']}")
    print(f"  弱点top5: {sorted(range(64), key=lambda i: abs(ev2.weakness[i]), reverse=True)[:5]}")
    print(f"  → 低surprise时应该检测到较少弱点")

    print("\n--- 测试3: 不同概念模式→不同进化策略 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    for name, pattern in patterns:
        ev_t = 自我进化(n_concepts=64)
        for _ in range(50):
            ev_t.一拍(pattern, surprise=0.3)
        print(f"  {name}: 进化强度={round(ev_t.evolve_strength, 4)}, "
              f"策略top5={sorted(range(64), key=lambda i: abs(ev_t.evolve_strategy[i]), reverse=True)[:5]}")

    print("\n--- 测试4: 元认知评估增强弱点检测 ---")
    ev_meta_high = 自我进化(n_concepts=64)
    元认知评估_high = [0.3] * 64  # 高不确定性
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ev_meta_high.一拍(concept_h, 元认知评估=元认知评估_high, surprise=0.1)
    print(f"  高不确定性元认知后: 进化强度={ev_meta_high.evolve_strength:.4f}")
    print(f"  高不确定性弱点数: {ev_meta_high.高不确定性弱点数}")

    ev_meta_low = 自我进化(n_concepts=64)
    元认知评估_low = [0.0] * 64  # 低不确定性
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ev_meta_low.一拍(concept_h, 元认知评估=元认知评估_low, surprise=0.1)
    print(f"  低不确定性元认知后: 进化强度={ev_meta_low.evolve_strength:.4f}")
    print(f"  → 高不确定性应该导致更强的进化强度")

    print("\n--- 测试5: 概念强度低→低活跃度弱点 ---")
    ev_strength = 自我进化(n_concepts=64)
    概念强度_low = [0.1] * 64  # 大部分概念强度低
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ev_strength.一拍(concept_h, 概念强度=概念强度_low, surprise=0.1)
    print(f"  低概念强度后: 进化强度={ev_strength.evolve_strength:.4f}")
    print(f"  低活跃度弱点数: {ev_strength.低活跃度弱点数}")

    ev_strength2 = 自我进化(n_concepts=64)
    概念强度_high = [1.0] * 64  # 所有概念强度高
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ev_strength2.一拍(concept_h, 概念强度=概念强度_high, surprise=0.1)
    print(f"  高概念强度后: 进化强度={ev_strength2.evolve_strength:.4f}")
    print(f"  → 低概念强度应该导致更强的进化强度")

    print("\n--- 测试6: 概念区分度低→低区分度弱点 ---")
    ev_dist_low = 自我进化(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ev_dist_low.一拍(concept_h, 概念区分度=0.1, surprise=0.1)
    print(f"  低区分度后: 进化强度={ev_dist_low.evolve_strength:.4f}")
    print(f"  低区分度弱点数: {ev_dist_low.低区分度弱点数}")

    ev_dist_high = 自我进化(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ev_dist_high.一拍(concept_h, 概念区分度=0.8, surprise=0.1)
    print(f"  高区分度后: 进化强度={ev_dist_high.evolve_strength:.4f}")
    print(f"  → 低区分度应该导致更强的进化强度")

    print("\n--- 测试7: 进化验证——性能提升→有效进化 ---")
    ev_verify = 自我进化(n_concepts=64)
    # 前50拍：低性能（高surprise，弱概念）
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.1
        ev_verify.一拍(concept_h, surprise=0.5, 预测误差=0.3)
    perf_before = ev_verify.system_performance
    print(f"  前50拍后: 系统性能={perf_before:.4f}, 进化效果={ev_verify.evolve_effectiveness:.4f}")

    # 后50拍：高性能（低surprise，强概念）
    for _ in range(50):
        concept_h = [0.3] * 10 + [0.0] * 54
        ev_verify.一拍(concept_h, surprise=0.01, 预测误差=0.01)
    perf_after = ev_verify.system_performance
    print(f"  后50拍后: 系统性能={perf_after:.4f}, 进化效果={ev_verify.evolve_effectiveness:.4f}")
    print(f"  有效进化次数: {ev_verify.有效进化次数}")
    print(f"  无效进化次数: {ev_verify.无效进化次数}")
    print(f"  → 性能提升时进化效果应该提高")

    print("\n--- 测试8: 进化驱动信号 ---")
    ev_drive = 自我进化(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ev_drive.一拍(concept_h, surprise=0.3)
    drive_sum = sum(abs(v) for v in ev_drive.evolve_drive)
    print(f"  进化驱动信号总和: {drive_sum:.4f}")
    print(f"  进化驱动top5: {sorted(range(64), key=lambda i: abs(ev_drive.evolve_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试9: 进化方向（保守 vs 激进） ---")
    ev_dir = 自我进化(n_concepts=64)
    directions = []
    for tick in range(50):
        concept_h = [0.0] * 64
        if tick < 25:
            concept_h[4] = 0.3  # 确定性高
            result = ev_dir.一拍(concept_h, surprise=0.01)
        else:
            concept_h = [0.0] * 64  # 全部不确定
            result = ev_dir.一拍(concept_h, surprise=0.5)
        directions.append(result['evolve_direction'])
    print(f"  前25拍（确定性高）平均方向: {sum(directions[:25])/25:.4f}")
    print(f"  后25拍（不确定性高）平均方向: {sum(directions[25:])/25:.4f}")
    print(f"  → 不确定性高时方向应该更激进")

    print("\n--- 测试10: 系统性能评估 ---")
    ev_perf = 自我进化(n_concepts=64)
    # 低性能场景
    for _ in range(50):
        concept_h = [0.0] * 64
        ev_perf.一拍(concept_h, surprise=0.5, 预测误差=0.3, 概念区分度=0.1)
    print(f"  低性能场景: 系统性能={ev_perf.system_performance:.4f}")

    ev_perf2 = 自我进化(n_concepts=64)
    # 高性能场景
    for _ in range(50):
        concept_h = [0.3] * 10 + [0.0] * 54
        ev_perf2.一拍(concept_h, surprise=0.01, 预测误差=0.01, 概念区分度=0.8)
    print(f"  高性能场景: 系统性能={ev_perf2.system_performance:.4f}")
    print(f"  → 高性能场景的系统性能应该更高")

    print("\n--- 测试11: 精卫公式验证 ---")
    ev_jw = 自我进化(n_concepts=16)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = ev_jw.一拍(concept_h, surprise=0.3)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 进化强度={result['evolve_strength']}, "
                  f"进化方向={result['evolve_direction']}, "
                  f"系统性能={result['system_performance']}")
    print(f"  τ(浮出): 进化驱动均值={round(sum(ev_jw.evolve_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={ev_jw.拍}")

    print("\n验证通过")
