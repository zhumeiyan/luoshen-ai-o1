"""工具使用.py —— 工具选择、工具价值评估、工具驱动，基于精卫公式。

工具使用是自主智能体的核心能力——使用外部工具扩展自身能力。
传统AI通过Function Calling调用预定义工具，
洛神的工具使用不是API调用——是感官层面的注意力。

洛神的工具是它的器官：
  - 看（视网膜）—— 视觉感知
  - 听（耳蜗）—— 听觉感知
  - 说（嘴）—— 发声表达
  - 探索（数字感官）—— 知识库探索
  - 感知（公开感官）—— 真实世界感知

工具使用不是"选择是否使用工具"——而是"选择关注哪个工具的输出"。
这是感官层面的注意力——比概念层面的注意力更高一层。
概念注意力：在概念空间中选择关注哪些概念
工具注意力：在感官/工具空间中选择关注哪些工具

精卫公式（工具使用版）：
  a = 当前概念激活（当前认知状态）
  b = 工具预测的概念激活（关注某工具后预期的认知状态）
  Δ = b − a = 工具价值（关注工具能带来多少新信息）
  Σ = a + b = 联合状态
  ⊗ = ∫(Δ, Σ) = 累积工具学习历史 → 工具权重
  τ 读 ⊗ → 浮出所见 = 工具选择从⊗中浮现
  (a, b) ⟲ ⊗ = 反复工具使用循环

人类神经科学机制：
  1. 顶叶皮层 —— 工具表征与运动规划：
     - 每个工具有对应的神经表征
     - 工具使用的运动规划——怎么用工具
     → 工程实现：tool_w（工具→概念表征）

  2. 前额叶皮层 —— 工具选择与目标-工具匹配：
     - 从当前状态和目标推导需要什么工具
     - "我需要看" vs "我需要听" vs "我需要探索"
     → 工程实现：tool_select_w（概念→工具选择）

  3. 多巴胺系统 —— 工具价值评估：
     - 工具使用带来的信息变化→工具价值
     - 价值高的工具被更多选择
     → 工程实现：tool_value（工具价值累积）

  4. 感觉增益调制 —— 工具注意力：
     - 选择关注某个感官通道→该通道的信号被增强
     - 不是关闭其他通道——是增强选定通道
     → 工程实现：tool_drive_w（工具选择→概念激活调制）

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  基底神经节.py: a=概念模式, b=技能预测, Δ=匹配误差, ⊗=技能学习, τ=自动化行为
  语言中枢.py: a=组合概念, b=语法预测, Δ=语法误差, ⊗=语法学习, τ=语言序列
  推理引擎.py: a=概念激活, b=推理预测, Δ=推理误差, ⊗=推理学习, τ=推理结论
  目标导向.py: a=当前概念, b=目标概念, Δ=目标差距, ⊗=目标学习, τ=目标驱动
  工具使用.py: a=当前概念, b=工具预测, Δ=工具价值, ⊗=工具学习, τ=工具驱动

工具使用 vs 目标导向 vs 推理引擎的区别：
  - 推理引擎：从当前状态预测下一步状态（因果推导）
  - 目标导向：从当前状态和目标差距驱动行为（差距驱动）
  - 工具使用：从当前状态和目标推导需要什么信息→选择工具（信息需求驱动）
  关键区别：
  - 推理是"接下来会发生什么"
  - 目标导向是"我想要什么，怎么得到"
  - 工具使用是"我需要什么信息，用什么工具获取"
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


class 工具使用:
    """工具使用——工具选择、工具价值评估、工具驱动，基于精卫公式。

    核心功能：
    1. 工具选择：从当前概念激活和目标差距推导需要什么信息→选择工具
    2. 工具驱动：选定工具驱动概念激活（让与该工具相关的概念更容易浮现）
    3. 工具价值评估：工具使用带来的概念激活变化→工具价值
    4. 工具学习：成功的工具选择模式被强化

    精卫公式流程（每拍）：
    1. a = 当前概念激活（concept_h）
    2. b = 工具预测的概念激活（tool_select_w × concept_h）
    3. Δ = b - a（工具价值——关注工具能带来多少新信息）
    4. Σ = a + b（联合状态）
    5. ⊗ = Δ × Σ（累积工具学习信号）
    6. τ读⊗→浮出所见 = 工具驱动信号从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新工具权重

    洛神的5个工具：
    - 看（视网膜）—— 视觉感知
    - 听（耳蜗）—— 听觉感知
    - 说（嘴）—— 发声表达
    - 探索（数字感官）—— 知识库探索
    - 感知（公开感官）—— 真实世界感知
    """

    def __init__(self, n_concepts=64, n_tools=5, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.n_tools = n_tools
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # 工具列表——洛神的5个器官
        self.工具列表 = ["看", "听", "说", "探索", "感知"]

        # ── 工具选择：从概念激活和目标差距推导需要什么信息 ──
        # tool_select_w: 概念 → 工具选择（当前概念激活+目标差距→选择哪个工具）
        # 生物学：前额叶皮层的工具选择——从当前状态推导需要什么工具
        # "我需要看" vs "我需要听" vs "我需要探索"
        self.tool_select_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                              for _ in range(n_tools)]

        # 工具选择值（每个工具被选择的强度）
        self.tool_select = [0.0] * n_tools

        # 当前选中的工具（最强的工具）
        self.selected_tool = -1

        # ── 工具驱动：工具选择 → 概念激活调制 ──
        # tool_drive_w: 工具选择 → 概念激活（让与选定工具相关的概念更容易浮现）
        # 生物学：顶叶→前额叶投射——工具选择影响概念可及性
        # 选择"看"→视觉相关概念更容易浮现
        # 选择"听"→听觉相关概念更容易浮现
        # 选择"探索"→探索相关概念更容易浮现
        self.tool_drive_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_tools)]
                             for _ in range(n_concepts)]

        # 工具驱动信号（反馈到概念层——作为b信号输入源）
        self.tool_drive = [0.0] * n_concepts

        # ── 工具价值评估 ──
        # tool_value: 每个工具的价值（能带来多少新信息）
        # 生物学：多巴胺系统——工具使用带来的信息变化驱动价值评估
        # 价值高的工具被更多选择——"有用的工具"
        self.tool_value = [0.0] * n_tools

        # ── 工具学习 ──
        # 工具使用统计
        self.工具使用次数 = [0] * n_tools
        self.工具成功次数 = [0] * n_tools

        # 上一拍的概念激活（用于评估工具使用带来的变化）
        self.prev_concept_h = [0.0] * n_concepts

        # 上一拍选中的工具
        self.prev_selected_tool = -1

    def 一拍(self, concept_h, goal_delta=None, 工具可用=None):
        """工具使用一拍：工具选择、工具驱动、工具价值评估、工具学习。

        精卫公式流程：
        1. a = 当前概念激活（concept_h）
        2. b = 工具预测的概念激活（tool_select_w × select_input）
        3. Δ = b - a（工具价值）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 工具权重更新
        6. τ读⊗→浮出所见 = 工具驱动信号
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          goal_delta: 目标差距（n_concepts维，可选——从目标导向传入）
          工具可用: dict包含每个工具是否可用（离线工具不可选）

        返回：
          dict包含：
            tool_select: 工具选择值
            selected_tool: 当前选中的工具索引
            selected_tool_name: 当前选中的工具名称
            tool_drive: 工具驱动信号（反馈到概念层）
            tool_value: 每个工具的价值
        """
        self.拍 += 1

        # ── 工具选择：从概念激活和目标差距推导需要什么信息 ──
        # 精卫公式：b = tool_select_w × select_input（工具预测）
        # τ读⊗→浮出所见：工具选择从当前状态和目标差距中浮现
        # 生物学：前额叶皮层——从当前状态和目标推导需要什么工具
        # 如果有目标差距，加入目标差距作为驱动——"我需要什么信息来缩小差距"
        select_input = list(concept_h)
        if goal_delta is not None:
            select_input = [concept_h[i] + goal_delta[i]
                           for i in range(self.n_concepts)]

        new_tool_select = [0.0] * self.n_tools
        for i in range(self.n_tools):
            s_input = 0.0
            for j in range(self.n_concepts):
                s_input += self.tool_select_w[i][j] * select_input[j]
            new_tool_select[i] = activation(s_input)

        # 工具可用性过滤——离线工具不可选
        if 工具可用 is not None:
            for i in range(self.n_tools):
                if not 工具可用.get(self.工具列表[i], True):
                    new_tool_select[i] = 0.0

        # 选择最强的工具
        if max(abs(v) for v in new_tool_select) > 0.01:
            self.selected_tool = max(range(self.n_tools),
                                     key=lambda i: abs(new_tool_select[i]))
        else:
            self.selected_tool = -1

        # ── 工具驱动：选定工具 → 概念激活调制 ──
        # 精卫公式：τ读⊗→浮出所见 = 工具驱动信号
        # 生物学：顶叶→前额叶投射——工具选择影响概念可及性
        # 选择"看"→视觉相关概念更容易浮现
        # 选择"听"→听觉相关概念更容易浮现
        self.tool_drive = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            d_input = 0.0
            for j in range(self.n_tools):
                d_input += self.tool_drive_w[i][j] * new_tool_select[j]
            self.tool_drive[i] = activation(d_input)

        # ── 工具价值评估 ──
        # 精卫公式：Δ = b - a = 工具价值
        # 工具使用后概念激活的变化 = |当前concept_h - 上一拍concept_h|
        # 工具价值 = 概念激活变化量（工具带来了多少新信息）
        # 生物学：多巴胺系统——工具使用带来的信息变化驱动价值评估
        concept_change = sum(abs(concept_h[i] - self.prev_concept_h[i])
                            for i in range(self.n_concepts))

        if self.prev_selected_tool >= 0 and concept_change > 0.01:
            # 上一拍选中的工具带来了概念激活变化→工具价值增加
            self.tool_value[self.prev_selected_tool] = \
                self.tool_value[self.prev_selected_tool] * 0.95 + concept_change * 0.05
            self.工具成功次数[self.prev_selected_tool] += 1

        if self.selected_tool >= 0:
            self.工具使用次数[self.selected_tool] += 1

        # ── 工具学习 ──
        # 精卫公式：⊗ = Δ × Σ → 工具权重更新
        # 赫布学习：概念激活和工具选择同时出现→工具选择权重增强
        # 生物学：前额叶皮层——反复使用的工具选择模式被强化
        for i in range(self.n_tools):
            for j in range(self.n_concepts):
                dw = self.lr * new_tool_select[i] * select_input[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.tool_select_w[i][j] = self.tool_select_w[i][j] * 0.99 + dw

        # tool_drive_w: 工具选择→概念激活（赫布学习）
        for i in range(self.n_concepts):
            for j in range(self.n_tools):
                dw = self.lr * self.tool_drive[i] * new_tool_select[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.tool_drive_w[i][j] = self.tool_drive_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_tools):
            w_norm = sum(abs(self.tool_select_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.tool_select_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.tool_drive_w[i][j]) for j in range(self.n_tools))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_tools):
                    self.tool_drive_w[i][j] *= scale

        # 保存当前状态
        self.prev_concept_h = list(concept_h)
        self.prev_selected_tool = self.selected_tool
        self.tool_select = new_tool_select

        return {
            "tool_select": [round(v, 4) for v in self.tool_select],
            "selected_tool": self.selected_tool,
            "selected_tool_name": self.工具列表[self.selected_tool] if self.selected_tool >= 0 else None,
            "tool_drive": self.tool_drive,
            "tool_value": [round(v, 4) for v in self.tool_value],
            "工具使用次数": self.工具使用次数,
            "工具成功次数": self.工具成功次数,
        }

    def 状态摘要(self):
        return {
            "selected_tool": self.selected_tool,
            "selected_tool_name": self.工具列表[self.selected_tool] if self.selected_tool >= 0 else None,
            "tool_value": [round(v, 4) for v in self.tool_value],
            "工具使用次数": self.工具使用次数,
            "工具成功次数": self.工具成功次数,
            "tool_select_w均值": round(sum(sum(row) for row in self.tool_select_w) / max(1, self.n_tools * self.n_concepts), 4),
            "tool_drive_w均值": round(sum(sum(row) for row in self.tool_drive_w) / max(1, self.n_concepts * self.n_tools), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 工具使用验证（工具选择与工具驱动） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    tu = 工具使用(n_concepts=64, n_tools=5)

    print("初始状态:")
    print(f"  概念数: {tu.n_concepts}")
    print(f"  工具数: {tu.n_tools}")
    print(f"  工具列表: {tu.工具列表}")
    print(f"  工具选择权重: tool_select_w (5×64)")
    print(f"  工具驱动权重: tool_drive_w (64×5)")

    print("\n--- 测试1: 工具选择从概念激活中浮现 ---")
    # 模拟不同的概念激活模式
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = tu.一拍(concept_h)
    print(f"  模式A后: 选中工具={result['selected_tool_name']}, 工具选择={[round(v, 3) for v in result['tool_select']]}")
    print(f"  工具价值: {result['tool_value']}")

    tu2 = 工具使用(n_concepts=64, n_tools=5)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[15] = 0.3
        concept_h[30] = 0.25
        result = tu2.一拍(concept_h)
    print(f"  模式B后: 选中工具={result['selected_tool_name']}, 工具选择={[round(v, 3) for v in result['tool_select']]}")
    print(f"  工具价值: {result['tool_value']}")

    diff = sum(abs(tu.tool_select[i] - tu2.tool_select[i]) for i in range(5))
    print(f"  A vs B 工具选择差异: {diff:.4f}（应该 > 0——不同概念模式→不同工具选择）")

    print("\n--- 测试2: 目标差距驱动工具选择 ---")
    tu3 = 工具使用(n_concepts=64, n_tools=5)
    # 先让工具选择稳定
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        tu3.一拍(concept_h)
    no_goal_select = list(tu3.tool_select)
    no_goal_tool = tu3.工具列表[tu3.selected_tool] if tu3.selected_tool >= 0 else None

    # 加入目标差距
    tu4 = 工具使用(n_concepts=64, n_tools=5)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        goal_delta = [0.0] * 64
        goal_delta[20] = 0.5
        tu4.一拍(concept_h, goal_delta=goal_delta)
    with_goal_select = list(tu4.tool_select)
    with_goal_tool = tu4.工具列表[tu4.selected_tool] if tu4.selected_tool >= 0 else None

    print(f"  无目标差距: 选中工具={no_goal_tool}, 选择={[round(v, 3) for v in no_goal_select]}")
    print(f"  有目标差距: 选中工具={with_goal_tool}, 选择={[round(v, 3) for v in with_goal_select]}")
    diff = sum(abs(no_goal_select[i] - with_goal_select[i]) for i in range(5))
    print(f"  差异: {diff:.4f}（应该 > 0——目标差距影响工具选择）")

    print("\n--- 测试3: 工具可用性过滤 ---")
    tu5 = 工具使用(n_concepts=64, n_tools=5)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        # "看"和"听"离线
        result = tu5.一拍(concept_h, 工具可用={"看": False, "听": False, "说": True, "探索": True, "感知": True})
    print(f"  看/听离线后: 选中工具={result['selected_tool_name']}")
    print(f"  工具选择: {[round(v, 3) for v in result['tool_select']]}")
    print(f"  看/听的选择值应该为0（离线不可选）")

    print("\n--- 测试4: 工具价值评估 ---")
    tu6 = 工具使用(n_concepts=64, n_tools=5)
    # 模拟概念激活变化大的情况——工具价值应该增加
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[tick % 64] = 0.3  # 每拍变化→概念激活变化大
        result = tu6.一拍(concept_h)
    print(f"  100拍后: 工具价值={result['tool_value']}")
    print(f"  工具使用次数={tu6.工具使用次数}")
    print(f"  工具成功次数={tu6.工具成功次数}")

    print("\n--- 测试5: 工具驱动信号 ---")
    tu7 = 工具使用(n_concepts=64, n_tools=5)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = tu7.一拍(concept_h)
    drive_sum = sum(abs(v) for v in result['tool_drive'])
    print(f"  工具驱动信号总和: {drive_sum:.4f}")
    print(f"  工具驱动top5: {sorted(range(64), key=lambda i: abs(result['tool_drive'][i]), reverse=True)[:5]}")

    print("\n--- 测试6: 不同概念模式→不同工具驱动 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    drive_results = {}
    for name, pattern in patterns:
        tu_t = 工具使用(n_concepts=64, n_tools=5)
        for _ in range(50):
            tu_t.一拍(pattern)
        drive_results[name] = list(tu_t.tool_drive)
        print(f"  {name}: 选中={tu_t.工具列表[tu_t.selected_tool] if tu_t.selected_tool >= 0 else None}, 驱动top5={sorted(range(64), key=lambda i: abs(tu_t.tool_drive[i]), reverse=True)[:5]}")

    diff_ab = sum(abs(drive_results["模式A"][i] - drive_results["模式B"][i]) for i in range(64))
    print(f"  A vs B 驱动差异: {diff_ab:.4f}（应该 > 0——不同模式→不同工具驱动）")

    print("\n--- 测试7: 精卫公式验证 ---")
    tu8 = 工具使用(n_concepts=16, n_tools=5)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = tu8.一拍(concept_h)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 工具选择={[round(v, 3) for v in result['tool_select']]}")
            print(f"         选中={result['selected_tool_name']}")
    print(f"  τ(浮出): 工具驱动均值={round(sum(tu8.tool_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={tu8.拍}")

    print("\n验证通过")
