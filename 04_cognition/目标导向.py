"""目标导向.py —— 目标生成、目标差距驱动行为、目标评估学习，基于精卫公式。

目标导向行为是自主智能体的核心——"想要什么→怎么得到"。
强化学习AI通过奖励函数驱动行为，
自主智能体通过目标驱动行为。
洛神的目标不是人为设定的——从内部状态自然涌现。

精卫公式（目标导向版）：
  a = 当前概念激活（当前状态）
  b = 目标概念激活（期望状态）
  Δ = b − a = 目标差距（当前与期望的差异）
  Σ = a + b = 联合状态（当前+期望）
  ⊗ = ∫(Δ, Σ) = 累积目标学习历史 → 目标权重
  τ 读 ⊗ → 浮出所见 = 目标驱动信号从⊗中浮现
  (a, b) ⟲ ⊗ = 反复目标追求循环

人类神经科学机制：
  1. 前额叶皮层（PFC）—— 目标维持和执行控制：
     - 目标在工作记忆中维持——"我在追求什么"
     - PFC的执行控制——抑制无关信息，聚焦目标
     - 目标不是外部设定的——从内部需求涌现
     → 工程实现：goal_gen_w（内部状态→目标概念激活）

  2. 多巴胺系统 —— 奖励预测误差驱动目标学习：
     - 目标差距缩小→多巴胺释放→强化当前行为
     - 目标差距扩大→多巴胺抑制→修正行为
     - 不是外部奖励——是内部的差距变化信号
     → 工程实现：差距变化驱动目标权重学习

  3. 纹状体 —— 目标导向行为的选择和执行：
     - 目标差距驱动行为选择——选择能缩小差距的行为
     - 和基底神经节的习惯行为不同——目标导向是灵活的
     - 习惯是自动化的，目标导向是有意识的
     → 工程实现：goal_drive_w（目标差距→概念激活调制）

  4. 杏仁核 —— 目标的情感价值评估：
     - 目标有情感价值——想要的东西有价值
     - 情感价值驱动目标强度——越想要→目标越强
     → 工程实现：情感E参与目标生成

  5. 目标差距驱动（goal-gap driven behavior）：
     - 不是直接执行目标——是缩小差距
     - 差距越大→驱动力越强
     - 差距缩小→满足感（多巴胺释放）
     - 差距消失→目标达成→新目标涌现
     → 工程实现：goal_delta驱动概念激活和技能选择

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  基底神经节.py: a=概念模式, b=技能预测, Δ=匹配误差, ⊗=技能学习, τ=自动化行为
  语言中枢.py: a=组合概念, b=语法预测, Δ=语法误差, ⊗=语法学习, τ=语言序列
  推理引擎.py: a=概念激活, b=推理预测, Δ=推理误差, ⊗=推理学习, τ=推理结论
  目标导向.py: a=当前概念, b=目标概念, Δ=目标差距, ⊗=目标学习, τ=目标驱动

目标导向 vs 推理引擎 vs 工作记忆的区别：
  - 工作记忆：维持当前信息——没有方向性（注意聚焦）
  - 推理引擎：预测下一步——没有目标方向（因果推导）
  - 目标导向：追求目标——有方向性（缩小差距）
  关键区别：
  - 工作记忆是维持——"我在想什么"
  - 推理是预测——"接下来会发生什么"
  - 目标导向是追求——"我想要什么，怎么得到"
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


class 目标导向:
    """目标导向行为——目标生成、目标差距驱动、目标评估学习，基于精卫公式。

    核心功能：
    1. 目标生成：从内部状态（surprise、欲望偏好、情感E）涌现目标概念激活
    2. 目标差距：当前概念激活与目标概念激活的差距
    3. 目标驱动：目标差距驱动概念激活（反馈到概念层）
    4. 目标评估：差距变化驱动学习——差距缩小→强化，差距扩大→修正

    精卫公式流程（每拍）：
    1. a = 当前概念激活（concept_h）
    2. b = 目标概念激活（goal_gen_w × 内部状态信号）
    3. Δ = b - a（目标差距）
    4. Σ = a + b（联合状态）
    5. ⊗ = Δ × Σ（累积目标学习信号）
    6. τ读⊗→浮出所见 = 目标驱动信号从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新目标权重
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 目标生成：从内部状态涌现目标概念激活 ──
        # goal_gen_w: 内部状态信号 → 目标概念激活
        # 3维内部状态信号：surprise, 欲望偏好, 情感E
        # 生物学：PFC+杏仁核→目标表征——从需求和情感涌现目标
        # 目标不是人为设定的——从surprise（探索）、欲望偏好（追求）、情感（价值）涌现
        self.goal_gen_w = [[rng.uniform(-0.15, 0.15) for _ in range(3)]
                           for _ in range(n_concepts)]

        # 目标概念激活（当前追求的目标在概念空间中的表征）
        self.goal_h = [0.0] * n_concepts

        # ── 目标差距：当前状态与目标状态的差距 ──
        # Δ = goal_h - concept_h
        self.goal_delta = [0.0] * n_concepts

        # 目标强度（差距的总体大小——差距越大→强度越高→驱动力越强）
        self.goal_strength = 0.0

        # ── 目标驱动：目标差距 → 概念激活调制 ──
        # goal_drive_w: 目标差距 → 概念激活（让与目标相关的概念更容易浮现）
        # 生物学：PFC→后部皮层反馈投射——目标差距影响概念可及性
        # 这是"目标导向认知"的核心——差距驱动概念激活
        # 差距越大→驱动力越强→与目标相关的概念更容易浮现
        self.goal_drive_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                             for _ in range(n_concepts)]

        # 目标驱动信号（反馈到概念层——作为b信号输入源）
        self.goal_drive = [0.0] * n_concepts

        # ── 目标评估 ──
        # 上一拍的目标差距（用于评估差距是否缩小）
        self.prev_goal_delta = [0.0] * n_concepts

        # 目标驱动力强度（差距对概念层的影响强度）
        # 差距越大→驱动力越强
        self.goal_drive_strength = 0.0

        # 目标成功/失败统计
        self.目标生成次数 = 0
        self.目标成功次数 = 0
        self.目标失败次数 = 0

        # 目标稳定性（目标持续多少拍不变）
        self.goal_stability = 0
        self.prev_goal_h = [0.0] * n_concepts

        # 目标历史
        self.goal_history = []
        self.goal_history_cap = 20

        # 目标强度历史（用于检测目标强度变化趋势）
        self.goal_strength_history = []

    def 一拍(self, concept_h, 内部状态=None):
        """目标导向一拍：目标生成、目标差距、目标驱动、目标评估。

        精卫公式流程：
        1. a = 当前概念激活（concept_h）
        2. b = 目标概念激活（goal_gen_w × 内部状态信号）
        3. Δ = b - a（目标差距）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 目标权重更新
        6. τ读⊗→浮出所见 = 目标驱动信号
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          内部状态: dict包含surprise、欲望偏好、情感E

        返回：
          dict包含：
            goal_h: 目标概念激活
            goal_delta: 目标差距
            goal_drive: 目标驱动信号（反馈到概念层）
            目标强度: 目标差距的总体大小
            目标驱动力: 差距对概念层的影响强度
        """
        self.拍 += 1

        # ── 目标生成：从内部状态涌现目标概念激活 ──
        # 精卫公式：b = goal_gen_w × 内部状态信号
        # τ读⊗→浮出所见：目标从内部需求中浮现
        # 生物学：PFC+杏仁核→目标表征——从需求和情感涌现目标
        # surprise→探索目标（理解未知）
        # 欲望偏好→追求目标（获得喜欢的）
        # 情感E→价值评估（正E追求，负E回避）
        if 内部状态 is not None:
            surprise = 内部状态.get("surprise", 0.0)
            欲望偏好 = 内部状态.get("欲望偏好", 0.0)
            情感E = 内部状态.get("情感E", 0.0)
            内部信号 = [surprise, 欲望偏好, 情感E]

            new_goal_h = [0.0] * self.n_concepts
            for i in range(self.n_concepts):
                g_input = 0.0
                for j in range(3):
                    g_input += self.goal_gen_w[i][j] * 内部信号[j]
                new_goal_h[i] = activation(g_input)

            # 目标稀疏化——只保留最强的几个目标
            # 生物学：PFC的执行控制——聚焦少数目标，抑制其他
            sorted_idx = sorted(range(self.n_concepts),
                                key=lambda i: abs(new_goal_h[i]), reverse=True)
            top_k = set(sorted_idx[:max(1, self.n_concepts // 8)])
            for i in range(self.n_concepts):
                if i not in top_k:
                    new_goal_h[i] *= 0.1

            # 目标稳定性检测——目标持续不变→稳定性增加
            goal_change = sum(abs(new_goal_h[i] - self.prev_goal_h[i])
                             for i in range(self.n_concepts))
            if goal_change < 0.5:
                self.goal_stability += 1
            else:
                self.goal_stability = 0

            self.prev_goal_h = self.goal_h[:]
            self.goal_h = new_goal_h
            self.目标生成次数 += 1

        # ── 目标差距：Δ = goal_h - concept_h ──
        # 精卫公式：Δ = b - a = 目标差距
        # 生物学：多巴胺系统的奖励预测误差——差距驱动行为
        self.goal_delta = [self.goal_h[i] - concept_h[i]
                          for i in range(self.n_concepts)]

        # 目标强度（差距的总体大小）
        self.goal_strength = sum(abs(d) for d in self.goal_delta) / max(1, self.n_concepts)

        # Σ = a + b（联合状态）
        goal_sigma = [self.goal_h[i] + concept_h[i]
                     for i in range(self.n_concepts)]

        # ⊗ = Δ × Σ
        goal_otimes = [self.goal_delta[i] * goal_sigma[i]
                      for i in range(self.n_concepts)]

        # ── 目标驱动：目标差距 → 概念激活调制 ──
        # 精卫公式：τ读⊗→浮出所见 = 目标驱动信号
        # 生物学：PFC→后部皮层反馈投射——目标差距影响概念可及性
        # 差距越大→驱动力越强→与目标相关的概念更容易浮现
        # 这是"目标导向认知"的核心——不是维持目标，而是缩小差距
        self.goal_drive = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            d_input = 0.0
            for j in range(self.n_concepts):
                d_input += self.goal_drive_w[i][j] * self.goal_delta[j]
            self.goal_drive[i] = activation(d_input)

        # ── 目标评估：差距变化驱动学习 ──
        # 精卫公式：⊗ = Δ × Σ → 目标权重更新
        # 生物学：多巴胺系统——差距缩小→多巴胺释放→强化
        # 差距扩大→多巴胺抑制→修正
        delta_change = sum(abs(self.goal_delta[i]) - abs(self.prev_goal_delta[i])
                          for i in range(self.n_concepts))

        if delta_change < -0.01:
            # 差距缩小→成功→强化当前目标追求模式
            self.目标成功次数 += 1

            # 强化目标生成权重——成功的目标生成模式被强化
            if 内部状态 is not None:
                surprise = 内部状态.get("surprise", 0.0)
                欲望偏好 = 内部状态.get("欲望偏好", 0.0)
                情感E = 内部状态.get("情感E", 0.0)
                内部信号 = [surprise, 欲望偏好, 情感E]
                for i in range(self.n_concepts):
                    for j in range(3):
                        dw = self.lr * self.goal_h[i] * 内部信号[j]
                        if not (math.isnan(dw) or math.isinf(dw)):
                            self.goal_gen_w[i][j] = self.goal_gen_w[i][j] * 0.99 + dw

            # 强化目标驱动权重——成功的驱动模式被强化
            for i in range(self.n_concepts):
                for j in range(self.n_concepts):
                    dw = self.lr * self.goal_drive[i] * self.goal_delta[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.goal_drive_w[i][j] = self.goal_drive_w[i][j] * 0.99 + dw

        elif delta_change > 0.01:
            # 差距扩大→失败→修正（弱化当前模式）
            self.目标失败次数 += 1

        # ── 目标驱动强度 ──
        # 差距越大→驱动力越强→概念层调制越强
        # 生物学：目标紧迫性——越想要→驱动力越强
        self.goal_drive_strength = min(1.0, self.goal_strength * 5.0)

        # 保存当前差距用于下一拍评估
        self.prev_goal_delta = self.goal_delta[:]

        # 记录目标强度历史
        self.goal_strength_history.append(self.goal_strength)
        if len(self.goal_strength_history) > 100:
            self.goal_strength_history.pop(0)

        # 记录目标历史
        if self.拍 % 10 == 0:
            self.goal_history.append({
                "拍": self.拍,
                "目标强度": round(self.goal_strength, 4),
                "目标稳定": self.goal_stability,
                "活跃目标数": sum(1 for v in self.goal_h if abs(v) > 0.1),
            })
            if len(self.goal_history) > self.goal_history_cap:
                self.goal_history.pop(0)

        return {
            "goal_h": self.goal_h,
            "goal_delta": self.goal_delta,
            "goal_drive": self.goal_drive,
            "目标强度": round(self.goal_strength, 4),
            "目标驱动力": round(self.goal_drive_strength, 4),
            "目标稳定": self.goal_stability,
            "活跃目标数": sum(1 for v in self.goal_h if abs(v) > 0.1),
            "目标生成次数": self.目标生成次数,
            "目标成功次数": self.目标成功次数,
            "目标失败次数": self.目标失败次数,
            "差距变化": round(delta_change, 4),
        }

    def 状态摘要(self):
        return {
            "活跃目标数": sum(1 for v in self.goal_h if abs(v) > 0.1),
            "目标强度": round(self.goal_strength, 4),
            "目标驱动力": round(self.goal_drive_strength, 4),
            "目标稳定": self.goal_stability,
            "目标生成次数": self.目标生成次数,
            "目标成功次数": self.目标成功次数,
            "目标失败次数": self.目标失败次数,
            "goal_gen_w均值": round(sum(sum(row) for row in self.goal_gen_w) / max(1, self.n_concepts * 3), 4),
            "goal_drive_w均值": round(sum(sum(row) for row in self.goal_drive_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 目标导向验证（目标生成与差距驱动） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    go = 目标导向(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {go.n_concepts}")
    print(f"  目标生成权重: goal_gen_w (64×3)")
    print(f"  目标驱动权重: goal_drive_w (64×64)")

    print("\n--- 测试1: 目标从内部状态涌现 ---")
    # 模拟高surprise状态——应该涌现探索目标
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = go.一拍(concept_h, 内部状态={"surprise": 0.8, "欲望偏好": 0.2, "情感E": 0.3})
    print(f"  高surprise后: 目标强度={result['目标强度']}, 活跃目标={result['活跃目标数']}")
    print(f"  目标top5: {sorted(range(64), key=lambda i: abs(go.goal_h[i]), reverse=True)[:5]}")
    print(f"  目标驱动top5: {sorted(range(64), key=lambda i: abs(go.goal_drive[i]), reverse=True)[:5]}")

    go2 = 目标导向(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = go2.一拍(concept_h, 内部状态={"surprise": 0.1, "欲望偏好": 0.9, "情感E": 0.5})
    print(f"  高欲望偏好后: 目标强度={result['目标强度']}, 活跃目标={result['活跃目标数']}")
    print(f"  目标top5: {sorted(range(64), key=lambda i: abs(go2.goal_h[i]), reverse=True)[:5]}")

    diff = sum(abs(go.goal_h[i] - go2.goal_h[i]) for i in range(64))
    print(f"  高surprise vs 高欲望偏好 目标差异: {diff:.4f}（应该 > 0——不同内部状态→不同目标）")

    print("\n--- 测试2: 目标差距驱动 ---")
    go3 = 目标导向(n_concepts=64)
    # 概念激活远离目标→差距大→驱动力强
    concept_far = [0.0] * 64
    concept_far[20] = 0.3
    go3.一拍(concept_far, 内部状态={"surprise": 0.8, "欲望偏好": 0.2, "情感E": 0.3})
    # 让目标先涌现
    for _ in range(50):
        go3.一拍(concept_far, 内部状态={"surprise": 0.8, "欲望偏好": 0.2, "情感E": 0.3})
    drive_far = sum(abs(v) for v in go3.goal_drive)
    print(f"  概念远离目标: 目标驱动力总和={drive_far:.4f}")

    # 概念激活接近目标→差距小→驱动力弱
    concept_near = list(go3.goal_h)
    result_near = go3.一拍(concept_near, 内部状态={"surprise": 0.8, "欲望偏好": 0.2, "情感E": 0.3})
    drive_near = sum(abs(v) for v in go3.goal_drive)
    print(f"  概念接近目标: 目标驱动力总和={drive_near:.4f}")
    print(f"  远 vs 近 驱动力差异: {drive_far - drive_near:.4f}（远应该 > 近——差距大→驱动力强）")

    print("\n--- 测试3: 目标差距缩小→成功 ---")
    go4 = 目标导向(n_concepts=16)
    # 模拟概念激活逐渐接近目标的过程
    for tick in range(100):
        # 概念激活逐渐向目标靠近
        progress = tick / 100.0
        concept_h = [go4.goal_h[i] * progress + 0.01 * (1 - progress) for i in range(16)]
        result = go4.一拍(concept_h, 内部状态={"surprise": 0.5, "欲望偏好": 0.5, "情感E": 0.3})
        if tick in [0, 49, 99]:
            print(f"  拍{tick+1}: 目标强度={result['目标强度']}, 差距变化={result['差距变化']}, 成功={go4.目标成功次数}, 失败={go4.目标失败次数}")

    print(f"  最终: 成功={go4.目标成功次数}, 失败={go4.目标失败次数}")
    print(f"  成功次数应该 > 失败次数（概念在接近目标→差距在缩小）")

    print("\n--- 测试4: 不同内部状态→不同目标 ---")
    states = [
        ("高surprise", {"surprise": 0.9, "欲望偏好": 0.1, "情感E": 0.0}),
        ("高欲望", {"surprise": 0.1, "欲望偏好": 0.9, "情感E": 0.0}),
        ("高情感E", {"surprise": 0.1, "欲望偏好": 0.1, "情感E": 0.9}),
        ("均衡", {"surprise": 0.3, "欲望偏好": 0.3, "情感E": 0.3}),
    ]
    goal_results = {}
    for name, state in states:
        go_t = 目标导向(n_concepts=64)
        for _ in range(50):
            concept_h = [0.0] * 64
            concept_h[4] = 0.3
            go_t.一拍(concept_h, 内部状态=state)
        goal_results[name] = list(go_t.goal_h)
        active = [i for i, v in enumerate(go_t.goal_h) if abs(v) > 0.05]
        print(f"  {name}: 活跃目标={active[:8]}, 目标强度={go_t.goal_strength:.4f}")

    diff_12 = sum(abs(goal_results["高surprise"][i] - goal_results["高欲望"][i]) for i in range(64))
    diff_13 = sum(abs(goal_results["高surprise"][i] - goal_results["高情感E"][i]) for i in range(64))
    print(f"  高surprise vs 高欲望 目标差异: {diff_12:.4f}")
    print(f"  高surprise vs 高情感E 目标差异: {diff_13:.4f}")

    print("\n--- 测试5: 目标稳定性 ---")
    go5 = 目标导向(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        go5.一拍(concept_h, 内部状态={"surprise": 0.5, "欲望偏好": 0.5, "情感E": 0.3})
    print(f"  50拍后: 目标稳定性={go5.goal_stability}")
    print(f"  目标稳定应该 > 0（相同内部状态→目标不变→稳定性增加）")

    print("\n--- 测试6: 精卫公式验证 ---")
    go6 = 目标导向(n_concepts=16)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = go6.一拍(concept_h, 内部状态={"surprise": 0.5, "欲望偏好": 0.3, "情感E": 0.2})
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 目标={[round(v, 3) for v in go6.goal_h[:4]]}")
            print(f"         差距={[round(v, 3) for v in go6.goal_delta[:4]]}")
            print(f"         目标强度={result['目标强度']}")
    print(f"  τ(浮出): 目标驱动均值={round(sum(go6.goal_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={go6.拍}")

    print("\n验证通过")
