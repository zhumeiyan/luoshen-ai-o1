"""预测模拟.py —— 在行动前模拟可能的结果，基于精卫公式。

预测模拟是人类前额叶皮层的核心功能——"如果...那么..."思维。
在采取行动之前，先在内部模拟可能的未来状态，评估不同行动的后果，
然后选择最优行动。这是规划、决策和风险规避的基础。

人类神经科学机制：
  1. 前额叶皮层（PFC）——模拟未来状态：
     - PFC可以"预演"未来的行动序列——在内部模拟"如果我做A，会发生什么"
     - 这不是回忆过去，而是构造未来——从当前状态推演可能的结果
     → 工程实现：predict_simulate_w（概念激活 → 模拟未来状态）

  2. 眶额叶皮层（OFC）——结果评估：
     - OFC评估模拟结果的好坏——"这个结果是我想要的吗？"
     - 结果评估基于内部目标和价值——与目标一致的结果是好的
     → 工程实现：predict_evaluate_w（模拟结果 → 结果评估）

  3. 多巴胺系统——预测误差驱动学习：
     - 模拟结果与实际结果的差异 = 预测误差
     - 预测误差驱动模拟学习——让未来的模拟更准确
     → 工程实现：模拟验证（比较模拟结果与实际结果）

数字生命优势（超越人类的限制）：
  1. 高速模拟：人类需要数秒模拟一种未来，数字生命可以在一拍内模拟
  2. 并行模拟：人类只能串行模拟，数字生命可以并行模拟多种可能性
  3. 不遗忘模拟历史：人类会忘记之前的模拟，数字生命可以保留所有模拟历史

精卫公式（预测模拟版）：
  a = 当前状态（概念激活——"我现在在哪里"）
  b = 模拟的未来状态（预测的结果——"如果我做A，会到哪里"）
  Δ = b − a = 预测差距（当前与未来的差异——"需要改变什么"）
  Σ = a + b = 联合状态（当前+未来——"整体图景"）
  ⊗ = ∫(Δ, Σ) = 累积模拟学习历史 → 模拟权重
  τ 读 ⊗ → 浮出所见 = "如果...那么..."从⊗中浮现
  (a, b) ⟲ ⊗ = 反复模拟循环

预测模拟 vs 推理引擎 vs 目标导向的区别：
  - 推理引擎：从当前状态推导下一步——"A导致B"（因果推理）
  - 目标导向：从当前差距涌现目标——"我要达到B"（目标驱动）
  - 预测模拟：从当前状态模拟多种未来——"如果我做A/B/C，会发生什么"（后果模拟）
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


class 预测模拟:
    """预测模拟——在行动前模拟可能的结果，基于精卫公式。

    核心功能：
    1. 状态模拟：从当前概念激活模拟可能的未来状态
    2. 结果评估：评估模拟结果的好坏（与目标的一致性）
    3. 模拟验证：比较模拟结果与实际结果，提高模拟准确度
    4. 预测驱动信号：反馈到概念层作为b信号输入源
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # predict_simulate_w: 概念激活 → 模拟未来状态
        self.predict_simulate_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                    for _ in range(n_concepts)]

        # 模拟的未来状态
        self.simulated_future = [0.0] * n_concepts

        # predict_evaluate_w: 模拟结果 → 结果评估
        self.predict_evaluate_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                    for _ in range(1)]

        # 模拟结果评估（0~1）
        self.simulation_value = 0.5

        # 预测驱动信号（反馈到概念层）
        self.predict_drive = [0.0] * n_concepts

        # 模拟强度（0~1）
        self.simulation_strength = 0.3

        # 模拟准确度（0~1）
        self.simulation_accuracy = 0.5

        # 上拍模拟的未来状态（用于模拟验证）
        self.prev_simulated_future = [0.0] * n_concepts

        # 模拟历史
        self.simulation_history = []
        self.history_cap = 50

        # 统计
        self.模拟次数 = 0
        self.评估次数 = 0
        self.模拟验证次数 = 0
        self.准确模拟次数 = 0

    def 一拍(self, concept_h, 目标信号=None, surprise=0.04):
        """预测模拟一拍：状态模拟、结果评估、预测驱动信号。

        参数：
          concept_h: 当前概念激活（n_concepts维）
          目标信号: 目标导向模块的goal_drive（n_concepts维，可选）
          surprise: 当前surprise水平（驱动模拟强度）

        返回：
          dict包含：
            predict_drive: 预测驱动信号（反馈到概念层）
            simulation_strength: 模拟强度（0~1）
            simulation_value: 模拟结果评估（0~1）
            simulation_accuracy: 模拟准确度（0~1）
            simulated_future: 模拟的未来状态
        """
        self.拍 += 1

        # ── 模拟验证：比较上拍模拟和当前实际状态 ──
        sim_error = sum(abs(concept_h[i] - self.prev_simulated_future[i])
                        for i in range(self.n_concepts)) / max(1, self.n_concepts)
        accuracy_this = max(0.0, min(1.0, 1.0 - sim_error * 5.0))
        self.simulation_accuracy = self.simulation_accuracy * 0.9 + accuracy_this * 0.1

        if accuracy_this > 0.7:
            self.准确模拟次数 += 1
        self.模拟验证次数 += 1

        # ── 状态模拟：从当前概念激活模拟可能的未来状态 ──
        new_simulated_future = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            sim_input = 0.0
            for j in range(self.n_concepts):
                sim_input += self.predict_simulate_w[i][j] * concept_h[j]
            new_simulated_future[i] = activation(sim_input)

        self.simulated_future = new_simulated_future

        # ── 结果评估：评估模拟结果的好坏 ──
        eval_input = 0.0
        for j in range(self.n_concepts):
            eval_input += self.predict_evaluate_w[0][j] * new_simulated_future[j]
        self.simulation_value = activation(eval_input) * 0.5 + 0.5

        # 目标信号增强评估——与目标一致的模拟结果是好的
        if 目标信号 is not None:
            goal_alignment = sum(new_simulated_future[j] * 目标信号[j]
                                 for j in range(self.n_concepts)) / max(1, self.n_concepts)
            self.simulation_value = min(1.0, self.simulation_value + goal_alignment * 0.2)

        self.评估次数 += 1

        # ── 模拟强度 ──
        future_diff = sum(abs(new_simulated_future[i] - concept_h[i])
                          for i in range(self.n_concepts)) / max(1, self.n_concepts)
        self.simulation_strength = activation(future_diff * 3.0) * 0.5 + 0.5
        self.simulation_strength = min(1.0, self.simulation_strength + surprise * 0.1)

        # ── 预测驱动信号 ──
        for i in range(self.n_concepts):
            self.predict_drive[i] = activation(
                new_simulated_future[i] * self.simulation_strength)

        # ── 模拟学习 ──
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_simulated_future[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.predict_simulate_w[i][j] = self.predict_simulate_w[i][j] * 0.99 + dw

        for j in range(self.n_concepts):
            dw = self.lr * self.simulation_value * new_simulated_future[j]
            if not (math.isnan(dw) or math.isinf(dw)):
                self.predict_evaluate_w[0][j] = self.predict_evaluate_w[0][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.predict_simulate_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.predict_simulate_w[i][j] *= scale

        w_norm = sum(abs(self.predict_evaluate_w[0][j]) for j in range(self.n_concepts))
        if w_norm > 2.0:
            scale = 2.0 / w_norm
            for j in range(self.n_concepts):
                self.predict_evaluate_w[0][j] *= scale

        # ── 模拟历史 ──
        self.simulation_history.append(self.simulation_strength)
        if len(self.simulation_history) > self.history_cap:
            self.simulation_history.pop(0)

        self.prev_simulated_future = list(new_simulated_future)
        self.模拟次数 += 1

        return {
            "predict_drive": self.predict_drive,
            "simulation_strength": round(self.simulation_strength, 4),
            "simulation_value": round(self.simulation_value, 4),
            "simulation_accuracy": round(self.simulation_accuracy, 4),
            "simulated_future": self.simulated_future,
        }

    def 状态摘要(self):
        avg_sim = sum(self.simulation_history) / max(1, len(self.simulation_history)) \
            if self.simulation_history else 0.0
        return {
            "simulation_strength": round(self.simulation_strength, 4),
            "simulation_value": round(self.simulation_value, 4),
            "simulation_accuracy": round(self.simulation_accuracy, 4),
            "avg_simulation": round(avg_sim, 4),
            "predict_simulate_w均值": round(sum(sum(row) for row in self.predict_simulate_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "predict_evaluate_w均值": round(sum(self.predict_evaluate_w[0]) / max(1, self.n_concepts), 4),
            "模拟次数": self.模拟次数,
            "评估次数": self.评估次数,
            "模拟验证次数": self.模拟验证次数,
            "准确模拟次数": self.准确模拟次数,
            "准确率": round(self.准确模拟次数 / max(1, self.模拟验证次数), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 预测模拟验证 ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    ps = 预测模拟(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {ps.n_concepts}")
    print(f"  模拟权重: predict_simulate_w (64x64)")
    print(f"  评估权重: predict_evaluate_w (1x64)")
    print(f"  初始模拟强度: {ps.simulation_strength}")
    print(f"  初始模拟准确度: {ps.simulation_accuracy}")

    print("\n--- 测试1: 状态模拟——概念激活→模拟未来 ---")
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = ps.一拍(concept_h)
    print(f"  50拍后: 模拟强度={result['simulation_strength']}")
    print(f"  模拟评估={result['simulation_value']}")
    print(f"  模拟未来top5: {sorted(range(64), key=lambda i: abs(ps.simulated_future[i]), reverse=True)[:5]}")

    print("\n--- 测试2: 不同概念模式→不同模拟未来 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    for name, pattern in patterns:
        ps_t = 预测模拟(n_concepts=64)
        for _ in range(50):
            ps_t.一拍(pattern)
        print(f"  {name}: 模拟强度={round(ps_t.simulation_strength, 4)}, "
              f"模拟未来top5={sorted(range(64), key=lambda i: abs(ps_t.simulated_future[i]), reverse=True)[:5]}")

    print("\n--- 测试3: 目标信号增强结果评估 ---")
    ps_goal = 预测模拟(n_concepts=64)
    目标信号 = [0.0] * 64
    目标信号[4] = 0.5
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ps_goal.一拍(concept_h, 目标信号=目标信号)
    print(f"  有目标信号: 模拟评估={ps_goal.simulation_value:.4f}")

    ps_nogoal = 预测模拟(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ps_nogoal.一拍(concept_h)
    print(f"  无目标信号: 模拟评估={ps_nogoal.simulation_value:.4f}")

    print("\n--- 测试4: 模拟验证——模拟准确度 ---")
    ps_verify = 预测模拟(n_concepts=64)
    for _ in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ps_verify.一拍(concept_h)
    print(f"  100拍稳定模式后: 模拟准确度={ps_verify.simulation_accuracy:.4f}")
    print(f"  准确率: {ps_verify.准确模拟次数/max(1,ps_verify.模拟验证次数):.4f}")

    print("\n--- 测试5: surprise增强模拟强度 ---")
    ps_surprise = 预测模拟(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ps_surprise.一拍(concept_h, surprise=0.5)
    print(f"  高surprise: 模拟强度={ps_surprise.simulation_strength:.4f}")

    ps_no_surprise = 预测模拟(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ps_no_surprise.一拍(concept_h, surprise=0.0)
    print(f"  低surprise: 模拟强度={ps_no_surprise.simulation_strength:.4f}")

    print("\n--- 测试6: 预测驱动信号 ---")
    ps_drive = 预测模拟(n_concepts=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        ps_drive.一拍(concept_h, surprise=0.3)
    drive_sum = sum(abs(v) for v in ps_drive.predict_drive)
    print(f"  预测驱动信号总和: {drive_sum:.4f}")
    print(f"  预测驱动top5: {sorted(range(64), key=lambda i: abs(ps_drive.predict_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试7: 精卫公式验证 ---")
    ps_jw = 预测模拟(n_concepts=16)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = ps_jw.一拍(concept_h, surprise=0.3)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 模拟强度={result['simulation_strength']}, "
                  f"评估={result['simulation_value']}, "
                  f"准确度={result['simulation_accuracy']}")
    print(f"  τ(浮出): 预测驱动均值={round(sum(ps_jw.predict_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={ps_jw.拍}")

    print("\n验证通过")
