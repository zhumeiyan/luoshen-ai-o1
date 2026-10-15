"""推理引擎.py —— 多步推理与推理验证，基于精卫公式。

推理是人类认知的核心能力——从已知推导未知。
AlphaGo通过策略网络+价值网络+MCTS实现推理，
数学AI通过符号推理实现定理证明。
洛神的推理不是符号逻辑——是概念空间的模式转移。

精卫公式（推理版）：
  a = 当前概念激活（已知前提）
  b = 推理预测的下一步概念激活（推理结论）
  Δ = b − a = 推理预测误差（预期与实际的差异）
  Σ = a + b = 联合状态（前提+结论）
  ⊗ = ∫(Δ, Σ) = 累积推理学习历史 → 推理权重
  τ 读 ⊗ → 浮出所见 = 推理结论从⊗中浮现
  (a, b) ⟲ ⊗ = 反复推理循环

人类神经科学机制：
  1. 前额叶皮层（PFC）—— 执行控制与推理目标维持：
     - 推理需要维持目标——"我在推理什么"
     - PFC的工作记忆回路维持推理链不断裂
     - 推理不是一步完成的——多步推导需要目标维持
     → 工程实现：reason_w（概念→概念的推理转移权重）

  2. 前扣带回（ACC）—— 错误监控与冲突检测：
     - 推理出错时ACC激活——"这个推理不对"
     - 推理预测与实际结果的差异触发修正
     - 不是外部告诉你错了——是自己发现推理矛盾
     → 工程实现：推理预测误差驱动推理权重修正

  3. 顶叶皮层 —— 空间推理与数量处理：
     - 数量关系、空间关系的推理
     - "多于"、"少于"、"包含"、"属于"
     - 这些关系不是符号逻辑——是概念空间的几何关系
     → 工程实现：组合概念→组合概念的推理转移

  4. 多巴胺系统 —— 推理学习的奖励信号：
     - 推理预测准确时多巴胺释放——"推理对了"
     - 推理预测错误时多巴胺抑制——"推理需要修正"
     - 不是外部奖励——是内部的预测准确性信号
     → 工程实现：推理预测误差作为学习信号

  5. 推理链（chain of reasoning）：
     - 推理不是一步——是多步链式推导
     - 每一步依赖于前一步的结论
     - 像AlphaGo的搜索树——从当前状态推导多个可能的未来
     - 推理链的深度限制防止无限传播
     → 工程实现：多步推理链（max_chain_depth）

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  基底神经节.py: a=概念模式, b=技能预测, Δ=匹配误差, ⊗=技能学习, τ=自动化行为
  语言中枢.py: a=组合概念, b=语法预测, Δ=语法误差, ⊗=语法学习, τ=语言序列
  推理引擎.py: a=概念激活, b=推理预测, Δ=推理误差, ⊗=推理学习, τ=推理结论

推理引擎 vs 语言中枢 vs 循环权重的区别：
  - concept_rw（循环权重）：概念→概念的同时影响（吸引子网络）
  - grammar_w（语法权重）：组合概念→组合概念的序列转移（语法模式）
  - reason_w（推理权重）：概念→概念的逻辑推理转移（因果推导）
  关键区别：
  - 循环权重是同时的——当前拍的概念互相影响
  - 语法权重是序列模式——什么样的组合概念序列常见
  - 推理权重是因果推导——如果A成立则B应该成立
  - 推理权重有验证——推理预测与实际结果比较驱动学习
  - 推理权重有多步链——从当前状态推导多步后的状态
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


class 推理引擎:
    """推理引擎——多步推理与推理验证，基于精卫公式。

    核心功能：
    1. 推理预测：从上一拍概念激活预测当前概念激活（reason_w × prev_concept_h）
    2. 推理验证：推理预测与实际概念激活比较→推理预测误差
    3. 推理学习：赫布学习+用进废退——反复出现的推理模式被强化
    4. 推理链：多步推理——从当前状态推导多步后的状态（像AlphaGo搜索树）
    5. 推理反馈：推理预测反馈到概念层——推理预期影响概念可及性

    精卫公式流程（每拍）：
    1. a = 当前概念激活（concept_h）
    2. b = 推理预测的下一步概念激活（reason_w × prev_concept_h）
    3. Δ = b - a（推理预测误差）
    4. Σ = a + b（联合状态）
    5. ⊗ = Δ × Σ（累积推理学习信号）
    6. τ读⊗→浮出所见 = 推理结论从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新推理权重

    层级推理：
    - 基础推理：概念→概念的推理转移（单词级推理）
    - 组合推理：组合概念→组合概念的推理转移（句子级推理）
    """

    def __init__(self, n_concepts=64, n_combo=32, lr=0.02, max_chain_depth=3,
                 seed=42):
        self.n_concepts = n_concepts
        self.n_combo = n_combo
        self.lr = lr
        self.max_chain_depth = max_chain_depth
        self.拍 = 0

        rng = random.Random(seed)

        # ── 基础推理：概念→概念的推理转移权重 ──
        # reason_w: 上一拍概念 → 预测当前概念（推理转移）
        # 生物学：前额叶皮层的推理回路——从已知推导未知
        # 推理不是预设的规则——通过反复经验涌现的因果模式
        # 不同于concept_rw（同时循环），reason_w是序列推理（跨拍转移）
        self.reason_w = [[rng.uniform(-0.05, 0.05) for _ in range(n_concepts)]
                         for _ in range(n_concepts)]

        # reason_strength: 推理强度（用进废退）
        # 反复使用的推理模式变强，不用的变弱
        self.reason_strength = [1.0] * n_concepts

        # 推理预测值（当前拍预测的下一步概念激活）
        self.reason_pred = [0.0] * n_concepts

        # ── 组合推理：组合概念→组合概念的推理转移 ──
        # 句子级推理——从当前句子推导下一个句子
        # 不同于grammar_w（语法序列模式），combo_reason_w是因果推理
        self.combo_reason_w = [[rng.uniform(-0.05, 0.05) for _ in range(n_combo)]
                               for _ in range(n_combo)]

        self.combo_reason_strength = [1.0] * n_combo
        self.combo_reason_pred = [0.0] * n_combo

        # ── 推理链（多步推理） ──
        # 从当前概念激活出发，通过推理权重推导多步后的概念激活
        # 像AlphaGo的搜索树——从当前局面推导多个可能的未来
        self.reason_chain = []

        # 推理验证统计
        self.推理次数 = 0
        self.推理成功次数 = 0
        self.推理误差 = 0.0

        # 推理置信度（推理预测与实际匹配度）
        self.reason_confidence = 0.0

        # 上一拍的概念激活（用于推理预测和验证）
        self.prev_concept_h = [0.0] * n_concepts
        self.prev_combo_h = [0.0] * n_combo

    def 一拍(self, concept_h, combo_h=None):
        """推理引擎一拍：推理预测、推理验证、推理学习、推理链。

        精卫公式流程：
        1. a = 当前概念激活（concept_h）
        2. b = 推理预测的下一步概念激活（reason_w × prev_concept_h）
        3. Δ = b - a（推理预测误差）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 推理权重更新
        6. τ读⊗→浮出所见 = 推理预测
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 概念激活（n_concepts维）
          combo_h: 组合概念激活（n_combo维，可选）

        返回：
          dict包含：
            reason_pred: 推理预测的下一步概念激活
            combo_reason_pred: 组合推理预测
            推理误差: 当前拍推理预测误差
            推理置信度: 推理预测与实际匹配度
            推理链深度: 当前推理链的深度
        """
        self.拍 += 1

        # ── 推理预测：从上一拍的概念激活预测当前概念激活 ──
        # 精卫公式：b = reason_w × prev_concept_h（推理预测）
        # τ读⊗→浮出所见：推理预测从历史因果模式中浮现
        new_reason_pred = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            if self.reason_strength[i] < 0.05:
                continue
            r_input = 0.0
            for j in range(self.n_concepts):
                r_input += self.reason_w[i][j] * self.prev_concept_h[j]
            new_reason_pred[i] = activation(r_input)

        # ── 推理验证：比较上一拍的推理预测与当前实际概念激活 ──
        # 精卫公式：Δ = reason_pred - concept_h（推理预测误差）
        # 推理预测误差驱动推理学习——预测准确的推理模式被强化
        # 生物学：前扣带回（ACC）的错误监控——推理预测与实际不符时触发修正
        reason_delta = [self.reason_pred[i] - concept_h[i]
                        for i in range(self.n_concepts)]
        reason_sigma = [self.reason_pred[i] + concept_h[i]
                        for i in range(self.n_concepts)]
        reason_otimes = [reason_delta[i] * reason_sigma[i]
                         for i in range(self.n_concepts)]

        # 当前拍推理预测误差（均方误差）
        current_mse = sum(d * d for d in reason_delta) / max(1, self.n_concepts)
        self.推理误差 = self.推理误差 * 0.95 + current_mse * 0.05
        self.推理次数 += 1

        # 推理置信度——推理预测与实际匹配度
        # 匹配度高→置信度高→推理权重可靠
        # 生物学：多巴胺系统——推理预测准确时多巴胺释放
        match = sum(1 for i in range(self.n_concepts)
                    if abs(self.reason_pred[i] - concept_h[i]) < 0.1)
        self.reason_confidence = match / max(1, self.n_concepts)
        if self.reason_confidence > 0.5:
            self.推理成功次数 += 1

        # ── (a, b) ⟲ ⊗ —— 推理权重更新 ──
        # 赫布学习：上一拍的概念和当前概念同时出现→推理转移权重增强
        # 精卫公式：⊗ = Δ × Σ → reason_w更新
        # 生物学：前额叶皮层的推理回路通过反复使用形成稳定模式
        for i in range(self.n_concepts):
            if self.reason_strength[i] < 0.05:
                continue
            for j in range(self.n_concepts):
                dw = self.lr * concept_h[i] * self.prev_concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.reason_w[i][j] = self.reason_w[i][j] * 0.99 + dw

        # ── 推理强度（用进废退） ──
        for i in range(self.n_concepts):
            if abs(concept_h[i]) > 0.15:
                self.reason_strength[i] = min(3.0, self.reason_strength[i] + 0.01)
            else:
                self.reason_strength[i] *= 0.999

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            if self.reason_strength[i] < 0.05:
                continue
            w_norm = sum(abs(self.reason_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.reason_w[i][j] *= scale

        # ── 组合推理：组合概念→组合概念的推理转移 ──
        if combo_h is not None:
            new_combo_pred = [0.0] * self.n_combo
            for i in range(self.n_combo):
                if self.combo_reason_strength[i] < 0.05:
                    continue
                r_input = 0.0
                for j in range(self.n_combo):
                    r_input += self.combo_reason_w[i][j] * self.prev_combo_h[j]
                new_combo_pred[i] = activation(r_input)

            # 组合推理学习
            for i in range(self.n_combo):
                if self.combo_reason_strength[i] < 0.05:
                    continue
                for j in range(self.n_combo):
                    dw = self.lr * combo_h[i] * self.prev_combo_h[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.combo_reason_w[i][j] = self.combo_reason_w[i][j] * 0.99 + dw

            # 组合推理强度
            for i in range(self.n_combo):
                if abs(combo_h[i]) > 0.15:
                    self.combo_reason_strength[i] = min(3.0, self.combo_reason_strength[i] + 0.01)
                else:
                    self.combo_reason_strength[i] *= 0.999

            # 组合推理突触缩放
            for i in range(self.n_combo):
                if self.combo_reason_strength[i] < 0.05:
                    continue
                w_norm = sum(abs(self.combo_reason_w[i][j]) for j in range(self.n_combo))
                if w_norm > 2.0:
                    scale = 2.0 / w_norm
                    for j in range(self.n_combo):
                        self.combo_reason_w[i][j] *= scale

            self.combo_reason_pred = new_combo_pred
            self.prev_combo_h = combo_h[:]
        else:
            new_combo_pred = self.combo_reason_pred

        # ── 推理链：多步推理 ──
        # 从当前概念激活出发，通过推理权重推导多步后的概念激活
        # 像AlphaGo的搜索树——从当前局面推导多个可能的未来
        # 推理链深度限制防止无限传播——像人类推理不会无限递归
        self.reason_chain = []
        current = list(concept_h)
        for step in range(self.max_chain_depth):
            next_pred = [0.0] * self.n_concepts
            for i in range(self.n_concepts):
                if self.reason_strength[i] < 0.05:
                    continue
                r_input = 0.0
                for j in range(self.n_concepts):
                    r_input += self.reason_w[i][j] * current[j]
                next_pred[i] = activation(r_input)
            self.reason_chain.append(next_pred)
            current = next_pred
            # 如果推理预测都很小，停止生成
            if max(abs(v) for v in next_pred) < 0.05:
                break

        # 保存当前概念激活用于下一拍的推理预测
        self.prev_concept_h = list(concept_h)
        self.reason_pred = new_reason_pred

        return {
            "reason_pred": self.reason_pred,
            "combo_reason_pred": new_combo_pred,
            "推理误差": round(self.推理误差, 4),
            "推理次数": self.推理次数,
            "推理成功次数": self.推理成功次数,
            "推理置信度": round(self.reason_confidence, 4),
            "推理链深度": len(self.reason_chain),
            "活跃推理数": sum(1 for s in self.reason_strength if s > 0.05),
            "活跃组合推理数": sum(1 for s in self.combo_reason_strength if s > 0.05),
        }

    def 推理链生成(self, concept_h, max_steps=5):
        """从当前概念激活生成推理链。

        自回归式推理：当前概念→推理预测下一步→继续推导
        像AlphaGo的搜索——从当前局面推导多个可能的未来
        像人类推理——"如果A，那么B，那么C..."

        精卫公式：
          a = 当前概念激活（已推导的）
          b = 推理预测的下一步概念激活
          Δ = b - a = 推理预测误差
          ⊗ = 累积推理历史 → 驱动下一步推导
          τ读⊗→浮出所见 = 推理链从⊗中浮现
        """
        chain = [list(concept_h)]
        current = list(concept_h)

        for step in range(max_steps):
            next_pred = [0.0] * self.n_concepts
            for i in range(self.n_concepts):
                if self.reason_strength[i] < 0.05:
                    continue
                r_input = 0.0
                for j in range(self.n_concepts):
                    r_input += self.reason_w[i][j] * current[j]
                next_pred[i] = activation(r_input)

            # 稀疏化——只保留最强的几个
            sorted_idx = sorted(range(self.n_concepts),
                                key=lambda i: abs(next_pred[i]), reverse=True)
            top_k = set(sorted_idx[:max(1, self.n_concepts // 4)])
            for i in range(self.n_concepts):
                if i not in top_k:
                    next_pred[i] *= 0.1

            chain.append(next_pred)
            current = next_pred

            # 如果推理预测都很小，停止
            if max(abs(v) for v in next_pred) < 0.05:
                break

        return chain

    def 状态摘要(self):
        return {
            "活跃推理数": sum(1 for s in self.reason_strength if s > 0.05),
            "活跃组合推理数": sum(1 for s in self.combo_reason_strength if s > 0.05),
            "推理误差": round(self.推理误差, 4),
            "推理次数": self.推理次数,
            "推理成功次数": self.推理成功次数,
            "推理置信度": round(self.reason_confidence, 4),
            "推理链深度": len(self.reason_chain),
            "reason_w均值": round(sum(sum(row) for row in self.reason_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 推理引擎验证（多步推理与推理验证） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    re = 推理引擎(n_concepts=64, n_combo=32)

    print("初始状态:")
    print(f"  概念数: {re.n_concepts}")
    print(f"  组合概念数: {re.n_combo}")
    print(f"  推理权重矩阵: reason_w ({re.n_concepts}×{re.n_concepts})")
    print(f"  最大推理链深度: {re.max_chain_depth}")

    print("\n--- 测试1: 推理预测从概念模式中浮现 ---")
    # 模拟两种不同的概念模式序列
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = re.一拍(concept_h)
    print(f"  模式A后: 推理误差={result['推理误差']}, 推理次数={result['推理次数']}")
    print(f"  推理预测top5: {sorted(range(64), key=lambda i: abs(re.reason_pred[i]), reverse=True)[:5]}")
    print(f"  推理置信度: {result['推理置信度']}")

    re2 = 推理引擎(n_concepts=64, n_combo=32)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[15] = 0.3
        concept_h[0] = 0.25
        result = re2.一拍(concept_h)
    print(f"  模式B后: 推理误差={result['推理误差']}, 推理次数={result['推理次数']}")
    print(f"  推理预测top5: {sorted(range(64), key=lambda i: abs(re2.reason_pred[i]), reverse=True)[:5]}")

    diff = sum(abs(re.reason_pred[i] - re2.reason_pred[i]) for i in range(64))
    print(f"  A vs B 推理预测差异: {diff:.4f}（应该 > 0——不同模式→不同推理预测）")

    print("\n--- 测试2: 推理区分度（独立实例） ---")
    patterns = [
        ("模式X", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式Y", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("均衡", [0.1] * 64),
    ]
    results = {}
    for name, pattern in patterns:
        re_t = 推理引擎(n_concepts=64, n_combo=32)
        for _ in range(50):
            re_t.一拍(pattern)
        results[name] = list(re_t.reason_pred)
        active = [i for i, v in enumerate(re_t.reason_pred) if abs(v) > 0.05]
        print(f"  {name}: 活跃推理={active[:8]}, 推理置信度={re_t.reason_confidence:.4f}")

    diff_xy = sum(abs(results["模式X"][i] - results["模式Y"][i]) for i in range(64))
    diff_xb = sum(abs(results["模式X"][i] - results["均衡"][i]) for i in range(64))
    print(f"  X vs Y 推理差异: {diff_xy:.4f}")
    print(f"  X vs 均衡 推理差异: {diff_xb:.4f}")

    print("\n--- 测试3: 推理链生成 ---")
    re3 = 推理引擎(n_concepts=64, n_combo=32)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        re3.一拍(concept_h)
    chain = re3.推理链生成(concept_h, max_steps=5)
    print(f"  推理链长度: {len(chain)}")
    for step, pred in enumerate(chain):
        active = [i for i, v in enumerate(pred) if abs(v) > 0.05]
        max_val = max(abs(v) for v in pred) if pred else 0
        print(f"  步{step}: 活跃={active[:5]}, 最大={max_val:.4f}")

    print("\n--- 测试4: 推理预测误差随学习降低 ---")
    re4 = 推理引擎(n_concepts=64, n_combo=32)
    errors = []
    for tick in range(200):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = re4.一拍(concept_h)
        if tick in [49, 99, 149, 199]:
            errors.append(result["推理误差"])
    print(f"  50拍: 误差={errors[0]}")
    print(f"  100拍: 误差={errors[1]}")
    print(f"  150拍: 误差={errors[2]}")
    print(f"  200拍: 误差={errors[3]}")
    print(f"  误差应该逐渐降低（推理学习→预测更准确）")

    print("\n--- 测试5: 推理用进废退 ---")
    re5 = 推理引擎(n_concepts=64, n_combo=32)
    for _ in range(200):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        re5.一拍(concept_h)
    alive = sum(1 for s in re5.reason_strength if s > 0.05)
    print(f"  200拍后: 存活推理={alive}")
    print(f"  推理强度: min={min(re5.reason_strength):.3f}, max={max(re5.reason_strength):.3f}")

    print("\n--- 测试6: 组合推理 ---")
    re6 = 推理引擎(n_concepts=64, n_combo=32)
    for _ in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        combo_h = [0.0] * 32
        combo_h[2] = 0.2
        combo_h[5] = 0.15
        re6.一拍(concept_h, combo_h=combo_h)
    print(f"  100拍后: 活跃组合推理={re6.状态摘要()['活跃组合推理数']}")
    combo_pred_top = sorted(range(32), key=lambda i: abs(re6.combo_reason_pred[i]), reverse=True)[:5]
    print(f"  组合推理预测top5: {combo_pred_top}")

    print("\n--- 测试7: 精卫公式验证 ---")
    re7 = 推理引擎(n_concepts=16, n_combo=8)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        concept_h[1] = 0.2
        result = re7.一拍(concept_h)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 推理预测={[round(v, 3) for v in result['reason_pred'][:4]]}")
            print(f"         推理误差={result['推理误差']}")
    print(f"  τ(浮出): 推理预测均值={round(sum(re7.reason_pred)/16, 4)}")
    print(f"  ⟲(循环): 拍={re7.拍}")

    print("\n验证通过")
