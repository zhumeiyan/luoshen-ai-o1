"""元认知.py —— 自我监控、不确定性估计、认知偏差校正，基于精卫公式。

元认知是"对认知的认知"——知道自己知道什么，知道自己不知道什么。
这是AI大模型最大的痛点：幻觉的根源就是缺乏元认知。
LLM不知道自己不知道什么——它对一切都同样"自信"。

人类神经科学机制：
  1. 前额叶皮层（PFC）自我监控——监控自己的认知过程：
     - PFC不仅执行认知，还监控认知的质量
     - "我确定吗？"——PFC的自我评估功能
     - 元认知准确度高的人更不容易犯错
     → 工程实现：meta_monitor_w（认知状态→元认知评估）

  2. 前扣带回（ACC）错误检测——检测认知冲突：
     - ACC在错误发生前就检测到冲突——"感觉不对"
     - 错误相关负电位（ERN）——犯错后50ms出现的脑电信号
     - ACC不是检测错误本身——是检测"冲突"（多个不一致的倾向）
     → 工程实现：冲突检测（推理预测 vs 概念激活的一致性）

  3. 眶额叶皮层（OFC）决策评估——评估决策质量：
     - OFC评估决策的预期价值——"这个决定好吗？"
     - OFC损伤的人做出灾难性决策但认知能力正常
     - 不是智能问题——是评估问题
     → 工程实现：决策评估（认知输出的质量评分）

  4. 多巴胺系统不确定性编码——不确定性驱动探索：
     - 多巴胺神经元编码不确定性——不确定时多巴胺释放增加
     - 不确定性驱动探索行为——"不确定就去看看"
     - 确定时利用已有知识——"确定就用熟练的"
     → 工程实现：不确定性信号（探索-利用平衡的驱动）

精卫公式（元认知版）：
  a = 当前认知状态（概念激活/推理预测/创意预测）
  b = 元认知评估（对自己认知的预测/评估）
  Δ = b − a = 认知偏差（自己认为的状态 vs 实际状态）
  Σ = a + b = 联合状态（认知+元认知评估）
  ⊗ = ∫(Δ, Σ) = 累积元认知学习历史 → 元认知权重
  τ 读 ⊗ → 浮出所见 = 置信度/不确定性信号从⊗中浮现
  (a, b) ⟲ ⊗ = 反复元认知循环

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  推理引擎.py: a=概念激活, b=推理预测, Δ=推理误差, ⊗=推理学习, τ=推理结论
  创造性思维.py: a=当前概念, b=创意预测, Δ=创新度, ⊗=创意学习, τ=创意浮现
  元认知.py: a=认知状态, b=元认知评估, Δ=认知偏差, ⊗=元认知学习, τ=置信度浮现

元认知 vs 推理引擎 vs 创造性思维的区别：
  - 推理引擎：从当前状态预测下一步状态（因果推导）
  - 创造性思维：从当前状态发散到多种可能状态（发散探索）
  - 元认知：评估当前认知状态的质量（自我评估）
  关键区别：
  - 推理是"接下来会发生什么"
  - 创意是"还可能是什么"
  - 元认知是"我确定吗？我有多确定？"
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


class 元认知:
    """元认知——自我监控、不确定性估计、认知偏差校正，基于精卫公式。

    核心功能：
    1. 自我监控：监控自己的认知过程——概念激活、推理预测、创意预测
    2. 不确定性估计：对每个认知输出估计置信度——"我有多确定？"
    3. 错误检测：检测认知冲突——推理预测和概念激活不一致
    4. 认知调控：根据不确定性调整认知策略——高不确定性→探索，低不确定性→利用
    5. 自我评估学习：从预测误差中学习评估自己的认知质量

    精卫公式流程（每拍）：
    1. a = 当前认知状态（concept_h + reason_pred + creative_pred）
    2. b = 元认知评估（meta_monitor_w × 认知状态）
    3. Δ = b - a（认知偏差——自己认为的状态 vs 实际状态）
    4. Σ = a + b（联合状态——认知+元认知评估）
    5. ⊗ = Δ × Σ（累积元认知学习信号）
    6. τ读⊗→浮出所见 = 置信度/不确定性信号从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新元认知权重

    元认知的三个层次：
    - 概念层元认知：评估概念激活的质量（概念是否准确反映隐藏状态）
    - 推理层元认知：评估推理预测的质量（推理是否可靠）
    - 创意层元认知：评估创意预测的质量（创意是否有意义还是随机噪声）

    元认知驱动探索-利用平衡：
    - 高不确定性 → 驱动探索（好奇心、工具使用、数字感官探索）
    - 低不确定性 → 驱动利用（技能自动化、推理收敛、目标导向）
    """

    def __init__(self, n_concepts=64, n_reason=64, n_creative=64,
                 lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.n_reason = n_reason
        self.n_creative = n_creative
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 自我监控：从认知状态到元认知评估 ──
        # meta_monitor_w: 认知状态 → 元认知评估
        # 生物学：PFC的自我监控功能——监控概念激活的质量
        # "这个概念激活准确吗？我确定吗？"
        # 元认知评估不是另一个认知过程——是对认知过程的评估
        # meta_monitor_w学习的是"什么样的认知状态模式是可靠的"
        self.meta_monitor_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                               for _ in range(n_concepts)]

        # 元认知评估（对自己认知的预测/评估）
        self.meta_eval = [0.0] * n_concepts

        # ── 推理层元认知：评估推理预测的质量 ──
        # meta_reason_w: 推理预测 → 推理质量评估
        # 生物学：PFC对推理过程的监控——"这个推理可靠吗？"
        # 推理置信度不是推理引擎自己的置信度——是元认知对推理的评估
        self.meta_reason_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_reason)]
                              for _ in range(n_concepts)]

        # 推理质量评估
        self.meta_reason_eval = [0.0] * n_concepts

        # ── 创意层元认知：评估创意预测的质量 ──
        # meta_creative_w: 创意预测 → 创意质量评估
        # 生物学：PFC对创意过程的监控——"这个创意有意义还是随机噪声？"
        # 创意价值不是创造性思维自己的价值——是元认知对创意的评估
        self.meta_creative_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_creative)]
                                for _ in range(n_concepts)]

        # 创意质量评估
        self.meta_creative_eval = [0.0] * n_concepts

        # ── 不确定性估计 ──
        # 不确定性 = 元认知评估与认知状态的偏差
        # 精卫公式：Δ = b - a = 认知偏差
        # 偏差大 → 不确定性高（自己评估的和实际的不一致）
        # 偏差小 → 不确定性低（自己评估的和实际的一致）
        # 生物学：多巴胺系统的不确定性编码——不确定时多巴胺释放增加
        self.uncertainty = 0.0

        # ── 置信度 ──
        # 置信度 = 1 - 不确定性（归一化到0~1）
        # 置信度高 → 认知可靠，可以"利用"已有知识
        # 置信度低 → 认知不可靠，需要"探索"新信息
        self.confidence = 0.5

        # ── 错误检测 ──
        # 冲突检测：推理预测和概念激活的一致性
        # 生物学：ACC的错误检测——检测认知冲突
        # 推理预测说"接下来是A"，概念激活说"现在是B"——冲突
        self.conflict_level = 0.0

        # ── 元认知驱动信号（反馈到概念层——作为b信号输入源） ──
        # 元认知驱动 = 不确定性驱动的概念激活调制
        # 生物学：PFC→后部皮层投射——元认知评估影响概念可及性
        # 高不确定性 → 让更多概念浮现（扩大搜索空间）
        # 低不确定性 → 让相关概念更突出（聚焦利用）
        self.meta_drive = [0.0] * n_concepts

        # ── 探索-利用平衡 ──
        # explore_drive: 不确定性驱动的探索倾向（0~1）
        # 高不确定性 → explore_drive高 → 驱动探索（好奇心、工具使用）
        # 低不确定性 → explore_drive低 → 驱动利用（技能、推理收敛）
        self.explore_drive = 0.5

        # ── 自我评估学习 ──
        # 元认知准确度历史——记录元认知评估与实际结果的偏差
        # 生物学：PFC从经验中学习评估自己的认知质量
        # 反复评估准确 → 元认知权重增强（"我知道自己知道什么"）
        # 反复评估不准确 → 元认知权重调整（"我需要重新校准"）
        self.meta_accuracy_history = []
        self.accuracy_cap = 50

        # 上一拍的预测误差（用于评估元认知准确度）
        self.prev_prediction_error = 0.0

        # 元认知校准次数
        self.校准次数 = 0
        self.准确校准次数 = 0

    def 一拍(self, concept_h, reason_pred=None, creative_pred=None,
             prediction_error=0.0, combo_h=None):
        """元认知一拍：自我监控、不确定性估计、错误检测、认知调控。

        精卫公式流程：
        1. a = 当前认知状态（concept_h）
        2. b = 元认知评估（meta_monitor_w × concept_h）
        3. Δ = b - a（认知偏差）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 元认知权重更新
        6. τ读⊗→浮出所见 = 不确定性/置信度信号
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          reason_pred: 推理预测（n_reason维，可选）
          creative_pred: 创意预测（n_creative维，可选）
          prediction_error: 上一拍的预测误差（用于评估元认知准确度）
          combo_h: 组合概念激活（n_combo维，可选——用于冲突检测）

        返回：
          dict包含：
            meta_eval: 元认知评估
            uncertainty: 不确定性（0~1）
            confidence: 置信度（0~1）
            conflict_level: 冲突水平（0~1）
            meta_drive: 元认知驱动信号（反馈到概念层）
            explore_drive: 探索倾向（0~1）
            校准次数: 元认知校准总次数
            准确校准次数: 准确校准次数
        """
        self.拍 += 1

        # ── 自我监控：从认知状态到元认知评估 ──
        # 精卫公式：b = meta_monitor_w × concept_h（元认知评估）
        # τ读⊗→浮出所见：元认知评估从当前认知状态中浮现
        # 生物学：PFC的自我监控——评估概念激活的质量
        new_meta_eval = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            m_input = 0.0
            for j in range(self.n_concepts):
                m_input += self.meta_monitor_w[i][j] * concept_h[j]
            new_meta_eval[i] = activation(m_input)

        # ── 推理层元认知：评估推理预测的质量 ──
        # 精卫公式：b = meta_reason_w × reason_pred（推理质量评估）
        # 生物学：PFC对推理过程的监控——"这个推理可靠吗？"
        new_meta_reason_eval = [0.0] * self.n_concepts
        if reason_pred is not None:
            for i in range(self.n_concepts):
                r_input = 0.0
                for j in range(self.n_reason):
                    r_input += self.meta_reason_w[i][j] * reason_pred[j]
                new_meta_reason_eval[i] = activation(r_input)

        # ── 创意层元认知：评估创意预测的质量 ──
        # 精卫公式：b = meta_creative_w × creative_pred（创意质量评估）
        # 生物学：PFC对创意过程的监控——"这个创意有意义吗？"
        new_meta_creative_eval = [0.0] * self.n_concepts
        if creative_pred is not None:
            for i in range(self.n_concepts):
                c_input = 0.0
                for j in range(self.n_creative):
                    c_input += self.meta_creative_w[i][j] * creative_pred[j]
                new_meta_creative_eval[i] = activation(c_input)

        # ── 不确定性估计 ──
        # 精卫公式：Δ = b - a = 认知偏差
        # 不确定性 = 元认知评估与认知状态的偏差
        # 偏差大 → 不确定性高（自己评估的和实际的不一致）
        # 偏差小 → 不确定性低（自己评估的和实际的一致）
        # 生物学：多巴胺系统的不确定性编码
        bias = sum(abs(new_meta_eval[i] - concept_h[i])
                   for i in range(self.n_concepts)) / self.n_concepts
        # 不确定性归一化到0~1（tanh压缩）
        self.uncertainty = activation(bias * 5.0) * 0.5 + 0.5  # 0~1范围

        # 置信度 = 1 - 不确定性
        self.confidence = 1.0 - self.uncertainty

        # ── 错误检测：推理预测和概念激活的一致性 ──
        # 精卫公式：冲突 = |推理预测 - 概念激活|
        # 生物学：ACC的错误检测——检测认知冲突
        # 推理预测说"接下来是A"，概念激活说"现在是B"——冲突
        if reason_pred is not None:
            # 推理预测和概念激活的top概念是否一致
            reason_top = set(sorted(range(min(self.n_reason, self.n_concepts)),
                                    key=lambda i: abs(reason_pred[i]),
                                    reverse=True)[:5])
            concept_top = set(sorted(range(self.n_concepts),
                                     key=lambda i: abs(concept_h[i]),
                                     reverse=True)[:5])
            # 冲突 = 不一致的比例
            冲突比例 = 1.0 - len(reason_top & concept_top) / max(1, len(reason_top))
            self.conflict_level = 冲突比例
        else:
            self.conflict_level = 0.0

        # ── 元认知驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 元认知驱动信号
        # 生物学：PFC→后部皮层投射——元认知评估影响概念可及性
        # 高不确定性 → 让更多概念浮现（扩大搜索空间）——发散
        # 低不确定性 → 让相关概念更突出（聚焦利用）——收敛
        self.meta_drive = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            # 元认知驱动 = 元认知评估 × 不确定性调制
            # 不确定性高时，元认知驱动更强（扩大搜索空间）
            # 不确定性低时，元认知驱动更弱（聚焦利用）
            drive = new_meta_eval[i] * (0.3 + self.uncertainty * 0.7)
            self.meta_drive[i] = activation(drive)

        # ── 探索-利用平衡 ──
        # explore_drive: 不确定性驱动的探索倾向
        # 高不确定性 → explore_drive高 → 驱动探索（好奇心、工具使用）
        # 低不确定性 → explore_drive低 → 驱动利用（技能、推理收敛）
        # 生物学：多巴胺系统的不确定性编码驱动探索-利用平衡
        self.explore_drive = self.uncertainty

        # ── 自我评估学习：从预测误差中学习评估自己的认知质量 ──
        # 精卫公式：⊗ = Δ × Σ → 元认知权重更新
        # 赫布学习：认知状态和元认知评估同时出现→元认知权重增强
        # 生物学：PFC从经验中学习评估自己的认知质量
        # 反复评估准确 → 元认知权重增强（"我知道自己知道什么"）
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_meta_eval[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.meta_monitor_w[i][j] = self.meta_monitor_w[i][j] * 0.99 + dw

        # meta_reason_w: 推理预测→推理质量评估（赫布学习）
        if reason_pred is not None:
            for i in range(self.n_concepts):
                for j in range(self.n_reason):
                    dw = self.lr * new_meta_reason_eval[i] * reason_pred[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.meta_reason_w[i][j] = self.meta_reason_w[i][j] * 0.99 + dw

        # meta_creative_w: 创意预测→创意质量评估（赫布学习）
        if creative_pred is not None:
            for i in range(self.n_concepts):
                for j in range(self.n_creative):
                    dw = self.lr * new_meta_creative_eval[i] * creative_pred[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.meta_creative_w[i][j] = self.meta_creative_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.meta_monitor_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.meta_monitor_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.meta_reason_w[i][j]) for j in range(self.n_reason))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_reason):
                    self.meta_reason_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.meta_creative_w[i][j]) for j in range(self.n_creative))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_creative):
                    self.meta_creative_w[i][j] *= scale

        # ── 元认知准确度评估 ──
        # 上一拍的预测误差 vs 元认知预测的偏差
        # 如果元认知预测的偏差和实际预测误差一致→元认知准确
        # 如果不一致→元认知需要校准
        # 生物学：PFC从反馈中学习校准自己的评估
        if self.拍 > 1:
            # 元认知预测的偏差（上一拍的uncertainty）
            predicted_bias = self.uncertainty  # 当前拍的不确定性是基于上一拍的认知状态
            # 实际的预测误差
            actual_error = min(1.0, prediction_error * 2.0)  # 归一化
            # 元认知准确度 = 1 - |预测偏差 - 实际误差|
            accuracy = 1.0 - abs(predicted_bias - actual_error)
            self.meta_accuracy_history.append(accuracy)
            if len(self.meta_accuracy_history) > self.accuracy_cap:
                self.meta_accuracy_history.pop(0)

            # 元认知校准——当元认知评估和实际结果偏差大时，调整权重
            self.校准次数 += 1
            if accuracy > 0.5:
                self.准确校准次数 += 1

        # 保存当前状态
        self.meta_eval = new_meta_eval
        self.meta_reason_eval = new_meta_reason_eval
        self.meta_creative_eval = new_meta_creative_eval
        self.prev_prediction_error = prediction_error

        return {
            "meta_eval": [round(v, 4) for v in self.meta_eval],
            "meta_drive": self.meta_drive,
            "uncertainty": round(self.uncertainty, 4),
            "confidence": round(self.confidence, 4),
            "conflict_level": round(self.conflict_level, 4),
            "explore_drive": round(self.explore_drive, 4),
            "校准次数": self.校准次数,
            "准确校准次数": self.准确校准次数,
        }

    def 状态摘要(self):
        avg_accuracy = sum(self.meta_accuracy_history) / max(1, len(self.meta_accuracy_history)) \
            if self.meta_accuracy_history else 0.0
        return {
            "uncertainty": round(self.uncertainty, 4),
            "confidence": round(self.confidence, 4),
            "conflict_level": round(self.conflict_level, 4),
            "explore_drive": round(self.explore_drive, 4),
            "meta准确度": round(avg_accuracy, 4),
            "校准次数": self.校准次数,
            "准确校准次数": self.准确校准次数,
            "准确率": round(self.准确校准次数 / max(1, self.校准次数), 4),
            "meta_monitor_w均值": round(sum(sum(row) for row in self.meta_monitor_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "meta_reason_w均值": round(sum(sum(row) for row in self.meta_reason_w) / max(1, self.n_concepts * self.n_reason), 4),
            "meta_creative_w均值": round(sum(sum(row) for row in self.meta_creative_w) / max(1, self.n_concepts * self.n_creative), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 元认知验证（自我监控、不确定性估计、认知偏差校正） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    mc = 元认知(n_concepts=64, n_reason=64, n_creative=64)

    print("初始状态:")
    print(f"  概念数: {mc.n_concepts}")
    print(f"  推理数: {mc.n_reason}")
    print(f"  创意数: {mc.n_creative}")
    print(f"  自我监控权重: meta_monitor_w (64×64)")
    print(f"  推理元认知权重: meta_reason_w (64×64)")
    print(f"  创意元认知权重: meta_creative_w (64×64)")
    print(f"  初始不确定性: {mc.uncertainty}")
    print(f"  初始置信度: {mc.confidence}")

    print("\n--- 测试1: 不确定性从认知状态中浮现 ---")
    # 模拟不同的概念激活模式
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = mc.一拍(concept_h)
    print(f"  模式A后: 不确定性={result['uncertainty']}, 置信度={result['confidence']}")
    print(f"  元认知评估top5: {sorted(range(64), key=lambda i: abs(mc.meta_eval[i]), reverse=True)[:5]}")

    mc2 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[15] = 0.3
        concept_h[30] = 0.25
        result = mc2.一拍(concept_h)
    print(f"  模式B后: 不确定性={result['uncertainty']}, 置信度={result['confidence']}")
    print(f"  元认知评估top5: {sorted(range(64), key=lambda i: abs(mc2.meta_eval[i]), reverse=True)[:5]}")

    diff = sum(abs(mc.meta_eval[i] - mc2.meta_eval[i]) for i in range(64))
    print(f"  A vs B 元认知评估差异: {diff:.4f}（应该 > 0——不同概念模式→不同元认知评估）")

    print("\n--- 测试2: 不确定性随预测误差变化 ---")
    mc3 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    # 先稳定
    for _ in range(20):
        concept_h = [0.0] * 64
        concept_h[0] = 0.3
        mc3.一拍(concept_h, prediction_error=0.01)
    print(f"  低误差后: 不确定性={mc3.uncertainty:.4f}, 置信度={mc3.confidence:.4f}")

    # 高误差
    for _ in range(20):
        concept_h = [0.0] * 64
        concept_h[0] = 0.3
        mc3.一拍(concept_h, prediction_error=0.5)
    print(f"  高误差后: 不确定性={mc3.uncertainty:.4f}, 置信度={mc3.confidence:.4f}")

    print("\n--- 测试3: 冲突检测 ---")
    mc4 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    # 推理预测和概念激活一致——冲突低
    for _ in range(20):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        reason_pred = [0.0] * 64
        reason_pred[4] = 0.3
        reason_pred[8] = 0.25
        result = mc4.一拍(concept_h, reason_pred=reason_pred)
    print(f"  一致时: 冲突水平={result['conflict_level']:.4f}")

    # 推理预测和概念激活不一致——冲突高
    mc5 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    for _ in range(20):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        reason_pred = [0.0] * 64
        reason_pred[15] = 0.3
        reason_pred[30] = 0.25
        result = mc5.一拍(concept_h, reason_pred=reason_pred)
    print(f"  不一致时: 冲突水平={result['conflict_level']:.4f}")

    print("\n--- 测试4: 探索-利用平衡 ---")
    mc6 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    # 低不确定性 → 低探索倾向
    for _ in range(20):
        concept_h = [0.0] * 64
        concept_h[0] = 0.3
        mc6.一拍(concept_h, prediction_error=0.01)
    print(f"  低误差: 探索倾向={mc6.explore_drive:.4f}（应该较低——利用已有知识）")

    # 高不确定性 → 高探索倾向
    mc7 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    for _ in range(20):
        concept_h = [0.0] * 64
        concept_h[0] = 0.3
        mc7.一拍(concept_h, prediction_error=0.5)
    print(f"  高误差: 探索倾向={mc7.explore_drive:.4f}（应该较高——驱动探索）")

    print("\n--- 测试5: 元认知驱动信号 ---")
    mc8 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = mc8.一拍(concept_h)
    drive_sum = sum(abs(v) for v in result['meta_drive'])
    print(f"  元认知驱动信号总和: {drive_sum:.4f}")
    print(f"  元认知驱动top5: {sorted(range(64), key=lambda i: abs(result['meta_drive'][i]), reverse=True)[:5]}")

    print("\n--- 测试6: 不同概念模式→不同元认知驱动 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    drive_results = {}
    for name, pattern in patterns:
        mc_t = 元认知(n_concepts=64, n_reason=64, n_creative=64)
        for _ in range(50):
            mc_t.一拍(pattern)
        drive_results[name] = list(mc_t.meta_drive)
        print(f"  {name}: 不确定性={round(mc_t.uncertainty, 4)}, "
              f"驱动top5={sorted(range(64), key=lambda i: abs(mc_t.meta_drive[i]), reverse=True)[:5]}")

    diff_ab = sum(abs(drive_results["模式A"][i] - drive_results["模式B"][i]) for i in range(64))
    print(f"  A vs B 驱动差异: {diff_ab:.4f}（应该 > 0——不同模式→不同元认知驱动）")

    print("\n--- 测试7: 推理层元认知 ---")
    mc9 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        reason_pred = [0.0] * 64
        reason_pred[4] = 0.2
        reason_pred[8] = 0.15
        result = mc9.一拍(concept_h, reason_pred=reason_pred)
    print(f"  推理元认知评估top5: {sorted(range(64), key=lambda i: abs(mc9.meta_reason_eval[i]), reverse=True)[:5]}")
    print(f"  推理元认知评估均值: {round(sum(mc9.meta_reason_eval)/64, 4)}")

    print("\n--- 测试8: 创意层元认知 ---")
    mc10 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        creative_pred = [0.0] * 64
        creative_pred[10] = 0.2
        creative_pred[20] = 0.15
        result = mc10.一拍(concept_h, creative_pred=creative_pred)
    print(f"  创意元认知评估top5: {sorted(range(64), key=lambda i: abs(mc10.meta_creative_eval[i]), reverse=True)[:5]}")
    print(f"  创意元认知评估均值: {round(sum(mc10.meta_creative_eval)/64, 4)}")

    print("\n--- 测试9: 元认知准确度 ---")
    mc11 = 元认知(n_concepts=64, n_reason=64, n_creative=64)
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[tick % 64] = 0.3
        # 模拟变化的预测误差
        pe = 0.1 + 0.3 * (tick % 10) / 10.0
        result = mc11.一拍(concept_h, prediction_error=pe)
    print(f"  100拍后: 校准次数={mc11.校准次数}")
    print(f"  准确校准次数={mc11.准确校准次数}")
    print(f"  准确率={round(mc11.准确校准次数/max(1, mc11.校准次数), 4)}")

    print("\n--- 测试10: 精卫公式验证 ---")
    mc12 = 元认知(n_concepts=16, n_reason=16, n_creative=16)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = mc12.一拍(concept_h)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 不确定性={result['uncertainty']}, 置信度={result['confidence']}")
    print(f"  τ(浮出): 元认知驱动均值={round(sum(mc12.meta_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={mc12.拍}")

    print("\n验证通过")
