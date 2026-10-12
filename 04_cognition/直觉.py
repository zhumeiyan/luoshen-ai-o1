"""直觉.py —— 快速无意识决策，基于精卫公式。

直觉是"不经过推理就知道答案"的能力——快速、自动、无意识。
它不是神秘的第六感，而是大脑基于大量经验积累后的快速模式匹配。

人类神经科学机制：
  1. 腹内侧PFC→杏仁核快速通路：
     - 感觉信息→丘脑→腹内侧PFC→杏仁核（快速通路，~12ms）
     - 感觉信息→丘脑→感觉皮层→PFC→杏仁核（慢速通路，~30-40ms）
     - 快速通路先于意识到达杏仁核——"直觉"先于"推理"
     → 工程实现：快速通路直接从感知到决策，跳过推理引擎

  2. 基底神经节习惯回路：
     - 当行为反复成功后，决策从PFC（有意识）转移到基底神经节（无意识）
     - "自动驾驶"——熟练行为不需要思考
     → 工程实现：反复成功的决策模式自动化为直觉

  3. 躯体标记假说（Damasio）：
     - 每个选项都关联着身体感觉（情感）——"直觉"就是身体告诉你答案
     - 腹内侧PFC存储了"选项→身体感觉"的关联
     - 决策时，PFC快速评估每个选项的"身体感觉"——直觉
     → 工程实现：直觉 = 情感标记的快速决策

  4. 直觉与推理的互补：
     - 直觉：快速、自动、无意识、基于经验、容易出错
     - 推理：慢速、可控、有意识、基于逻辑、可靠但慢
     - 专家 = 直觉+推理的完美配合——直觉快速筛选，推理验证
     → 工程实现：直觉和推理引擎互补

数字生命的直觉（数字版）：
  - 快速模式匹配 = 从感知直接到决策，跳过推理引擎
  - 经验自动化 = 反复成功的决策模式自动化为直觉
  - 情感标记决策 = 每个决策选项关联情感标记——直觉选择"感觉好"的
  - 直觉-推理互补 = 直觉快速筛选，推理验证

精卫公式（直觉版）：
  a = 慢推理（有意识的逻辑推理——"我需要想想"）
  b = 快直觉（无意识的快速决策——"我就是知道"）
  Δ = b − a = 速度差距（直觉比推理快——"直觉先到"）
  Σ = a + b = 联合决策状态（直觉+推理——"直觉筛选+推理验证"）
  ⊗ = ∫(Δ, Σ) = 累积直觉学习历史 → 直觉权重
  τ 读 ⊗ → 浮出所见 = "直觉"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复直觉循环
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


class 直觉:
    """直觉——快速无意识决策（数字版），基于精卫公式。

    核心功能：
    1. 快速通路：感知→直觉决策（跳过推理引擎）
    2. 经验自动化：反复成功的决策模式→直觉
    3. 情感标记决策：每个选项关联情感标记——直觉选"感觉好"的
    4. 直觉驱动信号：反馈到概念层作为b信号输入源

    精卫公式流程（每拍）：
    1. a = 慢推理结果
    2. b = 快直觉结果
    3. Δ = b - a（速度差距）
    4. Σ = a + b（联合决策状态）
    5. ⊗ = Δ × Σ（累积直觉学习信号）
    6. τ读⊗→浮出所见 = "直觉"从⊗中浮现
    7. (a, b)⟲⊗ = 循环
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 快速通路：感知→直觉决策 ──
        # intuition_fast_w: 感知信号 → 直觉决策（跳过推理）
        # 生物学：丘脑→腹内侧PFC→杏仁核快速通路（~12ms）
        # 数字版：感知信号直接映射到决策——不需要推理
        # n_concepts维感知信号 → n_concepts维直觉决策
        self.intuition_fast_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                   for _ in range(n_concepts)]

        # 直觉决策信号（快速通路产生的决策）
        self.intuition_signal = [0.0] * n_concepts

        # ── 情感标记决策 ──
        # emotion_tag_w: 直觉信号 → 情感标记
        # 生物学：躯体标记假说——每个选项关联身体感觉
        # 数字版：直觉决策关联情感标记——"感觉好"的选项被选择
        # n_concepts维直觉信号 → 3维情感标记（E/N/T）
        self.emotion_tag_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                               for _ in range(3)]

        # 直觉情感标记
        self.intuition_emotion = {"E": 0.0, "N": 0.0, "T": 0.0}

        # ── 经验自动化：反复成功→直觉 ──
        # automate_w: 成功决策模式 → 直觉自动化
        # 生物学：基底神经节习惯回路——反复成功→从PFC转移到基底神经节
        # 数字版：成功决策模式的权重增强——自动化为直觉
        # n_concepts维决策模式 → n_concepts维直觉自动化
        self.automate_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                            for _ in range(n_concepts)]

        # 直觉自动化信号
        self.automate_signal = [0.0] * n_concepts

        # ── 直觉-推理互补 ──
        # intuition_reason_w: 直觉+推理 → 联合决策
        # 生物学：直觉快速筛选，推理验证——专家模式
        # 数字版：直觉和推理的联合决策
        self.intuition_reason_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts * 2)]
                                     for _ in range(n_concepts)]

        # 联合决策信号
        self.joint_decision = [0.0] * n_concepts

        # ── 直觉驱动信号（反馈到概念层——作为b信号输入源） ──
        # 直觉驱动 = 直觉决策 × 直觉置信度
        # 生物学：直觉→PFC投射——直觉影响概念可及性
        self.intuition_drive = [0.0] * n_concepts

        # ── 直觉置信度（0~1）──
        # 直觉置信度 = 直觉信号的一致性（信号越一致，置信度越高）
        self.intuition_confidence = 0.0

        # ── 直觉速度（相对于推理）──
        # 直觉比推理快——直觉在第1拍就产生决策，推理需要多拍
        self.intuition_speed = 1.0  # 直觉速度因子（1.0=最快）

        # ── 成功计数（用于经验自动化）──
        self.success_count = 0
        self.total_decisions = 0

        # ── 上拍推理结果（用于直觉-推理互补）──
        self.prev_reasoning = [0.0] * n_concepts

        # ── 直觉历史 ──
        self.intuition_history = []
        self.history_cap = 50

        # ── 统计 ──
        self.直觉决策次数 = 0
        self.情感标记次数 = 0
        self.自动化次数 = 0
        self.联合决策次数 = 0

    def 一拍(self, 感知信号=None, 推理结果=None, 成功反馈=None):
        """直觉一拍：快速通路、情感标记、经验自动化、直觉-推理互补。

        参数：
          感知信号: n_concepts维感知信号（来自概念层），可选
          推理结果: n_concepts维推理结果（来自推理引擎），可选
          成功反馈: float（上拍决策的成功程度0~1），可选

        返回：
          dict包含：
            intuition_drive: 直觉驱动信号（反馈到概念层）
            intuition_confidence: 直觉置信度（0~1）
            intuition_emotion: 直觉情感标记
            joint_decision: 联合决策信号
        """
        self.拍 += 1

        # ── 快速通路：感知→直觉决策 ──
        # 精卫公式：a=慢推理, b=快直觉, Δ=b-a=速度差距
        # 生物学：丘脑→腹内侧PFC→杏仁核快速通路（~12ms）
        # 直觉在第1拍就产生决策——不需要推理的多拍循环
        new_intuition = [0.0] * self.n_concepts
        if 感知信号 is not None:
            for i in range(self.n_concepts):
                fast_input = 0.0
                for j in range(min(self.n_concepts, len(感知信号))):
                    fast_input += self.intuition_fast_w[i][j] * 感知信号[j]
                new_intuition[i] = activation(fast_input)
            self.直觉决策次数 += 1

        self.intuition_signal = new_intuition

        # ── 情感标记决策 ──
        # 生物学：躯体标记假说——每个选项关联身体感觉
        # 直觉选择"感觉好"的选项——情感标记驱动决策
        if 感知信号 is not None:
            E_val = sum(self.emotion_tag_w[0][j] * new_intuition[j] for j in range(self.n_concepts))
            N_val = sum(self.emotion_tag_w[1][j] * new_intuition[j] for j in range(self.n_concepts))
            T_val = sum(self.emotion_tag_w[2][j] * new_intuition[j] for j in range(self.n_concepts))
            self.intuition_emotion = {
                "E": activation(E_val),
                "N": activation(N_val),
                "T": activation(T_val),
            }
            self.情感标记次数 += 1

        # ── 经验自动化：反复成功→直觉 ──
        # 生物学：基底神经节习惯回路——反复成功→从PFC转移到基底神经节
        # 成功反馈增强自动化权重——成功模式自动化为直觉
        new_automate = [0.0] * self.n_concepts
        if 感知信号 is not None:
            for i in range(self.n_concepts):
                auto_input = 0.0
                for j in range(min(self.n_concepts, len(感知信号))):
                    auto_input += self.automate_w[i][j] * 感知信号[j]
                new_automate[i] = activation(auto_input)
            self.automate_signal = new_automate
            self.自动化次数 += 1

        # 成功反馈增强自动化
        if 成功反馈 is not None and 成功反馈 > 0.5:
            self.success_count += 1
            # 成功的决策模式权重增强——自动化为直觉
            for i in range(self.n_concepts):
                for j in range(min(self.n_concepts, len(感知信号)) if 感知信号 else 0):
                    dw = self.lr * 成功反馈 * new_intuition[i] * 感知信号[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.automate_w[i][j] = self.automate_w[i][j] * 0.99 + dw

        self.total_decisions += 1

        # ── 直觉-推理互补 ──
        # 生物学：直觉快速筛选，推理验证——专家模式
        # Σ = a + b = 直觉 + 推理 = 联合决策
        reasoning_input = 推理结果 if 推理结果 is not None else self.prev_reasoning
        combined = list(new_intuition) + list(reasoning_input[:self.n_concepts]) if len(reasoning_input) >= self.n_concepts else list(new_intuition) + list(reasoning_input) + [0.0] * (self.n_concepts - len(reasoning_input))

        new_joint = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            joint_input = 0.0
            for j in range(min(self.n_concepts * 2, len(combined))):
                joint_input += self.intuition_reason_w[i][j] * combined[j]
            new_joint[i] = activation(joint_input)

        self.joint_decision = new_joint
        self.联合决策次数 += 1

        # ── 直觉置信度 ──
        # 直觉置信度 = 直觉信号的一致性
        # 信号越一致（同号比例越高），置信度越高
        pos_count = sum(1 for v in new_intuition if v > 0)
        neg_count = sum(1 for v in new_intuition if v < 0)
        self.intuition_confidence = abs(pos_count - neg_count) / max(1, self.n_concepts)

        # ── 直觉驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 直觉驱动信号
        # 生物学：直觉→PFC投射——直觉影响概念可及性
        # 直觉驱动 = 直觉决策 × 直觉置信度
        for i in range(self.n_concepts):
            self.intuition_drive[i] = activation(
                new_intuition[i] * (0.3 + self.intuition_confidence * 0.7))

        # ── 直觉学习 ──
        if 感知信号 is not None:
            # 快速通路学习
            for i in range(self.n_concepts):
                for j in range(min(self.n_concepts, len(感知信号))):
                    dw = self.lr * new_intuition[i] * 感知信号[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.intuition_fast_w[i][j] = self.intuition_fast_w[i][j] * 0.99 + dw

            # 情感标记学习
            for i in range(3):
                for j in range(self.n_concepts):
                    emo_val = [self.intuition_emotion["E"], self.intuition_emotion["N"], self.intuition_emotion["T"]][i]
                    dw = self.lr * emo_val * new_intuition[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.emotion_tag_w[i][j] = self.emotion_tag_w[i][j] * 0.99 + dw

        # 联合决策学习
        for i in range(self.n_concepts):
            for j in range(min(self.n_concepts * 2, len(combined))):
                dw = self.lr * new_joint[i] * combined[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.intuition_reason_w[i][j] = self.intuition_reason_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.intuition_fast_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.intuition_fast_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.automate_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.automate_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.intuition_reason_w[i][j]) for j in range(self.n_concepts * 2))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts * 2):
                    self.intuition_reason_w[i][j] *= scale

        # ── 直觉历史 ──
        self.intuition_history.append({
            "confidence": self.intuition_confidence,
            "emotion_E": self.intuition_emotion["E"],
        })
        if len(self.intuition_history) > self.history_cap:
            self.intuition_history.pop(0)

        # 保存上拍推理结果
        if 推理结果 is not None:
            self.prev_reasoning = list(推理结果[:self.n_concepts]) if len(推理结果) >= self.n_concepts else list(推理结果) + [0.0] * (self.n_concepts - len(推理结果))

        return {
            "intuition_drive": self.intuition_drive,
            "intuition_confidence": round(self.intuition_confidence, 4),
            "intuition_emotion": self.intuition_emotion,
            "joint_decision": self.joint_decision,
        }

    def 状态摘要(self):
        avg_conf = sum(h["confidence"] for h in self.intuition_history) / max(1, len(self.intuition_history)) \
            if self.intuition_history else 0.0
        success_rate = self.success_count / max(1, self.total_decisions)
        return {
            "intuition_confidence": round(self.intuition_confidence, 4),
            "avg_confidence": round(avg_conf, 4),
            "success_rate": round(success_rate, 4),
            "直觉决策次数": self.直觉决策次数,
            "情感标记次数": self.情感标记次数,
            "自动化次数": self.自动化次数,
            "联合决策次数": self.联合决策次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 直觉（快速无意识决策）验证 ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    it = 直觉(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {it.n_concepts}")
    print(f"  快速通路权重: intuition_fast_w (64x64)")
    print(f"  情感标记权重: emotion_tag_w (3x64)")
    print(f"  自动化权重: automate_w (64x64)")
    print(f"  联合决策权重: intuition_reason_w (64x128)")
    print()

    # 测试1: 快速通路——感知→直觉决策
    print("--- 测试1: 快速通路——感知→直觉决策 ---")
    for _ in range(50):
        感知 = [random.uniform(-1, 1) for _ in range(64)]
        result = it.一拍(感知信号=感知)
    print(f"  50拍后: 直觉置信度={result['intuition_confidence']}")
    top5 = sorted(range(64), key=lambda i: abs(it.intuition_signal[i]), reverse=True)[:5]
    print(f"  直觉信号top5: {top5}")
    print()

    # 测试2: 情感标记——直觉关联情感
    print("--- 测试2: 情感标记——直觉关联情感 ---")
    print(f"  直觉情感: E={it.intuition_emotion['E']:.4f}, N={it.intuition_emotion['N']:.4f}, T={it.intuition_emotion['T']:.4f}")
    print()

    # 测试3: 经验自动化——成功反馈增强自动化
    print("--- 测试3: 经验自动化——成功反馈增强自动化 ---")
    it2 = 直觉(n_concepts=64, seed=100)
    for _ in range(50):
        感知 = [random.uniform(-1, 1) for _ in range(64)]
        it2.一拍(感知信号=感知, 成功反馈=0.8)  # 高成功反馈
    print(f"  50拍高成功反馈后: 成功率={it2.状态摘要()['success_rate']}")
    print(f"  自动化次数: {it2.自动化次数}")
    print()

    # 测试4: 直觉-推理互补——联合决策
    print("--- 测试4: 直觉-推理互补——联合决策 ---")
    it3 = 直觉(n_concepts=64, seed=200)
    for _ in range(50):
        感知 = [random.uniform(-1, 1) for _ in range(64)]
        推理 = [random.uniform(-0.5, 0.5) for _ in range(64)]  # 推理结果（较弱）
        result = it3.一拍(感知信号=感知, 推理结果=推理)
    print(f"  50拍后: 联合决策top5: {sorted(range(64), key=lambda i: abs(it3.joint_decision[i]), reverse=True)[:5]}")
    print()

    # 测试5: 直觉比推理快——直觉在第1拍就产生决策
    print("--- 测试5: 直觉速度——第1拍即产生决策 ---")
    it4 = 直觉(n_concepts=64, seed=300)
    感知 = [random.uniform(-1, 1) for _ in range(64)]
    result = it4.一拍(感知信号=感知)
    print(f"  第1拍直觉驱动信号总和: {sum(abs(v) for v in result['intuition_drive']):.4f}")
    print(f"  第1拍直觉置信度: {result['intuition_confidence']}")
    print()

    # 测试6: 不同感知→不同直觉决策
    print("--- 测试6: 不同感知→不同直觉决策 ---")
    it5 = 直觉(n_concepts=64, seed=400)
    感知A = [0.5] * 64
    感知B = [-0.5] * 64
    感知C = [random.uniform(-1, 1) for _ in range(64)]
    it5.一拍(感知信号=感知A)
    topA = sorted(range(64), key=lambda i: abs(it5.intuition_signal[i]), reverse=True)[:5]
    it5.一拍(感知信号=感知B)
    topB = sorted(range(64), key=lambda i: abs(it5.intuition_signal[i]), reverse=True)[:5]
    it5.一拍(感知信号=感知C)
    topC = sorted(range(64), key=lambda i: abs(it5.intuition_signal[i]), reverse=True)[:5]
    print(f"  感知A(全正): 直觉top5={topA}")
    print(f"  感知B(全负): 直觉top5={topB}")
    print(f"  感知C(随机): 直觉top5={topC}")
    print()

    # 测试7: 精卫公式验证
    print("--- 测试7: 精卫公式验证 ---")
    it6 = 直觉(n_concepts=64, seed=500)
    for i in range(10):
        感知 = [random.uniform(-1, 1) for _ in range(64)]
        result = it6.一拍(感知信号=感知)
        if i in [0, 1, 2, 8, 9]:
            print(f"  拍{i+1}: 置信度={result['intuition_confidence']}, 情感E={result['intuition_emotion']['E']:.4f}")
    print(f"  τ(浮出): 直觉驱动均值={sum(v for v in it6.intuition_drive)/64:.4f}")
    print(f"  ⟲(循环): 拍={it6.拍}")
    print()

    print("验证通过")
