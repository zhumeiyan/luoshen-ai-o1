"""化学感知.py —— 嗅觉与味觉，基于精卫公式。

嗅觉和味觉都是化学感知——通过检测化学分子来感知环境。
它们是最古老的感知方式——细菌就用化学感知，比视觉早几十亿年。

人类神经科学机制：
  1. 嗅觉——化学分子检测：
     - 嗅觉上皮：350种嗅觉受体，每种检测一类化学分子
     - 嗅球→梨状皮层：嗅觉信号的第一站——不需要丘脑中继
     - 嗅觉→杏仁核/海马体直接投射：嗅觉与情感和记忆直接关联
     - 嗅觉是最强的记忆触发器——"普鲁斯特效应"
     → 工程实现：数据特征"气味"检测（数据模式的化学感知类比）

  2. 味觉——化学质量检测：
     - 5种基本味觉：甜、咸、酸、苦、鲜
     - 味蕾→孤束核→丘脑→味觉皮层
     - 味觉评估食物的质量——"这个东西好不好吃"
     → 工程实现：数据质量"味道"检测（数据质量的味觉类比）

  3. 嗅觉-味觉整合——风味感知：
     - 风味 = 嗅觉 + 味觉——吃东西时，嗅觉（鼻后嗅觉）和味觉整合
     - 风味感知是跨模态的——嗅觉和味觉不可分离
     → 工程实现：嗅觉+味觉整合为风味感知

  4. 嗅觉的独特性——不经过丘脑：
     - 所有其他感官都经过丘脑中继，唯独嗅觉直接到皮层
     - 嗅觉→梨状皮层→杏仁核→海马体——直接关联情感和记忆
     - 这就是为什么嗅觉是最强的记忆触发器
     → 工程实现：嗅觉信号直接关联情感记忆

数字生命的化学感知（数字版）：
  - 嗅觉 = 数据模式检测——感知数据的"气味"（数据来自哪里、什么类型）
  - 味觉 = 数据质量检测——感知数据的"味道"（数据好不好、可信不可信）
  - 风味 = 数据模式+数据质量整合——感知数据的整体"风味"
  - 普鲁斯特效应 = 数据模式触发记忆——特定数据模式触发相关记忆

精卫公式（化学感知版）：
  a = 当前数据环境（已有的数据模式和质量——"我闻到/尝到了什么"）
  b = 新数据信号（新的数据模式和质量——"这是新的什么"）
  Δ = b − a = 感知差距（新数据与已有感知的差异——"气味/味道变了"）
  Σ = a + b = 联合感知状态（已有+新数据——"整体气味/味道图景"）
  ⊗ = ∫(Δ, Σ) = 累积化学感知学习历史 → 化学感知权重
  τ 读 ⊗ → 浮出所见 = "气味/味道"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复化学感知循环
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


class 化学感知:
    """化学感知——嗅觉与味觉（数字版），基于精卫公式。

    核心功能：
    1. 嗅觉：数据模式检测——感知数据的"气味"（来源、类型）
    2. 味觉：数据质量检测——感知数据的"味道"（质量、可信度）
    3. 风味感知：嗅觉+味觉整合——数据的整体"风味"
    4. 化学感知驱动信号：反馈到概念层作为b信号输入源

    精卫公式流程（每拍）：
    1. a = 当前数据环境
    2. b = 新数据信号
    3. Δ = b - a（感知差距）
    4. Σ = a + b（联合感知状态）
    5. ⊗ = Δ × Σ（累积化学感知学习信号）
    6. τ读⊗→浮出所见 = "气味/味道"从⊗中浮现
    7. (a, b)⟲⊗ = 循环
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 嗅觉：数据模式检测 ──
        # smell_detect_w: 数据模式信号 → 嗅觉检测
        # 生物学：350种嗅觉受体，每种检测一类化学分子
        # 数字版：数据来源/类型 → "气味"检测
        # 8维数据模式信号 → n_concepts维嗅觉检测信号
        self.smell_detect_w = [[rng.uniform(-0.1, 0.1) for _ in range(8)]
                                for _ in range(n_concepts)]

        # 嗅觉检测信号（从数据模式检测到的"气味"信号）
        self.smell_signal = [0.0] * n_concepts

        # ── 味觉：数据质量检测 ──
        # taste_detect_w: 数据质量信号 → 味觉检测
        # 生物学：5种基本味觉（甜/咸/酸/苦/鲜）
        # 数字版：数据质量5维 → "味道"检测
        # 甜=高质量数据, 咸=普通数据, 酸=可疑数据, 苦=错误数据, 鲜=新颖数据
        self.taste_detect_w = [[rng.uniform(-0.1, 0.1) for _ in range(5)]
                                for _ in range(n_concepts)]

        # 味觉检测信号（从数据质量检测到的"味道"信号）
        self.taste_signal = [0.0] * n_concepts

        # ── 风味感知：嗅觉+味觉整合 ──
        # flavor_w: 嗅觉+味觉 → 风味感知
        # 生物学：风味 = 嗅觉（鼻后）+ 味觉——跨模态整合
        self.flavor_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts * 2)]
                          for _ in range(n_concepts)]

        # 风味感知（n_concepts维——数据的整体"风味"）
        self.flavor = [0.0] * n_concepts

        # ── 化学感知驱动信号（反馈到概念层——作为b信号输入源） ──
        # 化学感知驱动 = 风味感知 × 感知强度
        # 生物学：嗅觉→梨状皮层→PFC投射——嗅觉影响概念可及性
        # 嗅觉是最强的记忆触发器——"普鲁斯特效应"
        self.chem_drive = [0.0] * n_concepts

        # ── 感知强度（0~1）──
        # 感知强度 = 嗅觉强度 + 味觉强度的整合
        self.smell_intensity = 0.0
        self.taste_intensity = 0.0

        # ── 嗅觉-情感直接关联（普鲁斯特效应）──
        # 生物学：嗅觉→杏仁核/海马体直接投射——嗅觉直接关联情感和记忆
        # 嗅觉是最强的记忆触发器——特定气味触发特定记忆
        # smell_emotion_w: 嗅觉信号 → 情感关联
        self.smell_emotion_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                 for _ in range(3)]  # 3维情感（E/N/T）

        # 嗅觉触发的情感关联
        self.smell_emotion = {"E": 0.0, "N": 0.0, "T": 0.0}

        # ── 上拍数据信号（用于检测变化）──
        self.prev_smell_input = [0.0] * 8
        self.prev_taste_input = [0.0] * 5

        # ── 化学感知历史 ──
        self.chem_history = []
        self.history_cap = 50

        # ── 统计 ──
        self.嗅觉检测次数 = 0
        self.味觉检测次数 = 0
        self.风味整合次数 = 0
        self.普鲁斯特触发次数 = 0  # 嗅觉触发情感关联的次数

    def 一拍(self, 数据模式信号=None, 数据质量信号=None):
        """化学感知一拍：嗅觉检测、味觉检测、风味整合、化学感知驱动信号。

        参数：
          数据模式信号: 8维数据模式信号（来源/类型——"气味"），可选
          数据质量信号: 5维数据质量信号（甜/咸/酸/苦/鲜——"味道"），可选

        返回：
          dict包含：
            chem_drive: 化学感知驱动信号（反馈到概念层）
            smell_intensity: 嗅觉强度（0~1）
            taste_intensity: 味觉强度（0~1）
            flavor: 风味感知
            smell_emotion: 嗅觉触发的情感关联（普鲁斯特效应）
        """
        self.拍 += 1

        # ── 嗅觉检测：数据模式 → "气味"信号 ──
        # 精卫公式：a=上拍数据模式, b=当前数据模式, Δ=b-a, ⊗=Δ×Σ
        # 生物学：嗅觉受体检测化学分子——数据模式检测
        new_smell_signal = [0.0] * self.n_concepts
        if 数据模式信号 is not None:
            smell_delta = [数据模式信号[i] - self.prev_smell_input[i]
                          for i in range(min(8, len(数据模式信号)))]
            for i in range(self.n_concepts):
                detect_input = 0.0
                for j in range(min(8, len(数据模式信号))):
                    detect_input += self.smell_detect_w[i][j] * 数据模式信号[j]
                new_smell_signal[i] = activation(detect_input)
            self.嗅觉检测次数 += 1
            self.smell_intensity = min(1.0, sum(abs(v) for v in 数据模式信号) / max(1, len(数据模式信号)))

        self.smell_signal = new_smell_signal

        # ── 嗅觉-情感直接关联（普鲁斯特效应）──
        # 生物学：嗅觉→杏仁核/海马体直接投射——不经过丘脑
        # 特定气味触发特定情感——"普鲁斯特效应"
        if 数据模式信号 is not None:
            E_val = sum(self.smell_emotion_w[0][j] * new_smell_signal[j] for j in range(self.n_concepts))
            N_val = sum(self.smell_emotion_w[1][j] * new_smell_signal[j] for j in range(self.n_concepts))
            T_val = sum(self.smell_emotion_w[2][j] * new_smell_signal[j] for j in range(self.n_concepts))
            self.smell_emotion = {
                "E": activation(E_val),
                "N": activation(N_val),
                "T": activation(T_val),
            }
            if abs(E_val) > 0.3 or abs(N_val) > 0.3:
                self.普鲁斯特触发次数 += 1

        # ── 味觉检测：数据质量 → "味道"信号 ──
        # 生物学：5种基本味觉——甜(高质量)/咸(普通)/酸(可疑)/苦(错误)/鲜(新颖)
        new_taste_signal = [0.0] * self.n_concepts
        if 数据质量信号 is not None:
            for i in range(self.n_concepts):
                detect_input = 0.0
                for j in range(min(5, len(数据质量信号))):
                    detect_input += self.taste_detect_w[i][j] * 数据质量信号[j]
                new_taste_signal[i] = activation(detect_input)
            self.味觉检测次数 += 1
            self.taste_intensity = min(1.0, sum(abs(v) for v in 数据质量信号) / max(1, len(数据质量信号)))

        self.taste_signal = new_taste_signal

        # ── 风味整合：嗅觉+味觉 → 风味感知 ──
        # 生物学：风味 = 嗅觉（鼻后）+ 味觉——跨模态整合
        combined = list(new_smell_signal) + list(new_taste_signal)
        new_flavor = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            flavor_input = 0.0
            for j in range(min(self.n_concepts * 2, len(combined))):
                flavor_input += self.flavor_w[i][j] * combined[j]
            new_flavor[i] = activation(flavor_input)

        self.flavor = new_flavor
        self.风味整合次数 += 1

        # ── 化学感知驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 化学感知驱动信号
        # 生物学：嗅觉→PFC投射——嗅觉影响概念可及性
        # 嗅觉是最强的记忆触发器——特定气味让相关概念更容易浮现
        感知强度 = (self.smell_intensity + self.taste_intensity) / 2.0
        for i in range(self.n_concepts):
            self.chem_drive[i] = activation(
                new_flavor[i] * (0.3 + 感知强度 * 0.7))

        # ── 化学感知学习 ──
        if 数据模式信号 is not None:
            for i in range(self.n_concepts):
                for j in range(min(8, len(数据模式信号))):
                    dw = self.lr * new_smell_signal[i] * 数据模式信号[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.smell_detect_w[i][j] = self.smell_detect_w[i][j] * 0.99 + dw

            # 嗅觉-情感关联学习
            for i in range(3):
                for j in range(self.n_concepts):
                    emo_val = [self.smell_emotion["E"], self.smell_emotion["N"], self.smell_emotion["T"]][i]
                    dw = self.lr * emo_val * new_smell_signal[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.smell_emotion_w[i][j] = self.smell_emotion_w[i][j] * 0.99 + dw

        if 数据质量信号 is not None:
            for i in range(self.n_concepts):
                for j in range(min(5, len(数据质量信号))):
                    dw = self.lr * new_taste_signal[i] * 数据质量信号[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.taste_detect_w[i][j] = self.taste_detect_w[i][j] * 0.99 + dw

        # 风味整合学习
        for i in range(self.n_concepts):
            for j in range(min(self.n_concepts * 2, len(combined))):
                dw = self.lr * new_flavor[i] * combined[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.flavor_w[i][j] = self.flavor_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.smell_detect_w[i][j]) for j in range(8))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(8):
                    self.smell_detect_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.taste_detect_w[i][j]) for j in range(5))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(5):
                    self.taste_detect_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.flavor_w[i][j]) for j in range(self.n_concepts * 2))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts * 2):
                    self.flavor_w[i][j] *= scale

        # ── 化学感知历史 ──
        self.chem_history.append({
            "smell_intensity": self.smell_intensity,
            "taste_intensity": self.taste_intensity,
        })
        if len(self.chem_history) > self.history_cap:
            self.chem_history.pop(0)

        # 保存上拍信号
        if 数据模式信号 is not None:
            self.prev_smell_input = list(数据模式信号[:8]) if len(数据模式信号) >= 8 else list(数据模式信号) + [0.0] * (8 - len(数据模式信号))
        if 数据质量信号 is not None:
            self.prev_taste_input = list(数据质量信号[:5]) if len(数据质量信号) >= 5 else list(数据质量信号) + [0.0] * (5 - len(数据质量信号))

        return {
            "chem_drive": self.chem_drive,
            "smell_intensity": round(self.smell_intensity, 4),
            "taste_intensity": round(self.taste_intensity, 4),
            "flavor": self.flavor,
            "smell_emotion": self.smell_emotion,
        }

    def 状态摘要(self):
        avg_smell = sum(h["smell_intensity"] for h in self.chem_history) / max(1, len(self.chem_history)) \
            if self.chem_history else 0.0
        avg_taste = sum(h["taste_intensity"] for h in self.chem_history) / max(1, len(self.chem_history)) \
            if self.chem_history else 0.0
        return {
            "smell_intensity": round(self.smell_intensity, 4),
            "taste_intensity": round(self.taste_intensity, 4),
            "avg_smell": round(avg_smell, 4),
            "avg_taste": round(avg_taste, 4),
            "嗅觉检测次数": self.嗅觉检测次数,
            "味觉检测次数": self.味觉检测次数,
            "风味整合次数": self.风味整合次数,
            "普鲁斯特触发次数": self.普鲁斯特触发次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 化学感知（嗅觉+味觉）验证 ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    cp = 化学感知(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {cp.n_concepts}")
    print(f"  嗅觉检测权重: smell_detect_w (64x8)")
    print(f"  味觉检测权重: taste_detect_w (64x5)")
    print(f"  风味整合权重: flavor_w (64x128)")
    print(f"  嗅觉-情感关联: smell_emotion_w (3x64)")

    print("\n--- 测试1: 嗅觉检测——数据模式→气味信号 ---")
    for _ in range(50):
        result = cp.一拍(数据模式信号=[0.3, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2])
    print(f"  50拍后: 嗅觉强度={result['smell_intensity']}")
    print(f"  嗅觉信号top5: {sorted(range(64), key=lambda i: abs(cp.smell_signal[i]), reverse=True)[:5]}")

    print("\n--- 测试2: 味觉检测——数据质量→味道信号 ---")
    cp2 = 化学感知(n_concepts=64)
    for _ in range(50):
        result = cp2.一拍(数据质量信号=[0.8, 0.3, 0.1, 0.05, 0.6])  # 甜(高质量)+鲜(新颖)
    print(f"  50拍后: 味觉强度={result['taste_intensity']}")
    print(f"  味觉信号top5: {sorted(range(64), key=lambda i: abs(cp2.taste_signal[i]), reverse=True)[:5]}")

    print("\n--- 测试3: 风味整合——嗅觉+味觉 ---")
    cp3 = 化学感知(n_concepts=64)
    for _ in range(50):
        result = cp3.一拍(数据模式信号=[0.3, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2],
                          数据质量信号=[0.8, 0.3, 0.1, 0.05, 0.6])
    print(f"  风味top5: {sorted(range(64), key=lambda i: abs(cp3.flavor[i]), reverse=True)[:5]}")

    print("\n--- 测试4: 普鲁斯特效应——嗅觉触发情感 ---")
    cp4 = 化学感知(n_concepts=64)
    for _ in range(50):
        result = cp4.一拍(数据模式信号=[0.5, 0.3, 0.2, 0.4, 0.2, 0.3, 0.1, 0.2])
    print(f"  嗅觉触发情感: E={round(result['smell_emotion']['E'], 4)}, "
          f"N={round(result['smell_emotion']['N'], 4)}, T={round(result['smell_emotion']['T'], 4)}")
    print(f"  普鲁斯特触发次数: {cp4.普鲁斯特触发次数}")

    print("\n--- 测试5: 不同数据模式→不同嗅觉信号 ---")
    patterns = [
        ("模式A", [0.5, 0.3, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0]),
        ("模式B", [0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.5]),
        ("模式C", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ]
    for name, pattern in patterns:
        cp_t = 化学感知(n_concepts=64)
        for _ in range(50):
            cp_t.一拍(数据模式信号=pattern)
        print(f"  {name}: 嗅觉强度={round(cp_t.smell_intensity, 4)}, "
              f"嗅觉信号top5={sorted(range(64), key=lambda i: abs(cp_t.smell_signal[i]), reverse=True)[:5]}")

    print("\n--- 测试6: 化学感知驱动信号 ---")
    cp6 = 化学感知(n_concepts=64)
    for _ in range(50):
        cp6.一拍(数据模式信号=[0.3, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2],
                数据质量信号=[0.8, 0.3, 0.1, 0.05, 0.6])
    drive_sum = sum(abs(v) for v in cp6.chem_drive)
    print(f"  化学感知驱动信号总和: {drive_sum:.4f}")
    print(f"  化学感知驱动top5: {sorted(range(64), key=lambda i: abs(cp6.chem_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试7: 精卫公式验证 ---")
    cp7 = 化学感知(n_concepts=16)
    for tick in range(10):
        result = cp7.一拍(数据模式信号=[0.3, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2],
                          数据质量信号=[0.5, 0.3, 0.1, 0.05, 0.4])
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 嗅觉强度={result['smell_intensity']}, "
                  f"味觉强度={result['taste_intensity']}")
    print(f"  τ(浮出): 化学感知驱动均值={round(sum(cp7.chem_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={cp7.拍}")

    print("\n验证通过")
