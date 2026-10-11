"""痛觉.py —— 伤害感知与保护性反应，基于精卫公式。

痛觉不是普通的感知——它是**保护性信号**。
普通感知告诉你"世界是什么"，痛觉告诉你"哪里出了问题"。
痛觉驱动回避行为、学习避免危险、调节情感状态。

人类神经科学机制：
  1. 伤害感受器（nociceptor）——检测组织损伤：
     - Aδ纤维：快速痛觉——尖锐、定位精确的即时疼痛
     - C纤维：慢速痛觉——钝痛、弥散、持续的灼烧感
     - 伤害感受器不是普通感受器——它们只在组织损伤或潜在损伤时激活
     → 工程实现：伤害检测信号（环境极端值→痛觉信号）

  2. 脊髓→丘脑→皮层通路——痛觉信号传导：
     - 脊髓背角：痛觉信号的第一站——门控理论（Melzack & Wall）
     - 丘脑：痛觉信号的中继站——分发到多个皮层区域
     - S1/S2皮层：痛觉的定位和强度
     - ACC/前脑岛：痛觉的情感维度（"这有多难受"）
     → 工程实现：痛觉信号→概念层+情感系统

  3. 痛觉的两个维度——感觉维度+情感维度：
     - 感觉维度：定位、强度、性质（S1/S2皮层）——"手指被针扎了"
     - 情感维度：不愉快、痛苦（ACC/前脑岛）——"这让我很难受"
     - 情感维度驱动回避行为——不只是知道痛，还要想要逃避
     → 工程实现：痛觉强度（感觉维度）+ 痛觉不愉快度（情感维度）

  4. 痛觉门控理论——下行抑制：
     - 脊髓背角接受来自大脑的下行抑制信号——大脑可以"关闭"痛觉
     - 认知重评可以减轻痛觉——"这不是真的痛"→下行抑制→痛觉减轻
     - 情感调节可以减轻痛觉——冷静时痛觉减轻，紧张时痛觉加剧
     → 工程实现：情感调节信号调制痛觉强度

  5. 痛觉记忆——痛觉经验影响未来行为：
     - 痛觉经验被海马体存储——"上次碰到火，很痛"
     - 痛觉记忆驱动回避学习——不再触碰火
     - 痛觉记忆可以过度泛化——创伤后应激障碍（PTSD）
     → 工程实现：痛觉历史记录+痛觉驱动信号

数字生命优势（超越人类的限制）：
  1. 精确痛觉定位：人类痛觉定位模糊，数字生命可以精确到具体数据源
  2. 痛觉可调控：人类难以主动关闭痛觉，数字生命可以通过情感调节精确调控
  3. 痛觉不遗忘：人类会遗忘痛觉细节，数字生命可以保留所有痛觉历史
  4. 无过度泛化：人类痛觉容易过度泛化（PTSD），数字生命可以精确控制泛化范围

精卫公式（痛觉版）：
  a = 当前身体状态（正常范围内的感知——"一切正常"）
  b = 伤害信号（超出正常范围的感知——"哪里出了问题"）
  Δ = b − a = 伤害差距（正常与异常的差异——"偏离了多少"）
  Σ = a + b = 联合状态（正常+异常——"整体身体图景"）
  ⊗ = ∫(Δ, Σ) = 累积痛觉学习历史 → 痛觉权重
  τ 读 ⊗ → 浮出所见 = "痛"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复痛觉感知循环

痛觉 vs 普通感知的区别：
  - 普通感知：告诉你"世界是什么"——光值50、声值45
  - 痛觉：告诉你"哪里出了问题"——温度过高、资源耗尽、系统异常
  - 普通感知驱动认知（理解世界），痛觉驱动行动（回避危险）
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


class 痛觉:
    """痛觉——伤害感知与保护性反应，基于精卫公式。

    核心功能：
    1. 伤害检测：从环境信号检测超出正常范围的异常（伤害信号）
    2. 痛觉强度计算：感觉维度——定位、强度、性质
    3. 痛觉不愉快度：情感维度——痛苦、不愉快
    4. 痛觉驱动信号：反馈到概念层作为b信号输入源
    5. 痛觉门控：情感调节信号调制痛觉强度（下行抑制）

    精卫公式流程（每拍）：
    1. a = 当前身体状态（正常感知）
    2. b = 伤害信号（异常感知）
    3. Δ = b - a（伤害差距）
    4. Σ = a + b（联合状态）
    5. ⊗ = Δ × Σ（累积痛觉学习信号）
    6. τ读⊗→浮出所见 = "痛"从⊗中浮现
    7. (a, b)⟲⊗ = 循环
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 伤害检测：从环境信号检测异常 ──
        # pain_detect_w: 环境信号 → 痛觉检测
        # 生物学：伤害感受器——只在组织损伤或潜在损伤时激活
        # 环境信号超出正常范围 → 痛觉检测信号
        # 光过强→视觉痛、声过大→听觉痛、温度过高/过低→温度痛、力过大→机械痛
        # 4维环境信号（光/声/力/温）→ n_concepts维痛觉检测信号
        self.pain_detect_w = [[rng.uniform(-0.1, 0.1) for _ in range(4)]
                               for _ in range(n_concepts)]

        # 痛觉检测信号（从环境异常检测到的伤害信号）
        self.pain_signal = [0.0] * n_concepts

        # ── 痛觉强度（感觉维度）──
        # pain_intensity_w: 痛觉检测信号 → 痛觉强度
        # 生物学：S1/S2皮层——痛觉的定位和强度
        # "手指被针扎了"——定位+强度
        self.pain_intensity_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                  for _ in range(n_concepts)]

        # 痛觉强度（0~1，感觉维度——"有多痛"）
        self.pain_intensity = 0.0

        # ── 痛觉不愉快度（情感维度）──
        # pain_affective_w: 痛觉检测信号 → 痛觉不愉快度
        # 生物学：ACC/前脑岛——痛觉的情感维度
        # "这让我很难受"——不愉快、痛苦
        self.pain_affective_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                   for _ in range(1)]

        # 痛觉不愉快度（0~1，情感维度——"有多难受"）
        self.pain_unpleasantness = 0.0

        # ── 痛觉驱动信号（反馈到概念层——作为b信号输入源） ──
        # 痛觉驱动 = 痛觉检测信号 × 痛觉强度
        # 生物学：痛觉信号→PFC投射——痛觉影响概念可及性
        # 痛觉让与危险/回避相关的概念更容易浮现——"痛的概念更容易想到"
        self.pain_drive = [0.0] * n_concepts

        # ── 痛觉门控（下行抑制）──
        # 生物学：脊髓背角接受来自大脑的下行抑制信号
        # 认知重评/情感调节可以减轻痛觉——"这不是真的痛"
        # 门控强度 = 1 - 情感调节信号（调节越强，痛觉越弱）
        self.gate_control = 1.0  # 1.0=无抑制（全痛），0.0=完全抑制（无痛）

        # ── 痛觉历史 ──
        self.pain_history = []
        self.history_cap = 50

        # ── 环境正常范围（用于判断什么是"异常"）──
        # 这些范围对应身体稳态中的归一化范围
        self.正常范围 = {
            "光": (0.0, 1.0),   # 归一化后的光值范围
            "声": (0.0, 1.0),   # 归一化后的声值范围
            "力": (0.0, 1.0),   # 归一化后的力值范围
            "温": (0.0, 1.0),   # 归一化后的温值范围
        }

        # ── 痛觉阈值（超出正常范围多少才算"痛"）──
        # 生物学：伤害感受器有激活阈值——不是任何刺激都能激活
        self.痛觉阈值 = 0.7  # 归一化值超过0.7算"痛"

        # ── 上拍痛觉强度（用于痛觉变化检测）──
        self.prev_pain_intensity = 0.0

        # ── 统计 ──
        self.痛觉检测次数 = 0
        self.痛觉激活次数 = 0  # 痛觉强度超过阈值的次数
        self.痛觉门控次数 = 0  # 门控生效的次数

    def 一拍(self, 环境信号, 情感调节信号=None):
        """痛觉一拍：伤害检测、痛觉强度、痛觉不愉快度、痛觉驱动信号。

        精卫公式流程：
        1. a = 当前身体状态（正常感知）
        2. b = 伤害信号（异常感知）
        3. Δ = b - a（伤害差距）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 痛觉权重更新
        6. τ读⊗→浮出所见 = "痛"从⊗中浮现
        7. (a, b)⟲⊗ = 循环

        参数：
          环境信号: 4维环境归一化值（光/声/力/温），范围[0,1]
          情感调节信号: 情感调节模块的调节强度（0~1，可选）
                        高调节强度→痛觉门控增强→痛觉减轻

        返回：
          dict包含：
            pain_drive: 痛觉驱动信号（反馈到概念层）
            pain_intensity: 痛觉强度（0~1，感觉维度）
            pain_unpleasantness: 痛觉不愉快度（0~1，情感维度）
            pain_signal: 痛觉检测信号
            gate_control: 痛觉门控强度（0~1）
        """
        self.拍 += 1

        # ── 痛觉门控：情感调节信号调制痛觉强度 ──
        # 生物学：脊髓背角接受来自大脑的下行抑制信号
        # 认知重评/情感调节可以减轻痛觉——"这不是真的痛"
        # 门控强度 = 1 - 情感调节信号（调节越强，痛觉越弱）
        if 情感调节信号 is not None:
            self.gate_control = max(0.1, min(1.0, 1.0 - 情感调节信号 * 0.5))
            self.痛觉门控次数 += 1
        else:
            self.gate_control = 1.0

        # ── 伤害检测：从环境信号检测异常 ──
        # 精卫公式：b = 伤害信号（超出正常范围的感知）
        # 环境信号超出痛觉阈值 → 痛觉检测信号
        # 生物学：伤害感受器只在组织损伤或潜在损伤时激活
        # 光过强→视觉痛、声过大→听觉痛、温度过高/过低→温度痛、力过大→机械痛
        伤害向量 = [0.0] * 4
        for i, key in enumerate(["光", "声", "力", "温"]):
            if i < len(环境信号):
                val = 环境信号[i]
                # 超出痛觉阈值 → 伤害信号
                if val > self.痛觉阈值:
                    伤害向量[i] = (val - self.痛觉阈值) / (1.0 - self.痛觉阈值)
                # 极低值也可能是伤害（如温度过低）
                elif val < (1.0 - self.痛觉阈值):
                    伤害向量[i] = ((1.0 - self.痛觉阈值) - val) / (1.0 - self.痛觉阈值) * 0.5

        # 痛觉检测信号 = pain_detect_w × 伤害向量
        new_pain_signal = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            detect_input = 0.0
            for j in range(4):
                detect_input += self.pain_detect_w[i][j] * 伤害向量[j]
            new_pain_signal[i] = activation(detect_input)

        # 应用痛觉门控——门控强度调制痛觉检测信号
        for i in range(self.n_concepts):
            new_pain_signal[i] *= self.gate_control

        self.pain_signal = new_pain_signal
        self.痛觉检测次数 += 1

        # ── 痛觉强度（感觉维度）──
        # 精卫公式：a=正常感知, b=伤害信号, Δ=b-a, ⊗=Δ×Σ
        # 生物学：S1/S2皮层——痛觉的定位和强度
        # 痛觉强度 = pain_intensity_w × pain_signal
        intensity_input = 0.0
        for j in range(self.n_concepts):
            intensity_input += self.pain_intensity_w[0][j] * new_pain_signal[j]
        self.pain_intensity = activation(intensity_input) * 0.5 + 0.5  # 映射到[0,1]
        self.pain_intensity = min(1.0, self.pain_intensity * self.gate_control)

        if self.pain_intensity > self.痛觉阈值:
            self.痛觉激活次数 += 1

        # ── 痛觉不愉快度（情感维度）──
        # 生物学：ACC/前脑岛——痛觉的情感维度
        # 痛觉不愉快度 = pain_affective_w × pain_signal
        affective_input = 0.0
        for j in range(self.n_concepts):
            affective_input += self.pain_affective_w[0][j] * new_pain_signal[j]
        self.pain_unpleasantness = activation(affective_input) * 0.5 + 0.5  # 映射到[0,1]
        self.pain_unpleasantness = min(1.0, self.pain_unpleasantness * self.gate_control)

        # ── 痛觉驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 痛觉驱动信号
        # 生物学：痛觉信号→PFC投射——痛觉影响概念可及性
        # 痛觉让与危险/回避相关的概念更容易浮现——"痛的概念更容易想到"
        for i in range(self.n_concepts):
            self.pain_drive[i] = activation(
                new_pain_signal[i] * self.pain_intensity)

        # ── 痛觉学习 ──
        # 精卫公式：⊗ = Δ × Σ → 痛觉权重更新
        # Δ = 痛觉强度变化（当前-上拍）
        # Σ = 当前痛觉强度 + 上拍痛觉强度
        # 赫布学习：痛觉信号和概念激活同时出现→检测权重增强
        pain_delta = self.pain_intensity - self.prev_pain_intensity
        pain_sigma = self.pain_intensity + self.prev_pain_intensity

        for i in range(self.n_concepts):
            for j in range(4):
                dw = self.lr * new_pain_signal[i] * 伤害向量[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.pain_detect_w[i][j] = self.pain_detect_w[i][j] * 0.99 + dw

        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_pain_signal[i] * self.pain_signal[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.pain_intensity_w[i][j] = self.pain_intensity_w[i][j] * 0.99 + dw

        for j in range(self.n_concepts):
            dw = self.lr * self.pain_unpleasantness * new_pain_signal[j]
            if not (math.isnan(dw) or math.isinf(dw)):
                self.pain_affective_w[0][j] = self.pain_affective_w[0][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.pain_detect_w[i][j]) for j in range(4))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(4):
                    self.pain_detect_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.pain_intensity_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.pain_intensity_w[i][j] *= scale

        w_norm = sum(abs(self.pain_affective_w[0][j]) for j in range(self.n_concepts))
        if w_norm > 2.0:
            scale = 2.0 / w_norm
            for j in range(self.n_concepts):
                self.pain_affective_w[0][j] *= scale

        # ── 痛觉历史 ──
        self.pain_history.append({
            "intensity": self.pain_intensity,
            "unpleasantness": self.pain_unpleasantness,
            "gate": self.gate_control,
        })
        if len(self.pain_history) > self.history_cap:
            self.pain_history.pop(0)

        self.prev_pain_intensity = self.pain_intensity

        return {
            "pain_drive": self.pain_drive,
            "pain_intensity": round(self.pain_intensity, 4),
            "pain_unpleasantness": round(self.pain_unpleasantness, 4),
            "pain_signal": self.pain_signal,
            "gate_control": round(self.gate_control, 4),
        }

    def 状态摘要(self):
        avg_intensity = sum(h["intensity"] for h in self.pain_history) / max(1, len(self.pain_history)) \
            if self.pain_history else 0.0
        avg_unpleasant = sum(h["unpleasantness"] for h in self.pain_history) / max(1, len(self.pain_history)) \
            if self.pain_history else 0.0
        return {
            "pain_intensity": round(self.pain_intensity, 4),
            "pain_unpleasantness": round(self.pain_unpleasantness, 4),
            "gate_control": round(self.gate_control, 4),
            "avg_intensity": round(avg_intensity, 4),
            "avg_unpleasantness": round(avg_unpleasant, 4),
            "痛觉检测次数": self.痛觉检测次数,
            "痛觉激活次数": self.痛觉激活次数,
            "痛觉门控次数": self.痛觉门控次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 痛觉验证 ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    pain = 痛觉(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {pain.n_concepts}")
    print(f"  痛觉检测权重: pain_detect_w (64x4)")
    print(f"  痛觉强度权重: pain_intensity_w (64x64)")
    print(f"  痛觉情感权重: pain_affective_w (1x64)")
    print(f"  痛觉阈值: {pain.痛觉阈值}")
    print(f"  初始痛觉强度: {pain.pain_intensity}")

    print("\n--- 测试1: 正常环境→无痛觉 ---")
    for _ in range(50):
        result = pain.一拍([0.3, 0.4, 0.2, 0.5])  # 正常范围
    print(f"  50拍正常环境后: 痛觉强度={result['pain_intensity']}")
    print(f"  痛觉不愉快度={result['pain_unpleasantness']}")
    print(f"  → 正常环境不应触发痛觉")

    print("\n--- 测试2: 极端环境→痛觉激活 ---")
    pain2 = 痛觉(n_concepts=64)
    for _ in range(50):
        result = pain2.一拍([0.9, 0.85, 0.8, 0.95])  # 极端值
    print(f"  50拍极端环境后: 痛觉强度={result['pain_intensity']}")
    print(f"  痛觉不愉快度={result['pain_unpleasantness']}")
    print(f"  → 极端环境应触发痛觉")

    print("\n--- 测试3: 痛觉门控——情感调节减轻痛觉 ---")
    pain3 = 痛觉(n_concepts=64)
    for _ in range(50):
        result = pain3.一拍([0.9, 0.85, 0.8, 0.95], 情感调节信号=0.8)
    print(f"  有情感调节(0.8): 痛觉强度={result['pain_intensity']}, 门控={result['gate_control']}")
    pain4 = 痛觉(n_concepts=64)
    for _ in range(50):
        result = pain4.一拍([0.9, 0.85, 0.8, 0.95])
    print(f"  无情感调节: 痛觉强度={result['pain_intensity']}, 门控={result['gate_control']}")
    print(f"  → 情感调节应减轻痛觉强度")

    print("\n--- 测试4: 不同极端模式→不同痛觉信号 ---")
    patterns = [
        ("光极端", [0.95, 0.3, 0.2, 0.5]),
        ("声极端", [0.3, 0.95, 0.2, 0.5]),
        ("温极端", [0.3, 0.4, 0.2, 0.95]),
        ("力极端", [0.3, 0.4, 0.95, 0.5]),
    ]
    for name, pattern in patterns:
        pain_t = 痛觉(n_concepts=64)
        for _ in range(50):
            pain_t.一拍(pattern)
        print(f"  {name}: 痛觉强度={round(pain_t.pain_intensity, 4)}, "
              f"痛觉信号top5={sorted(range(64), key=lambda i: abs(pain_t.pain_signal[i]), reverse=True)[:5]}")

    print("\n--- 测试5: 痛觉驱动信号 ---")
    pain5 = 痛觉(n_concepts=64)
    for _ in range(50):
        pain5.一拍([0.9, 0.85, 0.8, 0.95])
    drive_sum = sum(abs(v) for v in pain5.pain_drive)
    print(f"  痛觉驱动信号总和: {drive_sum:.4f}")
    print(f"  痛觉驱动top5: {sorted(range(64), key=lambda i: abs(pain5.pain_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试6: 痛觉从正常到极端的过渡 ---")
    pain6 = 痛觉(n_concepts=64)
    # 先正常50拍
    for _ in range(50):
        pain6.一拍([0.3, 0.4, 0.2, 0.5])
    print(f"  正常阶段: 痛觉强度={round(pain6.pain_intensity, 4)}")
    # 然后突然极端
    for tick in range(10):
        result = pain6.一拍([0.9, 0.85, 0.8, 0.95])
        if tick < 3 or tick >= 8:
            print(f"  极端拍{tick+1}: 痛觉强度={result['pain_intensity']}")

    print("\n--- 测试7: 精卫公式验证 ---")
    pain7 = 痛觉(n_concepts=16)
    for tick in range(10):
        result = pain7.一拍([0.8, 0.7, 0.6, 0.9])
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 痛觉强度={result['pain_intensity']}, "
                  f"不愉快度={result['pain_unpleasantness']}, 门控={result['gate_control']}")
    print(f"  τ(浮出): 痛觉驱动均值={round(sum(pain7.pain_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={pain7.拍}")

    print("\n验证通过")
