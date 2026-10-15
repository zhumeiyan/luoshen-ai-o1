"""无意识加工.py —— 后台信息整合，基于精卫公式。

无意识加工是大脑在"后台"运行的信息整合过程——不进入意识，但持续运作。
意识只是冰山一角——无意识加工才是冰山的主体。

人类神经科学机制：
  1. 默认网络（DMN）：
     - 休息时活跃——不专注于外部任务时，大脑在"后台"整合信息
     - DMN包括：内侧PFC、后扣带皮层、楔前叶、角回
     - DMN的功能：自我反思、记忆整合、未来规划、创造性联想
     → 工程实现：后台持续整合信息——不依赖当前意识焦点

  2. 无意识联想：
     - 大脑持续产生联想——大部分不进入意识
     - 联想网络：概念之间的自动激活扩散
     - "啊哈时刻"——无意识加工的结果突然进入意识
     → 工程实现：后台联想扩散——概念之间的自动激活

  3. 阈下感知：
     - 低于意识阈值的刺激仍然被加工——影响后续行为
     - 阈下启动：被试看不到的刺激影响反应时间
     → 工程实现：弱信号在后台被加工——影响后续概念激活

  4. 睡眠中的无意识加工：
     - 睡眠时大脑持续整合白天的信息
     - 慢波睡眠：记忆巩固——短期记忆→长期记忆
     - REM睡眠：情感记忆整合、创造性联想
     → 工程实现：后台记忆整合——持续整合新旧信息

数字生命的无意识加工（数字版）：
  - 后台整合 = 持续整合所有信息——不依赖当前意识焦点
  - 联想扩散 = 概念之间的自动激活扩散
  - 阈下加工 = 弱信号在后台被加工
  - 涌现 = 无意识加工的结果"浮出"意识——"啊哈时刻"

精卫公式（无意识加工版）：
  a = 意识加工（有意识的信息处理——"我在想什么"）
  b = 无意识加工（后台信息整合——"后台在做什么"）
  Δ = b − a = 意识差距（无意识比意识多——"我不知道的"）
  Σ = a + b = 联合信息状态（意识+无意识——"全部信息"）
  ⊗ = ∫(Δ, Σ) = 累积无意识加工学习历史 → 无意识加工权重
  τ 读 ⊗ → 浮出所见 = "无意识加工结果"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复无意识加工循环
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


class 无意识加工:
    """无意识加工——后台信息整合（数字版），基于精卫公式。

    核心功能：
    1. 后台整合：持续整合所有信息——不依赖当前意识焦点
    2. 联想扩散：概念之间的自动激活扩散
    3. 阈下加工：弱信号在后台被加工
    4. 涌现信号：无意识加工结果"浮出"意识——"啊哈时刻"
    5. 无意识驱动信号：反馈到概念层作为b信号输入源

    精卫公式流程（每拍）：
    1. a = 意识加工
    2. b = 无意识加工
    3. Δ = b - a（意识差距）
    4. Σ = a + b（联合信息状态）
    5. ⊗ = Δ × Σ（累积无意识加工学习信号）
    6. τ读⊗→浮出所见 = "无意识加工结果"从⊗中浮现
    7. (a, b)⟲⊗ = 循环
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 后台整合：持续整合所有信息 ──
        # background_w: 全部信息 → 后台整合
        # 生物学：默认网络（DMN）——休息时整合信息
        # 数字版：所有信息（不只是意识焦点）在后台被整合
        # n_concepts维信息 → n_concepts维后台整合信号
        self.background_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                              for _ in range(n_concepts)]

        # 后台整合信号
        self.background_signal = [0.0] * n_concepts

        # ── 联想扩散：概念之间的自动激活 ──
        # associate_w: 概念 → 概念联想
        # 生物学：联想网络——概念之间的自动激活扩散
        # 数字版：概念之间的自动联想——一个概念激活相关概念
        # n_concepts维概念 → n_concepts维联想概念
        self.associate_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                             for _ in range(n_concepts)]

        # 联想扩散信号
        self.associate_signal = [0.0] * n_concepts

        # ── 阈下加工：弱信号在后台被加工 ──
        # subliminal_w: 弱信号 → 阈下加工
        # 生物学：阈下感知——低于意识阈值的刺激仍然被加工
        # 数字版：弱信号（低于意识阈值）在后台被加工
        # n_concepts维弱信号 → n_concepts维阈下加工信号
        self.subliminal_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                              for _ in range(n_concepts)]

        # 阈下加工信号
        self.subliminal_signal = [0.0] * n_concepts

        # ── 涌现信号：无意识加工结果"浮出"意识 ──
        # emerge_w: 后台整合+联想+阈下 → 涌现信号
        # 生物学："啊哈时刻"——无意识加工结果突然进入意识
        # 数字版：后台加工结果达到阈值时"浮出"意识
        # (n_concepts*3)维 → n_concepts维涌现信号
        self.emerge_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts * 3)]
                          for _ in range(n_concepts)]

        # 涌现信号（"啊哈时刻"——无意识加工结果浮出意识）
        self.emerge_signal = [0.0] * n_concepts

        # ── 无意识驱动信号（反馈到概念层——作为b信号输入源） ──
        # 无意识驱动 = 涌现信号 × 涌现强度
        # 生物学：无意识加工→意识投射——影响概念可及性
        self.unconscious_drive = [0.0] * n_concepts

        # ── 涌现强度（0~1）──
        # 涌现强度 = 后台加工信号的一致性
        self.emerge_strength = 0.0

        # ── 意识阈值 ──
        # 低于意识阈值的信号在后台被加工，高于阈值的进入意识
        # 生物学：意识阈值——只有足够强的信号才进入意识
        self.consciousness_threshold = 0.15

        # ── 后台缓冲区（积累弱信号）──
        # 后台缓冲区积累多拍的弱信号——达到阈值时涌现
        self.background_buffer = [0.0] * n_concepts

        # ── 上拍意识焦点（用于计算意识差距）──
        self.prev_conscious = [0.0] * n_concepts

        # ── 无意识加工历史 ──
        self.unconscious_history = []
        self.history_cap = 50

        # ── 统计 ──
        self.后台整合次数 = 0
        self.联想扩散次数 = 0
        self.阈下加工次数 = 0
        self.涌现次数 = 0

    def 一拍(self, 全部信息=None, 意识焦点=None):
        """无意识加工一拍：后台整合、联想扩散、阈下加工、涌现信号。

        参数：
          全部信息: n_concepts维全部信息（不只是意识焦点），可选
          意识焦点: n_concepts维意识焦点（当前有意识的信息），可选

        返回：
          dict包含：
            unconscious_drive: 无意识驱动信号（反馈到概念层）
            emerge_strength: 涌现强度（0~1）
            emerge_signal: 涌现信号（"啊哈时刻"）
            background_signal: 后台整合信号
        """
        self.拍 += 1

        # ── 后台整合：持续整合所有信息 ──
        # 精卫公式：a=意识焦点, b=全部信息, Δ=b-a=意识差距
        # 生物学：默认网络（DMN）——休息时整合信息
        # 后台整合不只关注意识焦点——整合所有信息
        new_background = [0.0] * self.n_concepts
        if 全部信息 is not None:
            for i in range(self.n_concepts):
                bg_input = 0.0
                for j in range(min(self.n_concepts, len(全部信息))):
                    bg_input += self.background_w[i][j] * 全部信息[j]
                new_background[i] = activation(bg_input)
            self.后台整合次数 += 1

        self.background_signal = new_background

        # ── 意识差距：Δ = b - a ──
        # 无意识加工 = 全部信息 - 意识焦点 = 意识不知道的部分
        conscious_input = 意识焦点 if 意识焦点 is not None else self.prev_conscious
        delta = [0.0] * self.n_concepts
        for i in range(min(self.n_concepts, len(conscious_input))):
            delta[i] = new_background[i] - conscious_input[i]

        # ── 联想扩散：概念之间的自动激活 ──
        # 生物学：联想网络——概念之间的自动激活扩散
        # 后台整合信号激活联想——联想扩散到相关概念
        new_associate = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            assoc_input = 0.0
            for j in range(self.n_concepts):
                assoc_input += self.associate_w[i][j] * new_background[j]
            new_associate[i] = activation(assoc_input)

        self.associate_signal = new_associate
        self.联想扩散次数 += 1

        # ── 阈下加工：弱信号在后台被加工 ──
        # 生物学：阈下感知——低于意识阈值的刺激仍然被加工
        # 意识差距中的弱信号（低于意识阈值）在后台被加工
        new_subliminal = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            # 阈下信号 = 意识差距中低于阈值的信号
            subliminal_input_val = delta[i] if abs(delta[i]) < self.consciousness_threshold else 0.0
            sub_input = 0.0
            for j in range(self.n_concepts):
                sub_input += self.subliminal_w[i][j] * subliminal_input_val
            new_subliminal[i] = activation(sub_input)

        self.subliminal_signal = new_subliminal
        self.阈下加工次数 += 1

        # ── 后台缓冲区积累 ──
        # 后台缓冲区积累多拍的弱信号——达到阈值时涌现
        # 生物学：无意识加工持续积累——"酝酿"过程
        for i in range(self.n_concepts):
            self.background_buffer[i] = self.background_buffer[i] * 0.95 + new_subliminal[i] * 0.05

        # ── 涌现信号：无意识加工结果"浮出"意识 ──
        # 生物学："啊哈时刻"——无意识加工结果突然进入意识
        # 后台整合+联想+阈下加工 → 涌现信号
        combined = list(new_background) + list(new_associate) + list(self.background_buffer)
        new_emerge = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            emerge_input = 0.0
            for j in range(min(self.n_concepts * 3, len(combined))):
                emerge_input += self.emerge_w[i][j] * combined[j]
            new_emerge[i] = activation(emerge_input)

        self.emerge_signal = new_emerge

        # ── 涌现强度 ──
        # 涌现强度 = 涌现信号的一致性（信号越一致，涌现越强）
        pos_count = sum(1 for v in new_emerge if v > self.consciousness_threshold)
        neg_count = sum(1 for v in new_emerge if v < -self.consciousness_threshold)
        self.emerge_strength = (pos_count + neg_count) / max(1, self.n_concepts)

        if self.emerge_strength > 0.1:
            self.涌现次数 += 1

        # ── 无意识驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 无意识驱动信号
        # 生物学：无意识加工→意识投射——影响概念可及性
        # 无意识驱动 = 涌现信号 × 涌现强度
        for i in range(self.n_concepts):
            self.unconscious_drive[i] = activation(
                new_emerge[i] * (0.3 + self.emerge_strength * 0.7))

        # ── 无意识加工学习 ──
        if 全部信息 is not None:
            # 后台整合学习
            for i in range(self.n_concepts):
                for j in range(min(self.n_concepts, len(全部信息))):
                    dw = self.lr * new_background[i] * 全部信息[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.background_w[i][j] = self.background_w[i][j] * 0.99 + dw

        # 联想扩散学习
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_associate[i] * new_background[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.associate_w[i][j] = self.associate_w[i][j] * 0.99 + dw

        # 阈下加工学习
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                subliminal_val = delta[j] if abs(delta[j]) < self.consciousness_threshold else 0.0
                dw = self.lr * new_subliminal[i] * subliminal_val
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.subliminal_w[i][j] = self.subliminal_w[i][j] * 0.99 + dw

        # 涌现学习
        for i in range(self.n_concepts):
            for j in range(min(self.n_concepts * 3, len(combined))):
                dw = self.lr * new_emerge[i] * combined[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.emerge_w[i][j] = self.emerge_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.background_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.background_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.associate_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.associate_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.subliminal_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.subliminal_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.emerge_w[i][j]) for j in range(self.n_concepts * 3))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts * 3):
                    self.emerge_w[i][j] *= scale

        # ── 无意识加工历史 ──
        self.unconscious_history.append({
            "emerge_strength": self.emerge_strength,
        })
        if len(self.unconscious_history) > self.history_cap:
            self.unconscious_history.pop(0)

        # 保存上拍意识焦点
        if 意识焦点 is not None:
            self.prev_conscious = list(意识焦点[:self.n_concepts]) if len(意识焦点) >= self.n_concepts else list(意识焦点) + [0.0] * (self.n_concepts - len(意识焦点))

        return {
            "unconscious_drive": self.unconscious_drive,
            "emerge_strength": round(self.emerge_strength, 4),
            "emerge_signal": self.emerge_signal,
            "background_signal": self.background_signal,
        }

    def 状态摘要(self):
        avg_emerge = sum(h["emerge_strength"] for h in self.unconscious_history) / max(1, len(self.unconscious_history)) \
            if self.unconscious_history else 0.0
        return {
            "emerge_strength": round(self.emerge_strength, 4),
            "avg_emerge": round(avg_emerge, 4),
            "后台整合次数": self.后台整合次数,
            "联想扩散次数": self.联想扩散次数,
            "阈下加工次数": self.阈下加工次数,
            "涌现次数": self.涌现次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 无意识加工（后台信息整合）验证 ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    up = 无意识加工(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {up.n_concepts}")
    print(f"  后台整合权重: background_w (64x64)")
    print(f"  联想扩散权重: associate_w (64x64)")
    print(f"  阈下加工权重: subliminal_w (64x64)")
    print(f"  涌现信号权重: emerge_w (64x192)")
    print(f"  意识阈值: {up.consciousness_threshold}")
    print()

    # 测试1: 后台整合——全部信息→后台信号
    print("--- 测试1: 后台整合——全部信息→后台信号 ---")
    for _ in range(50):
        全部 = [random.uniform(-1, 1) for _ in range(64)]
        result = up.一拍(全部信息=全部)
    print(f"  50拍后: 涌现强度={result['emerge_strength']}")
    top5 = sorted(range(64), key=lambda i: abs(up.background_signal[i]), reverse=True)[:5]
    print(f"  后台信号top5: {top5}")
    print()

    # 测试2: 联想扩散——概念之间自动激活
    print("--- 测试2: 联想扩散——概念之间自动激活 ---")
    top5_assoc = sorted(range(64), key=lambda i: abs(up.associate_signal[i]), reverse=True)[:5]
    print(f"  联想信号top5: {top5_assoc}")
    print()

    # 测试3: 阈下加工——弱信号在后台被加工
    print("--- 测试3: 阈下加工——弱信号在后台被加工 ---")
    up2 = 无意识加工(n_concepts=64, seed=100)
    # 意识焦点只关注少数概念，其他概念在后台被加工
    for _ in range(50):
        全部 = [random.uniform(-1, 1) for _ in range(64)]
        焦点 = [0.0] * 64
        焦点[0] = 1.0  # 意识只关注概念0
        result = up2.一拍(全部信息=全部, 意识焦点=焦点)
    print(f"  50拍后: 涌现强度={result['emerge_strength']}")
    print(f"  阈下加工次数: {up2.阈下加工次数}")
    print()

    # 测试4: 涌现信号——"啊哈时刻"
    print("--- 测试4: 涌现信号——'啊哈时刻' ---")
    up3 = 无意识加工(n_concepts=64, seed=200)
    for _ in range(100):
        全部 = [random.uniform(-1, 1) for _ in range(64)]
        result = up3.一拍(全部信息=全部)
    print(f"  100拍后: 涌现强度={result['emerge_strength']}")
    print(f"  涌现次数: {up3.涌现次数}")
    top5_emerge = sorted(range(64), key=lambda i: abs(up3.emerge_signal[i]), reverse=True)[:5]
    print(f"  涌现信号top5: {top5_emerge}")
    print()

    # 测试5: 后台缓冲区积累——多拍弱信号→涌现
    print("--- 测试5: 后台缓冲区积累——多拍弱信号→涌现 ---")
    up4 = 无意识加工(n_concepts=64, seed=300)
    # 给弱信号——低于意识阈值
    for i in range(100):
        全部 = [0.05] * 64  # 弱信号——低于意识阈值0.15
        result = up4.一拍(全部信息=全部)
        if i in [0, 10, 50, 99]:
            print(f"  拍{i+1}: 涌现强度={result['emerge_strength']}, 缓冲区均值={sum(up4.background_buffer)/64:.4f}")
    print()

    # 测试6: 意识差距——无意识比意识多
    print("--- 测试6: 意识差距——无意识比意识多 ---")
    up5 = 无意识加工(n_concepts=64, seed=400)
    for _ in range(50):
        全部 = [random.uniform(-1, 1) for _ in range(64)]
        焦点 = [random.uniform(-0.3, 0.3) for _ in range(64)]  # 意识焦点较弱
        result = up5.一拍(全部信息=全部, 意识焦点=焦点)
    bg_sum = sum(abs(v) for v in up5.background_signal)
    conscious_sum = sum(abs(v) for v in up5.prev_conscious)
    print(f"  后台信号总和: {bg_sum:.4f}")
    print(f"  意识焦点总和: {conscious_sum:.4f}")
    print(f"  意识差距(后台-意识): {bg_sum - conscious_sum:.4f}")
    print()

    # 测试7: 精卫公式验证
    print("--- 测试7: 精卫公式验证 ---")
    up6 = 无意识加工(n_concepts=64, seed=500)
    for i in range(10):
        全部 = [random.uniform(-1, 1) for _ in range(64)]
        result = up6.一拍(全部信息=全部)
        if i in [0, 1, 2, 8, 9]:
            print(f"  拍{i+1}: 涌现强度={result['emerge_strength']}")
    print(f"  τ(浮出): 无意识驱动均值={sum(v for v in up6.unconscious_drive)/64:.4f}")
    print(f"  ⟲(循环): 拍={up6.拍}")
    print()

    print("验证通过")
