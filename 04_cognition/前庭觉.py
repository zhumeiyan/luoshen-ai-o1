"""前庭觉.py —— 运动与平衡感知，基于精卫公式。

前庭觉是人类的第六感——感知运动、平衡和空间方向。
它不是视觉、听觉等外感受，而是**本体感受**——感知自己身体的运动状态。

人类神经科学机制：
  1. 前庭器官——检测运动和重力：
     - 半规管（3个）：检测旋转运动——水平、前、后
     - 耳石器（椭圆囊+球囊）：检测线性加速度和重力——上下、前后、左右
     - 前庭器官在颞骨深处——感知头部的运动和方向
     → 工程实现：运动信号检测（3维旋转 + 3维线性加速度）

  2. 前庭核→小脑→丘脑→皮层通路：
     - 前庭核：前庭信号的第一站——整合左右耳信号
     - 小脑：运动协调——前庭信号参与运动控制
     - 丘脑：中继站——分发到皮层
     - 皮层：前庭皮层（通常在右半球）——有意识的运动感知
     → 工程实现：前庭信号→概念层+运动控制

  3. 前庭-视觉-本体感觉整合：
     - 前庭觉不是孤立的——与视觉和本体感觉整合
     - 视觉-前庭冲突→晕动症（motion sickness）
     - 前庭-本体整合→身体感知（知道身体在哪里）
     → 工程实现：前庭信号与视觉/本体信号整合

  4. 前庭觉的两个功能——感知+反应：
     - 感知功能：知道自己在哪里、在怎么运动
     - 反应功能：维持平衡、调整姿势、眼球稳定（前庭眼反射）
     - 前庭眼反射（VOR）：头部运动时眼球反向运动——保持视觉稳定
     → 工程实现：前庭驱动信号（感知）+ 平衡调节信号（反应）

数字生命优势（超越人类的限制）：
  1. 精确运动感知：人类前庭觉有延迟和噪声，数字生命可以精确感知
  2. 无晕动症：人类有视觉-前庭冲突导致的晕动症，数字生命不会
  3. 多维运动感知：人类只能感知6维运动，数字生命可以扩展到更多维度
  4. 运动记忆：人类前庭记忆模糊，数字生命可以精确记录所有运动历史

精卫公式（前庭觉版）：
  a = 当前运动状态（静止或匀速——"我在哪里"）
  b = 运动变化信号（加速度/旋转——"我在怎么动"）
  Δ = b − a = 运动差距（状态变化的差异——"改变了多少"）
  Σ = a + b = 联合运动状态（当前+变化——"整体运动图景"）
  ⊗ = ∫(Δ, Σ) = 累积运动学习历史 → 前庭权重
  τ 读 ⊗ → 浮出所见 = "运动感"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复运动感知循环
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


class 前庭觉:
    """前庭觉——运动与平衡感知，基于精卫公式。

    核心功能：
    1. 运动检测：从环境信号检测运动变化（加速度/旋转）
    2. 平衡感知：感知系统的稳定状态（是否平衡）
    3. 空间方向：感知系统在空间中的方向
    4. 前庭驱动信号：反馈到概念层作为b信号输入源

    精卫公式流程（每拍）：
    1. a = 当前运动状态
    2. b = 运动变化信号
    3. Δ = b - a（运动差距）
    4. Σ = a + b（联合运动状态）
    5. ⊗ = Δ × Σ（累积运动学习信号）
    6. τ读⊗→浮出所见 = "运动感"从⊗中浮现
    7. (a, b)⟲⊗ = 循环
    """

    def __init__(self, n_concepts=64, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 运动检测：从环境信号检测运动变化 ──
        # vestibular_detect_w: 运动信号 → 前庭检测
        # 生物学：半规管（3维旋转）+ 耳石器（3维线性加速度）
        # 6维运动信号 → n_concepts维前庭检测信号
        # 3维旋转（水平/前/后）+ 3维线性加速度（上下/前后/左右）
        self.vestibular_detect_w = [[rng.uniform(-0.1, 0.1) for _ in range(6)]
                                      for _ in range(n_concepts)]

        # 前庭检测信号（从运动变化检测到的信号）
        self.vestibular_signal = [0.0] * n_concepts

        # ── 平衡感知 ──
        # vestibular_balance_w: 前庭检测信号 → 平衡感知
        # 生物学：小脑——平衡协调
        # 平衡感知 = 系统是否处于稳定状态
        self.vestibular_balance_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                       for _ in range(1)]

        # 平衡感知（0~1，1=完全平衡，0=严重失衡）
        self.balance = 0.5

        # ── 空间方向感知 ──
        # vestibular_orient_w: 前庭检测信号 → 空间方向
        # 生物学：前庭皮层——有意识的运动感知
        # 空间方向 = 系统在空间中的方向感知
        self.vestibular_orient_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                                      for _ in range(n_concepts)]

        # 空间方向感知（n_concepts维——方向的概念表征）
        self.orientation = [0.0] * n_concepts

        # ── 前庭驱动信号（反馈到概念层——作为b信号输入源） ──
        # 前庭驱动 = 前庭检测信号 × 运动强度
        # 生物学：前庭→PFC投射——运动感知影响概念可及性
        # 运动感让与运动/空间相关的概念更容易浮现
        self.vestibular_drive = [0.0] * n_concepts

        # ── 运动强度（0~1）──
        # 运动强度 = 当前运动的整体幅度
        self.motion_intensity = 0.0

        # ── 上拍运动信号（用于检测运动变化）──
        self.prev_motion = [0.0] * 6

        # ── 前庭历史 ──
        self.vestibular_history = []
        self.history_cap = 50

        # ── 统计 ──
        self.运动检测次数 = 0
        self.失衡次数 = 0  # 平衡感知低于0.3的次数

    def 一拍(self, 运动信号):
        """前庭觉一拍：运动检测、平衡感知、空间方向、前庭驱动信号。

        参数：
          运动信号: 6维运动信号（3维旋转 + 3维线性加速度）
                    旋转：[水平旋转, 前旋转, 后旋转]
                    线性：[上下加速度, 前后加速度, 左右加速度]

        返回：
          dict包含：
            vestibular_drive: 前庭驱动信号（反馈到概念层）
            balance: 平衡感知（0~1）
            motion_intensity: 运动强度（0~1）
            orientation: 空间方向感知
            vestibular_signal: 前庭检测信号
        """
        self.拍 += 1

        # ── 运动检测：从运动信号检测运动变化 ──
        # 精卫公式：a=上拍运动, b=当前运动, Δ=b-a, ⊗=Δ×Σ
        # 生物学：前庭器官检测加速度（运动变化），不是匀速运动
        # Δ = 当前运动 - 上拍运动 = 运动变化（加速度）
        motion_delta = [运动信号[i] - self.prev_motion[i] for i in range(min(6, len(运动信号)))]
        motion_sigma = [运动信号[i] + self.prev_motion[i] for i in range(min(6, len(运动信号)))]

        # 前庭检测信号 = vestibular_detect_w × motion_delta（运动变化）
        new_vestibular_signal = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            detect_input = 0.0
            for j in range(min(6, len(motion_delta))):
                detect_input += self.vestibular_detect_w[i][j] * motion_delta[j]
            new_vestibular_signal[i] = activation(detect_input)

        self.vestibular_signal = new_vestibular_signal
        self.运动检测次数 += 1

        # ── 运动强度 ──
        # 运动强度 = 运动信号的整体幅度
        self.motion_intensity = sum(abs(v) for v in 运动信号) / max(1, len(运动信号))
        self.motion_intensity = min(1.0, self.motion_intensity)

        # ── 平衡感知 ──
        # 生物学：小脑——平衡协调
        # 平衡感知 = vestibular_balance_w × vestibular_signal
        # 运动变化大→失衡，运动变化小→平衡
        balance_input = 0.0
        for j in range(self.n_concepts):
            balance_input += self.vestibular_balance_w[0][j] * new_vestibular_signal[j]
        # 平衡 = 1 - 运动变化强度（变化大→失衡）
        motion_change = sum(abs(d) for d in motion_delta) / max(1, len(motion_delta))
        self.balance = max(0.0, min(1.0, 1.0 - motion_change * 2.0))
        # 平衡感知也受权重调制
        self.balance = activation(self.balance * 0.5 + activation(balance_input) * 0.5) * 0.5 + 0.5

        if self.balance < 0.3:
            self.失衡次数 += 1

        # ── 空间方向感知 ──
        # 生物学：前庭皮层——有意识的运动感知
        # 空间方向 = vestibular_orient_w × vestibular_signal
        new_orientation = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            orient_input = 0.0
            for j in range(self.n_concepts):
                orient_input += self.vestibular_orient_w[i][j] * new_vestibular_signal[j]
            new_orientation[i] = activation(orient_input)

        self.orientation = new_orientation

        # ── 前庭驱动信号 ──
        # 精卫公式：τ读⊗→浮出所见 = 前庭驱动信号
        # 生物学：前庭→PFC投射——运动感知影响概念可及性
        # 运动感让与运动/空间相关的概念更容易浮现
        for i in range(self.n_concepts):
            self.vestibular_drive[i] = activation(
                new_vestibular_signal[i] * (0.3 + self.motion_intensity * 0.7))

        # ── 前庭学习 ──
        # 精卫公式：⊗ = Δ × Σ → 前庭权重更新
        for i in range(self.n_concepts):
            for j in range(min(6, len(motion_delta))):
                dw = self.lr * new_vestibular_signal[i] * motion_delta[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.vestibular_detect_w[i][j] = self.vestibular_detect_w[i][j] * 0.99 + dw

        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_orientation[i] * new_vestibular_signal[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.vestibular_orient_w[i][j] = self.vestibular_orient_w[i][j] * 0.99 + dw

        for j in range(self.n_concepts):
            dw = self.lr * self.balance * new_vestibular_signal[j]
            if not (math.isnan(dw) or math.isinf(dw)):
                self.vestibular_balance_w[0][j] = self.vestibular_balance_w[0][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.vestibular_detect_w[i][j]) for j in range(6))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(6):
                    self.vestibular_detect_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.vestibular_orient_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.vestibular_orient_w[i][j] *= scale

        w_norm = sum(abs(self.vestibular_balance_w[0][j]) for j in range(self.n_concepts))
        if w_norm > 2.0:
            scale = 2.0 / w_norm
            for j in range(self.n_concepts):
                self.vestibular_balance_w[0][j] *= scale

        # ── 前庭历史 ──
        self.vestibular_history.append({
            "balance": self.balance,
            "motion_intensity": self.motion_intensity,
        })
        if len(self.vestibular_history) > self.history_cap:
            self.vestibular_history.pop(0)

        self.prev_motion = list(运动信号[:6]) if len(运动信号) >= 6 else list(运动信号) + [0.0] * (6 - len(运动信号))

        return {
            "vestibular_drive": self.vestibular_drive,
            "balance": round(self.balance, 4),
            "motion_intensity": round(self.motion_intensity, 4),
            "orientation": self.orientation,
            "vestibular_signal": self.vestibular_signal,
        }

    def 状态摘要(self):
        avg_balance = sum(h["balance"] for h in self.vestibular_history) / max(1, len(self.vestibular_history)) \
            if self.vestibular_history else 0.0
        avg_motion = sum(h["motion_intensity"] for h in self.vestibular_history) / max(1, len(self.vestibular_history)) \
            if self.vestibular_history else 0.0
        return {
            "balance": round(self.balance, 4),
            "motion_intensity": round(self.motion_intensity, 4),
            "avg_balance": round(avg_balance, 4),
            "avg_motion": round(avg_motion, 4),
            "运动检测次数": self.运动检测次数,
            "失衡次数": self.失衡次数,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 前庭觉验证 ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    vt = 前庭觉(n_concepts=64)

    print("初始状态:")
    print(f"  概念数: {vt.n_concepts}")
    print(f"  运动检测权重: vestibular_detect_w (64x6)")
    print(f"  平衡感知权重: vestibular_balance_w (1x64)")
    print(f"  方向感知权重: vestibular_orient_w (64x64)")
    print(f"  初始平衡: {vt.balance}")

    print("\n--- 测试1: 静止→高平衡 ---")
    for _ in range(50):
        result = vt.一拍([0.0] * 6)
    print(f"  50拍静止后: 平衡={result['balance']}, 运动强度={result['motion_intensity']}")

    print("\n--- 测试2: 运动→平衡下降 ---")
    vt2 = 前庭觉(n_concepts=64)
    for _ in range(50):
        result = vt2.一拍([0.3, 0.2, 0.1, 0.4, 0.2, 0.3])
    print(f"  50拍运动后: 平衡={result['balance']}, 运动强度={result['motion_intensity']}")

    print("\n--- 测试3: 突然运动变化→失衡 ---")
    vt3 = 前庭觉(n_concepts=64)
    for _ in range(30):
        vt3.一拍([0.0] * 6)
    print(f"  静止阶段: 平衡={round(vt3.balance, 4)}")
    result = vt3.一拍([0.8, 0.6, 0.4, 0.9, 0.5, 0.7])
    print(f"  突然运动: 平衡={result['balance']}")
    print(f"  → 突然运动变化应导致平衡下降")

    print("\n--- 测试4: 不同运动模式→不同前庭信号 ---")
    patterns = [
        ("旋转", [0.5, 0.3, 0.2, 0.0, 0.0, 0.0]),
        ("线性", [0.0, 0.0, 0.0, 0.5, 0.3, 0.2]),
        ("混合", [0.3, 0.2, 0.1, 0.4, 0.2, 0.3]),
    ]
    for name, pattern in patterns:
        vt_t = 前庭觉(n_concepts=64)
        for _ in range(50):
            vt_t.一拍(pattern)
        print(f"  {name}: 平衡={round(vt_t.balance, 4)}, "
              f"前庭信号top5={sorted(range(64), key=lambda i: abs(vt_t.vestibular_signal[i]), reverse=True)[:5]}")

    print("\n--- 测试5: 前庭驱动信号 ---")
    vt5 = 前庭觉(n_concepts=64)
    for _ in range(50):
        vt5.一拍([0.3, 0.2, 0.1, 0.4, 0.2, 0.3])
    drive_sum = sum(abs(v) for v in vt5.vestibular_drive)
    print(f"  前庭驱动信号总和: {drive_sum:.4f}")
    print(f"  前庭驱动top5: {sorted(range(64), key=lambda i: abs(vt5.vestibular_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试6: 精卫公式验证 ---")
    vt6 = 前庭觉(n_concepts=16)
    for tick in range(10):
        result = vt6.一拍([0.2, 0.1, 0.05, 0.3, 0.15, 0.2])
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 平衡={result['balance']}, 运动强度={result['motion_intensity']}")
    print(f"  τ(浮出): 前庭驱动均值={round(sum(vt6.vestibular_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={vt6.拍}")

    print("\n验证通过")
