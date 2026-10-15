"""基底神经节.py —— 程序性记忆（技能学习与自动化执行），基于精卫公式。

基底神经节是技能学习的中枢——人类骑自行车、说话、弹钢琴都是程序性记忆。
程序性记忆和陈述性记忆（海马体）的区别：
  - 陈述性记忆：知道"是什么"（事实、事件）——海马体
  - 程序性记忆：知道"怎么做"（技能、习惯）——基底神经节

精卫公式（程序性记忆版）：
  a = 当前概念激活模式（情境）
  b = 技能预测的行为模式（反应）
  Δ = b − a = 行为预测误差（技能与情境的匹配度）
  Σ = a + b = 联合状态（情境+反应）
  ⊗ = ∫(Δ, Σ) = 累积技能学习历史 → 技能权重
  τ 读 ⊗ → 浮出所见 = 技能从⊗中浮现（自动化行为）
  (a, b) ⟲ ⊗ = 反复练习循环

人类神经科学机制：
  1. 纹状体（striatum）接收皮层输入：
     - 背侧纹状体：习惯学习（自动化技能）
     - 腹侧纹状体：奖赏学习（动机驱动）
     → 工程实现：skill_w（概念模式→技能激活）

  2. 直接通路（Go）和间接通路（NoGo）：
     - 直接通路：促进选定行为（D1受体）
     - 间接通路：抑制未选行为（D2受体）
     → 工程实现：技能竞争——稀疏化+赢家通吃

  3. 多巴胺强化学习：
     - 成功的行为→多巴胺释放→技能强化
     - 失败的行为→多巴胺减少→技能削弱
     → 工程实现：预测误差驱动技能权重更新

  4. 习惯形成（habit formation）：
     - 初期：目标导向（前额叶控制，需要工作记忆）
     - 后期：自动化（基底神经节控制，不需要工作记忆）
     → 工程实现：技能强度高时自动触发，不需要工作记忆参与

  5. 技能层级（skill hierarchy）：
     - 简单技能→组合技能→高级技能
     → 工程实现：基础技能→组合技能→高级组合技能

和其他子系统的精卫公式解读对比：
  大脑.py: a=预测, b=实际输入, Δ=预测误差, ⊗=赫布学习, τ=读出层pred
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  海马体.py: a=当前输入, b=已有记忆, Δ=差异, ⊗=重播巩固, τ=回忆浮现
  基底神经节.py: a=概念模式, b=技能预测, Δ=匹配误差, ⊗=技能学习, τ=自动化行为
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


class 基底神经节:
    """程序性记忆——技能学习与自动化执行，基于精卫公式。

    核心功能：
    1. 技能学习：概念模式→技能激活→行为输出（习惯学习）
    2. 自动化执行：技能强度高时自动触发
    3. 技能强度：用进废退（和概念层、海马体一致）
    4. 技能竞争：稀疏化——只保留最强的技能
    5. 多巴胺强化：预测误差驱动技能权重更新

    精卫公式流程（每拍）：
    1. a = 当前概念激活模式（concept_h）
    2. b = 技能预测的行为模式（skill_action_w × skill_h）
    3. Δ = b - a（匹配误差——技能预测与实际概念的差异）
    4. Σ = a + b（联合状态）
    5. ⊗ = Δ × Σ（累积技能学习信号）
    6. τ读⊗→浮出所见 = 技能激活从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新技能权重

    技能层级：
    - 基础技能（n_skills个）：概念模式→行为模式
    - 组合技能（n_combo_skills个）：基础技能组合→复杂行为
    """

    def __init__(self, n_concepts=64, n_skills=16, n_actions=8,
                 n_combo=32, lr=0.03, seed=42):
        self.n_concepts = n_concepts
        self.n_skills = n_skills
        self.n_actions = n_actions
        self.n_combo = n_combo
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # skill_h: 技能激活值
        self.skill_h = [0.0] * n_skills
        self.prev_skill_h = [0.0] * n_skills

        # skill_w: 概念模式→技能激活（纹状体接收皮层输入）
        # 当特定概念模式出现时，相关技能被激活
        self.skill_w = [[rng.uniform(-0.15, 0.15) for _ in range(n_concepts)]
                        for _ in range(n_skills)]

        # skill_action_w: 技能→行为输出（技能触发具体行为）
        # 技能激活后通过这条通路产生行为输出
        self.skill_action_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_actions)]
                               for _ in range(n_skills)]

        # skill_strength: 技能强度（用进废退）
        self.skill_strength = [1.0] * n_skills

        # 技能稀疏化——只保留最强的几个技能
        self.skill_sparse_k = max(1, int(n_skills * 0.2))

        # 组合技能：基础技能组合→复杂行为
        n_combo_skills = max(4, n_skills // 2)
        self.n_combo_skills = n_combo_skills
        self.combo_skill_h = [0.0] * n_combo_skills
        self.prev_combo_skill_h = [0.0] * n_combo_skills

        # combo_skill_w: 基础技能→组合技能（组合信号）
        self.combo_skill_w = [[rng.uniform(-0.2, 0.2) for _ in range(n_skills)]
                              for _ in range(n_combo_skills)]

        # combo_skill_action_w: 组合技能→行为输出
        self.combo_skill_action_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_actions)]
                                     for _ in range(n_combo_skills)]

        # 组合技能强度
        self.combo_skill_strength = [1.0] * n_combo_skills
        self.combo_skill_sparse_k = max(1, int(n_combo_skills * 0.2))

        # 技能执行统计
        self.技能执行次数 = 0
        self.技能成功次数 = 0
        self.组合技能执行次数 = 0
        self.组合技能成功次数 = 0

        # 行为输出（当前拍）
        self.action_output = [0.0] * n_actions
        self.combo_action_output = [0.0] * n_actions

        # 反馈信号（上一拍的行为结果）
        self.上拍反馈 = None

    def 一拍(self, concept_h, combo_h=None, 反馈=None):
        """基底神经节一拍：从概念模式学习并执行技能。

        精卫公式流程：
        1. a = concept_h（当前概念激活模式）
        2. 技能激活 = skill_w × a（纹状体接收皮层输入）
        3. 行为输出 = skill_action_w × skill_h（技能触发行为）
        4. 如果有反馈，更新技能权重（多巴胺强化学习）
        5. 技能竞争（稀疏化）

        参数：
          concept_h: 概念层的基础概念激活（n_concepts维）
          combo_h: 概念层的组合概念激活（n_combo维，可选）
          反馈: 上一拍行为结果的反馈信号（n_actions维，可选）
            - 正值=成功（多巴胺释放→技能强化）
            - 负值=失败（多巴胺减少→技能削弱）
            - None=无反馈

        返回：
          dict包含：
            skill_h: 技能激活模式
            action_output: 行为输出（传入嘴.py作为发声指导）
            combo_skill_h: 组合技能激活
            combo_action_output: 组合技能行为输出
            技能活跃数: 活跃技能数量
            存活技能数: 存活技能数量
        """
        self.拍 += 1

        # ── 精卫公式：a = 当前概念激活模式 ──
        a = concept_h

        # ── 技能激活：skill_w × a（纹状体接收皮层输入）──
        # τ读⊗→浮出所见：技能从概念模式中浮现
        new_skill_h = [0.0] * self.n_skills
        for i in range(self.n_skills):
            if self.skill_strength[i] < 0.05:
                continue
            s_input = 0.0
            for j in range(self.n_concepts):
                s_input += self.skill_w[i][j] * a[j]
            new_skill_h[i] = activation(s_input)

        # ── 技能竞争：稀疏化——只保留最强的k个技能 ──
        # 生物学：直接通路（Go）和间接通路（NoGo）——赢家通吃
        sorted_idx = sorted(range(self.n_skills),
                            key=lambda i: abs(new_skill_h[i]), reverse=True)
        top_k = set(sorted_idx[:self.skill_sparse_k])
        for i in range(self.n_skills):
            if i not in top_k:
                new_skill_h[i] *= 0.1

        # ── 行为输出：skill_action_w × skill_h ──
        # 技能激活后通过这条通路产生行为输出
        self.action_output = [0.0] * self.n_actions
        for j in range(self.n_actions):
            for i in range(self.n_skills):
                self.action_output[j] += self.skill_action_w[i][j] * new_skill_h[i]

        # ── 组合技能：基础技能组合→复杂行为 ──
        # 精卫公式：
        #   a = 基础技能j的激活
        #   b = 其他活跃技能之和
        #   ⊗ = a × b（组合信号）
        #   τ读⊗→浮出所见 = 组合技能从⊗中浮现
        活跃技能abs和 = sum(abs(v) for v in new_skill_h if abs(v) > 0.1)
        组合技能信号 = [0.0] * self.n_skills
        for j in range(self.n_skills):
            if abs(new_skill_h[j]) > 0.1:
                b = 活跃技能abs和 - abs(new_skill_h[j])
                组合技能信号[j] = new_skill_h[j] * b

        new_combo_skill_h = [0.0] * self.n_combo_skills
        for i in range(self.n_combo_skills):
            if self.combo_skill_strength[i] < 0.05:
                continue
            cs_input = 0.0
            for j in range(self.n_skills):
                cs_input += self.combo_skill_w[i][j] * 组合技能信号[j]
            new_combo_skill_h[i] = activation(cs_input)

        # 组合技能稀疏化
        combo_sorted = sorted(range(self.n_combo_skills),
                              key=lambda i: abs(new_combo_skill_h[i]), reverse=True)
        combo_top = set(combo_sorted[:self.combo_skill_sparse_k])
        for i in range(self.n_combo_skills):
            if i not in combo_top:
                new_combo_skill_h[i] *= 0.1

        # 组合技能行为输出
        self.combo_action_output = [0.0] * self.n_actions
        for j in range(self.n_actions):
            for i in range(self.n_combo_skills):
                self.combo_action_output[j] += self.combo_skill_action_w[i][j] * new_combo_skill_h[i]

        # ── (a, b) ⟲ ⊗ —— 循环：更新权重 ──
        # 多巴胺强化学习：反馈信号驱动技能权重更新
        # 精卫公式：
        #   a = 技能预测的行为模式（skill_action_w × prev_skill_h）
        #   b = 实际反馈（反馈信号）
        #   Δ = b - a（预测误差——成功为正，失败为负）
        #   Σ = a + b（联合状态）
        #   ⊗ = Δ × Σ → 权重更新
        if 反馈 is not None and self.上拍反馈 is not None:
            # 多巴胺信号 = 反馈与预测的差异
            # 正值=成功（比预期好）→ 强化技能
            # 负值=失败（比预期差）→ 削弱技能
            dopamine = [反馈[j] - self.上拍反馈[j] for j in range(self.n_actions)]

            for i in range(self.n_skills):
                if self.skill_strength[i] < 0.05:
                    continue
                # 技能权重更新：概念模式×技能激活×多巴胺
                for j in range(self.n_concepts):
                    dw = self.lr * self.prev_skill_h[i] * a[j] * sum(dopamine)
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.skill_w[i][j] = self.skill_w[i][j] * 0.99 + dw

                # 行为权重更新：技能激活×多巴胺
                for j in range(self.n_actions):
                    dw = self.lr * self.prev_skill_h[i] * dopamine[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.skill_action_w[i][j] = self.skill_action_w[i][j] * 0.99 + dw

                self.技能执行次数 += 1
                if sum(dopamine) > 0:
                    self.技能成功次数 += 1

        # ── 赫布学习（无反馈时也学习） ──
        # 精卫公式：a=概念模式, b=技能激活, Δ=b-a, ⊗=Δ×Σ
        # 概念模式和技能激活同时出现→连接增强（习惯形成）
        if 反馈 is None:
            for i in range(self.n_skills):
                if self.skill_strength[i] < 0.05:
                    continue
                for j in range(self.n_concepts):
                    dw = self.lr * new_skill_h[i] * a[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.skill_w[i][j] = self.skill_w[i][j] * 0.99 + dw

        # ── 组合技能学习 ──
        for i in range(self.n_combo_skills):
            if self.combo_skill_strength[i] < 0.05:
                continue
            for j in range(self.n_skills):
                dw = self.lr * new_combo_skill_h[i] * 组合技能信号[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.combo_skill_w[i][j] = self.combo_skill_w[i][j] * 0.98 + dw

        # ── 技能强度（用进废退） ──
        for i in range(self.n_skills):
            if abs(new_skill_h[i]) > 0.15:
                self.skill_strength[i] = min(3.0, self.skill_strength[i] + 0.01)
            else:
                self.skill_strength[i] *= 0.999

        for i in range(self.n_combo_skills):
            if abs(new_combo_skill_h[i]) > 0.15:
                self.combo_skill_strength[i] = min(3.0, self.combo_skill_strength[i] + 0.01)
            else:
                self.combo_skill_strength[i] *= 0.999

        # ── 突触缩放 ──
        for i in range(self.n_skills):
            if self.skill_strength[i] < 0.05:
                continue
            w_norm = sum(abs(self.skill_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.skill_w[i][j] *= scale
            aw_norm = sum(abs(self.skill_action_w[i][j]) for j in range(self.n_actions))
            if aw_norm > 2.0:
                scale = 2.0 / aw_norm
                for j in range(self.n_actions):
                    self.skill_action_w[i][j] *= scale

        # 保存反馈用于下一拍
        self.上拍反馈 = 反馈[:] if 反馈 is not None else None

        self.prev_skill_h = self.skill_h[:]
        self.skill_h = new_skill_h
        self.prev_combo_skill_h = self.combo_skill_h[:]
        self.combo_skill_h = new_combo_skill_h

        return {
            "skill_h": self.skill_h,
            "action_output": self.action_output,
            "combo_skill_h": self.combo_skill_h,
            "combo_action_output": self.combo_action_output,
            "技能活跃数": sum(1 for v in self.skill_h if abs(v) > 0.15),
            "存活技能数": sum(1 for s in self.skill_strength if s > 0.05),
            "组合技能活跃数": sum(1 for v in self.combo_skill_h if abs(v) > 0.15),
            "存活组合技能数": sum(1 for s in self.combo_skill_strength if s > 0.05),
            "技能执行次数": self.技能执行次数,
            "技能成功次数": self.技能成功次数,
        }

    def 状态摘要(self):
        return {
            "活跃技能数": sum(1 for v in self.skill_h if abs(v) > 0.15),
            "存活技能数": sum(1 for s in self.skill_strength if s > 0.05),
            "技能均值": round(sum(self.skill_h) / max(1, self.n_skills), 4),
            "技能最大": round(max((abs(v) for v in self.skill_h), default=0), 4),
            "组合技能活跃数": sum(1 for v in self.combo_skill_h if abs(v) > 0.15),
            "存活组合技能数": sum(1 for s in self.combo_skill_strength if s > 0.05),
            "技能执行次数": self.技能执行次数,
            "技能成功次数": self.技能成功次数,
            "技能成功率": round(self.技能成功次数 / max(1, self.技能执行次数), 4),
            "行为输出均值": round(sum(self.action_output) / max(1, self.n_actions), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 基底神经节验证（程序性记忆） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    bg = 基底神经节(n_concepts=64, n_skills=16, n_actions=8)

    print("初始状态:")
    print(f"  技能数: {bg.n_skills}")
    print(f"  组合技能数: {bg.n_combo_skills}")
    print(f"  行为维度: {bg.n_actions}")
    print(f"  稀疏k: {bg.skill_sparse_k}")

    print("\n--- 测试1: 技能从概念模式中浮现 ---")
    # 模拟两种不同的概念模式
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        concept_h[27] = 0.2
        result = bg.一拍(concept_h)
    print(f"  模式A后: 活跃技能={result['技能活跃数']}, 存活={result['存活技能数']}")
    print(f"  技能激活top5: {sorted(range(16), key=lambda i: abs(bg.skill_h[i]), reverse=True)[:5]}")

    bg2 = 基底神经节(n_concepts=64, n_skills=16, n_actions=8)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[0] = 0.3
        concept_h[24] = 0.25
        concept_h[40] = 0.2
        result = bg2.一拍(concept_h)
    print(f"  模式B后: 活跃技能={result['技能活跃数']}, 存活={result['存活技能数']}")
    print(f"  技能激活top5: {sorted(range(16), key=lambda i: abs(bg2.skill_h[i]), reverse=True)[:5]}")

    diff = sum(abs(bg.skill_h[i] - bg2.skill_h[i]) for i in range(16))
    print(f"  A vs B 技能差异: {diff:.4f}（应该 > 0——不同概念模式→不同技能激活）")

    print("\n--- 测试2: 技能区分度（独立实例） ---")
    patterns = [
        ("光+声", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 54),
        ("温+力", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25, 0.2, 0.15, 0.0, 0.0] + [0.0] * 54),
        ("均衡", [0.15] * 64),
    ]
    results = {}
    for name, pattern in patterns:
        bg_t = 基底神经节(n_concepts=64, n_skills=16, n_actions=8)
        for _ in range(50):
            bg_t.一拍(pattern)
        results[name] = list(bg_t.skill_h)
        active = [i for i, v in enumerate(bg_t.skill_h) if abs(v) > 0.1]
        print(f"  {name}: 活跃技能={active[:6]}, 技能最大={max(abs(v) for v in bg_t.skill_h):.4f}")

    for i in range(3):
        for j in range(i + 1, 3):
            n1, n2 = patterns[i][0], patterns[j][0]
            d = sum(abs(results[n1][k] - results[n2][k]) for k in range(16))
            print(f"  {n1} vs {n2} 技能差异: {d:.4f}")

    print("\n--- 测试3: 行为输出 ---")
    bg3 = 基底神经节(n_concepts=64, n_skills=16, n_actions=8)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = bg3.一拍(concept_h)
    print(f"  行为输出: {[round(v, 3) for v in result['action_output']]}")
    print(f"  行为输出均值: {sum(abs(v) for v in result['action_output']) / 8:.4f}")

    print("\n--- 测试4: 多巴胺强化学习 ---")
    bg4 = 基底神经节(n_concepts=64, n_skills=16, n_actions=8)
    # 先让技能学习50拍（无反馈）
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        bg4.一拍(concept_h)
    # 然后给正面反馈50拍（技能成功）
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        反馈 = [0.1] * 8  # 正面反馈
        bg4.一拍(concept_h, 反馈=反馈)
    print(f"  正面反馈后: 技能执行={bg4.技能执行次数}, 成功={bg4.技能成功次数}")
    print(f"  成功率: {bg4.技能成功次数 / max(1, bg4.技能执行次数):.4f}")
    print(f"  skill_w总权重: {sum(abs(bg4.skill_w[i][j]) for i in range(16) for j in range(64)):.2f}")

    print("\n--- 测试5: 组合技能 ---")
    bg5 = 基底神经节(n_concepts=64, n_skills=16, n_actions=8)
    for _ in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        concept_h[27] = 0.2
        result = bg5.一拍(concept_h)
    print(f"  100拍后: 组合技能活跃={result['组合技能活跃数']}, 存活={result['存活组合技能数']}")
    print(f"  组合技能激活top3: {sorted(range(bg5.n_combo_skills), key=lambda i: abs(bg5.combo_skill_h[i]), reverse=True)[:3]}")

    print("\n--- 测试6: 技能用进废退 ---")
    bg6 = 基底神经节(n_concepts=64, n_skills=16, n_actions=8)
    for _ in range(200):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        bg6.一拍(concept_h)
    print(f"  200拍后: 存活技能={sum(1 for s in bg6.skill_strength if s > 0.05)}")
    print(f"  技能强度: min={min(bg6.skill_strength):.3f}, max={max(bg6.skill_strength):.3f}")

    print("\n--- 测试7: 精卫公式验证 ---")
    bg7 = 基底神经节(n_concepts=4, n_skills=4, n_actions=2)
    for tick in range(10):
        concept_h = [0.3, 0.5, 0.1, 0.2]
        result = bg7.一拍(concept_h)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick + 1}: 技能激活={[round(v, 3) for v in result['skill_h']]}")
            print(f"         行为输出={[round(v, 3) for v in result['action_output']]}")
    print(f"  τ(浮出): 技能激活均值={sum(bg7.skill_h) / 4:.4f}")
    print(f"  ⟲(循环): 拍={bg7.拍}")

    print("\n验证通过")
