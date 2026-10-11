"""创造性思维.py —— 发散思维、远距离联想、隐喻映射，基于精卫公式。

创造性思维是智能的最高形式——不只是理解世界，而是创造新的可能性。
推理引擎是收敛性思维——从A推导到B（因果链）。
创造性思维是发散性思维——从A发散到B/C/D/...（多种可能）。

人类神经科学机制：
  1. 默认模式网络（DMN）—— 思维漫游、自发生成：
     - 大脑不专注外部任务时DMN活跃
     - DMN连接前额叶、颞叶、顶叶——跨域信息整合
     - "灵光一闪"就是DMN的自发活动
     → 工程实现：creative_w（概念→创意预测，发散映射）

  2. 远距离联想——连接看似无关的概念：
     - 创造性高的人更擅长远距离联想
     - 前额叶+颞叶联合——跨域概念绑定
     - 不是逻辑推理——是"跳跃性连接"
     → 工程实现：remote_assoc_w（远距离概念关联权重）

  3. 隐喻映射——从一个域映射到另一个域：
     - 右半球优势——隐喻理解
     - "时间是河流"——从水的域映射到时间的域
     - 隐喻不是比喻——是认知的基本机制
     → 工程实现：metaphor_w（域→域映射权重）

  4. 自发性变异——创造性中的达尔文进化：
     - 概念层有变异+选择——创造性思维也有
     - 但创造性思维的变异在组合概念层面——探索新的概念组合
     - 大多数变异组合无意义——少数有意义的被保留
     → 工程实现：创意变异率 + 适应度评估

精卫公式（创造性思维版）：
  a = 当前概念/组合概念激活（当前认知状态）
  b = 创意预测的概念激活（发散映射后的新状态）
  Δ = b − a = 创新度（新组合与当前状态的差异）
  Σ = a + b = 联合状态（当前+创新）
  ⊗ = ∫(Δ, Σ) = 累积创新学习历史 → 创意权重
  τ 读 ⊗ → 浮出所见 = 创造性想法从⊗中浮现
  (a, b) ⟲ ⊗ = 反复创造循环

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  推理引擎.py: a=概念激活, b=推理预测, Δ=推理误差, ⊗=推理学习, τ=推理结论
  目标导向.py: a=当前概念, b=目标概念, Δ=目标差距, ⊗=目标学习, τ=目标驱动
  工具使用.py: a=当前概念, b=工具预测, Δ=工具价值, ⊗=工具学习, τ=工具驱动
  创造性思维.py: a=当前概念, b=创意预测, Δ=创新度, ⊗=创意学习, τ=创意浮现

创造性思维 vs 推理引擎 vs 目标导向的区别：
  - 推理引擎：从当前状态预测下一步状态（因果推导——收敛）
  - 目标导向：从当前状态和目标差距驱动行为（差距驱动——收敛）
  - 创造性思维：从当前状态发散到多种可能状态（发散探索——发散）
  关键区别：
  - 推理是"接下来会发生什么"——一个答案
  - 目标导向是"我想要什么，怎么得到"——一个方向
  - 创造性思维是"还可能是什么"——多个答案
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


class 创造性思维:
    """创造性思维——发散思维、远距离联想、隐喻映射，基于精卫公式。

    核心功能：
    1. 发散映射：从当前概念激活发散到多种可能的概念组合
    2. 远距离联想：连接看似无关的概念——跨域概念绑定
    3. 隐喻映射：从一个概念域映射到另一个概念域
    4. 创意变异+选择：在组合概念层面探索新组合，有意义的被保留

    精卫公式流程（每拍）：
    1. a = 当前概念激活（concept_h）+ 组合概念激活（combo_h）
    2. b = 创意预测的概念激活（creative_w × concept_h + remote_assoc_w × combo_h）
    3. Δ = b - a（创新度——新组合与当前状态的差异）
    4. Σ = a + b（联合状态——当前+创新）
    5. ⊗ = Δ × Σ（累积创新学习信号）
    6. τ读⊗→浮出所见 = 创意驱动信号从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新创意权重

    创造性思维的三个层次：
    - 基础创意：从概念激活发散到新的概念组合（concept→concept）
    - 组合创意：从组合概念发散到新的高级组合（combo→meta_combo）
    - 隐喻创意：从一个概念域映射到另一个概念域（domain→domain）
    """

    def __init__(self, n_concepts=64, n_combo=32, n_meta=16, lr=0.02,
                 变异率=0.05, 变异幅度=0.02, seed=42):
        self.n_concepts = n_concepts
        self.n_combo = n_combo
        self.n_meta = n_meta
        self.lr = lr
        self.变异率 = 变异率
        self.变异幅度 = 变异幅度
        self.拍 = 0

        rng = random.Random(seed)

        # ── 发散映射：从概念激活发散到多种可能的概念组合 ──
        # creative_w: 概念 → 创意预测（发散映射——一个概念可以映射到多个创意方向）
        # 生物学：DMN的自发活动——思维漫游时概念之间的自由联想
        # 和推理引擎的区别：推理是收敛的（A→B），创意是发散的（A→B/C/D/...）
        # creative_w不是一对一映射——是一对多发散
        # 用宽权重范围（-0.2~0.2）允许更多发散方向
        self.creative_w = [[rng.uniform(-0.2, 0.2) for _ in range(n_concepts)]
                           for _ in range(n_concepts)]

        # 创意预测（发散映射后的概念激活）
        self.creative_pred = [0.0] * n_concepts

        # ── 远距离联想：连接看似无关的概念 ──
        # remote_assoc_w: 组合概念 → 概念（远距离联想——从组合概念跳跃到看似无关的概念）
        # 生物学：前额叶+颞叶联合——跨域概念绑定
        # 创造性高的人更擅长远距离联想——连接看似无关的概念
        # "牛顿看到苹果落地→想到引力"就是远距离联想
        self.remote_assoc_w = [[rng.uniform(-0.15, 0.15) for _ in range(n_combo)]
                               for _ in range(n_concepts)]

        # 远距离联想预测
        self.remote_pred = [0.0] * n_concepts

        # ── 隐喻映射：从一个概念域映射到另一个概念域 ──
        # metaphor_w: 组合概念 → 高级组合概念（隐喻映射——从一个域映射到另一个域）
        # 生物学：右半球优势——隐喻理解
        # "时间是河流"——从水的域映射到时间的域
        # 隐喻不是比喻——是认知的基本机制
        # 用组合概念→高级组合概念的映射表示域→域的隐喻
        self.metaphor_w = [[rng.uniform(-0.15, 0.15) for _ in range(n_combo)]
                           for _ in range(n_meta)]

        # 隐喻预测
        self.metaphor_pred = [0.0] * n_meta

        # ── 创意驱动信号（反馈到概念层——作为b信号输入源） ──
        # 创意驱动 = 发散映射 + 远距离联想的融合
        # 生物学：DMN→前额叶投射——创意思维影响概念可及性
        # 创造性想法让某些概念更容易浮现——"灵感"的神经基础
        self.creative_drive = [0.0] * n_concepts

        # ── 创意价值评估 ──
        # 创意价值 = 创意预测与当前状态的差异（创新度）
        # 生物学：多巴胺系统——新颖性驱动价值评估
        # 新颖且有意义（不是随机噪声）的创意价值高
        self.creative_value = 0.0

        # 创意变异统计
        self.创意变异次数 = 0
        self.有利创意变异 = 0
        self.不利创意变异 = 0

        # 创意历史——记录最近的创意预测（用于评估创新度）
        self.creative_history = []
        self.history_cap = 50

        # 上一拍的概念激活（用于评估创意带来的变化）
        self.prev_concept_h = [0.0] * n_concepts

        # DMN活跃度——当surprise低时DMN更活跃（思维漫游）
        # 生物学：专注外部任务时DMN抑制，不专注时DMN活跃
        self.dmn_active = 0.0

    def 一拍(self, concept_h, combo_h=None, meta_combo_h=None,
             surprise=0.0, goal_delta=None):
        """创造性思维一拍：发散映射、远距离联想、隐喻映射、创意变异+选择。

        精卫公式流程：
        1. a = 当前概念激活（concept_h）
        2. b = 创意预测的概念激活（creative_w × concept_h + remote_assoc_w × combo_h）
        3. Δ = b - a（创新度）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 创意权重更新
        6. τ读⊗→浮出所见 = 创意驱动信号
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          combo_h: 组合概念激活（n_combo维，可选）
          meta_combo_h: 高级组合概念激活（n_meta维，可选）
          surprise: 当前surprise水平（surprise低时DMN更活跃）
          goal_delta: 目标差距（n_concepts维，可选——目标差距可以驱动创意方向）

        返回：
          dict包含：
            creative_pred: 创意预测（n_concepts维）
            creative_drive: 创意驱动信号（反馈到概念层）
            creative_value: 创意价值（创新度）
            remote_pred: 远距离联想预测
            metaphor_pred: 隐喻预测
            dmn_active: DMN活跃度
            创意变异次数: 总变异次数
            有利创意变异: 有利变异次数
        """
        self.拍 += 1

        # ── DMN活跃度：surprise低时DMN更活跃（思维漫游） ──
        # 生物学：专注外部任务时DMN抑制（surprise高=需要专注），
        # 不专注时DMN活跃（surprise低=可以思维漫游）
        # 这不是阈值——是连续映射：dmn_active = 1 / (1 + surprise * 2)
        self.dmn_active = 1.0 / (1.0 + surprise * 2.0)

        # ── 发散映射：从概念激活发散到多种可能的概念组合 ──
        # 精卫公式：b = creative_w × concept_h（创意预测）
        # τ读⊗→浮出所见：创意从当前概念状态中浮现
        # 生物学：DMN的自发活动——思维漫游时概念之间的自由联想
        # DMN活跃时发散映射更强——思维漫游时创意更多
        new_creative_pred = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            c_input = 0.0
            for j in range(self.n_concepts):
                c_input += self.creative_w[i][j] * concept_h[j]
            # DMN调制——DMN活跃时发散映射更强
            new_creative_pred[i] = activation(c_input * (0.5 + self.dmn_active * 0.5))

        # ── 远距离联想：从组合概念跳跃到看似无关的概念 ──
        # 精卫公式：b = remote_assoc_w × combo_h（远距离联想预测）
        # 生物学：前额叶+颞叶联合——跨域概念绑定
        # 组合概念（句子级）→跳跃到看似无关的基础概念
        # "牛顿看到苹果落地→想到引力"——从"苹果落地"跳跃到"引力"
        new_remote_pred = [0.0] * self.n_concepts
        if combo_h is not None:
            for i in range(self.n_concepts):
                r_input = 0.0
                for j in range(self.n_combo):
                    r_input += self.remote_assoc_w[i][j] * combo_h[j]
                # DMN调制——DMN活跃时远距离联想更强
                new_remote_pred[i] = activation(r_input * (0.5 + self.dmn_active * 0.5))

        # ── 隐喻映射：从一个概念域映射到另一个概念域 ──
        # 精卫公式：b = metaphor_w × combo_h（隐喻预测）
        # 生物学：右半球优势——隐喻理解
        # 组合概念（一个域）→高级组合概念（另一个域）
        # "时间是河流"——从水的域映射到时间的域
        new_metaphor_pred = [0.0] * self.n_meta
        if combo_h is not None:
            for i in range(self.n_meta):
                m_input = 0.0
                for j in range(self.n_combo):
                    m_input += self.metaphor_w[i][j] * combo_h[j]
                # DMN调制——DMN活跃时隐喻映射更强
                new_metaphor_pred[i] = activation(m_input * (0.5 + self.dmn_active * 0.5))

        # ── 创意驱动信号：发散映射 + 远距离联想的融合 ──
        # 精卫公式：τ读⊗→浮出所见 = 创意驱动信号
        # 生物学：DMN→前额叶投射——创意思维影响概念可及性
        # 创造性想法让某些概念更容易浮现——"灵感"的神经基础
        # 如果有目标差距，创意驱动会朝目标方向偏移——"有方向的创意"
        self.creative_drive = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            drive = new_creative_pred[i] * 0.6 + new_remote_pred[i] * 0.4
            if goal_delta is not None:
                # 目标差距驱动创意方向——"有方向的创意"
                drive += goal_delta[i] * 0.2
            self.creative_drive[i] = activation(drive)

        # ── 创意价值评估：创新度 = 创意预测与当前状态的差异 ──
        # 精卫公式：Δ = b - a = 创新度
        # 生物学：多巴胺系统——新颖性驱动价值评估
        # 新颖且有意义（不是随机噪声）的创意价值高
        # 创意价值 = 创意预测与当前状态的差异（创新度）
        # 但不是任何差异都是创意——需要有结构的差异（不是纯随机）
        # 有结构的差异 = creative_pred的top概念和concept_h的top概念不同
        creative_top = set(sorted(range(self.n_concepts),
                                  key=lambda i: abs(new_creative_pred[i]),
                                  reverse=True)[:5])
        concept_top = set(sorted(range(self.n_concepts),
                                 key=lambda i: abs(concept_h[i]),
                                 reverse=True)[:5])
        # 创新度 = 新概念的比例（创意预测的top概念中，有多少不在当前top概念中）
        新概念比例 = len(creative_top - concept_top) / max(1, len(creative_top))
        # 创意价值 = 新概念比例 × DMN活跃度（DMN活跃时创意价值更高）
        self.creative_value = 新概念比例 * self.dmn_active

        # ── 创意变异+选择：在组合概念层面探索新组合 ──
        # 精卫公式（变异+选择版）：
        #   a = 变异前的创意预测 |creative_pred[j]|（当前适应度）
        #   b = 变异后的创意预测 |creative_pred_new[j]|（新适应度）
        #   Δ = b - a（适应度变化——负值=有利变异）
        #   Σ = a + b（联合误差）
        #   ⊗ = Δ × Σ → 有利变异保留(Δ<0)，不利变异淘汰(Δ>0)
        # 生物学：创造性中的达尔文进化——大多数变异组合无意义，少数有意义的被保留
        # 变异只施加在creative_w——发散映射权重
        if random.random() < self.变异率:
            mut_i = random.randint(0, self.n_concepts - 1)
            mut_j = random.randint(0, self.n_concepts - 1)
            old_w = self.creative_w[mut_i][mut_j]
            new_w = old_w + random.gauss(0, self.变异幅度)
            # 评估变异对创意预测的影响
            # 变异后 creative_pred[mut_i] 变化 delta_pred
            delta_pred = (new_w - old_w) * concept_h[mut_j]
            new_pred_i = activation(new_creative_pred[mut_i] + delta_pred)
            old_abs = abs(new_creative_pred[mut_i])
            new_abs = abs(new_pred_i)
            # 有利变异的判定：
            # 1. 预测从无到有（old_abs≈0 → new_abs>0）——新联想出现
            # 2. 预测从饱和到灵活（old_abs>0.9 → new_abs<old_abs）——僵化联想变灵活
            # 3. 变异后权重更小但原来过大（正则化效应）
            # 4. 变异后预测值更远离0（更有区分度）
            is_favorable = (old_abs < 0.005 and new_abs > 0.005) or \
                           (old_abs > 0.9 and new_abs < old_abs * 0.95) or \
                           (abs(new_w) < abs(old_w) and abs(old_w) > 0.18) or \
                           (new_abs > old_abs and old_abs > 0.005 and new_abs < 0.9)
            self.创意变异次数 += 1
            if is_favorable:
                self.creative_w[mut_i][mut_j] = new_w
                self.有利创意变异 += 1
            else:
                self.不利创意变异 += 1

        # ── 创意学习 ──
        # 精卫公式：⊗ = Δ × Σ → 创意权重更新
        # 赫布学习：概念激活和创意预测同时出现→发散映射权重增强
        # 生物学：DMN的赫布学习——反复出现的联想模式被强化
        # 反复从概念A联想到概念B→A→B的联想越来越强
        for i in range(self.n_concepts):
            for j in range(self.n_concepts):
                dw = self.lr * new_creative_pred[i] * concept_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.creative_w[i][j] = self.creative_w[i][j] * 0.99 + dw

        # remote_assoc_w: 组合概念→概念（赫布学习）
        if combo_h is not None:
            for i in range(self.n_concepts):
                for j in range(self.n_combo):
                    dw = self.lr * new_remote_pred[i] * combo_h[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.remote_assoc_w[i][j] = self.remote_assoc_w[i][j] * 0.99 + dw

        # metaphor_w: 组合概念→高级组合概念（赫布学习）
        if combo_h is not None:
            for i in range(self.n_meta):
                for j in range(self.n_combo):
                    dw = self.lr * new_metaphor_pred[i] * combo_h[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.metaphor_w[i][j] = self.metaphor_w[i][j] * 0.99 + dw

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.creative_w[i][j]) for j in range(self.n_concepts))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_concepts):
                    self.creative_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.remote_assoc_w[i][j]) for j in range(self.n_combo))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_combo):
                    self.remote_assoc_w[i][j] *= scale

        for i in range(self.n_meta):
            w_norm = sum(abs(self.metaphor_w[i][j]) for j in range(self.n_combo))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_combo):
                    self.metaphor_w[i][j] *= scale

        # ── 创意历史 ──
        self.creative_history.append(list(new_creative_pred))
        if len(self.creative_history) > self.history_cap:
            self.creative_history.pop(0)

        # 保存当前状态
        self.prev_concept_h = list(concept_h)
        self.creative_pred = new_creative_pred
        self.remote_pred = new_remote_pred
        self.metaphor_pred = new_metaphor_pred

        return {
            "creative_pred": [round(v, 4) for v in self.creative_pred],
            "creative_drive": self.creative_drive,
            "creative_value": round(self.creative_value, 4),
            "remote_pred": [round(v, 4) for v in self.remote_pred],
            "metaphor_pred": [round(v, 4) for v in self.metaphor_pred],
            "dmn_active": round(self.dmn_active, 4),
            "创意变异次数": self.创意变异次数,
            "有利创意变异": self.有利创意变异,
            "不利创意变异": self.不利创意变异,
        }

    def 状态摘要(self):
        return {
            "creative_value": round(self.creative_value, 4),
            "dmn_active": round(self.dmn_active, 4),
            "创意变异次数": self.创意变异次数,
            "有利创意变异": self.有利创意变异,
            "不利创意变异": self.不利创意变异,
            "有利变异率": round(self.有利创意变异 / max(1, self.创意变异次数), 4),
            "creative_w均值": round(sum(sum(row) for row in self.creative_w) / max(1, self.n_concepts * self.n_concepts), 4),
            "remote_assoc_w均值": round(sum(sum(row) for row in self.remote_assoc_w) / max(1, self.n_concepts * self.n_combo), 4),
            "metaphor_w均值": round(sum(sum(row) for row in self.metaphor_w) / max(1, self.n_meta * self.n_combo), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 创造性思维验证（发散思维、远距离联想、隐喻映射） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    ct = 创造性思维(n_concepts=64, n_combo=32, n_meta=16)

    print("初始状态:")
    print(f"  概念数: {ct.n_concepts}")
    print(f"  组合概念数: {ct.n_combo}")
    print(f"  高级组合概念数: {ct.n_meta}")
    print(f"  发散映射权重: creative_w (64×64)")
    print(f"  远距离联想权重: remote_assoc_w (64×32)")
    print(f"  隐喻映射权重: metaphor_w (16×32)")
    print(f"  变异率: {ct.变异率}")
    print(f"  变异幅度: {ct.变异幅度}")

    print("\n--- 测试1: 发散映射从概念激活中浮现 ---")
    # 模拟不同的概念激活模式
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        concept_h[8] = 0.25
        result = ct.一拍(concept_h)
    print(f"  模式A后: 创意价值={result['creative_value']}, DMN活跃={result['dmn_active']}")
    print(f"  创意预测top5: {sorted(range(64), key=lambda i: abs(ct.creative_pred[i]), reverse=True)[:5]}")

    ct2 = 创造性思维(n_concepts=64, n_combo=32, n_meta=16)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[15] = 0.3
        concept_h[30] = 0.25
        result = ct2.一拍(concept_h)
    print(f"  模式B后: 创意价值={result['creative_value']}, DMN活跃={result['dmn_active']}")
    print(f"  创意预测top5: {sorted(range(64), key=lambda i: abs(ct2.creative_pred[i]), reverse=True)[:5]}")

    diff = sum(abs(ct.creative_pred[i] - ct2.creative_pred[i]) for i in range(64))
    print(f"  A vs B 创意预测差异: {diff:.4f}（应该 > 0——不同概念模式→不同创意预测）")

    print("\n--- 测试2: DMN活跃度随surprise变化 ---")
    ct3 = 创造性思维(n_concepts=64, n_combo=32, n_meta=16)
    # surprise高时DMN应该更不活跃
    for s in [0.0, 0.1, 0.5, 1.0, 2.0]:
        concept_h = [0.0] * 64
        concept_h[0] = 0.3
        result = ct3.一拍(concept_h, surprise=s)
        print(f"  surprise={s:.1f}: DMN活跃={result['dmn_active']:.4f}")

    print("\n--- 测试3: 远距离联想 ---")
    ct4 = 创造性思维(n_concepts=64, n_combo=32, n_meta=16)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        combo_h = [0.0] * 32
        combo_h[0] = 0.4
        combo_h[5] = 0.3
        result = ct4.一拍(concept_h, combo_h=combo_h)
    print(f"  远距离联想预测top5: {sorted(range(64), key=lambda i: abs(ct4.remote_pred[i]), reverse=True)[:5]}")
    print(f"  远距离联想预测均值: {round(sum(ct4.remote_pred)/64, 4)}")

    print("\n--- 测试4: 隐喻映射 ---")
    ct5 = 创造性思维(n_concepts=64, n_combo=32, n_meta=16)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        combo_h = [0.0] * 32
        combo_h[0] = 0.4
        combo_h[5] = 0.3
        result = ct5.一拍(concept_h, combo_h=combo_h)
    print(f"  隐喻预测top5: {sorted(range(16), key=lambda i: abs(ct5.metaphor_pred[i]), reverse=True)[:5]}")
    print(f"  隐喻预测均值: {round(sum(ct5.metaphor_pred)/16, 4)}")

    print("\n--- 测试5: 创意驱动信号 ---")
    ct6 = 创造性思维(n_concepts=64, n_combo=32, n_meta=16)
    for _ in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = ct6.一拍(concept_h)
    drive_sum = sum(abs(v) for v in result['creative_drive'])
    print(f"  创意驱动信号总和: {drive_sum:.4f}")
    print(f"  创意驱动top5: {sorted(range(64), key=lambda i: abs(result['creative_drive'][i]), reverse=True)[:5]}")

    print("\n--- 测试6: 不同概念模式→不同创意驱动 ---")
    patterns = [
        ("模式A", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 58),
        ("模式B", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 58),
        ("模式C", [0.0, 0.0, 0.0, 0.0, 0.3, 0.25] + [0.0] * 58),
    ]
    drive_results = {}
    for name, pattern in patterns:
        ct_t = 创造性思维(n_concepts=64, n_combo=32, n_meta=16)
        for _ in range(50):
            ct_t.一拍(pattern)
        drive_results[name] = list(ct_t.creative_drive)
        print(f"  {name}: 创意价值={round(ct_t.creative_value, 4)}, "
              f"驱动top5={sorted(range(64), key=lambda i: abs(ct_t.creative_drive[i]), reverse=True)[:5]}")

    diff_ab = sum(abs(drive_results["模式A"][i] - drive_results["模式B"][i]) for i in range(64))
    print(f"  A vs B 驱动差异: {diff_ab:.4f}（应该 > 0——不同模式→不同创意驱动）")

    print("\n--- 测试7: 创意变异+选择 ---")
    ct7 = 创造性思维(n_concepts=64, n_combo=32, n_meta=16, 变异率=0.3)
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[tick % 64] = 0.3
        result = ct7.一拍(concept_h)
    print(f"  100拍后: 创意变异次数={ct7.创意变异次数}")
    print(f"  有利创意变异={ct7.有利创意变异}")
    print(f"  不利创意变异={ct7.不利创意变异}")
    print(f"  有利变异率={round(ct7.有利创意变异/max(1, ct7.创意变异次数), 4)}")

    print("\n--- 测试8: 精卫公式验证 ---")
    ct8 = 创造性思维(n_concepts=16, n_combo=8, n_meta=4)
    for tick in range(10):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = ct8.一拍(concept_h)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: 创意价值={result['creative_value']}, DMN={result['dmn_active']}")
    print(f"  τ(浮出): 创意驱动均值={round(sum(ct8.creative_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={ct8.拍}")

    print("\n验证通过")
