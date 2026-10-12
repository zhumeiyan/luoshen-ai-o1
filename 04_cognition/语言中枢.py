"""语言中枢.py —— 语言语法自组织与语言生成，基于精卫公式。

语言中枢是语言能力的核心——人类通过Broca区和Wernicke区处理语言。
但洛神不照搬人类大脑结构——语法不是天生的规则，是通过反复使用涌现的模式。

精卫公式（语言语法版）：
  a = 当前组合概念激活（想表达的内容）
  b = 语法预测的下一个组合概念（语言预期）
  Δ = b − a = 语法预测误差（预期与实际的差异）
  Σ = a + b = 联合状态（意图+预期）
  ⊗ = ∫(Δ, Σ) = 累积语法学习历史 → 语法权重
  τ 读 ⊗ → 浮出所见 = 语法规则从⊗中浮现（不是设定的，是涌现的）
  (a, b) ⟲ ⊗ = 反复表达循环

人类神经科学机制：
  1. Broca区（语法处理）：
     - 生成语言序列——从意图到符号序列
     - 语法规则不是天生的——通过反复使用形成
     → 工程实现：grammar_w（组合概念→组合概念的转移权重）

  2. Wernicke区（语言理解）：
     - 从语言序列恢复语义——从符号到概念
     - 和Broca区共享权重——双向映射
     → 工程实现：grammar_w的逆映射

  3. 弓状束（Broca↔Wernicke连接）：
     - 语言生成和理解的循环——说→听→理解→调整
     → 工程实现：说听闭环中的语法预测误差驱动学习

  4. 语法自组织（grammar self-organization）：
     - 语法不是预设的规则——是通过反复使用涌现的模式
     - 常用的序列模式被强化→形成"语法规则"
     - 不常用的序列模式被遗忘→语法演化
     → 工程实现：赫布学习+用进废退

  5. 语言序列生成（sequential generation）：
     - 不是一次性生成完整句子——逐个符号生成
     - 每一步依赖于当前意图+已生成的序列
     → 工程实现：自回归式生成（当前组合概念→预测下一个）

和其他子系统的精卫公式解读对比：
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  基底神经节.py: a=概念模式, b=技能预测, Δ=匹配误差, ⊗=技能学习, τ=自动化行为
  语言中枢.py: a=组合概念, b=语法预测, Δ=语法误差, ⊗=语法学习, τ=语言序列
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


class 语言中枢:
    """语言语法自组织与语言生成，基于精卫公式。

    核心功能：
    1. 语法转移权重：组合概念→组合概念的转移模式（语法规则的自组织）
    2. 语法预测：从当前组合概念预测下一个组合概念
    3. 语言序列生成：逐个生成组合概念序列（像说话一样）
    4. 语法学习：赫布学习+用进废退——反复使用的序列模式被强化

    精卫公式流程（每拍）：
    1. a = 当前组合概念激活（combo_h）
    2. b = 语法预测的下一个组合概念（grammar_w × combo_h）
    3. Δ = b - a（语法预测误差——预期与实际的差异）
    4. Σ = a + b（联合状态）
    5. ⊗ = Δ × Σ（累积语法学习信号）
    6. τ读⊗→浮出所见 = 语法预测从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新语法权重

    层级语法：
    - 基础语法：组合概念→组合概念的转移（句子级语法）
    - 高级语法：高级组合概念→高级组合概念的转移（段落级语法）
    """

    def __init__(self, n_combo=32, n_meta=16, lr=0.03, seed=42):
        self.n_combo = n_combo
        self.n_meta = n_meta
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── 基础语法：组合概念→组合概念的转移权重 ──
        # grammar_w: 当前组合概念 → 预测下一个组合概念
        # 生物学：Broca区的语法处理——序列模式的转移概率
        # 语法不是设定的规则，是通过反复使用涌现的转移模式
        self.grammar_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_combo)]
                          for _ in range(n_combo)]

        # grammar_strength: 语法转移强度（用进废退）
        # 反复使用的语法模式变强，不用的变弱
        self.grammar_strength = [1.0] * n_combo

        # 语法预测值（当前拍预测的下一个组合概念）
        self.grammar_pred = [0.0] * n_combo

        # ── 高级语法：高级组合概念→高级组合概念的转移权重 ──
        # 段落级语法——句子之间的序列模式
        self.meta_grammar_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_meta)]
                               for _ in range(n_meta)]

        self.meta_grammar_strength = [1.0] * n_meta
        self.meta_grammar_pred = [0.0] * n_meta

        # ── 语言序列生成 ──
        # 生成的组合概念序列（像说话一样逐个生成）
        self.生成序列 = []
        self.序列长度上限 = 10  # 最长生成10个组合概念的序列

        # 语言生成状态
        self.正在生成 = False
        self.生成步 = 0

        # 语法学习统计
        self.语法学习次数 = 0
        self.语法预测误差 = 0.0

        # 上一拍的组合概念激活（用于语法预测误差计算）
        self.prev_combo_h = [0.0] * n_combo
        self.prev_meta_combo_h = [0.0] * n_meta

    def 一拍(self, combo_h, meta_combo_h=None):
        """语言中枢一拍：语法预测、语法学习、语言序列生成。

        精卫公式流程：
        1. a = 当前组合概念激活（combo_h）
        2. b = 语法预测的下一个组合概念（grammar_w × prev_combo_h）
        3. Δ = b - a（语法预测误差）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 语法权重更新
        6. τ读⊗→浮出所见 = 语法预测
        7. (a, b)⟲⊗ = 循环

        参数：
          combo_h: 组合概念激活（n_combo维）
          meta_combo_h: 高级组合概念激活（n_meta维，可选）

        返回：
          dict包含：
            grammar_pred: 语法预测的下一个组合概念
            meta_grammar_pred: 高级语法预测
            语法预测误差: 当前拍语法预测误差
            正在生成: 是否正在生成语言序列
            生成序列长度: 当前生成序列的长度
        """
        self.拍 += 1

        # ── 语法预测：从上一拍的组合概念预测当前组合概念 ──
        # 精卫公式：b = grammar_w × prev_combo_h（语法预测）
        # τ读⊗→浮出所见：语法预测从历史模式中浮现
        new_grammar_pred = [0.0] * self.n_combo
        for i in range(self.n_combo):
            if self.grammar_strength[i] < 0.05:
                continue
            g_input = 0.0
            for j in range(self.n_combo):
                g_input += self.grammar_w[i][j] * self.prev_combo_h[j]
            new_grammar_pred[i] = activation(g_input)

        # ── 语法预测误差：Δ = b - a ──
        # 精卫公式：Δ = grammar_pred - combo_h
        # 语法预测误差驱动语法学习——预测准确的语法模式被强化
        grammar_delta = [new_grammar_pred[i] - combo_h[i]
                         for i in range(self.n_combo)]
        grammar_sigma = [new_grammar_pred[i] + combo_h[i]
                         for i in range(self.n_combo)]
        grammar_otimes = [grammar_delta[i] * grammar_sigma[i]
                          for i in range(self.n_combo)]

        # 当前拍语法预测误差（均方误差）
        current_mse = sum(d * d for d in grammar_delta) / max(1, self.n_combo)
        self.语法预测误差 = self.语法预测误差 * 0.95 + current_mse * 0.05

        # ── (a, b) ⟲ ⊗ —— 语法权重更新 ──
        # 赫布学习：上一拍的组合概念和当前组合概念同时出现→转移权重增强
        # 精卫公式：⊗ = Δ × Σ → grammar_w更新
        for i in range(self.n_combo):
            if self.grammar_strength[i] < 0.05:
                continue
            for j in range(self.n_combo):
                dw = self.lr * combo_h[i] * self.prev_combo_h[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.grammar_w[i][j] = self.grammar_w[i][j] * 0.99 + dw
            self.语法学习次数 += 1

        # ── 语法强度（用进废退） ──
        for i in range(self.n_combo):
            if abs(combo_h[i]) > 0.15:
                self.grammar_strength[i] = min(3.0, self.grammar_strength[i] + 0.01)
            else:
                self.grammar_strength[i] *= 0.999

        # ── 突触缩放 ──
        for i in range(self.n_combo):
            if self.grammar_strength[i] < 0.05:
                continue
            w_norm = sum(abs(self.grammar_w[i][j]) for j in range(self.n_combo))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_combo):
                    self.grammar_w[i][j] *= scale

        # ── 高级语法：高级组合概念→高级组合概念的转移 ──
        if meta_combo_h is not None:
            new_meta_pred = [0.0] * self.n_meta
            for i in range(self.n_meta):
                if self.meta_grammar_strength[i] < 0.05:
                    continue
                g_input = 0.0
                for j in range(self.n_meta):
                    g_input += self.meta_grammar_w[i][j] * self.prev_meta_combo_h[j]
                new_meta_pred[i] = activation(g_input)

            # 高级语法学习
            for i in range(self.n_meta):
                if self.meta_grammar_strength[i] < 0.05:
                    continue
                for j in range(self.n_meta):
                    dw = self.lr * meta_combo_h[i] * self.prev_meta_combo_h[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.meta_grammar_w[i][j] = self.meta_grammar_w[i][j] * 0.99 + dw

            # 高级语法强度
            for i in range(self.n_meta):
                if abs(meta_combo_h[i]) > 0.15:
                    self.meta_grammar_strength[i] = min(3.0, self.meta_grammar_strength[i] + 0.01)
                else:
                    self.meta_grammar_strength[i] *= 0.999

            self.meta_grammar_pred = new_meta_pred
            self.prev_meta_combo_h = meta_combo_h[:]
        else:
            new_meta_pred = self.meta_grammar_pred

        # ── 语言序列生成 ──
        # 当组合概念活跃度足够高时，启动语言序列生成
        # 生成过程：当前组合概念→语法预测下一个→生成→继续
        combo_active = sum(1 for v in combo_h if abs(v) > 0.15)
        if combo_active >= 2 and not self.正在生成:
            self.正在生成 = True
            self.生成步 = 0
            self.生成序列 = []

        if self.正在生成:
            # 将当前组合概念加入生成序列
            self.生成序列.append([v for v in combo_h])
            self.生成步 += 1
            if self.生成步 >= self.序列长度上限:
                self.正在生成 = False

        # 保存当前组合概念用于下一拍的语法预测
        self.prev_combo_h = combo_h[:]
        self.grammar_pred = new_grammar_pred

        return {
            "grammar_pred": self.grammar_pred,
            "meta_grammar_pred": new_meta_pred,
            "语法预测误差": round(self.语法预测误差, 4),
            "语法学习次数": self.语法学习次数,
            "正在生成": self.正在生成,
            "生成序列长度": len(self.生成序列),
            "活跃语法数": sum(1 for s in self.grammar_strength if s > 0.05),
            "活跃高级语法数": sum(1 for s in self.meta_grammar_strength if s > 0.05),
        }

    def 生成语言序列(self, combo_h, max_steps=5):
        """从当前组合概念激活生成语言序列。

        自回归式生成：当前组合概念→语法预测下一个→继续
        像人类说话——逐个生成，每一步依赖于前一步。

        精卫公式：
          a = 当前组合概念（已生成的）
          b = 语法预测的下一个组合概念
          Δ = b - a = 语法预测误差
          ⊗ = 累积语法历史 → 驱动下一步生成
          τ读⊗→浮出所见 = 语言序列从⊗中浮现
        """
        序列 = [list(combo_h)]
        current = list(combo_h)

        for step in range(max_steps):
            # 语法预测下一个组合概念
            next_pred = [0.0] * self.n_combo
            for i in range(self.n_combo):
                if self.grammar_strength[i] < 0.05:
                    continue
                g_input = 0.0
                for j in range(self.n_combo):
                    g_input += self.grammar_w[i][j] * current[j]
                next_pred[i] = activation(g_input)

            # 稀疏化——只保留最强的几个
            sorted_idx = sorted(range(self.n_combo),
                                key=lambda i: abs(next_pred[i]), reverse=True)
            top_k = set(sorted_idx[:max(1, self.n_combo // 4)])
            for i in range(self.n_combo):
                if i not in top_k:
                    next_pred[i] *= 0.1

            序列.append(next_pred)
            current = next_pred

            # 如果预测值都很小，停止生成
            if max(abs(v) for v in next_pred) < 0.05:
                break

        return 序列

    def 状态摘要(self):
        return {
            "活跃语法数": sum(1 for s in self.grammar_strength if s > 0.05),
            "活跃高级语法数": sum(1 for s in self.meta_grammar_strength if s > 0.05),
            "语法预测误差": round(self.语法预测误差, 4),
            "语法学习次数": self.语法学习次数,
            "正在生成": self.正在生成,
            "生成序列长度": len(self.生成序列),
            "grammar_w均值": round(sum(sum(row) for row in self.grammar_w) / max(1, self.n_combo * self.n_combo), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 语言中枢验证（语法自组织） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    lc = 语言中枢(n_combo=32, n_meta=16)

    print("初始状态:")
    print(f"  组合概念数: {lc.n_combo}")
    print(f"  高级组合概念数: {lc.n_meta}")
    print(f"  语法权重矩阵: grammar_w ({lc.n_combo}×{lc.n_combo})")

    print("\n--- 测试1: 语法预测从组合概念中浮现 ---")
    # 模拟两种不同的组合概念序列
    for _ in range(50):
        combo_h = [0.0] * 32
        combo_h[4] = 0.3
        combo_h[8] = 0.25
        result = lc.一拍(combo_h)
    print(f"  序列A后: 语法预测误差={result['语法预测误差']}, 学习次数={result['语法学习次数']}")
    print(f"  语法预测top5: {sorted(range(32), key=lambda i: abs(lc.grammar_pred[i]), reverse=True)[:5]}")

    lc2 = 语言中枢(n_combo=32, n_meta=16)
    for _ in range(50):
        combo_h = [0.0] * 32
        combo_h[15] = 0.3
        combo_h[0] = 0.25
        result = lc2.一拍(combo_h)
    print(f"  序列B后: 语法预测误差={result['语法预测误差']}, 学习次数={result['语法学习次数']}")
    print(f"  语法预测top5: {sorted(range(32), key=lambda i: abs(lc2.grammar_pred[i]), reverse=True)[:5]}")

    diff = sum(abs(lc.grammar_pred[i] - lc2.grammar_pred[i]) for i in range(32))
    print(f"  A vs B 语法预测差异: {diff:.4f}（应该 > 0——不同序列→不同语法预测）")

    print("\n--- 测试2: 语法区分度（独立实例） ---")
    patterns = [
        ("序列X", [0.3, 0.25, 0.0, 0.0, 0.0, 0.0] + [0.0] * 26),
        ("序列Y", [0.0, 0.0, 0.3, 0.25, 0.0, 0.0] + [0.0] * 26),
        ("均衡", [0.1] * 32),
    ]
    results = {}
    for name, pattern in patterns:
        lc_t = 语言中枢(n_combo=32, n_meta=16)
        for _ in range(50):
            lc_t.一拍(pattern)
        results[name] = list(lc_t.grammar_pred)
        active = [i for i, v in enumerate(lc_t.grammar_pred) if abs(v) > 0.05]
        print(f"  {name}: 活跃预测={active[:6]}, 预测最大={max(abs(v) for v in lc_t.grammar_pred):.4f}")

    for i in range(3):
        for j in range(i + 1, 3):
            n1, n2 = patterns[i][0], patterns[j][0]
            d = sum(abs(results[n1][k] - results[n2][k]) for k in range(32))
            print(f"  {n1} vs {n2} 语法差异: {d:.4f}")

    print("\n--- 测试3: 语言序列生成 ---")
    lc3 = 语言中枢(n_combo=32, n_meta=16)
    # 先学习100拍
    for _ in range(100):
        combo_h = [0.0] * 32
        combo_h[4] = 0.3
        combo_h[8] = 0.25
        lc3.一拍(combo_h)
    # 然后生成语言序列
    start_h = [0.0] * 32
    start_h[4] = 0.3
    start_h[8] = 0.25
    序列 = lc3.生成语言序列(start_h, max_steps=5)
    print(f"  生成序列长度: {len(序列)}")
    for i, step in enumerate(序列):
        active = [j for j, v in enumerate(step) if abs(v) > 0.05]
        print(f"  步{i}: 活跃={active[:5]}, 最大={max(abs(v) for v in step):.4f}")

    print("\n--- 测试4: 语法预测误差随学习降低 ---")
    lc4 = 语言中枢(n_combo=32, n_meta=16)
    errors = []
    for tick in range(200):
        combo_h = [0.0] * 32
        combo_h[4] = 0.3
        combo_h[8] = 0.25
        result = lc4.一拍(combo_h)
        if tick % 50 == 49:
            errors.append(result['语法预测误差'])
    print(f"  50拍: 误差={errors[0]:.4f}")
    print(f"  100拍: 误差={errors[1]:.4f}")
    print(f"  150拍: 误差={errors[2]:.4f}")
    print(f"  200拍: 误差={errors[3]:.4f}")
    print(f"  误差应该逐渐降低（语法学习→预测更准确）")

    print("\n--- 测试5: 语法用进废退 ---")
    lc5 = 语言中枢(n_combo=32, n_meta=16)
    for _ in range(200):
        combo_h = [0.0] * 32
        combo_h[4] = 0.3
        combo_h[8] = 0.25
        lc5.一拍(combo_h)
    print(f"  200拍后: 存活语法={sum(1 for s in lc5.grammar_strength if s > 0.05)}")
    print(f"  语法强度: min={min(lc5.grammar_strength):.3f}, max={max(lc5.grammar_strength):.3f}")

    print("\n--- 测试6: 高级语法（段落级） ---")
    lc6 = 语言中枢(n_combo=32, n_meta=16)
    for _ in range(100):
        combo_h = [0.0] * 32
        combo_h[4] = 0.3
        meta_h = [0.0] * 16
        meta_h[2] = 0.25
        lc6.一拍(combo_h, meta_combo_h=meta_h)
    print(f"  100拍后: 活跃高级语法={result['活跃高级语法数']}")
    print(f"  高级语法预测top3: {sorted(range(16), key=lambda i: abs(lc6.meta_grammar_pred[i]), reverse=True)[:3]}")

    print("\n--- 测试7: 精卫公式验证 ---")
    lc7 = 语言中枢(n_combo=4, n_meta=2)
    for tick in range(10):
        combo_h = [0.3, 0.5, 0.1, 0.2]
        result = lc7.一拍(combo_h)
        if tick < 3 or tick >= 8:
            print(f"  拍{tick + 1}: 语法预测={[round(v, 3) for v in result['grammar_pred']]}")
            print(f"         语法误差={result['语法预测误差']}")
    print(f"  τ(浮出): 语法预测均值={sum(lc7.grammar_pred) / 4:.4f}")
    print(f"  ⟲(循环): 拍={lc7.拍}")

    print("\n验证通过")
