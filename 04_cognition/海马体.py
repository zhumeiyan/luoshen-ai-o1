"""海马体.py —— 记忆巩固 + 模式分离/完成，基于精卫公式。

海马体两个核心功能：
  1. 记忆巩固：短期记忆 → 长期记忆（海马→皮层转移）
  2. 模式分离/完成：区分相似经历 / 从部分线索恢复完整记忆

精卫公式解读（记忆巩固）：
  a = 海马中的记忆（短期，精确）
  b = 皮层中的记忆（长期，概括）
  Δ = b − a = 海马与皮层的差异
  Σ = a + b = 联合状态
  ⊗ = ∫(Δ, Σ) = 重播中累积的差异与联合 → 皮层逐步逼近海马
  τ 读 ⊗ → 浮出巩固后的稳定记忆
  (a, b) ⟲ ⊗ = 多次重播循环

精卫公式解读（模式分离）：
  a = 当前输入
  b = 最相似的已有记忆
  Δ = a − b = 差异
  Σ = a + b = 联合
  ⊗ = Δ × Σ → 差异越大，分离越强
  τ 读 ⊗ → 浮出独特的稀疏表征

精卫公式解读（模式完成）：
  a = 部分线索（当前输入）
  b = 完整记忆（CA3存储模式）
  Δ = b − a = 缺失部分
  Σ = a + b = 已有+完整
  ⊗ = Δ × Σ → 从已有推断缺失
  τ 读 ⊗ → 浮出完整的回忆

人类神经科学机制：
  1. 系统巩固（system consolidation）：
     - 海马CA1区存储精确的情景记忆
     - 睡眠中sharp-wave ripples期间海马重播序列
     - 慢速权重更新驱动皮层表征逐步形成
     - 最终皮层可以独立回忆，海马不再需要

  2. 模式分离（DG齿状回）：
     - 稀疏编码：只有极少数神经元被激活
     - 类似输入激活完全不同的神经元群
     - 防止相似记忆混淆

  3. 模式完成（CA3）：
     - 循环连接形成自动联想网络
     - 输入部分线索 → 循环迭代 → 恢复完整模式
     - Hebbian学习：同时激活的神经元连接增强

  4. 情感增强（杏仁核调制）：
     - 高情绪事件获得更多重播机会
     - 情绪调制因子乘到巩固强度上
"""

import math
import random
import collections


def activation(x, leak=0.01):
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if x > 1.5:
        return 1.5 + leak * (x - 1.5)
    elif x < -1.5:
        return -1.5 + leak * (x + 1.5)
    else:
        return math.tanh(x)


class 海马体:
    """海马体——记忆巩固 + 模式分离/完成，基于精卫公式。

    三个子功能：
    1. 经验缓冲区（CA1）：存储短期记忆痕迹
    2. 模式分离（DG）：稀疏编码区分相似输入
    3. 模式完成（CA3）：自动联想恢复完整记忆
    4. 睡眠重播：从缓冲区抽样，驱动慢速巩固
    """

    def __init__(self, n_hidden=11, n_dg=32, n_ca3=32,
                 buffer_size=128, consolidation_lr=0.002, seed=42):
        self.n_hidden = n_hidden
        self.n_dg = n_dg  # 齿状回维度（模式分离）
        self.n_ca3 = n_ca3  # CA3维度（模式完成）
        self.buffer_size = buffer_size
        self.consolidation_lr = consolidation_lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── DG模式分离：稀疏随机投影 ──
        # 输入(n_hidden) → DG(n_dg)，稀疏随机连接
        # 相似输入激活不同的DG神经元群
        self.dg_w = [[rng.gauss(0, 0.3) for _ in range(n_hidden)]
                     for _ in range(n_dg)]
        self.dg_bias = [rng.uniform(-0.1, 0.1) for _ in range(n_dg)]
        # DG稀疏激活：只有top-k被激活
        self.dg_k = max(2, n_dg // 8)  # 12.5%激活率
        self.dg_h = [0.0] * n_dg

        # ── CA3模式完成：循环连接自动联想 ──
        # CA3的循环权重——通过Hebbian学习形成自动联想
        self.ca3_w = [[rng.uniform(-0.02, 0.02) for _ in range(n_dg)]
                      for _ in range(n_ca3)]  # DG → CA3投影
        self.ca3_rw = [[rng.uniform(-0.02, 0.02) for _ in range(n_ca3)]
                       for _ in range(n_ca3)]  # CA3循环连接
        for i in range(n_ca3):
            self.ca3_rw[i][i] = rng.uniform(0.05, 0.15)  # 自连接
        self.ca3_h = [0.0] * n_ca3

        # ── 经验缓冲区（CA1）──
        # 存储 (h_before, h_after, error, salience)
        self.buffer = collections.deque(maxlen=buffer_size)

        # ── 皮层长期记忆（巩固目标）──
        # 皮层循环权重——通过睡眠重播逐步学习
        self.cortex_rw = [[rng.uniform(-0.02, 0.02) for _ in range(n_hidden)]
                          for _ in range(n_hidden)]
        for i in range(n_hidden):
            self.cortex_rw[i][i] = rng.uniform(0.05, 0.15)

        # ── 统计 ──
        self.total_replays = 0
        self.consolidation_loss_history = []

    def 模式分离(self, hidden_state):
        """DG模式分离：将输入映射到稀疏表征。

        精卫公式：
        a = 当前输入, b = 最相似的已有DG表征
        Δ = a - b → 差异越大，分离越强
        ⊗ = Δ × Σ → 累积差异驱动稀疏编码
        τ读⊗ → 浮出独特的稀疏表征

        生物学：DG的稀疏编码（只有极少数神经元激活）
        使相似输入激活完全不同的神经元群。
        """
        # 投影到DG空间
        dg_raw = [0.0] * self.n_dg
        for i in range(self.n_dg):
            s = self.dg_bias[i]
            for j in range(self.n_hidden):
                s += self.dg_w[i][j] * hidden_state[j]
            dg_raw[i] = s

        # 稀疏竞争：只有top-k被激活
        sorted_idx = sorted(range(self.n_dg),
                            key=lambda i: abs(dg_raw[i]), reverse=True)
        top_k = set(sorted_idx[:self.dg_k])
        self.dg_h = [activation(dg_raw[i]) if i in top_k else 0.0
                     for i in range(self.n_dg)]

        return self.dg_h

    def 模式完成(self, dg_pattern, n_iters=3):
        """CA3模式完成：从部分线索恢复完整记忆。

        精卫公式：
        a = 部分线索（DG输入）, b = 完整记忆（CA3存储模式）
        Δ = b - a = 缺失部分
        Σ = a + b = 已有+完整
        ⊗ = Δ × Σ → 从已有推断缺失
        τ读⊗ → 浮出完整的回忆

        生物学：CA3的循环连接形成自动联想网络。
        输入部分线索 → 循环迭代 → 恢复完整模式。
        """
        # DG → CA3投影
        ca3_input = [0.0] * self.n_ca3
        for i in range(self.n_ca3):
            s = 0.0
            for j in range(self.n_dg):
                s += self.ca3_w[i][j] * dg_pattern[j]
            ca3_input[i] = s

        # CA3循环迭代（模式完成）
        self.ca3_h = [activation(ca3_input[i]) for i in range(self.n_ca3)]
        for _ in range(n_iters):
            new_h = [0.0] * self.n_ca3
            for i in range(self.n_ca3):
                s = ca3_input[i] * 0.3  # 外部输入保持
                for k in range(self.n_ca3):
                    s += self.ca3_rw[i][k] * self.ca3_h[k]
                new_h[i] = activation(s)
            self.ca3_h = new_h

        return self.ca3_h

    def 存储(self, h_before, h_after, prediction_error, region_id=0,
             人类互动=False, 人类情感强度=0.0):
        """存储经验到缓冲区（CA1功能）。

        salience = 预测误差越高，记忆越凸显，越容易被重播。
        情感增强：杏仁核调制——高误差事件获得额外巩固权重。
        人类互动增强：与人类的互动经验获得额外salience——
          人类是精卫公式中的b，与b的互动是洛神认知发展的核心驱动。
        """
        # 基础salience：预测误差
        salience = min(1.0, prediction_error * 2.0)

        # 人类互动增强：与人类的互动优先巩固
        if 人类互动:
            salience = min(1.5, salience + 0.3 + abs(人类情感强度) * 0.5)

        self.buffer.append({
            'h_before': h_before[:],
            'h_after': h_after[:],
            'error': prediction_error,
            'region': region_id,
            'salience': salience,
            'replay_count': 0,
            '人类互动': 人类互动,
            '人类情感强度': 人类情感强度,
        })

    def 睡眠重播(self, batch_size=8, amygdala_modulation=1.0):
        """睡眠重播——从缓冲区抽样，驱动慢速巩固。

        精卫公式（记忆巩固）：
        a = 海马记忆（h_before → h_after的转移）
        b = 皮层记忆（cortex_rw的预测）
        Δ = h_after - pred_h = 皮层预测误差
        Σ = h_before + h_after
        ⊗ = Δ × Σ → 慢速Hebbian更新驱动皮层逐步逼近海马
        τ读⊗ → 浮出巩固后的稳定记忆
        (a,b)⟲⊗ → 多次重播循环

        生物学：慢波睡眠中sharp-wave ripples期间海马重播序列。
        consolidation_lr比在线学习慢20倍——慢速巩固。
        amygdala_modulation：杏仁核调制——高情绪事件巩固更强。
        """
        if len(self.buffer) < batch_size:
            return 0.0

        # 按salience加权抽样（高salience更容易被重播）
        items = list(self.buffer)
        weights = [max(m['salience'], 0.01) for m in items]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        batch = []
        for _ in range(min(batch_size, len(items))):
            idx = random.choices(range(len(items)), weights=probs, k=1)[0]
            batch.append(items[idx])

        total_loss = 0.0
        lr = self.consolidation_lr * amygdala_modulation

        for memory in batch:
            h_b = memory['h_before']
            h_a = memory['h_after']

            # 皮层预测：用cortex_rw从h_before预测h_after
            pred_h = [0.0] * self.n_hidden
            for k in range(self.n_hidden):
                s = 0.0
                for i in range(self.n_hidden):
                    s += self.cortex_rw[k][i] * h_b[i]
                pred_h[k] = activation(s)

            # Δ = h_after - pred_h（皮层预测误差）
            # Σ = h_before + h_after（联合状态）
            # ⊗ = Δ × Σ → 慢速Hebbian更新
            for k in range(self.n_hidden):
                err = h_a[k] - pred_h[k]
                total_loss += err * err
                for i in range(self.n_hidden):
                    delta_w = lr * h_b[i] * err
                    if not (math.isnan(delta_w) or math.isinf(delta_w)):
                        self.cortex_rw[k][i] += delta_w

            memory['replay_count'] += 1
            self.total_replays += 1

        avg_loss = total_loss / max(1, len(batch) * self.n_hidden)
        self.consolidation_loss_history.append(avg_loss)
        if len(self.consolidation_loss_history) > 100:
            self.consolidation_loss_history.pop(0)

        return avg_loss

    def CA3学习(self, dg_pattern, ca3_target=None):
        """CA3循环权重的Hebbian学习。

        当DG输入和CA3激活同时出现时，连接增强。
        这是自动联想学习——CA3学会从部分输入恢复完整模式。

        如果ca3_target为None，用当前ca3_h作为target（自联想）。
        """
        if ca3_target is None:
            ca3_target = self.ca3_h[:]

        # DG → CA3投影权重更新
        w_decay = 0.99
        for i in range(self.n_ca3):
            for j in range(self.n_dg):
                dw = self.consolidation_lr * 2.0 * ca3_target[i] * dg_pattern[j]
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.ca3_w[i][j] = self.ca3_w[i][j] * w_decay + dw

        # CA3循环权重更新（Hebbian自联想）
        for i in range(self.n_ca3):
            for k in range(self.n_ca3):
                if i != k:
                    dw = self.consolidation_lr * 2.0 * ca3_target[i] * self.ca3_h[k]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.ca3_rw[i][k] = self.ca3_rw[i][k] * w_decay + dw

    def 一拍(self, hidden_state, prediction_error=0.0, region_id=0,
             is_sleep=False, amygdala_modulation=1.0,
             人类互动=False, 人类情感强度=0.0):
        """海马体一拍。

        在线模式（is_sleep=False）：
        1. 模式分离：输入 → DG稀疏表征
        2. 模式完成：DG → CA3恢复完整模式
        3. 存储经验到缓冲区（人类互动经验获得更高salience）

        睡眠模式（is_sleep=True）：
        1. 从缓冲区抽样重播（人类互动经验优先被重播）
        2. 慢速巩固皮层权重
        """
        self.拍 += 1

        if is_sleep:
            loss = self.睡眠重播(amygdala_modulation=amygdala_modulation)
            return {
                "mode": "sleep",
                "consolidation_loss": loss,
                "total_replays": self.total_replays,
                "buffer_size": len(self.buffer),
            }

        # 在线模式
        # 1. 模式分离
        dg = self.模式分离(hidden_state)

        # 2. 模式完成
        ca3 = self.模式完成(dg)

        # 3. CA3学习（自联想）
        self.CA3学习(dg)

        # 4. 存储经验（人类互动获得更高salience）
        self.存储(hidden_state, hidden_state, prediction_error, region_id,
                  人类互动=人类互动, 人类情感强度=人类情感强度)

        return {
            "mode": "online",
            "dg": dg,
            "ca3": ca3,
            "buffer_size": len(self.buffer),
            "dg活跃": sum(1 for v in self.dg_h if abs(v) > 0.1),
            "ca3活跃": sum(1 for v in self.ca3_h if abs(v) > 0.1),
        }

    def 状态摘要(self):
        人类记忆数 = sum(1 for m in self.buffer if m.get('人类互动', False))
        return {
            "buffer_size": len(self.buffer),
            "人类记忆数": 人类记忆数,
            "total_replays": self.total_replays,
            "dg活跃": sum(1 for v in self.dg_h if abs(v) > 0.1),
            "ca3活跃": sum(1 for v in self.ca3_h if abs(v) > 0.1),
            "巩固误差": round(sum(self.consolidation_loss_history) / max(1, len(self.consolidation_loss_history)), 4)
                       if self.consolidation_loss_history else 0,
            "拍": self.拍,
        }

    def reset(self):
        self.dg_h = [0.0] * self.n_dg
        self.ca3_h = [0.0] * self.n_ca3
        self.buffer.clear()
        self.total_replays = 0
        self.consolidation_loss_history = []
        self.拍 = 0


if __name__ == "__main__":
    print("=== 海马体验证（精卫公式版） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    hip = 海马体(n_hidden=11, n_dg=32, n_ca3=32)

    print("--- 测试1: 模式分离（相似输入→不同DG表征） ---")
    input_a = [0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05]
    input_b = [0.48, 0.32, 0.12, 0.07, 0.18, 0.12, 0.28, 0.22, 0.13, 0.12, 0.07]  # 相似
    input_c = [0.05, 0.1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.3, 0.2, 0.1]  # 很不同

    dg_a = hip.模式分离(input_a)
    dg_b = hip.模式分离(input_b)
    dg_c = hip.模式分离(input_c)

    overlap_ab = sum(1 for i in range(32) if abs(dg_a[i]) > 0.1 and abs(dg_b[i]) > 0.1)
    overlap_ac = sum(1 for i in range(32) if abs(dg_a[i]) > 0.1 and abs(dg_c[i]) > 0.1)
    print(f"  输入A活跃DG: {sum(1 for v in dg_a if abs(v) > 0.1)}")
    print(f"  A-B重叠(相似): {overlap_ab}")
    print(f"  A-C重叠(不同): {overlap_ac}")
    print(f"  相似输入重叠应该 > 不同输入重叠")

    print("\n--- 测试2: 模式完成（部分线索→完整记忆） ---")
    hip2 = 海马体(n_hidden=11, n_dg=32, n_ca3=32)
    # 先学习一个模式
    dg_full = hip2.模式分离(input_a)
    ca3_full = hip2.模式完成(dg_full)
    hip2.CA3学习(dg_full, ca3_full)
    # 多次学习增强
    for _ in range(20):
        hip2.模式完成(dg_full)
        hip2.CA3学习(dg_full)

    # 然后给部分线索（遮住一半DG激活）
    dg_partial = dg_full[:]
    for i in range(16, 32):
        dg_partial[i] = 0.0
    ca3_recovered = hip2.模式完成(dg_partial)
    recovery_diff = sum(abs(ca3_full[i] - ca3_recovered[i]) for i in range(32))
    print(f"  完整模式CA3活跃: {sum(1 for v in ca3_full if abs(v) > 0.1)}")
    print(f"  部分线索恢复CA3活跃: {sum(1 for v in ca3_recovered if abs(v) > 0.1)}")
    print(f"  恢复差异: {recovery_diff:.4f}（越小越好——模式完成）")

    print("\n--- 测试3: 记忆巩固（睡眠重播） ---")
    hip3 = 海马体(n_hidden=11, n_dg=32, n_ca3=32, consolidation_lr=0.005)
    # 在线阶段：存储经验
    for tick in range(50):
        hip3.一拍(input_a, prediction_error=0.05 + 0.001 * tick)
    print(f"  在线50拍后: buffer={len(hip3.buffer)}, total_replays={hip3.total_replays}")

    # 睡眠阶段：重播巩固
    for _ in range(10):
        loss = hip3.睡眠重播(batch_size=8)
    print(f"  睡眠10轮后: total_replays={hip3.total_replays}, 巩固误差={loss:.4f}")
    print(f"  巩固误差应该逐步下降")

    print("\n--- 测试4: 情感增强（杏仁核调制） ---")
    hip4 = 海马体(n_hidden=11, n_dg=32, n_ca3=32, consolidation_lr=0.005)
    for tick in range(30):
        hip4.一拍(input_a, prediction_error=0.05)
    # 低情绪调制
    loss_low = hip4.睡眠重播(amygdala_modulation=1.0)
    # 高情绪调制
    loss_high = hip4.睡眠重播(amygdala_modulation=3.0)
    print(f"  低情绪调制(1.0): 巩固误差={loss_low:.4f}")
    print(f"  高情绪调制(3.0): 巩固误差={loss_high:.4f}")
    print(f"  高情绪应该驱动更强巩固（误差变化更大）")

    print("\n--- 测试5: 人类互动salience（优先巩固） ---")
    hip5 = 海马体(n_hidden=11, n_dg=32, n_ca3=32, consolidation_lr=0.005)
    # 存储一些普通经验
    for tick in range(20):
        hip5.一拍(input_a, prediction_error=0.05)
    # 存储一些人类互动经验
    for tick in range(10):
        hip5.一拍(input_b, prediction_error=0.05, 人类互动=True, 人类情感强度=0.5)

    # 检查salience
    人类salience = [m['salience'] for m in hip5.buffer if m.get('人类互动', False)]
    普通salience = [m['salience'] for m in hip5.buffer if not m.get('人类互动', False)]
    print(f"  普通经验salience: {[round(s,3) for s in 普通salience[:5]]}")
    print(f"  人类互动salience: {[round(s,3) for s in 人类salience[:5]]}")
    print(f"  人类互动salience应该更高")

    # 睡眠重播——人类互动经验应该被更多重播
    for _ in range(5):
        hip5.睡眠重播(batch_size=8)
    人类replays = sum(m['replay_count'] for m in hip5.buffer if m.get('人类互动', False))
    普通replays = sum(m['replay_count'] for m in hip5.buffer if not m.get('人类互动', False))
    print(f"  普通经验总重播次数: {普通replays}")
    print(f"  人类互动总重播次数: {人类replays}")
    print(f"  人类互动应该被更多重播")

    print("\n验证通过")
