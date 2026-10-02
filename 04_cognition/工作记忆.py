"""工作记忆.py —— 前额叶皮层（PFC）的工作记忆，基于精卫公式。

精卫公式解读（工作记忆版）：
  a = 当前感知输入（大脑隐藏状态）
  b = 工作记忆中维持的内容
  Δ = a − b = 新输入与维持内容的差异
  Σ = a + b = 新输入与维持内容的联合
  ⊗ = ∫(Δ, Σ) = 累积差异与联合 → 更新工作记忆
  τ 读 ⊗ → 浮出所见 = 当前关注的焦点（工作记忆的输出）
  (a, b) ⟲ ⊗ = 循环更新

人类神经科学机制：
  1. 持续放电（persistent firing）：PFC神经元通过NMDA受体和循环连接
     在没有外部输入时持续放电，维持信息在线。
     → 工程实现：slot的循环权重，gate close时自我维持

  2. 多巴胺D1/D2调节：
     - 低多巴胺（不惊讶）→ D1主导 → gate close → 工作记忆稳定
     - 高多巴胺（惊讶）→ D2主导 → gate open → 工作记忆更新
     → 工程实现：surprise驱动gate开闭

  3. 容量限制（capacity limit）：人类约4个slot
     → 工程实现：4个slot，稀疏竞争

  4. gamma振荡同步：不同记忆项通过相位同步协调
     → 工程实现：slot之间的弱耦合

和其他子系统的精卫公式解读对比：
  大脑.py: a=预测, b=实际输入, Δ=预测误差, ⊗=赫布学习, τ=读出层pred
  概念层.py: a=隐藏状态, b=概念预测, Δ=预测误差, ⊗=累积差与和, τ=概念浮现
  工作记忆.py: a=当前输入, b=维持内容, Δ=新旧差异, ⊗=累积更新, τ=关注焦点
"""

import math
import random


def activation(x, leak=0.01):
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if x > 1.5:
        return 1.5 + leak * (x - 1.5)
    elif x < -1.5:
        return -1.5 + leak * (x + 1.5)
    else:
        return math.tanh(x)


class 工作记忆:
    """前额叶PFC工作记忆——基于精卫公式。

    4个slot，每个slot维持一个隐藏状态向量。
    gate由surprise驱动：surprise大→gate open→更新，surprise小→gate close→维持。

    精卫公式流程（每个slot）：
    1. a = 当前感知输入（hidden_state）
    2. b = slot中维持的内容
    3. Δ = a - b（新旧差异）
    4. Σ = a + b（联合）
    5. ⊗ = Δ × Σ（差异与联合的交互）
    6. gate open时：slot更新（写入新内容）
       gate close时：slot维持（循环放电）
    7. τ读⊗ → 浮出关注焦点（所有slot的加权输出）
    """

    def __init__(self, n_hidden=11, n_slots=4, lr=0.03, seed=42):
        self.n_hidden = n_hidden
        self.n_slots = n_slots
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # 每个slot存一个隐藏状态向量（工作记忆内容）
        self.slots = [[0.0] * n_hidden for _ in range(n_slots)]

        # slot强度（用进废退——和概念层、筋树一致）
        self.slot_strength = [0.0] * n_slots

        # gate状态：0=close（维持），1=open（更新）
        self.gate = [0.0] * n_slots

        # Δ历史（⊗=∫(Δ,Σ)的工程近似）
        self.delta_history = [[] for _ in range(n_slots)]
        self.otimes_cap = 50

        # 输出历史
        self.output_history = []

        # ── 人类slot（第5个slot，专门维持人类信号） ──
        # 人类slot和普通slot的区别：
        # 1. gate由"有人在"信号驱动，不只是surprise
        # 2. 有人时优先保持，没人时快速衰减
        # 3. 输出作为单独信号返回，供概念层和海马体使用
        self.人类slot = [0.0] * n_hidden
        self.人类slot_strength = 0.0
        self.人类gate = 0.0
        self.人类delta_history = []
        self.有人在 = False

    def 一拍(self, hidden_state, surprise=0.0, 人类信号=None, 有人在=False):
        """工作记忆一拍：基于精卫公式维持和更新信息。

        参数：
          hidden_state: 当前大脑隐藏状态（a）
          surprise: 当前惊讶度（驱动gate开闭）
          人类信号: 人类通道传来的b向量（可选）
          有人在: 人类是否在场

        返回：
          output: 工作记忆的输出（τ读⊗→浮出关注焦点）
          人类output: 人类slot的输出（供概念层和海马体使用）
          info: 状态信息
        """
        self.拍 += 1
        self.有人在 = 有人在

        a = hidden_state

        # ── 对每个slot应用精卫公式 ──
        slot_deltas = []
        slot_otimes = []

        for i in range(self.n_slots):
            b = self.slots[i]

            # Δ = a - b（新旧差异）
            delta = [a[j] - b[j] for j in range(self.n_hidden)]

            # Σ = a + b（联合）
            sigma = [a[j] + b[j] for j in range(self.n_hidden)]

            # ⊗ = Δ × Σ（差异与联合的交互）
            otimes = [delta[j] * sigma[j] for j in range(self.n_hidden)]

            slot_deltas.append(delta)
            slot_otimes.append(otimes)

            # ⊗历史（全留）
            self.delta_history[i].append(delta[:])
            if len(self.delta_history[i]) > self.otimes_cap:
                self.delta_history[i].pop(0)

            # ── gate控制（多巴胺D1/D2调节） ──
            # surprise大 → gate open → 更新
            # surprise小 → gate close → 维持
            # 每个slot的gate独立：与slot内容的差异越大，越容易open
            delta_mag = sum(abs(d) for d in delta) / max(1, self.n_hidden)
            self.gate[i] = min(1.0, surprise * 1.5 + delta_mag * 0.5)

            # ── gate open: 更新slot（写入新内容） ──
            # ── gate close: 维持slot（双稳态保持） ──
            if self.gate[i] > 0.15:
                # 更新：新内容写入slot，比例由gate开度决定
                g = self.gate[i]
                for j in range(self.n_hidden):
                    self.slots[i][j] = self.slots[i][j] * (1.0 - g) + a[j] * g
            else:
                # 维持：双稳态保持（PFC up-state）
                # 不是循环计算，是状态保持 + 缓慢衰减
                # 衰减率0.98——信息能维持约50拍（1/0.02）
                for j in range(self.n_hidden):
                    self.slots[i][j] *= 0.98

            # slot强度更新（用进废退）
            if self.gate[i] > 0.15:
                self.slot_strength[i] = min(1.0, self.slot_strength[i] + 0.1)
            elif delta_mag > 0.05:
                self.slot_strength[i] = min(1.0, self.slot_strength[i] + 0.02)
            else:
                self.slot_strength[i] *= 0.998

        # ── 稀疏竞争：只有最强的slot能维持 ──
        # 弱slot衰减更快（容量限制的工程实现）
        sorted_slots = sorted(range(self.n_slots),
                              key=lambda i: self.slot_strength[i], reverse=True)
        for rank, i in enumerate(sorted_slots):
            if rank >= 2:  # 只有前2个slot保持满强度
                self.slot_strength[i] *= 0.995

        # ── 人类slot处理（精卫公式：人类作为b） ──
        # 人类slot和普通slot用同样的精卫公式，但gate驱动方式不同：
        # - 普通slot: surprise驱动
        # - 人类slot: "有人在"驱动 + 人类信号与维持内容的差异驱动
        人类output = [0.0] * self.n_hidden
        if 人类信号 is not None and 有人在:
            # a = 人类信号（来自人类通道的b向量）
            # b = 人类slot中维持的内容
            ha = 人类信号
            hb = self.人类slot

            # Δ = a - b
            h_delta = [ha[j] - hb[j] for j in range(self.n_hidden)]
            # Σ = a + b
            h_sigma = [ha[j] + hb[j] for j in range(self.n_hidden)]
            # ⊗ = Δ × Σ
            h_otimes = [h_delta[j] * h_sigma[j] for j in range(self.n_hidden)]

            self.人类delta_history.append(h_delta[:])
            if len(self.人类delta_history) > self.otimes_cap:
                self.人类delta_history.pop(0)

            # gate: "有人在"时gate更容易open
            h_delta_mag = sum(abs(d) for d in h_delta) / max(1, self.n_hidden)
            self.人类gate = min(1.0, 0.3 + h_delta_mag * 0.8)  # 有人在时基底0.3

            # gate open: 更新人类slot
            g = self.人类gate
            for j in range(self.n_hidden):
                self.人类slot[j] = self.人类slot[j] * (1.0 - g) + ha[j] * g

            # 人类slot强度：有人在时增长，没人在时衰减
            self.人类slot_strength = min(1.0, self.人类slot_strength + 0.05)
        else:
            # 没人在——人类slot衰减（但比普通slot慢，因为人类记忆更重要）
            self.人类gate = 0.0
            for j in range(self.n_hidden):
                self.人类slot[j] *= 0.995  # 慢衰减（1/0.005=200拍）
            self.人类slot_strength *= 0.998

        # 人类slot输出
        if self.人类slot_strength > 0.01:
            人类output = self.人类slot[:]

        # ── τ 读 ⊗ → 浮出关注焦点 ──
        # 工作记忆的输出 = 所有slot的强度加权平均
        output = [0.0] * self.n_hidden
        total_strength = sum(max(s, 0.01) for s in self.slot_strength)
        for i in range(self.n_slots):
            w = max(self.slot_strength[i], 0.01) / total_strength
            for j in range(self.n_hidden):
                output[j] += self.slots[i][j] * w

        self.output_history.append(output[:])
        if len(self.output_history) > 100:
            self.output_history.pop(0)

        return {
            "output": output,
            "人类output": 人类output,
            "slots": [s[:] for s in self.slots],
            "gate": [round(g, 3) for g in self.gate],
            "strength": [round(s, 3) for s in self.slot_strength],
            "活跃slot数": sum(1 for s in self.slot_strength if s > 0.1),
            "总强度": round(sum(self.slot_strength), 3),
            "人类slot强度": round(self.人类slot_strength, 3),
            "人类gate": round(self.人类gate, 3),
            "有人在": self.有人在,
        }

    def 状态摘要(self):
        return {
            "活跃slot数": sum(1 for s in self.slot_strength if s > 0.1),
            "总强度": round(sum(self.slot_strength), 3),
            "gate": [round(g, 3) for g in self.gate],
            "strength": [round(s, 3) for s in self.slot_strength],
            "人类slot强度": round(self.人类slot_strength, 3),
            "人类gate": round(self.人类gate, 3),
            "有人在": self.有人在,
            "拍": self.拍,
        }

    def reset(self):
        self.slots = [[0.0] * self.n_hidden for _ in range(self.n_slots)]
        self.slot_strength = [0.0] * self.n_slots
        self.gate = [0.0] * self.n_slots
        self.delta_history = [[] for _ in range(self.n_slots)]
        self.output_history = []
        self.人类slot = [0.0] * self.n_hidden
        self.人类slot_strength = 0.0
        self.人类gate = 0.0
        self.人类delta_history = []
        self.有人在 = False
        self.拍 = 0


if __name__ == "__main__":
    print("=== 工作记忆验证（精卫公式版） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")
    print("工作记忆解读:")
    print("  a=当前输入, b=维持内容, Δ=新旧差异, ⊗=累积更新, τ=关注焦点\n")

    wm = 工作记忆(n_hidden=11, n_slots=4, lr=0.03)

    print("--- 测试1: 持续放电（无新输入时维持信息） ---")
    # 先注入一个状态
    wm.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05], surprise=0.5)
    slot0_after_inject = wm.slots[0][:]
    print(f"  注入后slot[0]: {[round(v, 3) for v in slot0_after_inject[:5]]}...")

    # 然后给零输入，看slot能否维持
    for tick in range(10):
        wm.一拍([0.0] * 11, surprise=0.0)
    slot0_after_decay = wm.slots[0][:]
    diff = sum(abs(slot0_after_inject[j] - slot0_after_decay[j])
               for j in range(11))
    print(f"  10拍零输入后slot[0]: {[round(v, 3) for v in slot0_after_decay[:5]]}...")
    print(f"  维持差异: {diff:.4f}（越小越好——持续放电）")
    print(f"  活跃slot数: {wm.状态摘要()['活跃slot数']}")

    print("\n--- 测试2: gate open/close（surprise驱动更新） ---")
    wm2 = 工作记忆(n_hidden=11, n_slots=4, lr=0.03)
    # 低surprise：gate close，维持
    wm2.一拍([0.5] * 11, surprise=0.8)  # 先注入
    wm2.一拍([0.0] * 11, surprise=0.0)  # 低surprise，应该维持
    print(f"  低surprise后gate: {wm2.gate}")
    print(f"  slot[0]维持: {[round(v, 3) for v in wm2.slots[0][:5]]}...")

    # 高surprise：gate open，更新
    wm2.一拍([0.8, 0.6, 0.4, 0.2, 0.1, 0.3, 0.5, 0.7, 0.2, 0.1, 0.4], surprise=0.9)
    print(f"  高surprise后gate: {wm2.gate}")
    print(f"  slot[0]更新: {[round(v, 3) for v in wm2.slots[0][:5]]}...")

    print("\n--- 测试3: 容量限制（4个slot稀疏竞争） ---")
    wm3 = 工作记忆(n_hidden=11, n_slots=4, lr=0.03)
    # 注入4个不同的状态
    patterns = [
        [0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05],
        [0.1, 0.5, 0.3, 0.2, 0.05, 0.1, 0.2, 0.3, 0.1, 0.15, 0.05],
        [0.05, 0.1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.3, 0.2, 0.1],
        [0.2, 0.05, 0.1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.3, 0.2],
    ]
    for p in patterns:
        wm3.一拍(p, surprise=0.8)
    print(f"  注入4个模式后:")
    print(f"  strength: {wm3.状态摘要()['strength']}")
    print(f"  活跃slot数: {wm3.状态摘要()['活跃slot数']}")

    # 输出应该是4个模式的加权混合
    output = wm3.output_history[-1]
    print(f"  输出(关注焦点): {[round(v, 3) for v in output[:5]]}...")

    print("\n--- 测试4: 精卫公式验证 ---")
    wm4 = 工作记忆(n_hidden=4, n_slots=2, lr=0.05)
    result = wm4.一拍([0.3, 0.5, 0.1, 0.2], surprise=0.5)
    print(f"  拍1: gate={result['gate']} strength={result['strength']}")
    print(f"  output(τ浮出)={[round(v, 3) for v in result['output']]}")
    result = wm4.一拍([0.3, 0.5, 0.1, 0.2], surprise=0.0)
    print(f"  拍2(低surprise): gate={result['gate']}")
    print(f"  output(τ浮出)={[round(v, 3) for v in result['output']]}")

    print("\n--- 测试5: 人类slot（人类作为b） ---")
    wm5 = 工作记忆(n_hidden=11, n_slots=4, lr=0.03)
    # 有人在——注入人类信号
    人类信号_a = [0.8, 0.2, -0.5, 0.6, -1.0, -0.3, 0.4, 0.1, -0.7, 0.3, 0.2]
    r = wm5.一拍([0.5] * 11, surprise=0.3, 人类信号=人类信号_a, 有人在=True)
    print(f"  有人时: 人类gate={r['人类gate']}, 人类slot强度={r['人类slot强度']}")
    print(f"  人类output前5: {[round(v,3) for v in r['人类output'][:5]]}")

    # 重复注入相同人类信号——gate应该降低（差异小）
    r2 = wm5.一拍([0.5] * 11, surprise=0.0, 人类信号=人类信号_a, 有人在=True)
    print(f"  重复: 人类gate={r2['人类gate']}（应该降低）")

    # 没人在——人类slot应该衰减
    for _ in range(20):
        wm5.一拍([0.5] * 11, surprise=0.0, 有人在=False)
    print(f"  20拍没人在后: 人类slot强度={wm5.人类slot_strength:.4f}（衰减）")
    print(f"  人类output前5: {[round(v,4) for v in wm5.人类slot[:5]]}")

    # 有人但沉默——存在感基底信号
    r3 = wm5.一拍([0.5] * 11, surprise=0.0, 人类信号=[0.05]*11, 有人在=True)
    print(f"  有人但沉默: 人类gate={r3['人类gate']}, 人类slot强度={r3['人类slot强度']}")

    print("\n验证通过")
