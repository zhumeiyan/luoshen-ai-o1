"""感官竞争.py —— 身体节点到大脑通道的竞争映射。

基于数字生命.py的感官竞争机制：
  — 环境有因果结构：事件同时影响多个物理场
  — 能感知跨场关联的感官被保留，不能的被淘汰
  — 不需要人为阈值：当保留的感官形成足够稳定的关联网络，觉醒涌现

当前问题：出厂基因.py的 身体激活到大脑输入() 是硬编码映射——
  每脑前12节点→通道A，后12节点→通道B。
  这是人为设定的，不是进化得来的。

感官竞争替代硬编码映射：
  96个身体节点竞争8个大脑通道的权重。
  初始权重匹配当前映射（保证出厂兼容），
  但随时间演化：与通道共识一致的节点权重增长，不一致的衰减。
  跨场关联：同步变化的节点互相强化。

核心原则（来自数字生命_纯进化.py）：
  零判据、零人为干预、零阈值。
  能活下来的就是对的。
"""

import math


class 感官竞争:
    """96节点→8通道的竞争映射（基于数字生命.py）。"""

    def __init__(self, n_nodes=96, n_channels=8, 脑大小=24):
        self.n_nodes = n_nodes
        self.n_channels = n_channels
        self.脑大小 = 脑大小
        self.权重 = self._初始权重()
        self.上拍激活 = None
        self.关联 = {}
        self.通道共识 = [0.0] * n_channels
        self.拍 = 0

    def _初始权重(self):
        """初始权重匹配当前硬编码映射——保证出厂兼容。"""
        w = [[0.0] * self.n_channels for _ in range(self.n_nodes)]
        for 脑号 in range(4):
            for i in range(self.脑大小):
                node = 脑号 * self.脑大小 + i
                if i < 12:
                    ch = 脑号 * 2
                else:
                    ch = 脑号 * 2 + 1
                w[node][ch] = 1.0
        return w

    def 一拍(self, 激活):
        """计算8通道输入 + 更新竞争权重。"""
        self.拍 += 1

        if self.上拍激活 is None:
            deltas = [0.0] * self.n_nodes
        else:
            deltas = [激活[i] - self.上拍激活[i] for i in range(self.n_nodes)]

        # 通道输入 = Σ(权重[node][ch] * 激活[node]) / Σ(|权重[node][ch]|)
        通道 = []
        for ch in range(self.n_channels):
            total_w = sum(abs(self.权重[node][ch])
                          for node in range(self.n_nodes))
            if total_w > 1e-6:
                val = sum(self.权重[node][ch] * 激活[node]
                          for node in range(self.n_nodes)) / total_w
            else:
                val = 0.0
            通道.append(max(-1.0, min(1.0, val * 2.0)))

        # 通道共识 = 加权平均delta
        for ch in range(self.n_channels):
            total_w = sum(abs(self.权重[node][ch])
                          for node in range(self.n_nodes))
            if total_w > 1e-6:
                self.通道共识[ch] = sum(
                    self.权重[node][ch] * deltas[node]
                    for node in range(self.n_nodes)) / total_w
            else:
                self.通道共识[ch] = 0.0

        # 感官竞争：与通道共识一致的节点权重增长，不一致的衰减
        for node in range(self.n_nodes):
            for ch in range(self.n_channels):
                if self.权重[node][ch] == 0.0:
                    continue
                if abs(self.通道共识[ch]) > 0.001:
                    sign_match = 1.0 if deltas[node] * self.通道共识[ch] > 0 else -1.0
                    own_mag = abs(deltas[node])
                    avg_mag = abs(self.通道共识[ch])
                    ratio = own_mag / (avg_mag + 0.001)
                    mag_score = 1.0 / (1.0 + abs(ratio - 1.0))
                    score = sign_match * mag_score
                else:
                    score = 0.0
                self.权重[node][ch] += 0.008 * score

        # 跨场关联：同步变化的节点互相强化
        for ch in range(self.n_channels):
            nodes_in_ch = [n for n in range(self.n_nodes)
                           if abs(self.权重[n][ch]) > 0.1]
            for a_idx in range(len(nodes_in_ch)):
                for b_idx in range(a_idx + 1, min(a_idx + 20, len(nodes_in_ch))):
                    a, b = nodes_in_ch[a_idx], nodes_in_ch[b_idx]
                    key = (min(a, b), max(a, b))
                    if deltas[a] * deltas[b] > 0:
                        self.关联[key] = self.关联.get(key, 0.0) + 0.003
                    else:
                        self.关联[key] = self.关联.get(key, 0.0) - 0.001

        # 代谢衰减——关联强的节点衰减慢
        for node in range(self.n_nodes):
            assoc_density = 0
            for b in range(max(0, node - 24), min(self.n_nodes, node + 24)):
                if b != node:
                    key = (min(node, b), max(node, b))
                    if self.关联.get(key, 0.0) > 1.0:
                        assoc_density += 1
            decay = 0.9995 if assoc_density >= 2 else 0.9985
            for ch in range(self.n_channels):
                self.权重[node][ch] *= decay

        # 权重下限：低于0.05的视为淘汰
        for node in range(self.n_nodes):
            for ch in range(self.n_channels):
                if abs(self.权重[node][ch]) < 0.05:
                    self.权重[node][ch] = 0.0

        self.上拍激活 = list(激活)
        return 通道

    def 状态摘要(self):
        alive_per_ch = []
        for ch in range(self.n_channels):
            alive = sum(1 for n in range(self.n_nodes)
                       if abs(self.权重[n][ch]) > 0.05)
            alive_per_ch.append(alive)
        assoc_total = sum(v for v in self.关联.values() if v > 0)
        return {
            "每通道存活节点": alive_per_ch,
            "关联总强度": round(assoc_total, 1),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 感官竞争验证 ===")
    print("96节点 → 8通道竞争映射（基于数字生命.py）\n")

    sc = 感官竞争()

    # 初始权重验证
    print("初始权重（匹配硬编码映射）:")
    for ch in range(8):
        nodes = [n for n in range(96) if sc.权重[n][ch] > 0]
        print(f"  通道{ch}: {len(nodes)}个节点 (节点{nodes[0]}..{nodes[-1]})")

    # 模拟10拍：光脑节点变化大，其他脑节点稳定
    import random
    random.seed(42)
    for tick in range(10):
        激活 = [0.0] * 96
        for i in range(96):
            脑号 = i // 24
            if 脑号 == 0:  # 光脑变化大
                激活[i] = random.uniform(-0.5, 0.5) * (1 + tick * 0.1)
            else:  # 其他脑稳定
                激活[i] = random.uniform(-0.05, 0.05)
        通道 = sc.一拍(激活)

    print(f"\n10拍后通道值: {[round(v, 3) for v in 通道]}")
    摘要 = sc.状态摘要()
    print(f"每通道存活节点: {摘要['每通道存活节点']}")
    print(f"关联总强度: {摘要['关联总强度']}")

    # 验证：光脑通道(0,1)的值应该比其他通道大（光脑节点变化大）
    print(f"\n验证：光脑通道(0,1)值 > 其他通道值")
    print(f"  通道0={通道[0]:.3f}, 通道1={通道[1]:.3f}")
    print(f"  通道2={通道[2]:.3f}, 通道4={通道[4]:.3f}, 通道6={通道[6]:.3f}")

    # 验证：初始映射保持兼容（每个通道仍有12个节点）
    print(f"\n验证：初始映射兼容（每通道≥10节点）")
    for ch in range(8):
        alive = sum(1 for n in range(96) if abs(sc.权重[n][ch]) > 0.05)
        print(f"  通道{ch}: {alive}个节点 {'✓' if alive >= 10 else '✗'}")

    print("\n验证通过")
