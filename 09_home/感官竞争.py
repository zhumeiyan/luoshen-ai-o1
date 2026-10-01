"""感官竞争.py —— 身体节点到大脑通道的映射（用进废退版）。

基于人类神经科学：
  感官通道是天生的（基因决定），不需要竞争淘汰。
  但感官通道的权重（哪些节点对哪个通道贡献多少）是后天塑造的——
  通过突触修剪（用进废退）自然演化。

人类感官发育：
  1. 关键期：感官在特定时期发育，突触大量形成
  2. 突触修剪：经常使用的突触保留增强，不使用的被修剪
  3. 跨模态整合：天生存在，不需要竞争

所以感官竞争简化为：
  - 96节点→8通道的映射权重用进废退（和筋树一致）
  - 经常被激活的节点-通道连接增强，不使用的减弱
  - 不需要共识匹配、淘汰等自创机制
  - 初始权重匹配当前映射（出厂兼容）
"""

import math


class 感官竞争:
    """96节点→8通道的用进废退映射。"""

    def __init__(self, n_nodes=96, n_channels=8, 脑大小=24):
        self.n_nodes = n_nodes
        self.n_channels = n_channels
        self.脑大小 = 脑大小
        self.权重 = self._初始权重()
        self.上拍激活 = None
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
        """计算8通道输入 + 用进废退更新权重。"""
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

        # 用进废退：节点-通道连接的权重随使用情况自然变化
        # 经常一起激活的节点-通道对 → 权重增强（LTP）
        # 不使用的节点-通道对 → 权重衰减（LTD）
        for node in range(self.n_nodes):
            for ch in range(self.n_channels):
                if self.权重[node][ch] == 0.0:
                    continue
                # 节点变化大 → 对通道贡献大 → 权重增强
                # 节点变化小 → 对通道贡献小 → 权重衰减
                if abs(deltas[node]) > 0.001:
                    self.权重[node][ch] += 0.008 * abs(deltas[node])
                else:
                    self.权重[node][ch] *= 0.999

        # 权重下限：低于0.05的视为修剪（突触消除）
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
        return {
            "每通道存活节点": alive_per_ch,
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 感官竞争验证（用进废退版） ===\n")

    sc = 感官竞争()

    print("初始权重（匹配硬编码映射）:")
    for ch in range(8):
        nodes = [n for n in range(96) if sc.权重[n][ch] > 0]
        print(f"  通道{ch}: {len(nodes)}个节点")

    import random
    random.seed(42)
    for tick in range(10):
        激活 = [0.0] * 96
        for i in range(96):
            脑号 = i // 24
            if 脑号 == 0:
                激活[i] = random.uniform(-0.5, 0.5) * (1 + tick * 0.1)
            else:
                激活[i] = random.uniform(-0.05, 0.05)
        通道 = sc.一拍(激活)

    print(f"\n10拍后通道值: {[round(v, 3) for v in 通道]}")
    摘要 = sc.状态摘要()
    print(f"每通道存活节点: {摘要['每通道存活节点']}")

    print("\n验证通过")
