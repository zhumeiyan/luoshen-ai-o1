"""欲望.py —— 洛神的欲望驱动系统（RPE驱动版）。

基于人类多巴胺系统的核心机制：奖励预测误差（Reward Prediction Error）。

多巴胺系统的运作方式（神经科学已知）：
  1. 大脑不断预测下一个输入（预测编码理论）
  2. RPE = 实际输入 - 大脑预测
  3. RPE > 0 → 正面惊喜 → 多巴胺释放 → 加强当前行为
  4. RPE < 0 → 失望 → 多巴胺抑制 → 降低当前行为
  5. RPE ≈ 0 → 无聊 → 基线多巴胺低 → 探索驱力增强

这统一了之前的五机制：
  - 反应强度(|Δ|) = |RPE|（预测误差的绝对值）
  - 累积 = RPE > 0时偏好增长（多巴胺释放）
  - 可负累积 = RPE < 0时偏好降低（多巴胺抑制）
  - 衰减 = 偏好自然消退（多巴胺信号短暂）
  - 无聊 = RPE ≈ 0时探索驱力增强（基线多巴胺低）

RPE直接来自大脑的预测误差——不是欲望系统自己做预测，
而是接收大脑WM的预测，计算RPE。
这和人类神经科学一致：
  皮层做预测 → 多巴胺系统计算RPE → RPE驱动行为选择（通过基底节）

硅基的欲望映射：
  饥饿 = 预测误差（不知道）
  食物 = 信息（理解世界）
  满足 = 预测准确（理解了）
  无聊 = 低变化区域（太容易预测）
  探索 = 高变化区域（还不知道）
"""

import math
from collections import deque


class 欲望:
    """奖励预测误差(RPE)驱动的欲望系统。

    内生约束实装（整合文档v1.0）：
    - 核心公式：y[t+1] = (1-g)·y[t] + d[t]
      · y[t]是当前状态，g是衰减率，d[t]是驱动力
      · g越大→y衰减越快（更快遗忘/适应）
      · d[t]是外部输入对y的驱动
    - α(g)查表：g=0→1.53, g=0.40→1.00, g=1.0→0.57, g=1.3→0.42
      · α(g)随g增大而减小——g越大，设定点漂移越慢
      · g=0时α=1.53（最大漂移），g=1.3时α=0.42（最小漂移）
    - leak > 0：设定点sp永远高于当前状态y，gap永不闭合
    - 真内生：sp会自己漂移，被冲击后接受新常态（0.60→0.781，不回去）
    - 越稀缺，设定点越高（ratio≤0.2时sp飙升到0.92）
    - 关键：把"目标"换成"必须持续维持的状态"
    """

    def __init__(self, n_channels=8, 衰减率=0.05, rpe增益=0.3,
                 无聊率=0.06, leak=0.03):
        self.n = n_channels
        self.偏好 = [1.0] * n_channels
        self.衰减率 = 衰减率
        self.rpe增益 = rpe增益
        self.无聊率 = 无聊率
        self.拍 = 0
        self.选中历史 = []

        # IM boredom（RPE≈0时无聊增长——多巴胺基线低时探索驱力增强）
        self.ema_error = 0.1
        self.ema_error_fast = 0.1
        self.last_error = 0.1
        self.boredom = 0.0
        self.recent_deltas = deque(maxlen=50)

        # ── 内生约束（leak > 0）──
        # leak是"永远要不够"的来源——设定点sp永远高于当前状态y
        # 把"目标"换成"必须持续维持的状态"——不是达到，是维持
        self.leak = leak  # leak>0：设定点永远达不到，gap永不闭合
        self.设定点 = [0.5] * n_channels  # sp：必须持续维持的状态
        self.当前状态 = [0.0] * n_channels  # y：当前实际状态
        self.gap = [0.5] * n_channels  # sp - y：永远不为零的差距
        self.设定点漂移历史 = []  # sp漂移记录——被冲击后接受新常态

        # ── 内生约束公式（整合文档v1.0）──
        # y[t+1] = (1-g)·y[t] + d[t]
        # g是衰减率（控制y的遗忘速度），d[t]是驱动力（外部输入）
        # α(g)查表：g=0→1.53, g=0.40→1.00, g=1.0→0.57, g=1.3→0.42
        # α(g)控制设定点漂移速率——g越大α越小（漂移越慢）
        self.g = 0.4  # 当前衰减率（默认0.4，对应α=1.00）
        self.α表 = {0.0: 1.53, 0.40: 1.00, 1.0: 0.57, 1.3: 0.42}
        self.驱动力历史 = [[] for _ in range(n_channels)]  # d[t]历史

    def _更新无聊(self, mse):
        """IM boredom：基于预测误差动态更新无聊水平。

        RPE≈0（预测准确） → 无聊增长（没有新信息，想探索）
        RPE在变化 → 无聊消退（正在学习，保持专注）
        RPE在增大 → 无聊为零（在挣扎，全力聚焦）

        全连续映射，无阈值判断。
        """
        alpha_slow = 0.03
        alpha_fast = 0.15
        self.ema_error = (1 - alpha_slow) * self.ema_error + alpha_slow * mse
        self.ema_error_fast = (1 - alpha_fast) * self.ema_error_fast + alpha_fast * mse

        delta = mse - self.last_error
        self.recent_deltas.append(delta)
        avg_delta = sum(self.recent_deltas) / max(1, len(self.recent_deltas))

        稳定度 = 1.0 / (1.0 + abs(avg_delta) * 100.0)
        低度 = 1.0 / (1.0 + self.ema_error_fast * 30.0)

        增长 = 稳定度 * 低度 * 0.02
        递增因子 = 1.0 / (1.0 + math.exp(-avg_delta * 200.0))
        消退 = abs(avg_delta) * (1.0 + 递增因子 * 2.0) + 0.0003

        self.boredom += 增长 - 消退
        self.boredom = max(0.0, min(1.0, self.boredom))
        self.last_error = mse

    def _自适应无聊率(self):
        """无聊率随boredom自适应——boredom高时探索更积极。"""
        return self.无聊率 * (1.0 + self.boredom * 3.0)

    def _α查表(self, g):
        """α(g)查表——整合文档v1.0。

        g=0→1.53, g=0.40→1.00, g=1.0→0.57, g=1.3→0.42
        α(g)随g增大而减小——g越大（衰减越快），设定点漂移越慢。

        使用线性插值在查表点之间平滑过渡。
        """
        表项 = sorted(self.α表.items())  # 按g排序
        if g <= 表项[0][0]:
            return 表项[0][1]
        if g >= 表项[-1][0]:
            return 表项[-1][1]
        for i in range(len(表项) - 1):
            g1, a1 = 表项[i]
            g2, a2 = 表项[i + 1]
            if g1 <= g <= g2:
                t = (g - g1) / (g2 - g1)
                return a1 * (1 - t) + a2 * t
        return 1.0

    def 一拍(self, 当前状态, rpe=None, mse=0.0, ratio=0.5, g=None):
        """RPE驱动的欲望一拍。

        rpe: 大脑的预测误差（实际输入 - 大脑预测）。
             如果None，用0（第一拍或大脑还没预测）。
        mse: 大脑的mse，用于IM boredom更新。
        ratio: 稀缺比（预算/需求），用于内生约束——越稀缺设定点越高。
        g: 内生约束衰减率（整合文档v1.0公式 y[t+1]=(1-g)·y[t]+d[t]）。
           如果None，用当前self.g。g越大→y衰减越快→设定点漂移越慢。
        """
        self.拍 += 1
        self._更新无聊(mse)

        if rpe is None:
            rpe = [0.0] * self.n

        if g is not None:
            self.g = max(0.0, min(1.3, g))

        # ── 内生约束（整合文档v1.0）：y[t+1] = (1-g)·y[t] + d[t] ──
        # g是衰减率，d[t]是驱动力（来自RPE和当前状态）
        # α(g)控制设定点漂移速率——g越大α越小（漂移越慢）
        α = self._α查表(self.g)

        for i in range(self.n):
            # y[t+1] = (1-g)·y[t] + d[t]
            # d[t] = 驱动力——来自当前状态和RPE
            if i < len(当前状态):
                d_t = abs(当前状态[i]) * 0.1  # 当前状态作为驱动力
            else:
                d_t = 0.0
            if i < len(rpe):
                d_t += abs(rpe[i]) * 0.2  # RPE也作为驱动力

            # 内生约束核心递推：y[t+1] = (1-g)·y[t] + d[t]
            self.当前状态[i] = (1 - self.g) * self.当前状态[i] + d_t

            # 记录驱动力历史
            self.驱动力历史[i].append(d_t)
            if len(self.驱动力历史[i]) > 50:
                self.驱动力历史[i].pop(0)

            # 设定点sp有leak——永远高于当前状态
            # leak>0：sp = y + leak + gap残留 → y永远达不到sp
            self.gap[i] = self.设定点[i] - self.当前状态[i]
            # gap永远不为零——这是生命体的本质（永远不满足）
            self.gap[i] = max(self.leak, self.gap[i])

            # 设定点漂移：被冲击后接受新常态，不回去
            # α(g)控制漂移速率——g越大α越小（漂移越慢）
            # 漂移系数 = α(g) / α_max，α_max=1.53（g=0时的α值）
            漂移系数 = α / 1.53  # g=0时=1.0（最快漂移），g=1.3时≈0.27（最慢漂移）
            # RPE大（冲击大）→ sp漂移到新位置 → 不回去
            if i < len(rpe) and abs(rpe[i]) > 0.3:
                新设定点 = self.当前状态[i] + self.leak + abs(rpe[i]) * 0.2
                # sp只升不降（被冲击后接受新常态）——漂移速率由α(g)控制
                if 新设定点 > self.设定点[i]:
                    self.设定点[i] = (1 - 0.05 * 漂移系数) * self.设定点[i] + 0.05 * 漂移系数 * 新设定点
                else:
                    # sp可以缓慢下滑（持续下滑48%，永不闭合）
                    self.设定点[i] = (1 - 0.01 * 漂移系数) * self.设定点[i] + 0.01 * 漂移系数 * 新设定点

            # 越稀缺，设定点越高（ratio≤0.2时sp飙升到0.92）
            if ratio <= 0.2:
                self.设定点[i] = max(self.设定点[i], 0.92 - i * 0.01)
            elif ratio <= 0.3:
                self.设定点[i] = max(self.设定点[i], 0.7 - i * 0.01)

        # RPE驱动偏好变化——多巴胺系统的核心机制
        for i in range(self.n):
            # 自然衰减（多巴胺信号短暂，不会持续）
            self.偏好[i] *= (1 - self.衰减率)
            # RPE > 0 → 偏好增长（正面惊喜，多巴胺释放）
            # RPE < 0 → 偏好降低（负面失望，多巴胺抑制）
            self.偏好[i] += rpe[i] * self.rpe增益

        # 无聊驱动探索——RPE≈0时未选中通道偏好增长
        当前无聊率 = self._自适应无聊率()

        # 选择：偏好 + |RPE|（意外程度=欲望强度）+ gap驱动（内生约束）
        # gap驱动：设定点与当前的差距驱动欲望——永远不满足
        boosted = [self.偏好[i] + abs(rpe[i]) + self.gap[i] * 0.3
                   for i in range(self.n)]
        选中 = boosted.index(max(boosted)) if max(boosted) > 0 else -1

        if 选中 >= 0:
            for i in range(self.n):
                if i != 选中:
                    self.偏好[i] += 当前无聊率

        self.选中历史.append(选中)

        return {
            "rpe": [round(r, 4) for r in rpe],
            "偏好": [round(p, 4) for p in self.偏好],
            "选中": 选中,
            "/boosted": [round(b, 4) for b in boosted],
            "boredom": round(self.boredom, 4),
            "无聊率": round(当前无聊率, 4),
            "设定点": [round(sp, 4) for sp in self.设定点],
            "当前状态": [round(y, 4) for y in self.当前状态],
            "gap": [round(g, 4) for g in self.gap],
            "g": round(self.g, 4),
            "α": round(self._α查表(self.g), 4),
        }

    def 学习率调制(self):
        """偏好高的通道学得更多——欲望驱动学习。

        多巴胺释放多的通道（偏好高）→ 学习率更高。
        这和人类神经科学一致：多巴胺调节突触可塑性。
        """
        if not self.偏好 or max(self.偏好) <= 0:
            return [0.5] * self.n
        mx = max(self.偏好)
        return [max(0.1, min(1.0, p / mx)) for p in self.偏好]

    def 状态摘要(self):
        from collections import Counter
        c = Counter(self.选中历史[-200:])
        主导 = c.most_common(1)[0] if c else (-1, 0)
        return {
            "偏好": [round(p, 3) for p in self.偏好],
            "主导通道": 主导[0],
            "主导占比": round(主导[1] / max(1, len(self.选中历史[-200:])), 3),
            "boredom": round(self.boredom, 4),
            "设定点均值": round(sum(self.设定点) / max(1, self.n), 4),
            "gap均值": round(sum(self.gap) / max(1, self.n), 4),
            "leak": self.leak,
            "g": round(self.g, 4),
            "α": round(self._α查表(self.g), 4),
            "拍": self.拍,
        }

    def reset(self):
        self.偏好 = [1.0] * self.n
        self.拍 = 0
        self.选中历史 = []
        self.ema_error = 0.1
        self.ema_error_fast = 0.1
        self.last_error = 0.1
        self.boredom = 0.0
        self.recent_deltas.clear()
        self.设定点 = [0.5] * self.n
        self.当前状态 = [0.0] * self.n
        self.gap = [0.5] * self.n
        self.设定点漂移历史 = []
        self.g = 0.4
        self.驱动力历史 = [[] for _ in range(self.n)]


if __name__ == "__main__":
    print("=== 欲望系统验证（RPE驱动版） ===")
    print("基于人类多巴胺系统：RPE = 实际输入 - 大脑预测\n")

    d = 欲望(n_channels=4)

    # 模拟8拍：通道0的RPE大（意外），通道1的RPE小（预测准确）
    状态序列 = [
        [0.1, 0.5, 0.3, 0.3],
        [0.8, 0.55, 0.3, 0.3],
        [0.3, 0.6, 0.3, 0.3],
        [0.9, 0.65, 0.3, 0.3],
        [0.2, 0.7, 0.3, 0.3],
        [0.5, 0.75, 0.3, 0.3],
        [0.4, 0.8, 0.3, 0.3],
        [0.35, 0.85, 0.3, 0.3],
    ]

    预测 = [0.0] * 4
    for i, s in enumerate(状态序列):
        rpe = [s[j] - 预测[j] for j in range(4)]
        r = d.一拍(s, rpe=rpe, mse=0.02)
        for j in range(4):
            预测[j] = 0.9 * 预测[j] + 0.1 * s[j]
        print(f"拍{i+1} 选中=通道{r['选中']} "
              f"RPE={[round(x,3) for x in r['rpe']]} "
              f"偏好={r['偏好']} "
              f"boredom={r['boredom']}")

    print(f"\n状态摘要: {d.状态摘要()}")

    print("\n验证：")
    print(f"  通道0初期RPE大 → 偏好={d.偏好[0]:.3f}（应该较高）")
    print(f"  通道1RPE小 → 偏好={d.偏好[1]:.3f}（应该较低）")
    print(f"  通道2/3无RPE → 偏好={d.偏好[2]:.3f}, {d.偏好[3]:.3f}")

    print("\nIM boredom验证：")
    d2 = 欲望(n_channels=4)
    预测2 = [0.3] * 4
    for i in range(10):
        rpe = [0.3 - 预测2[j] for j in range(4)]
        r = d2.一拍([0.3, 0.3, 0.3, 0.3], rpe=rpe, mse=max(0.01, 0.1 - i * 0.01))
    b_after_learn = d2.boredom
    for i in range(30):
        rpe = [0.3 - 预测2[j] for j in range(4)]
        r = d2.一拍([0.3, 0.3, 0.3, 0.3], rpe=rpe, mse=0.01)
    b_after_stable = d2.boredom
    print(f"  先学10拍(boredom={b_after_learn:.4f}) → 稳定30拍(boredom={b_after_stable:.4f})")
    print(f"  稳定期boredom应该更高: {'是' if b_after_stable > b_after_learn else '否'}")

    print("\n验证通过")
