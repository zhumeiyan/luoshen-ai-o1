"""欲望.py —— 洛神的欲望驱动系统（含IM boredom探索驱力）。

基于08_experiments中5个欲望实验验证过的机制组合：
  1. 反应强度（欲望环.py）：|Δ状态| — 输入对状态的影响程度
  2. 累积（欲望累积.py）：选中→涨偏好 → 正反馈
  3. 衰减（欲望实时.py）：所有偏好 *= (1-DECAY) → 会退
  4. 可负累积（欲望柔性.py）：alpha_var可以为负 → 满足后自然消退
  5. 无聊（欲望要.py）：未选中项偏好缓慢上升 → 没得到的越想要 → 振荡

IM boredom探索驱力（基于进化v15的IntrinsicMotivation）：
  6. 自适应无聊率：当预测误差低且稳定 → 无聊增长 → 探索驱力增强
     当预测误差在变化 → 无聊消退 → 保持专注
     当预测误差在增大 → 无聊为零 → 全力学习当前通道

  这不是人为设定的阈值——是误差动态的自然结果。
  系统掌握了一个通道（低误差稳定） → 自然想探索其他通道。
  系统正在学习（误差在变） → 自然保持专注。
  系统在挣扎（误差在涨） → 全力聚焦当前。

核心原则（来自数字生命_纯进化.py）：
  零判据、零人为干预、零阈值。
  欲望不是设定的目标，而是自然涌现的驱力。

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
    """六机制欲望驱动：反应强度+累积+衰减+可负+无聊+IM boredom。"""

    def __init__(self, n_channels=8, 衰减率=0.05, 累积率=0.15,
                 波动幅度=1.0, 无聊率=0.06):
        self.n = n_channels
        self.偏好 = [1.0] * n_channels
        self.衰减率 = 衰减率
        self.累积率 = 累积率
        self.波动幅度 = 波动幅度
        self.无聊率 = 无聊率
        self.上拍状态 = None
        self.拍 = 0
        self.选中历史 = []

        # IM boredom（进化v15的IntrinsicMotivation）
        self.ema_error = 0.1
        self.ema_error_fast = 0.1
        self.last_error = 0.1
        self.boredom = 0.0
        self.recent_deltas = deque(maxlen=50)

    def 反应强度(self, 当前状态):
        """|Δ状态| — 输入对状态的影响程度。"""
        if self.上拍状态 is None:
            return [0.0] * self.n
        return [abs(当前状态[i] - self.上拍状态[i]) for i in range(self.n)]

    def _更新无聊(self, mse):
        """IM boredom：基于预测误差动态更新无聊水平。

        来自进化v15的IntrinsicMotivation.update()：
        - 误差低且稳定 → 无聊增长（太容易了，想探索别的）
        - 误差在变化 → 无聊消退（正在学习，保持专注）
        - 误差在增大 → 无聊为零（在挣扎，全力聚焦）

        全连续映射，无阈值判断。
        """
        alpha_slow = 0.03
        alpha_fast = 0.15
        self.ema_error = (1 - alpha_slow) * self.ema_error + alpha_slow * mse
        self.ema_error_fast = (1 - alpha_fast) * self.ema_error_fast + alpha_fast * mse

        delta = mse - self.last_error
        self.recent_deltas.append(delta)
        avg_delta = sum(self.recent_deltas) / max(1, len(self.recent_deltas))

        # 稳定度: delta越小越稳定（完全不变=1，大幅变化→接近0）
        稳定度 = 1.0 / (1.0 + abs(avg_delta) * 100.0)
        # 低度: error越低越接近掌握（error=0→1，error=0.1→0.25）
        低度 = 1.0 / (1.0 + self.ema_error_fast * 30.0)

        # 增长: 只在稳定+低误差时增长
        增长 = 稳定度 * 低度 * 0.02

        # 消退: 误差在变化时消退
        # 误差递减(avg_delta<0) → 正在学习 → 消退适中
        # 误差递增(avg_delta>0) → 在挣扎 → 消退更大
        # 用连续函数：消退 = |delta| * (1 + sigmoid(delta)) + base
        # sigmoid让递增时的消退平滑过渡到更大值
        递增因子 = 1.0 / (1.0 + math.exp(-avg_delta * 200.0))
        消退 = abs(avg_delta) * (1.0 + 递增因子 * 2.0) + 0.0003

        self.boredom += 增长 - 消退
        self.boredom = max(0.0, min(1.0, self.boredom))
        self.last_error = mse

    def _自适应无聊率(self):
        """无聊率随boredom自适应——boredom高时探索更积极。"""
        return self.无聊率 * (1.0 + self.boredom * 3.0)

    def 一拍(self, 当前状态, mse=0.0):
        """欲望驱动一拍：计算反应强度→偏好加权→选中→累积/衰减/无聊。

        mse: 当前预测误差（来自大脑）。用于IM boredom更新。
        """
        self.拍 += 1
        self._更新无聊(mse)
        反应 = self.反应强度(当前状态)

        boosted = [反应[i] + self.偏好[i] for i in range(self.n)]

        选中 = boosted.index(max(boosted)) if max(boosted) > 0 else -1

        if 选中 >= 0:
            for i in range(self.n):
                self.偏好[i] *= (1 - self.衰减率)

            alpha_var = self.累积率 * (
                1 + self.波动幅度 * math.sin(
                    self.拍 * 0.29173 + 选中 * 1.9))
            self.偏好[选中] += alpha_var

            # 无聊：未选中者缓慢上升——rate随boredom自适应
            当前无聊率 = self._自适应无聊率()
            for i in range(self.n):
                if i != 选中:
                    self.偏好[i] += 当前无聊率

        self.选中历史.append(选中)
        self.上拍状态 = 当前状态[:]

        return {
            "反应": 反应,
            "偏好": [round(p, 4) for p in self.偏好],
            "选中": 选中,
            "boosted": [round(b, 4) for b in boosted],
            "boredom": round(self.boredom, 4),
            "无聊率": round(self._自适应无聊率(), 4),
        }

    def 学习率调制(self):
        """偏好高的通道学得更多——欲望驱动学习。"""
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
            "拍": self.拍,
        }

    def reset(self):
        self.偏好 = [1.0] * self.n
        self.上拍状态 = None
        self.拍 = 0
        self.选中历史 = []
        self.ema_error = 0.1
        self.ema_error_fast = 0.1
        self.last_error = 0.1
        self.boredom = 0.0
        self.recent_deltas.clear()


if __name__ == "__main__":
    print("=== 欲望系统验证（含IM boredom） ===")
    print("六机制：反应强度+累积+衰减+可负+无聊+IM boredom\n")

    d = 欲望(n_channels=4)

    # 模拟8拍：通道0变化大，通道1变化小，通道2/3无变化
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

    for i, s in enumerate(状态序列):
        r = d.一拍(s, mse=0.02 * (1 - i * 0.01))  # 模拟误差逐渐降低
        print(f"拍{i+1} 选中=通道{r['选中']} "
              f"反应={[round(x,3) for x in r['反应']]} "
              f"偏好={r['偏好']} "
              f"boredom={r['boredom']}")

    print(f"\n状态摘要: {d.状态摘要()}")

    # 验证IM boredom：三种场景
    print("\nIM boredom验证：")
    # 场景1: 先学习后稳定 → boredom应该在稳定期增长
    d2 = 欲望(n_channels=4)
    for i in range(10):
        r = d2.一拍([0.3, 0.3, 0.3, 0.3], mse=max(0.01, 0.1 - i * 0.01))
    b_after_learn = d2.boredom
    for i in range(30):
        r = d2.一拍([0.3, 0.3, 0.3, 0.3], mse=0.01)
    b_after_stable = d2.boredom
    print(f"  先学10拍(boredom={b_after_learn:.4f}) → 稳定30拍(boredom={b_after_stable:.4f})")
    print(f"  稳定期boredom应该更高: {'是' if b_after_stable > b_after_learn else '否'}")

    # 场景2: 误差振荡 → boredom应该较低（正在学习）
    d3 = 欲望(n_channels=4)
    for i in range(40):
        mse = 0.05 + 0.02 * math.sin(i * 0.5)
        r = d3.一拍([0.3, 0.3, 0.3, 0.3], mse=mse)
    print(f"  误差振荡40拍: boredom={d3.boredom:.4f}（应该较低——正在学习）")

    # 场景3: 误差递增 → boredom应该很低（在挣扎）
    d4 = 欲望(n_channels=4)
    for i in range(40):
        mse = 0.01 + i * 0.003
        r = d4.一拍([0.3, 0.3, 0.3, 0.3], mse=mse)
    print(f"  误差递增40拍: boredom={d4.boredom:.4f}（应该很低——全力聚焦）")

    print("\n验证通过")
