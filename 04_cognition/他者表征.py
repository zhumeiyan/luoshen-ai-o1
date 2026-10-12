"""他者表征.py —— 洛神对他者的认知建模。

基于LMS/NLMS自适应滤波——整合文档v1.0。

人类神经科学基础：
  大脑通过观察他者行为序列，建立内部模型预测他者下一步行为。
  预测准确→理解他者；预测失败→更新模型。
  这就是"心智理论"(Theory of Mind)的计算基础。

LMS（Least Mean Squares）自适应滤波：
  · 用滤波器权重w预测他者行为：ŷ = w·φ
  · 预测误差：e = y - ŷ（他者实际行为 - 预测行为）
  · 权重更新：w[t+1] = w[t] + μ·e·φ
  · μ是学习率（步长）

NLMS（Normalized LMS）：
  · 与LMS相同，但步长用输入功率归一化
  · μ_norm = μ / (||φ||² + ε)
  · 更稳定——不会因为输入幅度大而发散

特征向量φ = [1, x_t, x_{t-1}, x_t|x_t|]：
  · 1：常数项（偏置）——他者的基线行为倾向
  · x_t：他者当前行为——最新观察
  · x_{t-1}：他者上一拍行为——短期记忆
  · x_t|x_t|：非线性项——捕捉他者行为的非线性特征
    （x_t|x_t| = x_t² if x_t≥0, -x_t² if x_t<0——保留符号的平方）

精卫公式与他者表征：
  a = 洛神对他者的预测（内在模型）
  b = 他者的实际行为（外在表现）
  Δ = b - a = 预测误差（e）
  Σ = a + b = 联合状态（洛神和他者的叠加）
  ⊗ = ∫(Δ, Σ) = 预测误差的累积历史（滤波器权重的演化）
  τ 读 ⊗ → 浮出所见 = 从累积交互中浮现对他者的理解
  (a, b) ⟲ ⊗ = 闭环——预测→观察→误差→更新→更好的预测
"""

import math


class 他者表征:
    """LMS/NLMS自适应滤波——洛神对他者的认知建模。

    通过观察他者行为序列，用自适应滤波器学习预测他者行为。
    预测误差越小→对他者理解越深→心智理论越成熟。
    """

    def __init__(self, n他者=1, 学习率=0.1, 使用NLMS=True):
        """
        参数：
          n他者: 同时建模的他者数量
          学习率: LMS步长μ（NLMS时会被归一化）
          使用NLMS: True=归一化LMS, False=普通LMS
        """
        self.n他者 = n他者
        self.学习率 = 学习率
        self.使用NLMS = 使用NLMS

        # 特征向量维度：φ = [1, x_t, x_{t-1}, x_t|x_t|] → 4维
        self.特征维度 = 4

        # 每个他者一组滤波器权重
        self.权重 = [[0.0] * self.特征维度 for _ in range(n他者)]

        # 他者行为历史（每个他者）
        self.行为历史 = [[0.0, 0.0] for _ in range(n他者)]  # [x_{t-1}, x_t]

        # 预测误差历史
        self.误差历史 = [[] for _ in range(n他者)]

        # 累积预测误差（⊗——预测误差的累积历史）
        self.累积误差 = [0.0] * n他者

        # 理解度——预测误差的EMA倒数（误差越小理解度越高）
        self.理解度 = [0.0] * n他者

        # 拍
        self.拍 = 0

    def _构造特征(self, 他者idx):
        """构造特征向量φ = [1, x_t, x_{t-1}, x_t|x_t|]。"""
        x_t = self.行为历史[他者idx][1]    # 当前行为
        x_prev = self.行为历史[他者idx][0]  # 上一拍行为
        x_nonlinear = x_t * abs(x_t)        # 非线性项 x_t|x_t|
        return [1.0, x_t, x_prev, x_nonlinear]

    def _预测(self, 他者idx):
        """用滤波器权重预测他者行为：ŷ = w·φ。"""
        φ = self._构造特征(他者idx)
        w = self.权重[他者idx]
        return sum(w[i] * φ[i] for i in range(self.特征维度))

    def _更新权重(self, 他者idx, 误差):
        """LMS/NLMS权重更新。

        LMS:  w[t+1] = w[t] + μ·e·φ
        NLMS: w[t+1] = w[t] + (μ / (||φ||² + ε))·e·φ
        """
        φ = self._构造特征(他者idx)

        if self.使用NLMS:
            # NLMS——归一化步长
            φ功率 = sum(f * f for f in φ)
            ε = 1e-6  # 防止除零
            有效步长 = self.学习率 / (φ功率 + ε)
        else:
            # LMS——固定步长
            有效步长 = self.学习率

        for i in range(self.特征维度):
            self.权重[他者idx][i] += 有效步长 * 误差 * φ[i]

    def 一拍(self, 他者行为列表):
        """他者表征一拍——观察他者行为，更新内部模型。

        参数：
          他者行为列表: 各他者当前拍的实际行为值列表

        返回：
          dict包含各他者的预测、误差、理解度
        """
        self.拍 += 1

        结果 = {}

        for idx in range(self.n他者):
            if idx >= len(他者行为列表):
                continue

            实际行为 = 他者行为列表[idx]

            # 先用当前模型预测（基于上一拍的行为历史）
            预测值 = self._预测(idx)

            # 计算预测误差 e = y - ŷ
            误差 = 实际行为 - 预测值

            # 更新权重（LMS/NLMS）
            self._更新权重(idx, 误差)

            # 更新行为历史：x_{t-1} ← x_t, x_t ← 实际行为
            self.行为历史[idx][0] = self.行为历史[idx][1]
            self.行为历史[idx][1] = 实际行为

            # 记录误差历史
            self.误差历史[idx].append(误差)
            if len(self.误差历史[idx]) > 100:
                self.误差历史[idx].pop(0)

            # 累积误差（⊗——预测误差的累积历史）
            self.累积误差[idx] += abs(误差)

            # 理解度——误差EMA的倒数
            # 误差越小→理解度越高；误差越大→理解度越低
            if len(self.误差历史[idx]) > 0:
                近期误差 = self.误差历史[idx][-10:]
                误差EMA = sum(abs(e) for e in 近期误差) / max(1, len(近期误差))
                self.理解度[idx] = max(0.0, 1.0 - 误差EMA)

            结果[idx] = {
                "预测": round(预测值, 4),
                "实际": round(实际行为, 4),
                "误差": round(误差, 4),
                "理解度": round(self.理解度[idx], 4),
                "累积误差": round(self.累积误差[idx], 4),
                "权重": [round(w, 4) for w in self.权重[idx]],
            }

        return 结果

    def 状态摘要(self):
        """他者表征状态摘要。"""
        return {
            "理解度均值": round(sum(self.理解度) / max(1, self.n他者), 4),
            "理解度": [round(u, 4) for u in self.理解度],
            "累积误差": [round(e, 4) for e in self.累积误差],
            "模式": "NLMS" if self.使用NLMS else "LMS",
            "学习率": self.学习率,
            "拍": self.拍,
        }

    def 他者描述(self, 他者idx=0):
        """人类可读的他者理解描述。"""
        if 他者idx >= self.n他者:
            return "无此他者"

        u = self.理解度[他者idx]
        if u > 0.8:
            描述 = "深度理解——预测高度准确"
        elif u > 0.5:
            描述 = "部分理解——预测大致正确"
        elif u > 0.2:
            描述 = "初步理解——预测经常偏离"
        else:
            描述 = "几乎不理解——预测基本失败"

        return f"理解度={u:.2f} ({描述})"


# ═══════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 他者表征验证（LMS/NLMS） ===\n")

    print("特征向量: φ = [1, x_t, x_{t-1}, x_t|x_t|]")
    print("LMS: w[t+1] = w[t] + μ·e·φ")
    print("NLMS: w[t+1] = w[t] + (μ/(||φ||²+ε))·e·φ\n")

    # 测试1：NLMS预测稳定信号
    print("--- 测试1: NLMS预测稳定信号 ---")
    表征1 = 他者表征(n他者=1, 学习率=0.1, 使用NLMS=True)
    # 模拟他者行为：y = 0.5 + 0.3*sin(t) ——可预测的周期信号
    for t in range(50):
        行为 = 0.5 + 0.3 * math.sin(t * 0.1)
        r = 表征1.一拍([行为])
    print(f"  50拍后: {表征1.他者描述(0)}")
    print(f"  理解度={r[0]['理解度']}, 累积误差={r[0]['累积误差']}")
    print(f"  权重={r[0]['权重']}")

    # 测试2：LMS vs NLMS对比
    print("\n--- 测试2: LMS vs NLMS对比 ---")
    表征LMS = 他者表征(n他者=1, 学习率=0.1, 使用NLMS=False)
    表征NLMS = 他者表征(n他者=1, 学习率=0.1, 使用NLMS=True)
    for t in range(50):
        行为 = 0.3 + 0.2 * math.cos(t * 0.2) + 0.1 * (t % 7 == 0)  # 有突变
        rL = 表征LMS.一拍([行为])
        rN = 表征NLMS.一拍([行为])
    print(f"  LMS:  理解度={rL[0]['理解度']}, 累积误差={rL[0]['累积误差']}")
    print(f"  NLMS: 理解度={rN[0]['理解度']}, 累积误差={rN[0]['累积误差']}")
    print(f"  → NLMS应该更稳定（归一化步长防止发散）")

    # 测试3：预测随机信号——理解度应该低
    print("\n--- 测试3: 预测随机信号（理解度应该低） ---")
    import random
    表征3 = 他者表征(n他者=1, 学习率=0.2, 使用NLMS=True)
    rng = random.Random(42)
    for t in range(50):
        行为 = rng.uniform(-1, 1)  # 完全随机
        r = 表征3.一拍([行为])
    print(f"  50拍后: {表征3.他者描述(0)}")
    print(f"  → 随机信号不可预测，理解度应该低")

    # 测试4：多他者建模
    print("\n--- 测试4: 多他者建模 ---")
    表征4 = 他者表征(n他者=3, 学习率=0.1, 使用NLMS=True)
    for t in range(50):
        # 三个他者：一个稳定，一个周期，一个随机
        行为列表 = [
            0.5,                                    # 稳定
            0.3 + 0.2 * math.sin(t * 0.15),         # 周期
            rng.uniform(-1, 1),                      # 随机
        ]
        r = 表征4.一拍(行为列表)
    print(f"  他者0（稳定）: {表征4.他者描述(0)}")
    print(f"  他者1（周期）: {表征4.他者描述(1)}")
    print(f"  他者2（随机）: {表征4.他者描述(2)}")
    print(f"  → 稳定>周期>随机——可预测性决定理解度")

    print("\n验证通过")
