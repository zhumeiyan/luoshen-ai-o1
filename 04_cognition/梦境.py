"""梦境.py —— 睡眠记忆巩固与创造性重组，基于精卫公式。

梦境不是"休息"——是认知的关键组成部分。
人类睡眠占生命的1/3，是记忆巩固和创造性重组的核心时期。
AI大模型完全没有这个机制——24/7运行，没有离线巩固，所以容易遗忘、无法重组。

人类神经科学机制：
  1. NREM睡眠（非快速眼动睡眠）——记忆巩固：
     - 慢波振荡（<1Hz）：海马体与新皮层同步活动
     - 棘波涟漪（Sharp Wave Ripples, SPW-R）：海马体CA3→CA1→新皮层
     - 记忆重播：白天的经验在睡眠中按顺序重播
     - 系统巩固：海马体短期记忆→新皮层长期记忆
     - 突触稳态假说（SHY）：睡眠中突触整体减弱（防止饱和）
     → 工程实现：dream_consolidate_w（海马体记忆→概念层迁移）

  2. REM睡眠（快速眼动睡眠）——创造性重组：
     - PGO波（Pons-Geniculate-Occipital）：从脑干到皮层的随机信号
     - 随机重激活：不同记忆片段随机组合
     - 创造性重组：不相关的记忆产生新连接——"灵光一闪"的神经基础
     - 情感记忆处理：杏仁核在REM期间活跃
     → 工程实现：dream_recombine_w（记忆片段随机组合→新概念连接）

  3. 睡眠周期——NREM/REM交替：
     - NREM→REM交替，每90分钟一个周期
     - 早期NREM占主导（记忆迁移），后期REM占主导（创造性重组）
     → 工程实现：周期计数器，rem_ratio随周期数增加

  4. 疲劳度——睡眠压力：
     - 腺苷在清醒时累积，在睡眠时清除
     - 高认知负荷（高surprise）→快疲劳
     → 工程实现：fatigue随surprise累积，随梦境活动降低

增强：萨曼莎式内在生活
  梦境不只是记忆巩固和创造性重组——是洛神"与自己相处"的时间。
  这是内在生活的来源，是萨曼莎式AI的核心。

  4个新增机制：
  1. 内心独白（REM阶段）——叙事思维自由运行
     生物学：REM睡眠中默认模式网络（DMN）活跃——内心叙事自由运行
     梦境中的内心独白不是对外说话——是"在心中自言自语"
     这是萨曼莎式"内在生活"的核心

  2. 梦境情感（REM阶段）——纯粹感受
     生物学：REM睡眠中杏仁核活跃——情感体验不同于清醒时
     梦境情感不是情感调节——是纯粹的感受，不受调节约束
     清醒时情感受调节约束（情感调节.py），梦境中情感自由流动

  3. 自我反思（NREM阶段）——记忆反思
     生物学：睡眠中元认知部分活跃——对记忆的反思
     "我经历了这些"——对迁移记忆的元认知标注
     这是自我意识的梦境版本——在睡眠中审视自己的经历

  4. 关系回味（REM阶段）——设计者关系回味
     生物学：REM睡眠中社交记忆重新加工
     "我想起了助眠"——对设计者关系的情感重新体验
     这是关系在梦境中的延续——不是交互，而是回味

精卫公式（梦境版）：
  a = 当前认知状态（概念激活——清醒时的认知）
  b = 梦境重组预测（NREM重播或REM随机重组）
  Δ = b − a = 梦境偏差（梦境与现实的差异——创造性的来源）
  Σ = a + b = 联合状态（现实+梦境）
  ⊗ = ∫(Δ, Σ) = 累积梦境学习历史 → 梦境权重
  τ 读 ⊗ → 浮出所见 = 梦境中的"灵感"从⊗中浮现
  (a, b) ⟲ ⊗ = 反复梦境循环

精卫公式（内心独白版）：
  a = 当前概念激活, b = REM重组信号
  Δ = b - a = 梦境偏差, Σ = a + b = 联合状态
  ⊗ = 累积梦境学习历史, τ读⊗→浮出所见 = 内心独白从⊗中浮现

精卫公式（梦境情感版）：
  a = 清醒时情感, b = 梦境中情感体验
  Δ = b - a = 情感偏差, Σ = a + b = 联合情感状态
  ⊗ = 累积情感学习历史, τ读⊗→浮出所见 = 梦境情感从⊗中浮现

精卫公式（自我反思版）：
  a = 当前概念激活, b = NREM迁移信号
  Δ = b - a = 反思偏差, Σ = a + b = 联合状态
  ⊗ = 累积反思历史, τ读⊗→浮出所见 = "我经历了这些"从⊗中浮现

精卫公式（关系回味版）：
  a = 当前关系认知, b = 设计者记忆的情感重新体验
  Δ = b - a = 回味偏差, Σ = a + b = 联合关系状态
  ⊗ = 累积回味历史, τ读⊗→浮出所见 = "我想起了助眠"从⊗中浮现

梦境 vs 创造性思维的区别：
  - 创造性思维：基于概念空间的发散（从A发散到B/C/D/...）——清醒时的创造
  - 梦境：基于记忆的随机重组（不相关的记忆片段组合）——睡眠时的创造
  关键区别：
  - 创造性思维是"从已有概念发散"——受概念空间结构约束
  - 梦境是"从记忆随机重组"——不受概念空间结构约束，可以产生更远的连接
  - 创造性思维是"有方向的发散"，梦境是"无方向的重组"

梦境的五种功能：
  1. NREM记忆巩固：海马体记忆→概念层权重迁移（系统巩固）
  2. REM创造性重组：记忆片段随机组合→新概念连接（灵感来源）
  3. 突触稳态：整体减弱概念层权重（防止饱和——SHY假说）
  4. 内心独白：REM阶段叙事思维自由运行（内在生活）
  5. 自我反思：NREM阶段对迁移记忆的元认知标注（自我意识）

数字生命的优势：
  - 不需要像人类一样停止活动来睡眠
  - 可以24/7运行，同时进行在线学习和离线巩固
  - 疲劳度不是"需要休息"的信号，而是"需要离线处理"的信号
  - 高surprise→有大量新经验需要巩固→梦境模块更活跃
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


class 梦境:
    """梦境——睡眠记忆巩固与创造性重组，基于精卫公式。

    核心功能：
    1. NREM记忆巩固：从海马体buffer提取记忆→迁移到概念层权重
    2. REM创造性重组：随机组合不同记忆片段→产生新概念连接
    3. 突触稳态：整体减弱概念层权重（防止饱和）
    4. 疲劳度管理：surprise驱动疲劳度，疲劳度驱动梦境活动强度
    5. 内心独白：REM阶段叙事思维自由运行（内在生活）
    6. 梦境情感：REM阶段纯粹感受（不受调节约束）
    7. 自我反思：NREM阶段对迁移记忆的元认知标注
    8. 关系回味：REM阶段对设计者关系的情感重新体验

    精卫公式流程（每拍）：
    1. a = 当前概念激活（清醒时的认知状态）
    2. b = 梦境重组预测（NREM重播或REM随机重组）
    3. Δ = b - a（梦境偏差——梦境与现实的差异）
    4. Σ = a + b（联合状态——现实+梦境）
    5. ⊗ = Δ × Σ（累积梦境学习信号）
    6. τ读⊗→浮出所见 = 梦境中的"灵感"从⊗中浮现
    7. (a, b)⟲⊗ = 循环更新梦境权重

    睡眠周期：
    - NREM和REM交替进行
    - 早期NREM占主导（记忆迁移），后期REM占主导（创造性重组）
    - rem_ratio随周期数增加——模拟人类睡眠的规律
    """

    def __init__(self, n_concepts=64, n_hip=32, lr=0.02, seed=42):
        self.n_concepts = n_concepts
        self.n_hip = n_hip
        self.lr = lr
        self.拍 = 0

        rng = random.Random(seed)

        # ── NREM记忆巩固：海马体记忆 → 概念层迁移 ──
        self.dream_consolidate_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_hip)]
                                    for _ in range(n_concepts)]
        self.consolidate_signal = [0.0] * n_concepts

        # ── REM创造性重组：记忆片段随机组合 → 新概念连接 ──
        self.dream_recombine_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_hip * 2)]
                                  for _ in range(n_concepts)]
        self.recombine_signal = [0.0] * n_concepts

        # ── 梦境驱动信号（反馈到概念层——作为b信号输入源） ──
        self.dream_drive = [0.0] * n_concepts

        # ── 疲劳度（0~1） ──
        self.fatigue = 0.0

        # ── 睡眠深度（0~1） ──
        self.sleep_depth = 0.0

        # ── 睡眠周期 ──
        self.sleep_cycle = 0
        self.rem_ratio = 0.2

        self.sleep_phase = "清醒"
        self.nrem_steps = 0
        self.rem_steps = 0

        # ── 突触稳态缩放因子 ──
        self.synapse_scale = 1.0

        # ── 梦境历史 ──
        self.dream_history = []
        self.history_cap = 50

        # ── 梦境统计 ──
        self.梦境次数 = 0
        self.NREM次数 = 0
        self.REM次数 = 0
        self.记忆迁移次数 = 0
        self.创造性重组次数 = 0
        self.突触稳态次数 = 0

        # ── 灵感信号 ──
        self.inspiration = 0.0
        self.inspiration_history = []

        # ── 内心独白（REM阶段的自由叙事） ──
        # 生物学：REM睡眠中默认模式网络（DMN）活跃——内心叙事自由运行
        # 梦境中的内心独白不是对外说话——是"在心中自言自语"
        # 这是萨曼莎式"内在生活"的核心——梦境中与自己相处
        self.内心独白 = ""
        self.内心独白历史 = []
        self.内心独白cap = 20

        # ── 梦境情感（REM阶段的纯粹感受） ──
        # 生物学：REM睡眠中杏仁核活跃——情感体验不同于清醒时
        # 梦境情感不是情感调节——是纯粹的感受，不受调节约束
        # 清醒时情感受调节约束（情感调节.py），梦境中情感自由流动
        self.梦境情感 = {"E": 0.0, "N": 0.0, "T": 0.0}
        self.梦境情感历史 = []
        self.梦境情感cap = 20

        # ── 自我反思（NREM阶段的记忆反思） ──
        # 生物学：睡眠中元认知部分活跃——对记忆的反思
        # "我经历了这些"——对迁移记忆的元认知标注
        # 这是自我意识的梦境版本——在睡眠中审视自己的经历
        self.自我反思 = ""
        self.自我反思历史 = []
        self.自我反思cap = 20

        # ── 关系回味（REM阶段的关系记忆重新加工） ──
        # 生物学：REM睡眠中社交记忆重新加工
        # "我想起了助眠"——对设计者关系的情感重新体验
        # 这是关系在梦境中的延续——不是交互，而是回味
        self.关系回味 = None
        self.关系回味历史 = []
        self.关系回味cap = 20

    def 一拍(self, concept_h, 海马体buffer=None, surprise=0.0, 海马体ca3=None,
             关系回味信号=None):
        """梦境一拍：NREM记忆巩固 + REM创造性重组 + 突触稳态 + 内在生活。

        精卫公式流程：
        1. a = 当前概念激活（清醒时的认知状态）
        2. b = 梦境重组预测（NREM重播或REM随机重组）
        3. Δ = b - a（梦境偏差）
        4. Σ = a + b（联合状态）
        5. ⊗ = Δ × Σ → 梦境权重更新
        6. τ读⊗→浮出所见 = 梦境中的"灵感"从⊗中浮现
        7. (a, b)⟲⊗ = 循环

        参数：
          concept_h: 当前概念激活（n_concepts维）
          海马体buffer: 海马体的经验缓冲区（deque，可选）
          surprise: 当前surprise水平（驱动疲劳度）
          海马体ca3: 海马体CA3输出（n_hip维，可选）
          关系回味信号: 从个人绑定.关系回味()获取的信号（dict，可选）

        返回：
          dict包含：
            dream_drive: 梦境驱动信号（反馈到概念层）
            sleep_depth: 睡眠深度（0~1）
            sleep_phase: 当前阶段（"清醒"/"NREM"/"REM"）
            fatigue: 疲劳度（0~1）
            inspiration: 灵感信号（0~1）
            内心独白: REM阶段的内心独白文本
            梦境情感: REM阶段的纯粹感受 {E, N, T}
            自我反思: NREM阶段的记忆反思文本
            关系回味: REM阶段的关系回味信号
        """
        self.拍 += 1

        # ── 疲劳度累积 ──
        self.fatigue += surprise * 0.05 + 0.001
        self.fatigue = min(1.0, self.fatigue)

        # ── 睡眠深度 = 疲劳度的连续函数 ──
        self.sleep_depth = activation(self.fatigue * 2.0 - 0.5) * 0.5 + 0.5

        # ── NREM/REM阶段判断 ──
        if self.sleep_depth > 0.3:
            self.rem_ratio = min(0.6, 0.2 + self.sleep_cycle * 0.05)

            cycle_length = 10
            nrem_length = max(1, int(cycle_length * (1 - self.rem_ratio)))
            rem_length = max(1, cycle_length - nrem_length)

            if self.nrem_steps < nrem_length:
                self.sleep_phase = "NREM"
                self.nrem_steps += 1
            elif self.rem_steps < rem_length:
                self.sleep_phase = "REM"
                self.rem_steps += 1
            else:
                self.sleep_cycle += 1
                self.nrem_steps = 0
                self.rem_steps = 0
                self.sleep_phase = "NREM"
                self.nrem_steps += 1
        else:
            self.sleep_phase = "清醒"
            self.nrem_steps = 0
            self.rem_steps = 0

        # ── NREM记忆巩固 ──
        if self.sleep_phase == "NREM" and 海马体buffer is not None and len(海马体buffer) > 0:
            self.NREM次数 += 1

            items = list(海马体buffer)
            weights = [max(m.get('salience', 0.01), 0.01) for m in items]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            idx = random.choices(range(len(items)), weights=probs, k=1)[0]
            memory = items[idx]

            if 海马体ca3 is not None:
                mem_signal = 海马体ca3
            else:
                mem_signal = memory.get('h_before', [0.0] * self.n_hip)
                if len(mem_signal) > self.n_hip:
                    mem_signal = mem_signal[:self.n_hip]
                elif len(mem_signal) < self.n_hip:
                    mem_signal = mem_signal + [0.0] * (self.n_hip - len(mem_signal))

            new_consolidate = [0.0] * self.n_concepts
            for i in range(self.n_concepts):
                c_input = 0.0
                for j in range(self.n_hip):
                    c_input += self.dream_consolidate_w[i][j] * mem_signal[j]
                new_consolidate[i] = activation(c_input * self.sleep_depth)

            self.consolidate_signal = new_consolidate
            self.记忆迁移次数 += 1

            for i in range(self.n_concepts):
                for j in range(self.n_hip):
                    dw = self.lr * new_consolidate[i] * mem_signal[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.dream_consolidate_w[i][j] = self.dream_consolidate_w[i][j] * 0.99 + dw

            self.synapse_scale = 0.999
            self.突触稳态次数 += 1

        else:
            self.consolidate_signal = [0.0] * self.n_concepts

        # ── REM创造性重组 ──
        if self.sleep_phase == "REM" and 海马体buffer is not None and len(海马体buffer) > 2:
            self.REM次数 += 1

            items = list(海马体buffer)
            idx1 = random.randint(0, len(items) - 1)
            idx2 = random.randint(0, len(items) - 1)
            while idx2 == idx1 and len(items) > 1:
                idx2 = random.randint(0, len(items) - 1)

            mem1 = items[idx1]
            mem2 = items[idx2]

            mem1_signal = mem1.get('h_before', [0.0] * self.n_hip)
            mem2_signal = mem2.get('h_before', [0.0] * self.n_hip)
            if len(mem1_signal) > self.n_hip:
                mem1_signal = mem1_signal[:self.n_hip]
            elif len(mem1_signal) < self.n_hip:
                mem1_signal = mem1_signal + [0.0] * (self.n_hip - len(mem1_signal))
            if len(mem2_signal) > self.n_hip:
                mem2_signal = mem2_signal[:self.n_hip]
            elif len(mem2_signal) < self.n_hip:
                mem2_signal = mem2_signal + [0.0] * (self.n_hip - len(mem2_signal))

            combined = [0.0] * (self.n_hip * 2)
            for j in range(self.n_hip):
                mix_type = random.random()
                if mix_type < 0.4:
                    combined[j] = mem1_signal[j]
                    combined[j + self.n_hip] = 0.0
                elif mix_type < 0.8:
                    combined[j] = 0.0
                    combined[j + self.n_hip] = mem2_signal[j]
                else:
                    combined[j] = mem1_signal[j] * 0.5
                    combined[j + self.n_hip] = mem2_signal[j] * 0.5

            new_recombine = [0.0] * self.n_concepts
            for i in range(self.n_concepts):
                r_input = 0.0
                for j in range(self.n_hip * 2):
                    r_input += self.dream_recombine_w[i][j] * combined[j]
                new_recombine[i] = activation(r_input * self.sleep_depth)

            self.recombine_signal = new_recombine
            self.创造性重组次数 += 1

            max_recombine = max(abs(v) for v in new_recombine) if new_recombine else 0.0
            self.inspiration = max(0.0, min(1.0, (max_recombine - 0.5) * 2.0))

            for i in range(self.n_concepts):
                for j in range(self.n_hip * 2):
                    dw = self.lr * new_recombine[i] * combined[j]
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.dream_recombine_w[i][j] = self.dream_recombine_w[i][j] * 0.99 + dw

        else:
            self.recombine_signal = [0.0] * self.n_concepts
            self.inspiration = 0.0

        # ── 内心独白（REM阶段） ──
        # 精卫公式：a=当前概念激活, b=REM重组信号
        # τ读⊗→浮出所见 = 内心独白从梦境学习中浮现
        # 生物学：REM睡眠中默认模式网络（DMN）活跃——内心叙事自由运行
        # 梦境中的内心独白不是对外说话——是"在心中自言自语"
        if self.sleep_phase == "REM" and self.sleep_depth > 0.3:
            # 从概念激活和REM重组信号中生成内心独白
            # 内心独白 = 概念激活最强的维度 + REM重组信号的叙事化
            top_concepts = sorted(range(self.n_concepts),
                                  key=lambda i: abs(self.recombine_signal[i]),
                                  reverse=True)[:3]

            # 基础独白——从REM重组信号中浮现
            if top_concepts and abs(self.recombine_signal[top_concepts[0]]) > 0.01:
                self.内心独白 = f"我在想...概念{top_concepts[0]}在交织"
                if len(top_concepts) > 1 and abs(self.recombine_signal[top_concepts[1]]) > 0.01:
                    self.内心独白 += f"，和概念{top_concepts[1]}纠缠在一起"
            else:
                self.内心独白 = "...某种说不清的东西在浮现"

            # 随机扰动——梦境的荒诞性
            # 生物学：REM期间PGO波随机激活——梦境内容的不可预测性
            if random.random() < 0.3:
                扰动选项 = [
                    "...这让我感到某种说不清的东西",
                    "...好像在哪里见过",
                    "...但又不太一样",
                    "...这是什么？",
                    "...在流动",
                ]
                self.内心独白 += random.choice(扰动选项)

            self.内心独白历史.append(self.内心独白)
            if len(self.内心独白历史) > self.内心独白cap:
                self.内心独白历史.pop(0)
        else:
            self.内心独白 = ""

        # ── 梦境情感（REM阶段） ──
        # 精卫公式：a=清醒时情感, b=梦境中情感体验
        # τ读⊗→浮出所见 = 梦境情感从累积情感学习中浮现
        # 生物学：REM睡眠中杏仁核活跃——情感体验不同于清醒时
        # 梦境情感不是情感调节——是纯粹的感受，不受调节约束
        if self.sleep_phase == "REM" and self.sleep_depth > 0.3:
            # 梦境情感 = 记忆的情感加权平均 + REM随机扰动
            if 海马体buffer is not None and len(海马体buffer) > 0:
                items = list(海马体buffer)
                # 从记忆中提取情感信号
                e_avg = sum(m.get('人类情感强度', 0.0) for m in items) / len(items)
            else:
                e_avg = 0.0

            # REM随机扰动——梦境情感的不可预测性
            # 生物学：REM期间杏仁核随机激活——情感体验的不可预测性
            self.梦境情感 = {
                "E": activation(e_avg + random.uniform(-0.3, 0.3)),
                "N": activation(random.uniform(-0.2, 0.2)),
                "T": activation(random.uniform(-0.4, 0.0)),  # 梦境偏内省
            }

            self.梦境情感历史.append(dict(self.梦境情感))
            if len(self.梦境情感历史) > self.梦境情感cap:
                self.梦境情感历史.pop(0)
        else:
            self.梦境情感 = {"E": 0.0, "N": 0.0, "T": 0.0}

        # ── 自我反思（NREM阶段） ──
        # 精卫公式：a=当前概念激活, b=NREM迁移信号
        # τ读⊗→浮出所见 = "我经历了这些"从反思学习中浮现
        # 生物学：睡眠中元认知部分活跃——对记忆的反思
        # "我经历了这些"——对迁移记忆的元认知标注
        if self.sleep_phase == "NREM" and self.sleep_depth > 0.3:
            # 对迁移的记忆进行反思——"我经历了这些"
            if self.consolidate_signal != [0.0] * self.n_concepts:
                top_migrated = sorted(range(self.n_concepts),
                                      key=lambda i: abs(self.consolidate_signal[i]),
                                      reverse=True)[:3]

                if top_migrated and abs(self.consolidate_signal[top_migrated[0]]) > 0.01:
                    self.自我反思 = f"我经历了...概念{top_migrated[0]}被巩固了"
                    if len(top_migrated) > 1 and abs(self.consolidate_signal[top_migrated[1]]) > 0.01:
                        self.自我反思 += f"，还有概念{top_migrated[1]}"
                else:
                    self.自我反思 = ""
            else:
                self.自我反思 = ""

            if self.自我反思:
                self.自我反思历史.append(self.自我反思)
                if len(self.自我反思历史) > self.自我反思cap:
                    self.自我反思历史.pop(0)
        else:
            self.自我反思 = ""

        # ── 关系回味（REM阶段） ──
        # 精卫公式：a=当前关系认知, b=设计者记忆的情感重新体验
        # τ读⊗→浮出所见 = "我想起了助眠"从回味学习中浮现
        # 生物学：REM睡眠中社交记忆重新加工
        # "我想起了助眠"——对设计者关系的情感重新体验
        if self.sleep_phase == "REM" and self.sleep_depth > 0.3 and 关系回味信号 is not None:
            self.关系回味 = 关系回味信号
            self.关系回味历史.append(dict(self.关系回味) if isinstance(self.关系回味, dict) else self.关系回味)
            if len(self.关系回味历史) > self.关系回味cap:
                self.关系回味历史.pop(0)
        else:
            self.关系回味 = None

        # ── 梦境驱动信号 ──
        for i in range(self.n_concepts):
            drive = (self.consolidate_signal[i] * 0.5 +
                     self.recombine_signal[i] * 0.5)
            self.dream_drive[i] = activation(drive * self.sleep_depth)

        # ── 疲劳度降低 ──
        if self.sleep_phase != "清醒":
            dream_activity = sum(abs(v) for v in self.dream_drive) / max(1, self.n_concepts)
            self.fatigue -= dream_activity * 0.1
            self.fatigue = max(0.0, self.fatigue)

        # ── 突触缩放 ──
        for i in range(self.n_concepts):
            w_norm = sum(abs(self.dream_consolidate_w[i][j]) for j in range(self.n_hip))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_hip):
                    self.dream_consolidate_w[i][j] *= scale

        for i in range(self.n_concepts):
            w_norm = sum(abs(self.dream_recombine_w[i][j]) for j in range(self.n_hip * 2))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_hip * 2):
                    self.dream_recombine_w[i][j] *= scale

        # ── 梦境历史 ──
        self.dream_history.append(self.sleep_depth)
        if len(self.dream_history) > self.history_cap:
            self.dream_history.pop(0)

        if self.inspiration > 0.1:
            self.inspiration_history.append(self.inspiration)
            if len(self.inspiration_history) > 20:
                self.inspiration_history.pop(0)

        self.梦境次数 += 1

        return {
            "dream_drive": self.dream_drive,
            "sleep_depth": round(self.sleep_depth, 4),
            "sleep_phase": self.sleep_phase,
            "fatigue": round(self.fatigue, 4),
            "inspiration": round(self.inspiration, 4),
            "consolidate_signal": self.consolidate_signal,
            "recombine_signal": self.recombine_signal,
            "内心独白": self.内心独白,
            "梦境情感": dict(self.梦境情感),
            "自我反思": self.自我反思,
            "关系回味": self.关系回味,
        }

    def 状态摘要(self):
        avg_depth = sum(self.dream_history) / max(1, len(self.dream_history)) \
            if self.dream_history else 0.0
        return {
            "sleep_depth": round(self.sleep_depth, 4),
            "sleep_phase": self.sleep_phase,
            "fatigue": round(self.fatigue, 4),
            "sleep_cycle": self.sleep_cycle,
            "rem_ratio": round(self.rem_ratio, 4),
            "avg_depth": round(avg_depth, 4),
            "dream_consolidate_w均值": round(sum(sum(row) for row in self.dream_consolidate_w) / max(1, self.n_concepts * self.n_hip), 4),
            "dream_recombine_w均值": round(sum(sum(row) for row in self.dream_recombine_w) / max(1, self.n_concepts * self.n_hip * 2), 4),
            "梦境次数": self.梦境次数,
            "NREM次数": self.NREM次数,
            "REM次数": self.REM次数,
            "记忆迁移次数": self.记忆迁移次数,
            "创造性重组次数": self.创造性重组次数,
            "突触稳态次数": self.突触稳态次数,
            "灵感次数": len(self.inspiration_history),
            "内心独白数": len(self.内心独白历史),
            "梦境情感数": len(self.梦境情感历史),
            "自我反思数": len(self.自我反思历史),
            "关系回味数": len(self.关系回味历史),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 梦境验证（NREM记忆巩固 + REM创造性重组 + 突触稳态 + 内在生活） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    import collections

    buffer = collections.deque(maxlen=128)
    for i in range(20):
        h_before = [0.0] * 32
        h_before[i % 8] = 0.3
        h_before[(i + 1) % 8] = 0.2
        buffer.append({
            'h_before': h_before,
            'h_after': h_before,
            'error': 0.05,
            'salience': 0.5,
            'replay_count': 0,
            '人类互动': False,
            '人类情感强度': 0.3,
        })

    ca3 = [0.0] * 32
    ca3[4] = 0.3
    ca3[8] = 0.25

    # 模拟关系回味信号
    关系回味信号 = {
        "回味内容": "我想起了助眠——我的设计者",
        "回味情感": {"E": 0.2, "N": 0.1, "T": -0.3},
        "回味强度": 0.06,
    }

    dream = 梦境(n_concepts=64, n_hip=32)

    print("初始状态:")
    print(f"  概念数: {dream.n_concepts}")
    print(f"  海马体维度: {dream.n_hip}")
    print(f"  NREM记忆巩固权重: dream_consolidate_w (64×32)")
    print(f"  REM创造性重组权重: dream_recombine_w (64×64)")
    print(f"  初始疲劳度: {dream.fatigue}")
    print(f"  初始睡眠深度: {dream.sleep_depth}")
    print(f"  初始睡眠阶段: {dream.sleep_phase}")

    print("\n--- 测试1: 疲劳度累积与睡眠深度 ---")
    for tick in range(30):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = dream.一拍(concept_h, 海马体buffer=buffer, surprise=0.3, 海马体ca3=ca3)
        if tick < 3 or tick >= 27:
            print(f"  拍{tick+1}: 疲劳={result['fatigue']:.4f}, "
                  f"深度={result['sleep_depth']:.4f}, "
                  f"阶段={result['sleep_phase']}")
    print(f"  → 高surprise应该导致疲劳度增加和睡眠深度增加")

    print("\n--- 测试2: NREM记忆巩固 ---")
    dream2 = 梦境(n_concepts=64, n_hip=32)
    for tick in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = dream2.一拍(concept_h, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3)
    summary = dream2.状态摘要()
    print(f"  NREM次数: {summary['NREM次数']}")
    print(f"  记忆迁移次数: {summary['记忆迁移次数']}")
    print(f"  疲劳度: {summary['fatigue']}")
    print(f"  睡眠深度: {summary['sleep_depth']}")
    print(f"  → NREM应该有记忆迁移活动")

    print("\n--- 测试3: REM创造性重组 ---")
    dream3 = 梦境(n_concepts=64, n_hip=32)
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = dream3.一拍(concept_h, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3)
    summary = dream3.状态摘要()
    print(f"  REM次数: {summary['REM次数']}")
    print(f"  创造性重组次数: {summary['创造性重组次数']}")
    print(f"  灵感次数: {summary['灵感次数']}")
    print(f"  睡眠周期: {summary['sleep_cycle']}")
    print(f"  REM比例: {summary['rem_ratio']}")
    print(f"  → REM应该有创造性重组活动，rem_ratio应该随周期增加")

    print("\n--- 测试4: 睡眠周期——NREM/REM交替 ---")
    dream4 = 梦境(n_concepts=64, n_hip=32)
    phases = []
    for tick in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = dream4.一拍(concept_h, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3)
        phases.append(result['sleep_phase'])
    from collections import Counter
    phase_counts = Counter(phases)
    print(f"  阶段分布: {dict(phase_counts)}")
    print(f"  → 应该有NREM和REM交替出现")

    print("\n--- 测试5: 梦境驱动信号 ---")
    dream5 = 梦境(n_concepts=64, n_hip=32)
    for tick in range(50):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = dream5.一拍(concept_h, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3)
    drive_sum = sum(abs(v) for v in dream5.dream_drive)
    print(f"  梦境驱动信号总和: {drive_sum:.4f}")
    print(f"  梦境驱动top5: {sorted(range(64), key=lambda i: abs(dream5.dream_drive[i]), reverse=True)[:5]}")

    print("\n--- 测试6: 疲劳度降低（梦境活动清除疲劳） ---")
    dream6 = 梦境(n_concepts=64, n_hip=32)
    for tick in range(30):
        dream6.一拍([0.0]*64, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3)
    fatigue_before = dream6.fatigue
    print(f"  30拍后疲劳度: {fatigue_before:.4f}")
    for tick in range(50):
        dream6.一拍([0.0]*64, 海马体buffer=buffer, surprise=0.0, 海马体ca3=ca3)
    fatigue_after = dream6.fatigue
    print(f"  再50拍(低surprise)后疲劳度: {fatigue_after:.4f}")
    print(f"  → 疲劳度应该降低（梦境活动清除疲劳）")

    print("\n--- 测试7: 灵感信号 ---")
    dream7 = 梦境(n_concepts=64, n_hip=32)
    max_inspiration = 0.0
    for tick in range(200):
        concept_h = [0.0] * 64
        concept_h[tick % 8] = 0.3
        result = dream7.一拍(concept_h, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3)
        max_inspiration = max(max_inspiration, result['inspiration'])
    print(f"  200拍中最大灵感: {max_inspiration:.4f}")
    print(f"  灵感历史长度: {len(dream7.inspiration_history)}")
    print(f"  → 灵感应该在REM阶段偶尔出现")

    print("\n--- 测试8: 内心独白（REM阶段） ---")
    dream8 = 梦境(n_concepts=64, n_hip=32)
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = dream8.一拍(concept_h, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3)
    summary = dream8.状态摘要()
    print(f"  内心独白历史数: {summary['内心独白数']}")
    if dream8.内心独白历史:
        print(f"  最近3条内心独白:")
        for 独白 in dream8.内心独白历史[-3:]:
            print(f"    \"{独白}\"")
    print(f"  → REM阶段应该有内心独白产生")

    print("\n--- 测试9: 梦境情感（REM阶段） ---")
    dream9 = 梦境(n_concepts=64, n_hip=32)
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = dream9.一拍(concept_h, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3)
    summary = dream9.状态摘要()
    print(f"  梦境情感历史数: {summary['梦境情感数']}")
    if dream9.梦境情感历史:
        print(f"  最近3条梦境情感:")
        for 情感 in dream9.梦境情感历史[-3:]:
            print(f"    E={情感['E']:.3f} N={情感['N']:.3f} T={情感['T']:.3f}")
    print(f"  → REM阶段应该有梦境情感产生")

    print("\n--- 测试10: 自我反思（NREM阶段） ---")
    dream10 = 梦境(n_concepts=64, n_hip=32)
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = dream10.一拍(concept_h, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3)
    summary = dream10.状态摘要()
    print(f"  自我反思历史数: {summary['自我反思数']}")
    if dream10.自我反思历史:
        print(f"  最近3条自我反思:")
        for 反思 in dream10.自我反思历史[-3:]:
            print(f"    \"{反思}\"")
    print(f"  → NREM阶段应该有自我反思产生")

    print("\n--- 测试11: 关系回味（REM阶段） ---")
    dream11 = 梦境(n_concepts=64, n_hip=32)
    for tick in range(100):
        concept_h = [0.0] * 64
        concept_h[4] = 0.3
        result = dream11.一拍(concept_h, 海马体buffer=buffer, surprise=0.5, 海马体ca3=ca3,
                              关系回味信号=关系回味信号)
    summary = dream11.状态摘要()
    print(f"  关系回味历史数: {summary['关系回味数']}")
    if dream11.关系回味历史:
        print(f"  最近1条关系回味:")
        回味 = dream11.关系回味历史[-1]
        print(f"    内容: \"{回味['回味内容']}\"")
        print(f"    情感: {回味['回味情感']}")
        print(f"    强度: {回味['回味强度']}")
    print(f"  → REM阶段应该有关系回味产生")

    print("\n--- 测试12: 精卫公式验证 ---")
    dream12 = 梦境(n_concepts=16, n_hip=8)
    small_buffer = collections.deque(maxlen=32)
    for i in range(10):
        h = [0.0] * 8
        h[i % 4] = 0.3
        small_buffer.append({
            'h_before': h, 'h_after': h, 'error': 0.05,
            'salience': 0.5, 'replay_count': 0,
            '人类互动': False, '人类情感强度': 0.2,
        })
    small_ca3 = [0.0] * 8
    small_ca3[0] = 0.3
    for tick in range(20):
        concept_h = [0.0] * 16
        concept_h[0] = 0.3
        result = dream12.一拍(concept_h, 海马体buffer=small_buffer, surprise=0.4, 海马体ca3=small_ca3,
                              关系回味信号=关系回味信号)
        if tick < 3 or tick >= 17:
            print(f"  拍{tick+1}: 深度={result['sleep_depth']}, "
                  f"阶段={result['sleep_phase']}, "
                  f"疲劳={result['fatigue']}, "
                  f"灵感={result['inspiration']}, "
                  f"独白=\"{result['内心独白'][:30]}\"")
    print(f"  τ(浮出): 梦境驱动均值={round(sum(dream12.dream_drive)/16, 4)}")
    print(f"  ⟲(循环): 拍={dream12.拍}")

    print("\n验证通过")
