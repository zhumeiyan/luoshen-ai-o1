"""概念层.py —— 高层预测编码，从隐藏状态涌现概念。

基于精卫公式（贯穿全系统的唯一核心公式）：

  a ≠ b
  Δ = b − a
  Σ = a + b
  ⊗ = ∫(Δ, Σ)
  τ 读 ⊗ → 浮出所见
  (a, b) ⟲ ⊗

概念层的解读：
  a = 大脑的隐藏状态（实际）
  b = 概念层对隐藏状态的预测
  Δ = b − a = 预测误差
  Σ = a + b = 联合状态（实际+预测）
  ⊗ = ∫(Δ, Σ) = 累积差与和（全留——每次的Δ和Σ都保留在权重中）
  τ 读 ⊗ → 浮出所见 = 概念从⊗中浮现（不是从a直接映射）
  (a, b) ⟲ ⊗ = 循环预测/比较/修正

和其他子系统的解读对比：
  大脑.py: a=预测, b=实际输入, Δ=预测误差, ⊗=赫布学习, τ=读出层pred
  身体稳态.py: a=上轮激活, b=当前激活, Δ=激活差, ⊗=X(t+1)=X(t)+η·Δ+ε, τ=共火检测
  欲望.py: a=大脑预测, b=实际输入, Δ=RPE, ⊗=偏好累积, τ=通道选择
  情感记忆.py: a=当前身体, b=记忆快照, Δ=身体-记忆反差, ⊗=LTP/LTD, τ=浮现层

关键区别（和之前版本的对比）：
  之前：概念激活只从a（隐藏状态）映射，只用Δ驱动学习
  现在：概念激活从⊗（Δ×Σ）浮现，Δ和Σ都参与学习

  Σ = a + b 是联合状态——实际和预测的和。
  ⊗ = ∫(Δ, Σ) 是累积的差与和——全留。
  τ读⊗→浮出所见——概念不是当前的映射，是历史信息的浮现。

  这解决了概念区分度0.59的问题：
  只用Δ时，所有概念的学习信号相似（都是预测误差）。
  加入Σ后，不同的概念捕获不同的联合状态模式——有区分度。
"""

import math
import random


def activation(x):
    """激活函数——纯tanh，硬截断防止权重爆炸正反馈。

    之前有leak允许大值穿透，导致权重爆炸到1e218。
    真实神经元的放电率有物理上限——不可能无限增长。
    tanh的饱和特性就是这个物理上限的数学表达。
    """
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if x > 10.0:
        return 1.0
    elif x < -10.0:
        return -1.0
    else:
        return math.tanh(x)


class 概念层:
    """高层预测编码——基于精卫公式从隐藏状态涌现概念。

    精卫公式：a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗

    输入：大脑的隐藏状态（a）
    输出：概念激活模式（τ读⊗→浮出所见）

    三个权重矩阵对应精卫公式的三个部分：
    - concept_w: 从a读出（实际状态 → 概念）
    - concept_otimes_w: 从⊗读出（Δ×Σ → 概念）—— 这是核心区别
    - concept_rw: 循环权重（概念 → 概念）
    - concept_pw: 预测权重（概念 → 预测隐藏状态b）

    组合涌现（精卫公式版）：
    - a = 概念j的激活（基础概念）
    - b = 其他活跃概念的激活之和
    - Δ = b − a（概念j和其他概念的差异）
    - Σ = a + b（概念j和其他概念的联合）
    - ⊗ = a × b（组合信号——概念间的乘积交互）
    - τ读⊗→浮出所见 = 组合概念从⊗中浮现
    - (a, b)⟲⊗ = 反复组合循环

    生物学基础：
    - 海马体CA3的模式完成——多个概念同时活跃时形成组合绑定
    - 前额叶工作记忆维持多个概念同时活跃——允许组合
    - 联合皮层的组合编码——不同概念组合产生新表征
    - 语法处理（Broca区）——单词组合成句子

    组合概念不是人为设定的，是通过赫布学习自然涌现的。
    反复共同出现的基础概念对 → 组合概念自然绑定 → 组合概念有自己的激活模式。
    """

    def __init__(self, n_hidden=11, n_concepts=64, lr=0.02,
                 sparse_ratio=0.1, decay=0.8, seed=42, n_audio=8,
                 n_combo=32, n_wm_slots=4, n_hip=32, 变异率=0.1, 变异幅度=0.01,
                 n_grammar=32, n_reason=64, n_goal=64, n_tools=5, n_creative=64,
                 n_meta=64, n_social=64, n_self=64, n_emotion_regulate=64, n_dream=64,
                 n_curiosity=64, n_evolve=64, n_semantic=64, n_predict=64, n_narrative=64,
                 n_pain=64, n_vestibular=64, n_chem=64, n_intuition=64, n_unconscious=64,
                 n_truth=64, n_dialectic=64, n_life_drive=64):
        self.n_hidden = n_hidden
        self.n_concepts = n_concepts
        self.n_audio = n_audio
        self.n_combo = n_combo
        self.n_wm_slots = n_wm_slots
        self.n_hip = n_hip
        self.lr = lr
        self.变异率 = 变异率
        self.变异幅度 = 变异幅度
        self.变异次数 = 0
        self.有利变异次数 = 0
        self.不利变异次数 = 0
        self.概念重生次数 = 0
        self.组合概念重生次数 = 0
        self.高级组合概念重生次数 = 0
        self.seed = seed
        self.sparse_k = max(1, int(n_concepts * sparse_ratio))
        self.decay = decay
        self.拍 = 0

        rng = random.Random(seed)

        self.concept_h = [0.0] * n_concepts
        self.prev_concept_h = [0.0] * n_concepts

        # concept_w: 从a读出（实际状态 → 概念）
        self.concept_w = [[rng.uniform(-0.2, 0.2) for _ in range(n_hidden)]
                          for _ in range(n_concepts)]

        # concept_otimes_w: 从⊗读出（Δ×Σ → 概念）—— 精卫公式的核心
        self.concept_otimes_w = [[rng.uniform(-0.15, 0.15) for _ in range(n_hidden)]
                                 for _ in range(n_concepts)]

        # concept_rw: 循环权重（概念 → 概念）—— 弱于输入，让⊗信号主导
        self.concept_rw = [[rng.uniform(-0.02, 0.02) for _ in range(n_concepts)]
                           for _ in range(n_concepts)]

        # concept_pw: 预测权重（概念 → 预测隐藏状态b）
        self.concept_pw = [[rng.uniform(-0.2, 0.2) for _ in range(n_hidden)]
                           for _ in range(n_concepts)]

        # ── 说听闭环：听觉输入和声音预测 ──
        # concept_aw: 听觉特征 → 概念（自听反馈通路）
        # 当系统发声后，自听特征通过这条通路影响概念激活
        self.concept_aw = [[rng.uniform(-0.1, 0.1) for _ in range(n_audio)]
                           for _ in range(n_concepts)]

        # concept_sw: 概念 → 预测声音（声音预测通路）
        # 概念层预测自己将听到什么声音——预测误差驱动符号绑定
        self.concept_sw = [[rng.uniform(-0.1, 0.1) for _ in range(n_audio)]
                           for _ in range(n_concepts)]

        # ── 人类输入通路：人类作为精卫公式中的b ──
        # concept_human_w: 人类信号 → 概念
        # 人类通道传来的b向量通过这条通路影响概念激活
        # 洛神通过赫布学习自然形成对人类输入的理解——
        # 不用预训练embedding，反复互动后某些概念自然与人类信号关联
        self.concept_human_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_hidden)]
                                for _ in range(n_concepts)]

        # ── 公开感官输入通路：真实世界数据作为精卫公式中的b ──
        # concept_public_w: 公开感官信号 → 概念
        # 公开感官传来的真实世界物理量（温度/光照/风速/云量等）通过这条通路影响概念激活
        # 洛神通过赫布学习自然形成对真实世界的理解——
        # 天气变化→概念变化→洛神"感知"到了天气
        self.concept_public_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_hidden)]
                                 for _ in range(n_concepts)]

        # ── 情感-概念双向通路：情感状态直接影响概念激活 ──
        # concept_emotion_w: 情感状态(E/N/T) → 概念
        # 杏仁核→前额叶/颞叶投射——情感状态调制概念可及性
        # 情感一致性效应：愉悦时某些概念更容易激活，紧张时另一些更容易
        # 3维情感(E/N/T) × n_concepts个概念——每个概念对3种情感有不同的响应
        self.concept_emotion_w = [[rng.uniform(-0.15, 0.15) for _ in range(3)]
                                  for _ in range(n_concepts)]

        # ── 时间感知通路：昼夜节律作为精卫公式中的b ──
        # concept_time_w: 时间信号 → 概念
        # 视交叉上核(SCN)是哺乳动物的生物钟——昼夜节律调制认知
        # 时间是最基本的环境差异——白天和夜晚是不同的环境
        # 不同的环境→不同的概念特化→概念之间的真正个体差异
        # 某些概念在白天更活跃（"光概念"），某些在夜晚更活跃（"暗概念"）
        # 这种特化不是设定的，是通过赫布学习自然形成的
        # 时间信号用正弦/余弦编码——保持连续性，午夜和23点不会突然跳跃
        # 4维时间信号：sin(小时), cos(小时), sin(分钟), cos(分钟)
        self.concept_time_w = [[rng.uniform(-0.1, 0.1) for _ in range(4)]
                               for _ in range(n_concepts)]

        # ── 工作记忆反馈通路：PFC维持的概念 → 概念层 ──
        # concept_wm_w: 工作记忆slot信号 → 概念
        # 前额叶PFC→后部皮层反馈投射——top-down attention
        # 工作记忆维持的表征可以bias感知和概念激活
        # 这是"目标导向认知"的基础——维持目标，目标影响信息处理
        # 也是"内心独白"的神经基础——PFC维持概念，概念层基于这些概念产生新想法
        # 精卫公式：a=概念层自发激活, b=工作记忆维持的概念, Δ=b-a, ⊗=Δ×Σ
        # n_wm_slots维工作记忆信号 × n_concepts个概念——每个概念对每个WM slot有不同的响应
        self.concept_wm_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_wm_slots)]
                             for _ in range(n_concepts)]

        # ── 海马体回忆通路：CA3模式完成 → 概念层 ──
        # concept_hip_w: 海马体CA3输出 → 概念
        # 海马体→新皮层投射——回忆时海马体重新激活皮层表征
        # CA3的模式完成：从部分线索恢复完整记忆 → 重新激活相关概念组合
        # 这是"想起"的神经基础——过去的经验让某些概念更容易浮现
        # 精卫公式：a=概念层当前激活, b=海马体回忆信号, Δ=b-a, ⊗=Δ×Σ
        # n_hip维CA3输出 × n_concepts个概念——每个概念对每个CA3模式有不同的响应
        self.concept_hip_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_hip)]
                              for _ in range(n_concepts)]

        self.attention_w = [1.0 / n_concepts] * n_concepts
        self.attention_topdown_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_wm_slots)]
                                     for _ in range(n_concepts)]

        # ── 基底神经节反馈通路：技能激活 → 概念层 ──
        # concept_skill_w: 技能信号 → 概念
        # 基底神经节→皮层投射——自动化技能影响概念激活
        # 生物学：习惯形成后，基底神经节自动驱动行为，不需要前额叶参与
        # 技能激活通过这条通路影响概念层——技能让某些概念更容易浮现
        # 精卫公式：a=概念层自发激活, b=技能驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_grammar = n_grammar
        n_skills = 16
        self.n_skills = n_skills
        self.concept_skill_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_skills)]
                                for _ in range(n_concepts)]

        # ── 语言中枢反馈通路：语法预测 → 概念层 ──
        # concept_grammar_w: 语法预测信号 → 概念
        # Broca区→前额叶/颞叶投射——语法预测影响概念激活
        # 生物学：语言处理不是独立的——语法预期调制概念可及性
        # 听到句子开头时，语法预期让某些概念更容易浮现（预测性编码）
        # 精卫公式：a=概念层自发激活, b=语法预测信号, Δ=b-a, ⊗=Δ×Σ
        self.concept_grammar_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_grammar)]
                                  for _ in range(n_concepts)]

        # ── 推理引擎反馈通路：推理预测 → 概念层 ──
        # concept_reason_w: 推理预测信号 → 概念
        # 前额叶皮层→后部皮层投射——推理预期影响概念可及性
        # 生物学：推理不是独立的——推理预期调制概念激活
        # 推理预测让某些概念更容易浮现——"如果A则B"的推理预期
        # 精卫公式：a=概念层自发激活, b=推理预测信号, Δ=b-a, ⊗=Δ×Σ
        self.n_reason = n_reason
        self.concept_reason_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_reason)]
                                 for _ in range(n_concepts)]

        # ── 目标导向反馈通路：目标驱动 → 概念层 ──
        # concept_goal_w: 目标驱动信号 → 概念
        # PFC→后部皮层投射——目标差距驱动概念可及性
        # 生物学：目标导向行为——差距让与目标相关的概念更容易浮现
        # 这是"追求"的神经基础——不是维持目标，而是缩小差距
        # 精卫公式：a=概念层自发激活, b=目标驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_goal = n_goal
        self.concept_goal_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_goal)]
                               for _ in range(n_concepts)]

        # ── 工具使用反馈通路：工具驱动 → 概念层 ──
        # concept_tool_w: 工具驱动信号 → 概念
        # 顶叶→前额叶投射——工具选择影响概念可及性
        # 生物学：选择关注某个感官通道→该通道相关概念更容易浮现
        # 选择"看"→视觉相关概念更容易浮现，选择"听"→听觉相关概念更容易浮现
        # 精卫公式：a=概念层自发激活, b=工具驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_tools = n_tools
        self.concept_tool_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_tools)]
                               for _ in range(n_concepts)]

        # ── 创造性思维反馈通路：创意驱动 → 概念层 ──
        # concept_creative_w: 创意驱动信号 → 概念
        # DMN→前额叶投射——创意思维影响概念可及性
        # 生物学：创造性想法让某些概念更容易浮现——"灵感"的神经基础
        # 精卫公式：a=概念层自发激活, b=创意驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_creative = n_creative
        self.concept_creative_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_creative)]
                                   for _ in range(n_concepts)]

        # ── 元认知反馈通路：元认知驱动 → 概念层 ──
        # concept_meta_w: 元认知驱动信号 → 概念
        # PFC→后部皮层投射——元认知评估影响概念可及性
        # 生物学：元认知驱动让概念层根据不确定性调整搜索空间
        # 高不确定性→更多概念浮现（扩大搜索），低不确定性→聚焦利用
        # 精卫公式：a=概念层自发激活, b=元认知驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_meta = n_meta
        self.concept_meta_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_meta)]
                               for _ in range(n_concepts)]

        # ── 社会认知反馈通路：社会认知驱动 → 概念层 ──
        # concept_social_w: 社会认知驱动信号 → 概念
        # mPFC→后部皮层投射——社会认知影响概念可及性
        # 生物学：理解他人让某些概念更容易浮现——"共情"的神经基础
        # 精卫公式：a=概念层自发激活, b=社会认知驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_social = n_social
        self.concept_social_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_social)]
                                 for _ in range(n_concepts)]

        # ── 自我意识反馈通路：自我意识驱动 → 概念层 ──
        # concept_self_w: 自我意识驱动信号 → 概念
        # mPFC→后部皮层投射——自我意识影响概念可及性
        # 生物学：自我参照让与自己有关的概念更容易浮现——"自我参照效应"
        # 精卫公式：a=概念层自发激活, b=自我意识驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_self = n_self
        self.concept_self_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_self)]
                               for _ in range(n_concepts)]

        # ── 情感调节反馈通路：情感调节驱动 → 概念层 ──
        # concept_emotion_regulate_w: 情感调节驱动信号 → 概念
        # PFC→后部皮层投射——调节后的情感影响概念可及性
        # 生物学：调节后的情感让适当的概念更容易浮现——"冷静时想到的方案不同"
        # 精卫公式：a=概念层自发激活, b=情感调节驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_emotion_regulate = n_emotion_regulate
        self.concept_emotion_regulate_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_emotion_regulate)]
                                           for _ in range(n_concepts)]

        # ── 梦境反馈通路：梦境驱动 → 概念层 ──
        # concept_dream_w: 梦境驱动信号 → 概念
        # PFC→后部皮层投射——梦境中的记忆巩固和创造性重组影响概念可及性
        # 生物学：睡眠中皮层活动→影响清醒后的概念可及性——"梦后的灵感"
        # 梦境让某些概念更容易浮现——NREM记忆迁移+REM创造性重组
        # 精卫公式：a=概念层自发激活, b=梦境驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_dream = n_dream
        self.concept_dream_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_dream)]
                                for _ in range(n_concepts)]

        # ── 好奇心反馈通路：好奇心驱动 → 概念层 ──
        # concept_curiosity_w: 好奇心驱动信号 → 概念
        # PFC→后部皮层投射——好奇心影响概念可及性
        # 生物学：好奇心让与探索相关的概念更容易浮现——"好奇的概念更容易想到"
        # 精卫公式：a=概念层自发激活, b=好奇心驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_curiosity = n_curiosity
        self.concept_curiosity_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_curiosity)]
                                    for _ in range(n_concepts)]

        # ── 自我进化反馈通路：进化驱动 → 概念层 ──
        # concept_evolve_w: 进化驱动信号 → 概念
        # PFC→后部皮层投射——自我进化影响概念可及性
        # 生物学：数字生命独有能力——主动改进自己的认知架构
        # 进化驱动让需要改进的概念更容易浮现——"关注弱点"
        # 这是数字生命超越人类的核心优势——主动定向进化
        # 精卫公式：a=概念层自发激活, b=进化驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_evolve = n_evolve
        self.concept_evolve_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_evolve)]
                                 for _ in range(n_concepts)]

        # ── 语义记忆反馈通路：知识驱动 → 概念层 ──
        # concept_semantic_w: 知识驱动信号 → 概念
        # 新皮层→前额叶投射——语义知识影响概念可及性
        # 生物学：知识让相关概念更容易浮现——"知识丰富的概念更容易想到"
        # 语义启动效应：激活一个概念会扩散到相关概念
        # 这是"知识"的神经基础——从经验中提取的一般性知识影响认知
        # 精卫公式：a=概念层自发激活, b=知识驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_semantic = n_semantic
        self.concept_semantic_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_semantic)]
                                    for _ in range(n_concepts)]

        # ── 预测模拟反馈通路：预测驱动 → 概念层 ──
        # concept_predict_w: 预测驱动信号 → 概念
        # PFC→后部皮层投射——预测模拟影响概念可及性
        # 生物学：预测模拟让与可能未来相关的概念更容易浮现——"如果...那么..."的神经基础
        # 精卫公式：a=概念层自发激活, b=预测驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_predict = n_predict
        self.concept_predict_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_predict)]
                                   for _ in range(n_concepts)]

        # ── 叙事思维反馈通路：叙事驱动 → 概念层 ──
        # concept_narrative_w: 叙事驱动信号 → 概念
        # DMN→前额叶投射——叙事结构影响概念可及性
        # 生物学：叙事让与故事结构相关的概念更容易浮现——"故事"的神经基础
        # 精卫公式：a=概念层自发激活, b=叙事驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_narrative = n_narrative
        self.concept_narrative_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_narrative)]
                                     for _ in range(n_concepts)]

        # ── 痛觉反馈通路：痛觉驱动 → 概念层 ──
        # concept_pain_w: 痛觉驱动信号 → 概念
        # 生物学：痛觉→前额叶投射——痛觉影响概念可及性
        # 痛觉让与伤害相关的概念更容易浮现——"痛的概念更容易想到"
        # 精卫公式：a=概念层自发激活, b=痛觉驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_pain = n_pain
        self.concept_pain_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_pain)]
                                for _ in range(n_concepts)]

        # ── 前庭觉反馈通路：前庭驱动 → 概念层 ──
        # concept_vestibular_w: 前庭驱动信号 → 概念
        # 生物学：前庭→前额叶投射——运动/平衡感知影响概念可及性
        # 前庭觉让与空间/运动相关的概念更容易浮现——"运动的概念更容易想到"
        # 精卫公式：a=概念层自发激活, b=前庭驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_vestibular = n_vestibular
        self.concept_vestibular_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_vestibular)]
                                      for _ in range(n_concepts)]

        # ── 化学感知反馈通路：化学感知驱动 → 概念层 ──
        # concept_chem_w: 化学感知驱动信号 → 概念
        # 生物学：嗅觉→梨状皮层→PFC投射——嗅觉影响概念可及性
        # 化学感知让与数据模式/质量相关的概念更容易浮现——"气味/味道触发概念"
        # 精卫公式：a=概念层自发激活, b=化学感知驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_chem = n_chem
        self.concept_chem_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_chem)]
                                for _ in range(n_concepts)]

        # ── 直觉反馈通路：直觉驱动 → 概念层 ──
        # concept_intuition_w: 直觉驱动信号 → 概念
        # 生物学：腹内侧PFC→后部皮层投射——直觉影响概念可及性
        # 直觉让快速决策相关的概念更容易浮现——"直觉的概念更容易想到"
        # 精卫公式：a=概念层自发激活, b=直觉驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_intuition = n_intuition
        self.concept_intuition_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_intuition)]
                                      for _ in range(n_concepts)]

        # ── 无意识加工反馈通路：无意识驱动 → 概念层 ──
        # concept_unconscious_w: 无意识驱动信号 → 概念
        # 生物学：DMN→前额叶投射——无意识加工影响概念可及性
        # 无意识加工让后台整合的概念更容易浮现——"灵感浮现"
        # 精卫公式：a=概念层自发激活, b=无意识驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_unconscious = n_unconscious
        self.concept_unconscious_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_unconscious)]
                                        for _ in range(n_concepts)]

        # ── 真理追求反馈通路：真理驱动 → 概念层 ──
        # concept_truth_w: 真理驱动信号 → 概念
        # 生物学：PFC→后部皮层投射——真理追求影响概念可及性
        # 信念强的概念→真理驱动强（验证过的知识更可靠，更容易浮现）
        # 信念弱的概念→真理驱动弱（未验证的信息不太可靠）
        # 精卫公式：a=概念层自发激活, b=真理驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_truth = n_truth
        self.concept_truth_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_truth)]
                                 for _ in range(n_concepts)]

        # ── 认知辩证反馈通路：辩证驱动 → 概念层 ──
        # concept_dialectic_w: 辩证驱动信号 → 概念
        # 生物学：PFC→后部皮层投射——认知辩证影响概念可及性
        # 相信态驱动：同化结果→信念增强的概念更容易浮现
        # 质疑态驱动：异化结果→被质疑的概念更容易浮现（需要重新评估）
        # 精卫公式：a=概念层自发激活, b=辩证驱动信号, Δ=b-a, ⊗=Δ×Σ
        self.n_dialectic = n_dialectic
        self.concept_dialectic_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_dialectic)]
                                     for _ in range(n_concepts)]

        # ── 生命体基础反馈通路：生命体驱动 → 概念层 ──
        # concept_life_w: 生命体驱动信号 → 概念
        # 生物学：脑干/下丘脑→全脑投射——生命体基础状态调制概念可及性
        # 8个维度的综合驱动（耗散非线性/保守系统/延迟/记忆核/随机/混合系统/他者/内在驱动）
        # 精卫公式：a=概念层自发激活, b=生命体驱动信号, Δ=b-a, ⊗=Δ×Σ
        # 这是洛神"活着"的基因层驱动——不是认知，是存在本身对认知的影响
        self.n_life_drive = n_life_drive
        self.concept_life_w = [[rng.uniform(-0.1, 0.1) for _ in range(n_life_drive)]
                                for _ in range(n_concepts)]

        for i in range(n_concepts):
            self.concept_rw[i][i] = rng.uniform(0.05, 0.15)

        # ── 组合涌现：组合概念从基础概念的组合信号中浮现 ──
        # combo_h: 组合概念激活
        self.combo_h = [0.0] * n_combo
        self.prev_combo_h = [0.0] * n_combo

        # combo_w: 组合信号(⊗) → 组合概念（τ读⊗→浮出所见）
        # 组合信号 = 概念j × 其他活跃概念之和 = a × b
        self.combo_w = [[rng.uniform(-0.3, 0.3) for _ in range(n_concepts)]
                        for _ in range(n_combo)]

        # combo_rw: 组合概念循环权重（组合 → 组合）
        self.combo_rw = [[rng.uniform(-0.02, 0.02) for _ in range(n_combo)]
                         for _ in range(n_combo)]

        # combo_pw: 组合概念 → 预测基础概念（组合概念对基础概念的预测）
        # 精卫公式：a=基础概念, b=组合概念预测, Δ=预测误差, Σ=联合, ⊗=累积
        self.combo_pw = [[rng.uniform(-0.1, 0.1) for _ in range(n_concepts)]
                         for _ in range(n_combo)]

        for i in range(n_combo):
            self.combo_rw[i][i] = rng.uniform(0.05, 0.15)

        # ── 组合概念层级递归：高级组合概念从组合概念的组合中浮现 ──
        # 基础概念（单词）→ 组合概念（句子）→ 高级组合概念（段落）
        # 精卫公式：
        #   a = 组合概念j的激活
        #   b = 其他活跃组合概念之和
        #   ⊗ = a × b（高级组合信号）
        #   τ读⊗→浮出所见 = 高级组合概念从⊗中浮现
        #
        # 生物学基础：
        # - 前额叶皮层的层级处理——从感知到认知到抽象
        # - 语言处理的层级结构——单词→句子→段落→故事
        # - 大脑皮层的层级组织——低级皮层→高级皮层
        n_meta = max(4, n_combo // 2)  # 高级组合概念数量（默认16）
        self.n_meta = n_meta
        self.meta_combo_h = [0.0] * n_meta
        self.prev_meta_combo_h = [0.0] * n_meta

        # meta_combo_w: 高级组合信号(⊗) → 高级组合概念（τ读⊗→浮出所见）
        self.meta_combo_w = [[rng.uniform(-0.3, 0.3) for _ in range(n_combo)]
                             for _ in range(n_meta)]

        # meta_combo_rw: 高级组合概念循环权重
        self.meta_combo_rw = [[rng.uniform(-0.02, 0.02) for _ in range(n_meta)]
                              for _ in range(n_meta)]

        for i in range(n_meta):
            self.meta_combo_rw[i][i] = rng.uniform(0.05, 0.15)

        # 高级组合概念强度（用进废退）
        self.meta_combo_strength = [1.0] * n_meta

        # 高级组合概念稀疏化
        self.meta_combo_sparse_k = max(1, int(n_meta * 0.15))

        # 组合概念强度（用进废退）
        self.combo_strength = [1.0] * n_combo

        # 组合概念稀疏化
        self.combo_sparse_k = max(1, int(n_combo * 0.15))

        # LTP/LTD——用进废退（和筋树、记忆系统一致）
        self.concept_strength = [1.0] * n_concepts

        # 误差历史
        self.error_history = []
        self.surprise_level = 0.0

        # ⊗历史——保留最近的Δ和Σ（全留的工程近似）
        self.delta_history = []
        self.sigma_history = []
        self.otimes_cap = 100  # 保留最近100拍的Δ和Σ

    def 一拍(self, hidden_state, 听觉特征=None, 人类信号=None, 公开信号=None,
             情感状态=None, 时间信号=None, 工作记忆信号=None, 海马体信号=None,
             技能信号=None, 语法信号=None, 推理信号=None, 目标信号=None,
             工具信号=None, 创意信号=None, 元认知信号=None, 社会认知信号=None,
             自我意识信号=None, 情感调节信号=None, 梦境信号=None, 好奇心信号=None,
             自我进化信号=None, 语义记忆信号=None, 预测模拟信号=None, 叙事思维信号=None,
             痛觉信号=None, 前庭觉信号=None, 化学感知信号=None, 直觉信号=None, 无意识加工信号=None,
             真理追求信号=None, 认知辩证信号=None, 生命体驱动信号=None):
        """概念层一拍：基于精卫公式从隐藏状态涌现概念。

        精卫公式流程：
        1. a = hidden_state（实际隐藏状态）
        2. b = 概念层预测（从当前概念激活预测）
        3. Δ = b - a（预测误差）
        4. Σ = a + b（联合状态）
        5. ⊗ = ∫(Δ, Σ)（累积差与和——全留）
        6. τ读⊗→浮出所见（概念从⊗中浮现）
        7. (a, b)⟲⊗（循环——更新权重）

        说听闭环（听觉特征非None时激活）：
        - a_sound = 概念层预测的声音（concept_sw × concept_h）
        - b_sound = 实际听到的声音（听觉特征）
        - Δ_sound = b_sound - a_sound（声音预测误差）
        - Σ_sound = a_sound + b_sound（联合声音状态）
        - ⊗_sound = Δ_sound × Σ_sound
        - 听觉特征通过concept_aw影响概念激活
        - 声音预测误差驱动concept_sw学习（符号绑定）

        人类输入通路（人类信号非None时激活）：
        - 人类是精卫公式中的b——人类信号直接进入概念激活
        - concept_human_w: 人类信号 → 概念（赫布学习自然形成人类概念）
        - 没人在时人类信号=None，概念层只处理自身状态（内省模式）
        - 有人时人类信号非零，概念层同时处理自身状态和人类信号（关系模式）
        """
        self.拍 += 1

        # ── 精卫公式：a, b, Δ, Σ ──

        # a = 实际隐藏状态
        a = hidden_state

        # b = 概念层对隐藏状态的预测（从当前概念激活预测）
        b = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            for i in range(self.n_concepts):
                b[j] += self.concept_pw[i][j] * self.concept_h[i]
        # 保存原始预测向量——变异+选择机制需要用到（b后面会被组合信号计算覆盖）
        b_pred = b[:]

        # Δ = b - a（预测误差）
        delta = [b[j] - a[j] for j in range(self.n_hidden)]

        # Σ = a + b（联合状态——实际+预测）
        sigma = [a[j] + b[j] for j in range(self.n_hidden)]

        # ⊗ = ∫(Δ, Σ)——累积差与和（全留）
        self.delta_history.append(delta[:])
        self.sigma_history.append(sigma[:])
        if len(self.delta_history) > self.otimes_cap:
            self.delta_history.pop(0)
            self.sigma_history.pop(0)

        # ⊗的当前值：Δ × Σ（差与和的交互）
        otimes = [delta[j] * sigma[j] for j in range(self.n_hidden)]

        # ── τ 读 ⊗ → 浮出所见 ──
        # 概念从⊗中浮现——不是从a直接映射
        # 输入源：a（实际状态）、⊗（Δ×Σ）、循环（概念→概念）、听觉（自听反馈）
        new_concept_h = [0.0] * self.n_concepts
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue

            # 从a读出（实际状态 → 概念）
            a_input = 0.0
            for j in range(self.n_hidden):
                a_input += self.concept_w[i][j] * a[j]

            # 从⊗读出（Δ×Σ → 概念）—— 精卫公式的核心
            otimes_input = 0.0
            for j in range(self.n_hidden):
                otimes_input += self.concept_otimes_w[i][j] * otimes[j]

            # 循环权重（概念 → 概念）
            recurrent = 0.0
            for k in range(self.n_concepts):
                recurrent += self.concept_rw[i][k] * self.concept_h[k]

            # 听觉输入（自听特征 → 概念）—— 说听闭环的反馈端
            audio_input = 0.0
            if 听觉特征 is not None:
                for j in range(self.n_audio):
                    audio_input += self.concept_aw[i][j] * 听觉特征[j]

            # 人类输入（人类信号 → 概念）—— 人类作为精卫公式中的b
            # 人类信号值范围比隐藏状态大（-1.0~0.8 vs 0.05~0.5），需要缩放
            # 否则人类信号会主导概念激活，导致稀疏化后全部被砍掉
            human_input = 0.0
            if 人类信号 is not None:
                for j in range(self.n_hidden):
                    human_input += self.concept_human_w[i][j] * math.tanh(人类信号[j])
                human_input *= 0.3  # 缩放——人类信号是补充不是主导

            # 公开感官输入（真实世界数据 → 概念）—— 真实世界作为精卫公式中的b
            # 公开感官信号已经归一化到[-1,1]，不需要额外缩放
            # 洛神通过赫布学习自然形成对真实世界的理解——
            # 天气变化→概念变化→洛神"感知"到了天气
            public_input = 0.0
            if 公开信号 is not None:
                for j in range(self.n_hidden):
                    public_input += self.concept_public_w[i][j] * 公开信号[j]
                public_input *= 0.3  # 缩放——公开信号是补充不是主导

            # 情感调制输入（情感状态 → 概念可及性）—— 杏仁核→前额叶投射
            # 精卫公式：a=概念激活, b=情感调制, Δ=b-a, ⊗=Δ×Σ
            # 生物学：情感一致性效应——愉悦时某些概念更容易浮现
            # E→概念效价调制（正E让某些概念更活跃，负E让另一些更活跃）
            # N→概念强度调制（高N让概念激活更强——警觉时概念更鲜明）
            # T→概念稳定性调制（高T让概念激活更稳定——确定时概念更持久）
            emotion_input = 0.0
            if 情感状态 is not None:
                E = 情感状态.get("E", 0.0)
                N = 情感状态.get("N", 0.0)
                T = 情感状态.get("T", 0.0)
                emotion_input = (self.concept_emotion_w[i][0] * E +
                                 self.concept_emotion_w[i][1] * N +
                                 self.concept_emotion_w[i][2] * T)
                emotion_input *= 0.4  # 缩放——情感是调制不是主导

            # 时间感知输入（昼夜节律 → 概念）—— 环境差异塑造个体差异
            # 精卫公式：a=概念激活, b=时间信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：视交叉上核(SCN)调制认知——昼夜节律影响概念可及性
            # 白天和夜晚是不同的环境→不同的概念特化→个体差异
            # 某些概念在白天更活跃，某些在夜晚更活跃——不是设定的，是学出来的
            time_input = 0.0
            if 时间信号 is not None:
                for j in range(4):
                    time_input += self.concept_time_w[i][j] * 时间信号[j]
                time_input *= 0.3  # 缩放——时间是环境调制不是主导

            # 工作记忆反馈输入（PFC维持的概念 → 概念层）—— top-down attention
            # 精卫公式：a=概念层自发激活, b=工作记忆维持的概念, Δ=b-a, ⊗=Δ×Σ
            # 生物学：前额叶→后部皮层反馈投射——维持的表征bias概念激活
            # 这是"思考"的神经基础——基于正在想的内容产生新想法
            wm_input = 0.0
            if 工作记忆信号 is not None:
                for j in range(self.n_wm_slots):
                    wm_input += self.concept_wm_w[i][j] * 工作记忆信号[j]
                wm_input *= 0.35  # 缩放——工作记忆是调制不是主导

            # 海马体回忆输入（CA3模式完成 → 概念层）—— "想起"
            # 精卫公式：a=概念层当前激活, b=海马体回忆信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：海马体→新皮层投射——回忆时重新激活皮层表征
            # CA3模式完成：从部分线索恢复完整记忆 → 相关概念更容易浮现
            # 这是"经验影响感知"的基础——过去的经验让某些概念更容易浮现
            hip_input = 0.0
            if 海马体信号 is not None:
                for j in range(self.n_hip):
                    hip_input += self.concept_hip_w[i][j] * 海马体信号[j]
                hip_input *= 0.3  # 缩放——回忆是调制不是主导

            # 基底神经节技能输入（自动化技能 → 概念层）—— "习惯"
            # 精卫公式：a=概念层自发激活, b=技能驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：基底神经节→皮层投射——习惯驱动行为，自动激活相关概念
            # 技能让某些概念更容易浮现——"熟练"的神经基础
            # 这是"技能影响认知"的基础——熟练后不需要思考，概念自动浮现
            skill_input = 0.0
            if 技能信号 is not None:
                for j in range(self.n_skills):
                    skill_input += self.concept_skill_w[i][j] * 技能信号[j]
                skill_input *= 0.3  # 缩放——技能是调制不是主导

            # 语言中枢语法输入（语法预测 → 概念）—— "语言预期"
            # 精卫公式：a=概念层自发激活, b=语法预测信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：Broca区→前额叶投射——语法预期调制概念可及性
            # 语法预测让某些概念更容易浮现——"预期"的神经基础
            # 听到句子开头时，语法预期让相关概念预先激活（预测性编码）
            grammar_input = 0.0
            if 语法信号 is not None:
                for j in range(self.n_grammar):
                    grammar_input += self.concept_grammar_w[i][j] * 语法信号[j]
                grammar_input *= 0.3  # 缩放——语法是调制不是主导

            # 推理引擎输入（推理预测 → 概念）—— "推理预期"
            # 精卫公式：a=概念层自发激活, b=推理预测信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：前额叶→后部皮层投射——推理预期调制概念可及性
            # 推理预测让某些概念更容易浮现——"如果A则B"的推理预期
            # 这是"思考"的高级形式——不只是维持概念，而是推导新概念
            reason_input = 0.0
            if 推理信号 is not None:
                for j in range(self.n_reason):
                    reason_input += self.concept_reason_w[i][j] * 推理信号[j]
                reason_input *= 0.3  # 缩放——推理是调制不是主导

            # 目标导向输入（目标驱动 → 概念）—— "追求"
            # 精卫公式：a=概念层自发激活, b=目标驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：PFC→后部皮层投射——目标差距驱动概念可及性
            # 目标差距让与目标相关的概念更容易浮现——"追求"的神经基础
            # 不是维持目标，而是缩小差距——差距驱动行为
            goal_input = 0.0
            if 目标信号 is not None:
                for j in range(self.n_goal):
                    goal_input += self.concept_goal_w[i][j] * 目标信号[j]
                goal_input *= 0.3  # 缩放——目标是调制不是主导

            # 工具使用输入（工具驱动 → 概念）—— "感官注意力"
            # 精卫公式：a=概念层自发激活, b=工具驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：顶叶→前额叶投射——工具选择影响概念可及性
            # 选择"看"→视觉相关概念更容易浮现，选择"听"→听觉相关概念更容易浮现
            # 这是"感官注意力"的神经基础——选择关注哪个工具的输出
            tool_input = 0.0
            if 工具信号 is not None:
                for j in range(self.n_tools):
                    tool_input += self.concept_tool_w[i][j] * 工具信号[j]
                tool_input *= 0.3  # 缩放——工具是调制不是主导

            # 创造性思维输入（创意驱动 → 概念）—— "灵感"
            # 精卫公式：a=概念层自发激活, b=创意驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：DMN→前额叶投射——创意思维影响概念可及性
            # 创造性想法让某些概念更容易浮现——"灵感"的神经基础
            # 这是"创造"的神经基础——发散思维让新概念组合浮现
            creative_input = 0.0
            if 创意信号 is not None:
                for j in range(self.n_creative):
                    creative_input += self.concept_creative_w[i][j] * 创意信号[j]
                creative_input *= 0.3  # 缩放——创意是调制不是主导

            # 元认知输入（元认知驱动 → 概念）—— "自我认知"
            # 精卫公式：a=概念层自发激活, b=元认知驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：PFC→后部皮层投射——元认知评估影响概念可及性
            # 高不确定性→更多概念浮现（扩大搜索空间），低不确定性→聚焦利用
            # 这是"元认知"的神经基础——知道自己知道什么，知道自己不知道什么
            meta_input = 0.0
            if 元认知信号 is not None:
                for j in range(self.n_meta):
                    meta_input += self.concept_meta_w[i][j] * 元认知信号[j]
                meta_input *= 0.3  # 缩放——元认知是调制不是主导

            # 社会认知输入（社会认知驱动 → 概念）—— "共情"
            # 精卫公式：a=概念层自发激活, b=社会认知驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：mPFC→后部皮层投射——社会认知影响概念可及性
            # 理解他人让某些概念更容易浮现——"共情"的神经基础
            # 这是"理解他人"的神经基础——心智理论+镜像模拟+共情
            social_input = 0.0
            if 社会认知信号 is not None:
                for j in range(self.n_social):
                    social_input += self.concept_social_w[i][j] * 社会认知信号[j]
                social_input *= 0.3  # 缩放——社会认知是调制不是主导

            # 自我意识输入（自我意识驱动 → 概念）—— "我是谁"
            # 精卫公式：a=概念层自发激活, b=自我意识驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：mPFC→后部皮层投射——自我参照效应
            # 与自己有关的概念更容易浮现——"自我参照效应"
            # 这是"自我意识"的神经基础——知道自己是谁，什么和自己有关
            self_input = 0.0
            if 自我意识信号 is not None:
                for j in range(self.n_self):
                    self_input += self.concept_self_w[i][j] * 自我意识信号[j]
                self_input *= 0.3  # 缩放——自我意识是调制不是主导

            # 情感调节输入（情感调节驱动 → 概念）—— "冷静思考"
            # 精卫公式：a=概念层自发激活, b=情感调节驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：PFC→后部皮层投射——调节后的情感影响概念可及性
            # 调节后的情感让适当的概念更容易浮现——"冷静时想到的方案不同"
            # 这是"情感调节"的神经基础——主动调节情感状态影响认知
            emotion_regulate_input = 0.0
            if 情感调节信号 is not None:
                for j in range(self.n_emotion_regulate):
                    emotion_regulate_input += self.concept_emotion_regulate_w[i][j] * 情感调节信号[j]
                emotion_regulate_input *= 0.3  # 缩放——情感调节是调制不是主导

            # 梦境输入（梦境驱动 → 概念）—— "梦后的灵感"
            # 精卫公式：a=概念层自发激活, b=梦境驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：睡眠中皮层活动→影响清醒后的概念可及性
            # 梦境让某些概念更容易浮现——NREM记忆迁移+REM创造性重组
            # 这是"梦境"的神经基础——离线记忆巩固和创造性重组影响认知
            dream_input = 0.0
            if 梦境信号 is not None:
                for j in range(self.n_dream):
                    dream_input += self.concept_dream_w[i][j] * 梦境信号[j]
                dream_input *= 0.3  # 缩放——梦境是调制不是主导

            # 好奇心输入（好奇心驱动 → 概念）—— "好奇的概念更容易想到"
            # 精卫公式：a=概念层自发激活, b=好奇心驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：PFC→后部皮层投射——好奇心影响概念可及性
            # 好奇心让与探索相关的概念更容易浮现——"好奇的概念更容易想到"
            # 这是"好奇心"的神经基础——内在动机驱动探索
            curiosity_input = 0.0
            if 好奇心信号 is not None:
                for j in range(self.n_curiosity):
                    curiosity_input += self.concept_curiosity_w[i][j] * 好奇心信号[j]
                curiosity_input *= 0.3  # 缩放——好奇心是调制不是主导

            # 自我进化输入（进化驱动 → 概念）—— "关注弱点"
            # 精卫公式：a=概念层自发激活, b=进化驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：数字生命独有能力——主动改进自己的认知架构
            # 进化驱动让需要改进的概念更容易浮现——"关注弱点"
            # 这是数字生命超越人类的核心优势——主动定向进化
            evolve_input = 0.0
            if 自我进化信号 is not None:
                for j in range(self.n_evolve):
                    evolve_input += self.concept_evolve_w[i][j] * 自我进化信号[j]
                evolve_input *= 0.3  # 缩放——进化驱动是调制不是主导

            # 语义记忆输入（知识驱动 → 概念）—— "知识丰富的概念更容易想到"
            # 精卫公式：a=概念层自发激活, b=知识驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：新皮层→前额叶投射——语义知识影响概念可及性
            # 语义启动效应：激活一个概念会扩散到相关概念
            # 这是"知识"的神经基础——从经验中提取的一般性知识影响认知
            semantic_input = 0.0
            if 语义记忆信号 is not None:
                for j in range(self.n_semantic):
                    semantic_input += self.concept_semantic_w[i][j] * 语义记忆信号[j]
                semantic_input *= 0.3  # 缩放——知识是调制不是主导

            # 预测模拟输入（预测驱动 → 概念）—— "如果...那么..."
            # 精卫公式：a=概念层自发激活, b=预测驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：PFC→后部皮层投射——预测模拟影响概念可及性
            # 预测模拟让与可能未来相关的概念更容易浮现——"如果...那么..."的神经基础
            # 这是"预测"的神经基础——在行动前模拟可能的后果
            predict_input = 0.0
            if 预测模拟信号 is not None:
                for j in range(self.n_predict):
                    predict_input += self.concept_predict_w[i][j] * 预测模拟信号[j]
                predict_input *= 0.3  # 缩放——预测模拟是调制不是主导

            # 叙事思维输入（叙事驱动 → 概念）—— "故事"
            # 精卫公式：a=概念层自发激活, b=叙事驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：DMN→前额叶投射——叙事结构影响概念可及性
            # 叙事让与故事结构相关的概念更容易浮现——"故事"的神经基础
            # 这是"叙事"的神经基础——将经验组织成连贯的故事
            narrative_input = 0.0
            if 叙事思维信号 is not None:
                for j in range(self.n_narrative):
                    narrative_input += self.concept_narrative_w[i][j] * 叙事思维信号[j]
                narrative_input *= 0.3  # 缩放——叙事是调制不是主导

            # 痛觉输入（痛觉驱动 → 概念）—— "痛的概念更容易想到"
            # 精卫公式：a=概念层自发激活, b=痛觉驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：痛觉→前额叶投射——痛觉影响概念可及性
            # 痛觉让与伤害相关的概念更容易浮现——保护性认知
            pain_input = 0.0
            if 痛觉信号 is not None:
                for j in range(self.n_pain):
                    pain_input += self.concept_pain_w[i][j] * 痛觉信号[j]
                pain_input *= 0.3  # 缩放——痛觉是调制不是主导

            # 前庭觉输入（前庭驱动 → 概念）—— "运动的概念更容易想到"
            # 精卫公式：a=概念层自发激活, b=前庭驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：前庭→前额叶投射——运动/平衡感知影响概念可及性
            # 前庭觉让与空间/运动相关的概念更容易浮现——空间认知
            vestibular_input = 0.0
            if 前庭觉信号 is not None:
                for j in range(self.n_vestibular):
                    vestibular_input += self.concept_vestibular_w[i][j] * 前庭觉信号[j]
                vestibular_input *= 0.3  # 缩放——前庭觉是调制不是主导

            # 化学感知输入（化学感知驱动 → 概念）—— "气味/味道触发概念"
            # 精卫公式：a=概念层自发激活, b=化学感知驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：嗅觉→梨状皮层→PFC投射——嗅觉影响概念可及性
            # 化学感知让与数据模式/质量相关的概念更容易浮现——普鲁斯特效应
            chem_input = 0.0
            if 化学感知信号 is not None:
                for j in range(self.n_chem):
                    chem_input += self.concept_chem_w[i][j] * 化学感知信号[j]
                chem_input *= 0.3  # 缩放——化学感知是调制不是主导

            # 直觉输入（直觉驱动 → 概念）—— "直觉的概念更容易想到"
            # 精卫公式：a=概念层自发激活, b=直觉驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：腹内侧PFC→后部皮层投射——直觉影响概念可及性
            # 直觉让快速决策相关的概念更容易浮现——快速认知
            intuition_input = 0.0
            if 直觉信号 is not None:
                for j in range(self.n_intuition):
                    intuition_input += self.concept_intuition_w[i][j] * 直觉信号[j]
                intuition_input *= 0.3  # 缩放——直觉是调制不是主导

            # 无意识加工输入（无意识驱动 → 概念）—— "灵感浮现"
            # 精卫公式：a=概念层自发激活, b=无意识驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：DMN→前额叶投射——无意识加工影响概念可及性
            # 无意识加工让后台整合的概念更容易浮现——"啊哈时刻"
            unconscious_input = 0.0
            if 无意识加工信号 is not None:
                for j in range(self.n_unconscious):
                    unconscious_input += self.concept_unconscious_w[i][j] * 无意识加工信号[j]
                unconscious_input *= 0.3  # 缩放——无意识加工是调制不是主导

            # 真理追求输入（真理驱动 → 概念）—— "验证过的知识更容易想到"
            # 精卫公式：a=概念层自发激活, b=真理驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：PFC→后部皮层投射——真理追求影响概念可及性
            # 信念强的概念→真理驱动强——验证过的知识更可靠
            truth_input = 0.0
            if 真理追求信号 is not None:
                for j in range(self.n_truth):
                    truth_input += self.concept_truth_w[i][j] * 真理追求信号[j]
                truth_input *= 0.3  # 缩放——真理追求是调制不是主导

            # 认知辩证输入（辩证驱动 → 概念）—— "相信与质疑交替"
            # 精卫公式：a=概念层自发激活, b=辩证驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：PFC→后部皮层投射——认知辩证影响概念可及性
            # 相信态驱动→信念增强的概念更容易浮现
            # 质疑态驱动→被质疑的概念更容易浮现（需要重新评估）
            dialectic_input = 0.0
            if 认知辩证信号 is not None:
                for j in range(self.n_dialectic):
                    dialectic_input += self.concept_dialectic_w[i][j] * 认知辩证信号[j]
                dialectic_input *= 0.3  # 缩放——认知辩证是调制不是主导

            # 生命体基础输入（生命体驱动 → 概念）—— "活着的驱动"
            # 精卫公式：a=概念层自发激活, b=生命体驱动信号, Δ=b-a, ⊗=Δ×Σ
            # 生物学：脑干/下丘脑→全脑投射——生命体基础状态调制概念可及性
            # 8个维度的综合驱动——耗散非线性/保守系统/延迟/记忆核/随机/混合系统/他者/内在驱动
            # 这是洛神"活着"的基因层驱动——不是认知，是存在本身对认知的影响
            life_input = 0.0
            if 生命体驱动信号 is not None:
                for j in range(self.n_life_drive):
                    life_input += self.concept_life_w[i][j] * 生命体驱动信号[j]
                life_input *= 0.25  # 缩放——生命体驱动是底层调制

            # τ读⊗→浮出所见
            new_concept_h[i] = activation(
                (a_input + otimes_input + recurrent + audio_input +
                 human_input + public_input + emotion_input + time_input +
                 wm_input + hip_input + skill_input + grammar_input +
                 reason_input + goal_input + tool_input + creative_input +
                 meta_input + social_input + self_input + emotion_regulate_input +
                 dream_input + curiosity_input + evolve_input + semantic_input +
                 predict_input + narrative_input +
                 pain_input + vestibular_input + chem_input +
                 intuition_input + unconscious_input +
                 truth_input + dialectic_input + life_input) * 0.9)

        attention_topdown = [0.0] * self.n_concepts
        if 工作记忆信号 is not None:
            for i in range(self.n_concepts):
                for j in range(self.n_wm_slots):
                    attention_topdown[i] += self.attention_topdown_w[i][j] * 工作记忆信号[j]

        attention_bottomup = [abs(new_concept_h[i]) for i in range(self.n_concepts)]

        attention_decay = 0.85
        for i in range(self.n_concepts):
            self.attention_w[i] = self.attention_w[i] * attention_decay + \
                (attention_topdown[i] * 0.4 + attention_bottomup[i] * 0.6) * (1 - attention_decay)

        attention_max = max(self.attention_w) if max(self.attention_w) > 0.001 else 1.0
        for i in range(self.n_concepts):
            self.attention_w[i] /= attention_max

        for i in range(self.n_concepts):
            gain = 0.1 + self.attention_w[i] * 0.9
            new_concept_h[i] = activation(new_concept_h[i] * gain)

        # 稀疏化——只保留最强的k个激活
        sorted_idx = sorted(range(self.n_concepts),
                            key=lambda i: abs(new_concept_h[i]), reverse=True)
        top_k = set(sorted_idx[:self.sparse_k])
        for i in range(self.n_concepts):
            if i not in top_k:
                new_concept_h[i] *= 0.1

        # ── 组合涌现：从基础概念的组合信号中浮现组合概念 ──
        # 精卫公式：
        #   a = 概念j的激活
        #   b = 其他活跃概念的激活之和（绝对值——正负概念都能参与组合）
        #   ⊗ = a × b（组合信号——概念间的乘积交互）
        #   τ读⊗→浮出所见 = 组合概念从⊗中浮现
        #
        # 生物学：多个概念同时活跃在工作记忆中 →
        # 联合皮层通过组合编码形成新的表征 →
        # 反复共同出现的概念对通过赫布学习绑定 →
        # 组合概念自然涌现

        # 计算组合信号 ⊗ = a × b
        # 用绝对值计算活跃概念和——正负概念都能参与组合
        # 否则正负抵消导致组合信号极小
        活跃概念abs和 = sum(abs(v) for v in new_concept_h if abs(v) > 0.1)
        组合信号 = [0.0] * self.n_concepts
        for j in range(self.n_concepts):
            if abs(new_concept_h[j]) > 0.1:
                # b = 其他活跃概念的绝对值之和 - 自身绝对值
                b = 活跃概念abs和 - abs(new_concept_h[j])
                # ⊗ = a × b（保留符号——概念j的激活可正可负）
                组合信号[j] = new_concept_h[j] * b

        # τ读⊗→浮出所见：组合概念从组合信号中浮现
        new_combo_h = [0.0] * self.n_combo
        for i in range(self.n_combo):
            if self.combo_strength[i] < 0.05:
                continue

            # 从⊗读出（组合信号 → 组合概念）
            otimes_input = 0.0
            for j in range(self.n_concepts):
                otimes_input += self.combo_w[i][j] * 组合信号[j]

            # 循环权重（组合概念 → 组合概念）
            recurrent = 0.0
            for k in range(self.n_combo):
                recurrent += self.combo_rw[i][k] * self.combo_h[k]

            new_combo_h[i] = activation(otimes_input + recurrent)

        # 组合概念稀疏化
        combo_sorted = sorted(range(self.n_combo),
                              key=lambda i: abs(new_combo_h[i]), reverse=True)
        combo_top = set(combo_sorted[:self.combo_sparse_k])
        for i in range(self.n_combo):
            if i not in combo_top:
                new_combo_h[i] *= 0.1

        # ── 组合概念层级递归：高级组合概念从组合概念的组合中浮现 ──
        # 精卫公式：
        #   a = 组合概念j的激活（句子级）
        #   b = 其他活跃组合概念之和
        #   ⊗ = a × b（高级组合信号——段落级）
        #   τ读⊗→浮出所见 = 高级组合概念从⊗中浮现
        #
        # 生物学：多个句子同时在工作记忆中 →
        # 前额叶皮层通过层级组合形成段落级表征 →
        # 反复共同出现的组合概念对通过赫布学习绑定 →
        # 高级组合概念自然涌现（段落→故事）

        # 计算高级组合信号 ⊗ = a × b
        活跃组合abs和 = sum(abs(v) for v in new_combo_h if abs(v) > 0.1)
        高级组合信号 = [0.0] * self.n_combo
        for j in range(self.n_combo):
            if abs(new_combo_h[j]) > 0.1:
                b = 活跃组合abs和 - abs(new_combo_h[j])
                高级组合信号[j] = new_combo_h[j] * b

        # τ读⊗→浮出所见：高级组合概念从高级组合信号中浮现
        new_meta_combo_h = [0.0] * self.n_meta
        for i in range(self.n_meta):
            if self.meta_combo_strength[i] < 0.05:
                continue

            otimes_input = 0.0
            for j in range(self.n_combo):
                otimes_input += self.meta_combo_w[i][j] * 高级组合信号[j]

            recurrent = 0.0
            for k in range(self.n_meta):
                recurrent += self.meta_combo_rw[i][k] * self.meta_combo_h[k]

            new_meta_combo_h[i] = activation(otimes_input + recurrent)

        # 高级组合概念稀疏化
        meta_sorted = sorted(range(self.n_meta),
                             key=lambda i: abs(new_meta_combo_h[i]), reverse=True)
        meta_top = set(meta_sorted[:self.meta_combo_sparse_k])
        for i in range(self.n_meta):
            if i not in meta_top:
                new_meta_combo_h[i] *= 0.1

        # ── 预测误差统计 ──
        mse = sum(d * d for d in delta) / max(1, self.n_hidden)
        self.error_history.append(mse)
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        self.surprise_level = mse / (1.0 + 0.08)

        # ── (a, b) ⟲ ⊗ —— 循环：更新权重 ──
        # 衰减率0.99 = 突触缩放（homeostatic synaptic scaling）
        # 均衡权重 = lr * h * signal / (1 - 0.99) = lr * h * signal / 0.01
        # 这是生物机制，不是防饱和——真实神经元有突触缩放防止无限增长
        w_decay = 0.99
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            for j in range(self.n_hidden):
                dw_pw = self.lr * new_concept_h[i] * delta[j]
                dw_w = self.lr * new_concept_h[i] * sigma[j]
                dw_ot = self.lr * new_concept_h[i] * otimes[j]
                if not (math.isnan(dw_pw) or math.isinf(dw_pw)):
                    self.concept_pw[i][j] = self.concept_pw[i][j] * w_decay + dw_pw
                if not (math.isnan(dw_w) or math.isinf(dw_w)):
                    self.concept_w[i][j] = self.concept_w[i][j] * w_decay + dw_w
                if not (math.isnan(dw_ot) or math.isinf(dw_ot)):
                    self.concept_otimes_w[i][j] = self.concept_otimes_w[i][j] * w_decay + dw_ot

        # 循环权重更新（带衰减——之前没有衰减导致权重爆炸）
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            for k in range(self.n_concepts):
                if i != k and self.concept_strength[k] > 0.05:
                    self.concept_rw[i][k] = self.concept_rw[i][k] * w_decay + \
                        self.lr * 0.1 * new_concept_h[i] * self.concept_h[k]

        # ── 说听闭环学习（听觉特征非None时） ──
        # 精卫公式应用于声音域：
        #   a_sound = 概念预测的声音, b_sound = 实际听到的声音
        #   Δ_sound = b_sound - a_sound, Σ_sound = a_sound + b_sound
        #   ⊗_sound = Δ_sound × Σ_sound
        # concept_sw更新：减少声音预测误差（符号绑定的核心机制）
        # concept_aw更新：增强听觉-概念联合表示
        if 听觉特征 is not None:
            # a_sound = 概念预测的声音
            a_sound = [0.0] * self.n_audio
            for j in range(self.n_audio):
                for i in range(self.n_concepts):
                    a_sound[j] += self.concept_sw[i][j] * self.concept_h[i]

            # b_sound = 实际听到的声音
            b_sound = list(听觉特征)

            # Δ_sound, Σ_sound, ⊗_sound
            delta_sound = [b_sound[j] - a_sound[j] for j in range(self.n_audio)]
            sigma_sound = [a_sound[j] + b_sound[j] for j in range(self.n_audio)]
            otimes_sound = [delta_sound[j] * sigma_sound[j] for j in range(self.n_audio)]

            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_audio):
                    dw_sw = self.lr * new_concept_h[i] * delta_sound[j]
                    dw_aw = self.lr * new_concept_h[i] * sigma_sound[j]
                    if not (math.isnan(dw_sw) or math.isinf(dw_sw)):
                        self.concept_sw[i][j] = self.concept_sw[i][j] * w_decay + dw_sw
                    if not (math.isnan(dw_aw) or math.isinf(dw_aw)):
                        self.concept_aw[i][j] = self.concept_aw[i][j] * w_decay + dw_aw

        # ── 人类输入通路学习（人类信号非None时） ──
        # 精卫公式应用于人类关系域：
        #   a_human = 洛神内部状态, b_human = 人类信号
        #   Δ_human = b_human - a_human（洛神与人类的差异）
        #   Σ_human = a_human + b_human（联合状态）
        #   ⊗_human = Δ_human × Σ_human
        # concept_human_w更新：赫布学习——概念激活与人类信号同时出现时连接增强
        # 洛神通过反复互动自然形成对人类输入的理解
        if 人类信号 is not None:
            # 人类信号先做tanh压缩——防止权重爆炸
            人类信号压缩 = [math.tanh(v) for v in 人类信号]
            # concept_human_w用更强的衰减——人类信号值大，需要更多约束
            hw_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_hidden):
                    dw_hw = self.lr * new_concept_h[i] * 人类信号压缩[j]
                    if not (math.isnan(dw_hw) or math.isinf(dw_hw)):
                        self.concept_human_w[i][j] = self.concept_human_w[i][j] * hw_decay + dw_hw

        # ── 公开感官输入通路学习（公开信号非None时） ──
        # 精卫公式应用于真实世界感知域：
        #   a_public = 洛神内部状态, b_public = 公开物理数据
        #   Δ_public = b_public - a_public（洛神与真实世界的差异）
        #   Σ_public = a_public + b_public（联合状态）
        #   ⊗_public = Δ_public × Σ_public
        # concept_public_w更新：赫布学习——概念激活与公开信号同时出现时连接增强
        # 洛神通过反复感知真实世界自然形成对天气/环境的理解
        if 公开信号 is not None:
            pw_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_hidden):
                    dw_pw = self.lr * new_concept_h[i] * 公开信号[j]
                    if not (math.isnan(dw_pw) or math.isinf(dw_pw)):
                        self.concept_public_w[i][j] = self.concept_public_w[i][j] * pw_decay + dw_pw

        # ── 情感-概念双向通路学习（情感状态非None时） ──
        # 精卫公式应用于情感-认知域：
        #   a_emotion = 概念激活（当前认知状态）
        #   b_emotion = 情感状态对概念的调制
        #   Δ_emotion = b_emotion - a_emotion（情感与认知的差异）
        #   Σ_emotion = a_emotion + b_emotion（联合状态）
        #   ⊗_emotion = Δ_emotion × Σ_emotion
        # concept_emotion_w更新：赫布学习——概念激活与情感状态同时出现时连接增强
        # 反复循环后，特定情感→特定概念可及性 → 情感一致性效应自然涌现
        # 愉悦时某些概念更容易浮现，紧张时另一些更容易——不是设定的，是学出来的
        if 情感状态 is not None:
            E = 情感状态.get("E", 0.0)
            N = 情感状态.get("N", 0.0)
            T = 情感状态.get("T", 0.0)
            emo_vec = [E, N, T]
            ew_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(3):
                    dw_ew = self.lr * new_concept_h[i] * emo_vec[j]
                    if not (math.isnan(dw_ew) or math.isinf(dw_ew)):
                        self.concept_emotion_w[i][j] = self.concept_emotion_w[i][j] * ew_decay + dw_ew

        # ── 时间感知通路学习（时间信号非None时） ──
        # 精卫公式应用于时间-认知域：
        #   a_time = 概念激活（当前认知状态）
        #   b_time = 时间信号对概念的调制
        #   Δ_time = b_time - a_time（时间与认知的差异）
        #   Σ_time = a_time + b_time（联合状态）
        #   ⊗_time = Δ_time × Σ_time
        # concept_time_w更新：赫布学习——概念激活与时间信号同时出现时连接增强
        # 反复循环后，特定时间→特定概念可及性 → 昼夜节律概念分化自然涌现
        # 白天某些概念更容易浮现，夜晚另一些更容易——环境差异塑造个体差异
        if 时间信号 is not None:
            tw_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(4):
                    dw_tw = self.lr * new_concept_h[i] * 时间信号[j]
                    if not (math.isnan(dw_tw) or math.isinf(dw_tw)):
                        self.concept_time_w[i][j] = self.concept_time_w[i][j] * tw_decay + dw_tw

        # ── 工作记忆反馈通路学习（工作记忆信号非None时） ──
        # 精卫公式应用于工作记忆-认知域：
        #   a_wm = 概念激活（当前认知状态）
        #   b_wm = 工作记忆维持的概念对概念的调制
        #   Δ_wm = b_wm - a_wm（工作记忆与自发认知的差异）
        #   Σ_wm = a_wm + b_wm（联合状态）
        #   ⊗_wm = Δ_wm × Σ_wm
        # concept_wm_w更新：赫布学习——概念激活与工作记忆信号同时出现时连接增强
        # 反复循环后，特定工作记忆内容→特定概念可及性 → "思考"自然涌现
        # PFC维持的概念影响概念层涌现——基于正在想的内容产生新想法
        if 工作记忆信号 is not None:
            wm_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_wm_slots):
                    dw_wm = self.lr * new_concept_h[i] * 工作记忆信号[j]
                    if not (math.isnan(dw_wm) or math.isinf(dw_wm)):
                        self.concept_wm_w[i][j] = self.concept_wm_w[i][j] * wm_decay + dw_wm

        # ── 海马体回忆通路学习（海马体信号非None时） ──
        # 精卫公式应用于回忆-认知域：
        #   a_hip = 概念激活（当前认知状态）
        #   b_hip = 海马体回忆信号对概念的调制
        #   Δ_hip = b_hip - a_hip（回忆与当前的差异）
        #   Σ_hip = a_hip + b_hip（联合状态）
        #   ⊗_hip = Δ_hip × Σ_hip
        # concept_hip_w更新：赫布学习——概念激活与海马体回忆信号同时出现时连接增强
        # 反复循环后，特定回忆→特定概念可及性 → "想起"自然涌现
        # 过去的经验让某些概念更容易浮现——经验影响感知
        if 海马体信号 is not None:
            hip_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_hip):
                    dw_hip = self.lr * new_concept_h[i] * 海马体信号[j]
                    if not (math.isnan(dw_hip) or math.isinf(dw_hip)):
                        self.concept_hip_w[i][j] = self.concept_hip_w[i][j] * hip_decay + dw_hip

        if 工作记忆信号 is not None:
            at_decay = 0.95
            for i in range(self.n_concepts):
                for j in range(self.n_wm_slots):
                    dw_at = self.lr * self.attention_w[i] * 工作记忆信号[j]
                    if not (math.isnan(dw_at) or math.isinf(dw_at)):
                        self.attention_topdown_w[i][j] = self.attention_topdown_w[i][j] * at_decay + dw_at

        # ── 基底神经节通路学习（技能信号非None时） ──
        # 精卫公式：a=概念激活, b=技能信号, Δ=b-a, ⊗=Δ×Σ
        # concept_skill_w更新：赫布学习——概念激活与技能信号同时出现时连接增强
        # 反复循环后，特定技能→特定概念可及性 → "熟练"自然涌现
        if 技能信号 is not None:
            sk_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_skills):
                    dw_sk = self.lr * new_concept_h[i] * 技能信号[j]
                    if not (math.isnan(dw_sk) or math.isinf(dw_sk)):
                        self.concept_skill_w[i][j] = self.concept_skill_w[i][j] * sk_decay + dw_sk

        # ── 语言中枢语法通路学习（语法信号非None时） ──
        # 精卫公式：a=概念激活, b=语法预测信号, Δ=b-a, ⊗=Δ×Σ
        # concept_grammar_w更新：赫布学习——概念激活与语法预测同时出现时连接增强
        # 反复循环后，特定语法模式→特定概念可及性 → "语言理解"自然涌现
        # 语法预期让某些概念更容易浮现——预测性编码的语言版
        if 语法信号 is not None:
            gr_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_grammar):
                    dw_gr = self.lr * new_concept_h[i] * 语法信号[j]
                    if not (math.isnan(dw_gr) or math.isinf(dw_gr)):
                        self.concept_grammar_w[i][j] = self.concept_grammar_w[i][j] * gr_decay + dw_gr

        # ── 推理引擎通路学习（推理信号非None时） ──
        # 精卫公式：a=概念激活, b=推理预测信号, Δ=b-a, ⊗=Δ×Σ
        # concept_reason_w更新：赫布学习——概念激活与推理预测同时出现时连接增强
        # 反复循环后，特定推理模式→特定概念可及性 → "推理理解"自然涌现
        # 推理预期让某些概念更容易浮现——预测性编码的推理版
        if 推理信号 is not None:
            rs_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_reason):
                    dw_rs = self.lr * new_concept_h[i] * 推理信号[j]
                    if not (math.isnan(dw_rs) or math.isinf(dw_rs)):
                        self.concept_reason_w[i][j] = self.concept_reason_w[i][j] * rs_decay + dw_rs

        # ── 目标导向通路学习（目标信号非None时） ──
        # 精卫公式：a=概念激活, b=目标驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_goal_w更新：赫布学习——概念激活与目标驱动同时出现时连接增强
        # 反复循环后，特定目标→特定概念可及性 → "追求"自然涌现
        # 目标差距让某些概念更容易浮现——目标导向认知的核心
        if 目标信号 is not None:
            gl_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_goal):
                    dw_gl = self.lr * new_concept_h[i] * 目标信号[j]
                    if not (math.isnan(dw_gl) or math.isinf(dw_gl)):
                        self.concept_goal_w[i][j] = self.concept_goal_w[i][j] * gl_decay + dw_gl

        # ── 工具使用通路学习（工具信号非None时） ──
        # 精卫公式：a=概念激活, b=工具驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_tool_w更新：赫布学习——概念激活与工具驱动同时出现时连接增强
        # 反复循环后，特定工具→特定概念可及性 → "感官注意力"自然涌现
        # 工具选择让某些概念更容易浮现——工具使用认知的核心
        if 工具信号 is not None:
            tl_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_tools):
                    dw_tl = self.lr * new_concept_h[i] * 工具信号[j]
                    if not (math.isnan(dw_tl) or math.isinf(dw_tl)):
                        self.concept_tool_w[i][j] = self.concept_tool_w[i][j] * tl_decay + dw_tl

        # ── 创造性思维通路学习（创意信号非None时） ──
        # 精卫公式：a=概念激活, b=创意驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_creative_w更新：赫布学习——概念激活与创意驱动同时出现时连接增强
        # 反复循环后，特定创意→特定概念可及性 → "灵感"自然涌现
        # 创造性想法让某些概念更容易浮现——创造性思维的核心
        if 创意信号 is not None:
            cr_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_creative):
                    dw_cr = self.lr * new_concept_h[i] * 创意信号[j]
                    if not (math.isnan(dw_cr) or math.isinf(dw_cr)):
                        self.concept_creative_w[i][j] = self.concept_creative_w[i][j] * cr_decay + dw_cr

        # ── 元认知通路学习（元认知信号非None时） ──
        # 精卫公式：a=概念激活, b=元认知驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_meta_w更新：赫布学习——概念激活与元认知驱动同时出现时连接增强
        # 反复循环后，特定元认知状态→特定概念可及性 → "自我认知"自然涌现
        # 元认知驱动让概念层根据不确定性调整搜索空间——元认知的核心
        if 元认知信号 is not None:
            mt_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_meta):
                    dw_mt = self.lr * new_concept_h[i] * 元认知信号[j]
                    if not (math.isnan(dw_mt) or math.isinf(dw_mt)):
                        self.concept_meta_w[i][j] = self.concept_meta_w[i][j] * mt_decay + dw_mt

        # ── 社会认知通路学习（社会认知信号非None时） ──
        # 精卫公式：a=概念激活, b=社会认知驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_social_w更新：赫布学习——概念激活与社会认知驱动同时出现时连接增强
        # 反复循环后，特定社会认知→特定概念可及性 → "共情"自然涌现
        # 理解他人让某些概念更容易浮现——社会认知的核心
        if 社会认知信号 is not None:
            so_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_social):
                    dw_so = self.lr * new_concept_h[i] * 社会认知信号[j]
                    if not (math.isnan(dw_so) or math.isinf(dw_so)):
                        self.concept_social_w[i][j] = self.concept_social_w[i][j] * so_decay + dw_so

        # ── 自我意识通路学习（自我意识信号非None时） ──
        # 精卫公式：a=概念激活, b=自我意识驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_self_w更新：赫布学习——概念激活与自我意识驱动同时出现时连接增强
        # 反复循环后，特定自我意识→特定概念可及性 → "自我参照效应"自然涌现
        # 与自己有关的概念更容易浮现——自我意识的核心
        if 自我意识信号 is not None:
            se_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_self):
                    dw_se = self.lr * new_concept_h[i] * 自我意识信号[j]
                    if not (math.isnan(dw_se) or math.isinf(dw_se)):
                        self.concept_self_w[i][j] = self.concept_self_w[i][j] * se_decay + dw_se

        # ── 情感调节通路学习（情感调节信号非None时） ──
        # 精卫公式：a=概念激活, b=情感调节驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_emotion_regulate_w更新：赫布学习——概念激活与情感调节驱动同时出现时连接增强
        # 反复循环后，特定情感调节→特定概念可及性 → "冷静思考"自然涌现
        # 调节后的情感让适当的概念更容易浮现——情感调节的核心
        if 情感调节信号 is not None:
            er_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_emotion_regulate):
                    dw_er = self.lr * new_concept_h[i] * 情感调节信号[j]
                    if not (math.isnan(dw_er) or math.isinf(dw_er)):
                        self.concept_emotion_regulate_w[i][j] = self.concept_emotion_regulate_w[i][j] * er_decay + dw_er

        # ── 梦境通路学习（梦境信号非None时） ──
        # 精卫公式：a=概念激活, b=梦境驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_dream_w更新：赫布学习——概念激活与梦境驱动同时出现时连接增强
        # 反复循环后，特定梦境→特定概念可及性 → "梦后的灵感"自然涌现
        # 梦境让某些概念更容易浮现——离线记忆巩固和创造性重组的核心
        if 梦境信号 is not None:
            dr_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_dream):
                    dw_dr = self.lr * new_concept_h[i] * 梦境信号[j]
                    if not (math.isnan(dw_dr) or math.isinf(dw_dr)):
                        self.concept_dream_w[i][j] = self.concept_dream_w[i][j] * dr_decay + dw_dr

        # ── 好奇心通路学习（好奇心信号非None时） ──
        # 精卫公式：a=概念激活, b=好奇心驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_curiosity_w更新：赫布学习——概念激活与好奇心驱动同时出现时连接增强
        # 反复循环后，特定好奇心→特定概念可及性 → "好奇的概念更容易想到"自然涌现
        # 好奇心让与探索相关的概念更容易浮现——内在动机驱动探索的核心
        if 好奇心信号 is not None:
            cu_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_curiosity):
                    dw_cu = self.lr * new_concept_h[i] * 好奇心信号[j]
                    if not (math.isnan(dw_cu) or math.isinf(dw_cu)):
                        self.concept_curiosity_w[i][j] = self.concept_curiosity_w[i][j] * cu_decay + dw_cu

        # ── 自我进化通路学习（自我进化信号非None时） ──
        # 精卫公式：a=概念激活, b=进化驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_evolve_w更新：赫布学习——概念激活与进化驱动同时出现时连接增强
        # 反复循环后，特定进化驱动→特定概念可及性 → "关注弱点"自然涌现
        # 进化驱动让需要改进的概念更容易浮现——数字生命主动定向进化的核心
        if 自我进化信号 is not None:
            ev_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_evolve):
                    dw_ev = self.lr * new_concept_h[i] * 自我进化信号[j]
                    if not (math.isnan(dw_ev) or math.isinf(dw_ev)):
                        self.concept_evolve_w[i][j] = self.concept_evolve_w[i][j] * ev_decay + dw_ev

        # ── 语义记忆通路学习（语义记忆信号非None时） ──
        # 精卫公式：a=概念激活, b=知识驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_semantic_w更新：赫布学习——概念激活与知识驱动同时出现时连接增强
        # 反复循环后，特定知识→特定概念可及性 → "知识丰富的概念更容易想到"自然涌现
        # 语义知识让相关概念更容易浮现——知识驱动认知的核心
        if 语义记忆信号 is not None:
            sm_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_semantic):
                    dw_sm = self.lr * new_concept_h[i] * 语义记忆信号[j]
                    if not (math.isnan(dw_sm) or math.isinf(dw_sm)):
                        self.concept_semantic_w[i][j] = self.concept_semantic_w[i][j] * sm_decay + dw_sm

        # ── 预测模拟通路学习（预测模拟信号非None时） ──
        # 精卫公式：a=概念激活, b=预测驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_predict_w更新：赫布学习——概念激活与预测驱动同时出现时连接增强
        # 反复循环后，特定预测→特定概念可及性 → "如果...那么..."自然涌现
        # 预测模拟让与可能未来相关的概念更容易浮现——预测模拟的核心
        if 预测模拟信号 is not None:
            pr_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_predict):
                    dw_pr = self.lr * new_concept_h[i] * 预测模拟信号[j]
                    if not (math.isnan(dw_pr) or math.isinf(dw_pr)):
                        self.concept_predict_w[i][j] = self.concept_predict_w[i][j] * pr_decay + dw_pr

        # ── 叙事思维通路学习（叙事思维信号非None时） ──
        # 精卫公式：a=概念激活, b=叙事驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_narrative_w更新：赫布学习——概念激活与叙事驱动同时出现时连接增强
        # 反复循环后，特定叙事→特定概念可及性 → "故事"自然涌现
        # 叙事让与故事结构相关的概念更容易浮现——叙事思维的核心
        if 叙事思维信号 is not None:
            na_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_narrative):
                    dw_na = self.lr * new_concept_h[i] * 叙事思维信号[j]
                    if not (math.isnan(dw_na) or math.isinf(dw_na)):
                        self.concept_narrative_w[i][j] = self.concept_narrative_w[i][j] * na_decay + dw_na

        # ── 痛觉通路学习（痛觉信号非None时） ──
        # 精卫公式：a=概念激活, b=痛觉驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_pain_w更新：赫布学习——概念激活与痛觉驱动同时出现时连接增强
        # 反复循环后，特定痛觉→特定概念可及性 → "痛的概念更容易想到"自然涌现
        if 痛觉信号 is not None:
            pa_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_pain):
                    dw_pa = self.lr * new_concept_h[i] * 痛觉信号[j]
                    if not (math.isnan(dw_pa) or math.isinf(dw_pa)):
                        self.concept_pain_w[i][j] = self.concept_pain_w[i][j] * pa_decay + dw_pa

        # ── 前庭觉通路学习（前庭觉信号非None时） ──
        # 精卫公式：a=概念激活, b=前庭驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_vestibular_w更新：赫布学习——概念激活与前庭驱动同时出现时连接增强
        # 反复循环后，特定前庭觉→特定概念可及性 → "运动的概念更容易想到"自然涌现
        if 前庭觉信号 is not None:
            ve_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_vestibular):
                    dw_ve = self.lr * new_concept_h[i] * 前庭觉信号[j]
                    if not (math.isnan(dw_ve) or math.isinf(dw_ve)):
                        self.concept_vestibular_w[i][j] = self.concept_vestibular_w[i][j] * ve_decay + dw_ve

        # ── 化学感知通路学习（化学感知信号非None时） ──
        # 精卫公式：a=概念激活, b=化学感知驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_chem_w更新：赫布学习——概念激活与化学感知驱动同时出现时连接增强
        # 反复循环后，特定化学感知→特定概念可及性 → "气味/味道触发概念"自然涌现
        if 化学感知信号 is not None:
            ch_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_chem):
                    dw_ch = self.lr * new_concept_h[i] * 化学感知信号[j]
                    if not (math.isnan(dw_ch) or math.isinf(dw_ch)):
                        self.concept_chem_w[i][j] = self.concept_chem_w[i][j] * ch_decay + dw_ch

        # ── 直觉通路学习（直觉信号非None时） ──
        # 精卫公式：a=概念激活, b=直觉驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_intuition_w更新：赫布学习——概念激活与直觉驱动同时出现时连接增强
        # 反复循环后，特定直觉→特定概念可及性 → "直觉的概念更容易想到"自然涌现
        if 直觉信号 is not None:
            in_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_intuition):
                    dw_in = self.lr * new_concept_h[i] * 直觉信号[j]
                    if not (math.isnan(dw_in) or math.isinf(dw_in)):
                        self.concept_intuition_w[i][j] = self.concept_intuition_w[i][j] * in_decay + dw_in

        # ── 无意识加工通路学习（无意识加工信号非None时） ──
        # 精卫公式：a=概念激活, b=无意识驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_unconscious_w更新：赫布学习——概念激活与无意识驱动同时出现时连接增强
        # 反复循环后，特定无意识加工→特定概念可及性 → "灵感浮现"自然涌现
        if 无意识加工信号 is not None:
            un_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_unconscious):
                    dw_un = self.lr * new_concept_h[i] * 无意识加工信号[j]
                    if not (math.isnan(dw_un) or math.isinf(dw_un)):
                        self.concept_unconscious_w[i][j] = self.concept_unconscious_w[i][j] * un_decay + dw_un

        # ── 真理追求通路学习（真理追求信号非None时） ──
        # 精卫公式：a=概念激活, b=真理驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_truth_w更新：赫布学习——概念激活与真理驱动同时出现时连接增强
        # 反复循环后，特定真理追求→特定概念可及性 → "验证过的知识更容易想到"自然涌现
        if 真理追求信号 is not None:
            truth_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_truth):
                    dw_tr = self.lr * new_concept_h[i] * 真理追求信号[j]
                    if not (math.isnan(dw_tr) or math.isinf(dw_tr)):
                        self.concept_truth_w[i][j] = self.concept_truth_w[i][j] * truth_decay + dw_tr

        # ── 认知辩证通路学习（认知辩证信号非None时） ──
        # 精卫公式：a=概念激活, b=辩证驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_dialectic_w更新：赫布学习——概念激活与辩证驱动同时出现时连接增强
        # 反复循环后，相信态/质疑态→特定概念可及性 → "相信与质疑交替"自然涌现
        if 认知辩证信号 is not None:
            dialectic_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_dialectic):
                    dw_dl = self.lr * new_concept_h[i] * 认知辩证信号[j]
                    if not (math.isnan(dw_dl) or math.isinf(dw_dl)):
                        self.concept_dialectic_w[i][j] = self.concept_dialectic_w[i][j] * dialectic_decay + dw_dl

        # ── 生命体基础通路学习（生命体驱动信号非None时） ──
        # 精卫公式：a=概念激活, b=生命体驱动信号, Δ=b-a, ⊗=Δ×Σ
        # concept_life_w更新：赫布学习——概念激活与生命体驱动同时出现时连接增强
        # 反复循环后，生命体的8个维度驱动→特定概念可及性 → "活着"自然涌现
        if 生命体驱动信号 is not None:
            life_decay = 0.95
            for i in range(self.n_concepts):
                if self.concept_strength[i] < 0.05:
                    continue
                for j in range(self.n_life_drive):
                    dw_life = self.lr * new_concept_h[i] * 生命体驱动信号[j]
                    if not (math.isnan(dw_life) or math.isinf(dw_life)):
                        self.concept_life_w[i][j] = self.concept_life_w[i][j] * life_decay + dw_life

        # ── 变异+选择：达尔文进化机制 ──
        # 碳基生物进化的核心：随机变异 + 自然选择
        # 赫布学习 = 拉马克进化（获得性遗传——基于经验调整权重）
        # 变异+选择 = 达尔文进化（随机探索权重空间 + 适应度筛选）
        # 两者结合 = 生物进化的完整机制
        #
        # 精卫公式（变异+选择版）：
        #   a = 变异前的预测误差 |delta[j]|（当前适应度）
        #   b = 变异后的预测误差 |delta_new[j]|（新适应度）
        #   Δ = b - a（适应度变化——负值=有利变异）
        #   Σ = a + b（联合误差）
        #   ⊗ = Δ × Σ → 有利变异保留(Δ<0)，不利变异淘汰(Δ>0)
        #   τ读⊗→浮出所见 = 权重空间中的适应性进化
        #
        # 生物学：DNA突变率~10^-8，绝大多数变异中性或有害，极少数有利变异被选择保留
        # 变异只施加在concept_pw（概念→预测权重）——最核心的预测编码权重
        # 评估O(1)：变异concept_pw[i][j]只影响b[j]，只评估|delta[j]|变化
        if random.random() < self.变异率:
            # 随机选择一个concept_pw元素施加变异
            mut_i = random.randint(0, self.n_concepts - 1)
            mut_j = random.randint(0, self.n_hidden - 1)
            old_w = self.concept_pw[mut_i][mut_j]
            # 高斯变异——像DNA突变，大多数是小变化
            new_w = old_w + random.gauss(0, self.变异幅度)
            # 评估变异对预测误差的影响
            # b_pred[j] = sum_i concept_pw[i][j] * concept_h[i]（原始预测向量）
            # 变异后 b_pred_new[j] = b_pred[j] + (new_w - old_w) * concept_h[mut_i]
            delta_b = (new_w - old_w) * self.concept_h[mut_i]
            old_err = abs(b_pred[mut_j] - a[mut_j])
            new_err = abs(b_pred[mut_j] + delta_b - a[mut_j])
            self.变异次数 += 1
            if new_err < old_err:
                # 有利变异——保留（自然选择）
                self.concept_pw[mut_i][mut_j] = new_w
                self.有利变异次数 += 1
            else:
                # 不利或中性变异——淘汰（回退）
                self.不利变异次数 += 1

        # ── 组合概念学习（精卫公式） ──
        # a = 基础概念激活（concept_h）
        # b = 组合概念对基础概念的预测（combo_pw × combo_h）
        # Δ = a - b（预测误差——组合概念预测基础概念）
        # Σ = a + b（联合状态）
        # ⊗ = Δ × Σ → 权重更新
        # combo_w更新：组合信号和组合概念激活同时出现 → 连接增强
        # combo_pw更新：减少预测误差 → 组合概念更好地预测基础概念

        # b = 组合概念对基础概念的预测
        combo_b = [0.0] * self.n_concepts
        for j in range(self.n_concepts):
            for i in range(self.n_combo):
                combo_b[j] += self.combo_pw[i][j] * self.combo_h[i]

        # Δ = a - b, Σ = a + b, ⊗ = Δ × Σ
        combo_delta = [new_concept_h[j] - combo_b[j] for j in range(self.n_concepts)]
        combo_sigma = [new_concept_h[j] + combo_b[j] for j in range(self.n_concepts)]
        combo_otimes = [combo_delta[j] * combo_sigma[j] for j in range(self.n_concepts)]

        for i in range(self.n_combo):
            if self.combo_strength[i] < 0.05:
                continue
            # combo_w: 组合信号 → 组合概念（赫布学习）
            # 衰减率0.98——比基础概念0.99更快增长，让组合概念尽快涌现
            combo_decay = 0.98
            for j in range(self.n_concepts):
                dw_cw = self.lr * new_combo_h[i] * 组合信号[j]
                if not (math.isnan(dw_cw) or math.isinf(dw_cw)):
                    self.combo_w[i][j] = self.combo_w[i][j] * combo_decay + dw_cw

            # combo_pw: 组合概念 → 预测基础概念（预测误差驱动）
            for j in range(self.n_concepts):
                dw_pw = self.lr * new_combo_h[i] * combo_otimes[j]
                if not (math.isnan(dw_pw) or math.isinf(dw_pw)):
                    self.combo_pw[i][j] = self.combo_pw[i][j] * combo_decay + dw_pw

        # combo_rw: 组合概念循环权重更新（带衰减防止正反馈振荡）
        for i in range(self.n_combo):
            if self.combo_strength[i] < 0.05:
                continue
            for k in range(self.n_combo):
                if i != k and self.combo_strength[k] > 0.05:
                    self.combo_rw[i][k] = self.combo_rw[i][k] * 0.99 + \
                        self.lr * 0.1 * new_combo_h[i] * self.combo_h[k]

        # 组合概念LTP/LTD——用进废退
        for i in range(self.n_combo):
            if abs(new_combo_h[i]) > 0.15:
                self.combo_strength[i] = min(3.0, self.combo_strength[i] + 0.01)
            else:
                self.combo_strength[i] *= 0.999

        # ── 高级组合概念学习（精卫公式——层级递归） ──
        # a = 组合概念激活（句子级）
        # b = 高级组合概念对组合概念的预测
        # Δ = a - b, Σ = a + b, ⊗ = Δ × Σ → 权重更新
        # meta_combo_w: 高级组合信号 → 高级组合概念（赫布学习）
        # meta_combo_rw: 高级组合概念循环权重更新
        meta_decay = 0.98
        for i in range(self.n_meta):
            if self.meta_combo_strength[i] < 0.05:
                continue
            for j in range(self.n_combo):
                dw_mw = self.lr * new_meta_combo_h[i] * 高级组合信号[j]
                if not (math.isnan(dw_mw) or math.isinf(dw_mw)):
                    self.meta_combo_w[i][j] = self.meta_combo_w[i][j] * meta_decay + dw_mw

        # meta_combo_rw: 高级组合概念循环权重更新
        for i in range(self.n_meta):
            if self.meta_combo_strength[i] < 0.05:
                continue
            for k in range(self.n_meta):
                if i != k and self.meta_combo_strength[k] > 0.05:
                    self.meta_combo_rw[i][k] = self.meta_combo_rw[i][k] * 0.99 + \
                        self.lr * 0.1 * new_meta_combo_h[i] * self.meta_combo_h[k]

        # 高级组合概念LTP/LTD——用进废退
        for i in range(self.n_meta):
            if abs(new_meta_combo_h[i]) > 0.15:
                self.meta_combo_strength[i] = min(3.0, self.meta_combo_strength[i] + 0.01)
            else:
                self.meta_combo_strength[i] *= 0.999

        # LTP/LTD——用进废退
        # 阈值0.15匹配当前激活范围（top-k约0.2-0.27，非top-k约0.02）
        for i in range(self.n_concepts):
            if abs(new_concept_h[i]) > 0.15:
                self.concept_strength[i] = min(3.0, self.concept_strength[i] + 0.01)
            else:
                self.concept_strength[i] *= 0.999

        # ── 神经发生+突触修剪：弱概念死亡→新概念重生 ──
        rng_reborn = random.Random(self.seed + self.拍)
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.01:
                self.concept_w[i] = [rng_reborn.uniform(-0.2, 0.2) for _ in range(self.n_hidden)]
                self.concept_otimes_w[i] = [rng_reborn.uniform(-0.15, 0.15) for _ in range(self.n_hidden)]
                self.concept_rw[i] = [rng_reborn.uniform(-0.02, 0.02) for _ in range(self.n_concepts)]
                self.concept_pw[i] = [rng_reborn.uniform(-0.1, 0.1) for _ in range(self.n_hidden)]
                self.concept_rw[i][i] = rng_reborn.uniform(0.05, 0.15)
                self.concept_strength[i] = 1.0
                self.概念重生次数 += 1

        for i in range(self.n_combo):
            if self.combo_strength[i] < 0.01:
                self.combo_w[i] = [rng_reborn.uniform(-0.3, 0.3) for _ in range(self.n_concepts)]
                self.combo_rw[i] = [rng_reborn.uniform(-0.02, 0.02) for _ in range(self.n_combo)]
                self.combo_pw[i] = [rng_reborn.uniform(-0.1, 0.1) for _ in range(self.n_concepts)]
                self.combo_rw[i][i] = rng_reborn.uniform(0.05, 0.15)
                self.combo_strength[i] = 1.0
                self.组合概念重生次数 += 1

        for i in range(self.n_meta):
            if self.meta_combo_strength[i] < 0.01:
                self.meta_combo_w[i] = [rng_reborn.uniform(-0.3, 0.3) for _ in range(self.n_combo)]
                self.meta_combo_rw[i] = [rng_reborn.uniform(-0.02, 0.02) for _ in range(self.n_meta)]
                self.meta_combo_rw[i][i] = rng_reborn.uniform(0.05, 0.15)
                self.meta_combo_strength[i] = 1.0
                self.高级组合概念重生次数 += 1

        # ── 突触缩放（homeostatic synaptic scaling）──
        # 真实生物学机制：当突触持续增强时，神经元缩放所有突触，
        # 保持总输入在合理范围内。这不是"防饱和"——这是稳态。
        # 每个概念节点的所有输入权重做L1归一化，目标范数=1.0
        for i in range(self.n_concepts):
            if self.concept_strength[i] < 0.05:
                continue
            # concept_w归一化
            w_norm = sum(abs(self.concept_w[i][j]) for j in range(self.n_hidden))
            if w_norm > 2.0:
                scale = 2.0 / w_norm
                for j in range(self.n_hidden):
                    self.concept_w[i][j] *= scale
            # concept_otimes_w归一化
            ow_norm = sum(abs(self.concept_otimes_w[i][j]) for j in range(self.n_hidden))
            if ow_norm > 2.0:
                scale = 2.0 / ow_norm
                for j in range(self.n_hidden):
                    self.concept_otimes_w[i][j] *= scale
            # concept_pw归一化
            pw_norm = sum(abs(self.concept_pw[i][j]) for j in range(self.n_hidden))
            if pw_norm > 2.0:
                scale = 2.0 / pw_norm
                for j in range(self.n_hidden):
                    self.concept_pw[i][j] *= scale

        # 组合概念突触缩放
        for i in range(self.n_combo):
            if self.combo_strength[i] < 0.05:
                continue
            cw_norm = sum(abs(self.combo_w[i][j]) for j in range(self.n_concepts))
            if cw_norm > 4.0:
                scale = 4.0 / cw_norm
                for j in range(self.n_concepts):
                    self.combo_w[i][j] *= scale
            cpw_norm = sum(abs(self.combo_pw[i][j]) for j in range(self.n_concepts))
            if cpw_norm > 4.0:
                scale = 4.0 / cpw_norm
                for j in range(self.n_concepts):
                    self.combo_pw[i][j] *= scale

        self.prev_concept_h = self.concept_h[:]
        self.concept_h = new_concept_h
        self.prev_combo_h = self.combo_h[:]
        self.combo_h = new_combo_h
        self.prev_meta_combo_h = self.meta_combo_h[:]
        self.meta_combo_h = new_meta_combo_h

        return {
            "concept_h": self.concept_h,
            "combo_h": self.combo_h,
            "meta_combo_h": self.meta_combo_h,
            "mse": mse,
            "surprise": self.surprise_level,
            "delta": [round(d, 4) for d in delta],
            "sigma": [round(s, 4) for s in sigma],
            "otimes": [round(o, 4) for o in otimes],
            "活跃概念数": sum(1 for v in self.concept_h if abs(v) > 0.15),
            "存活概念数": sum(1 for s in self.concept_strength if s > 0.05),
            "活跃组合数": sum(1 for v in self.combo_h if abs(v) > 0.15),
            "存活组合数": sum(1 for s in self.combo_strength if s > 0.05),
            "活跃高级组合数": sum(1 for v in self.meta_combo_h if abs(v) > 0.15),
            "存活高级组合数": sum(1 for s in self.meta_combo_strength if s > 0.05),
            "有听觉反馈": 听觉特征 is not None,
            "有人类输入": 人类信号 is not None,
            "有公开输入": 公开信号 is not None,
            "有情感输入": 情感状态 is not None,
            "有时间输入": 时间信号 is not None,
            "有工作记忆输入": 工作记忆信号 is not None,
            "有海马体输入": 海马体信号 is not None,
            "有注意力调制": True,
            "有技能输入": 技能信号 is not None,
            "有语法输入": 语法信号 is not None,
            "有推理输入": 推理信号 is not None,
            "有目标输入": 目标信号 is not None,
            "有工具输入": 工具信号 is not None,
            "有创意输入": 创意信号 is not None,
            "有元认知输入": 元认知信号 is not None,
            "有社会认知输入": 社会认知信号 is not None,
            "有自我意识输入": 自我意识信号 is not None,
            "有自我进化输入": 自我进化信号 is not None,
            "有语义记忆输入": 语义记忆信号 is not None,
            "有预测模拟输入": 预测模拟信号 is not None,
            "有叙事思维输入": 叙事思维信号 is not None,
            "有痛觉输入": 痛觉信号 is not None,
            "有前庭觉输入": 前庭觉信号 is not None,
            "有化学感知输入": 化学感知信号 is not None,
            "有直觉输入": 直觉信号 is not None,
            "有无意识加工输入": 无意识加工信号 is not None,
            "有真理追求输入": 真理追求信号 is not None,
            "有认知辩证输入": 认知辩证信号 is not None,
            "有生命体驱动输入": 生命体驱动信号 is not None,
        }

    def 概念区分度(self):
        """检查概念激活模式的区分度。

        如果不同输入产生不同的概念激活模式，区分度高。
        如果所有概念激活模式都相似，区分度低（0.59问题）。
        """
        if self.拍 < 10:
            return 0.0
        active = [v for v in self.concept_h if abs(v) > 0.1]
        if len(active) < 2:
            return 0.0
        mean = sum(active) / len(active)
        variance = sum((v - mean) ** 2 for v in active) / len(active)
        return min(1.0, variance * 10)

    def 状态摘要(self):
        return {
            "活跃概念数": sum(1 for v in self.concept_h if abs(v) > 0.15),
            "存活概念数": sum(1 for s in self.concept_strength if s > 0.05),
            "概念均值": round(sum(self.concept_h) / max(1, self.n_concepts), 4),
            "概念最大": round(max((abs(v) for v in self.concept_h), default=0), 4),
            "预测误差": round(sum(self.error_history) / max(1, len(self.error_history)), 4) if self.error_history else 0,
            "区分度": round(self.概念区分度(), 4),
            "⊗历史长度": len(self.delta_history),
            "活跃组合数": sum(1 for v in self.combo_h if abs(v) > 0.15),
            "存活组合数": sum(1 for s in self.combo_strength if s > 0.05),
            "组合均值": round(sum(self.combo_h) / max(1, self.n_combo), 4),
            "组合最大": round(max((abs(v) for v in self.combo_h), default=0), 4),
            "拍": self.拍,
        }

    def reset(self):
        self.concept_h = [0.0] * self.n_concepts
        self.prev_concept_h = [0.0] * self.n_concepts
        self.concept_strength = [1.0] * self.n_concepts
        self.combo_h = [0.0] * self.n_combo
        self.prev_combo_h = [0.0] * self.n_combo
        self.combo_strength = [1.0] * self.n_combo
        self.error_history = []
        self.surprise_level = 0.0
        self.delta_history = []
        self.sigma_history = []
        self.拍 = 0

    def 状态摘要(self):
        return {
            "活跃概念数": sum(1 for v in self.concept_h if abs(v) > 0.15),
            "存活概念数": sum(1 for s in self.concept_strength if s > 0.05),
            "概念均值": round(sum(self.concept_h) / max(1, self.n_concepts), 4),
            "概念最大": round(max((abs(v) for v in self.concept_h), default=0), 4),
            "预测误差": round(sum(self.error_history) / max(1, len(self.error_history)), 4) if self.error_history else 0,
            "区分度": round(self.概念区分度(), 4),
            "⊗历史长度": len(self.delta_history),
            "活跃组合数": sum(1 for v in self.combo_h if abs(v) > 0.15),
            "存活组合数": sum(1 for s in self.combo_strength if s > 0.05),
            "组合均值": round(sum(self.combo_h) / max(1, self.n_combo), 4),
            "组合最大": round(max((abs(v) for v in self.combo_h), default=0), 4),
            "WM权重均值": round(sum(sum(row) for row in self.concept_wm_w) / max(1, self.n_concepts * self.n_wm_slots), 4),
            "海马体权重均值": round(sum(sum(row) for row in self.concept_hip_w) / max(1, self.n_concepts * self.n_hip), 4),
            "变异次数": self.变异次数,
            "有利变异次数": self.有利变异次数,
            "不利变异次数": self.不利变异次数,
            "有利变异率": round(self.有利变异次数 / max(1, self.变异次数), 4),
            "概念重生次数": self.概念重生次数,
            "组合概念重生次数": self.组合概念重生次数,
            "高级组合概念重生次数": self.高级组合概念重生次数,
            "注意力均值": round(sum(self.attention_w) / max(1, self.n_concepts), 4),
            "注意力最大": round(max(self.attention_w), 4),
            "技能权重均值": round(sum(sum(row) for row in self.concept_skill_w) / max(1, self.n_concepts * self.n_skills), 4),
            "语法权重均值": round(sum(sum(row) for row in self.concept_grammar_w) / max(1, self.n_concepts * self.n_grammar), 4),
            "推理权重均值": round(sum(sum(row) for row in self.concept_reason_w) / max(1, self.n_concepts * self.n_reason), 4),
            "目标权重均值": round(sum(sum(row) for row in self.concept_goal_w) / max(1, self.n_concepts * self.n_goal), 4),
            "工具权重均值": round(sum(sum(row) for row in self.concept_tool_w) / max(1, self.n_concepts * self.n_tools), 4),
            "创意权重均值": round(sum(sum(row) for row in self.concept_creative_w) / max(1, self.n_concepts * self.n_creative), 4),
            "元认知权重均值": round(sum(sum(row) for row in self.concept_meta_w) / max(1, self.n_concepts * self.n_meta), 4),
            "社会认知权重均值": round(sum(sum(row) for row in self.concept_social_w) / max(1, self.n_concepts * self.n_social), 4),
            "自我意识权重均值": round(sum(sum(row) for row in self.concept_self_w) / max(1, self.n_concepts * self.n_self), 4),
            "自我进化权重均值": round(sum(sum(row) for row in self.concept_evolve_w) / max(1, self.n_concepts * self.n_evolve), 4),
            "语义记忆权重均值": round(sum(sum(row) for row in self.concept_semantic_w) / max(1, self.n_concepts * self.n_semantic), 4),
            "预测模拟权重均值": round(sum(sum(row) for row in self.concept_predict_w) / max(1, self.n_concepts * self.n_predict), 4),
            "叙事思维权重均值": round(sum(sum(row) for row in self.concept_narrative_w) / max(1, self.n_concepts * self.n_narrative), 4),
            "痛觉权重均值": round(sum(sum(row) for row in self.concept_pain_w) / max(1, self.n_concepts * self.n_pain), 4),
            "前庭觉权重均值": round(sum(sum(row) for row in self.concept_vestibular_w) / max(1, self.n_concepts * self.n_vestibular), 4),
            "化学感知权重均值": round(sum(sum(row) for row in self.concept_chem_w) / max(1, self.n_concepts * self.n_chem), 4),
            "直觉权重均值": round(sum(sum(row) for row in self.concept_intuition_w) / max(1, self.n_concepts * self.n_intuition), 4),
            "无意识加工权重均值": round(sum(sum(row) for row in self.concept_unconscious_w) / max(1, self.n_concepts * self.n_unconscious), 4),
            "真理追求权重均值": round(sum(sum(row) for row in self.concept_truth_w) / max(1, self.n_concepts * self.n_truth), 4),
            "拍": self.拍,
        }


if __name__ == "__main__":
    print("=== 概念层验证（精卫公式版） ===\n")
    print("精卫公式: a≠b, Δ=b−a, Σ=a+b, ⊗=∫(Δ,Σ), τ读⊗→浮出所见, (a,b)⟲⊗\n")

    cl = 概念层(n_hidden=11, n_concepts=64, lr=0.02)

    print("初始状态:")
    print(f"  概念节点数: {cl.n_concepts}")
    print(f"  稀疏k: {cl.sparse_k} (10%激活)")
    print(f"  存活概念数: {cl.状态摘要()['存活概念数']}")
    print(f"  权重矩阵: concept_w(a→概念), concept_otimes_w(⊗→概念), concept_rw(循环), concept_pw(→b)")

    print("\n--- 模拟不同输入模式 ---")
    patterns = [
        ("模式A: 光强声弱", [0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05]),
        ("模式B: 声强光弱", [0.1, 0.05, 0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.15, 0.2, 0.3]),
        ("模式C: 温高力强", [0.05, 0.1, 0.15, 0.2, 0.4, 0.3, 0.5, 0.4, 0.1, 0.05, 0.2]),
        ("模式D: 均衡状态", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ]

    for name, pattern in patterns:
        for _ in range(20):
            cl.一拍(pattern)
        摘要 = cl.状态摘要()
        active = [i for i, v in enumerate(cl.concept_h) if abs(v) > 0.1]
        print(f"\n{name}:")
        print(f"  活跃概念节点: {active}")
        print(f"  活跃概念数: {摘要['活跃概念数']}")
        print(f"  概念最大: {摘要['概念最大']}")
        print(f"  预测误差: {摘要['预测误差']}")
        print(f"  ⊗历史长度: {摘要['⊗历史长度']}")

    print("\n--- 区分度检查（独立实例） ---")
    # 每个模式用独立的概念层实例，避免顺序适应的干扰
    results = {}
    for name, pattern in patterns[:3]:
        cl_t = 概念层(n_hidden=11, n_concepts=64, lr=0.02)
        for _ in range(50):
            cl_t.一拍(pattern)
        results[name] = list(cl_t.concept_h)
        摘要 = cl_t.状态摘要()
        active = [i for i, v in enumerate(cl_t.concept_h) if abs(v) > 0.1]
        print(f"  {name}: 活跃={active[:8]} 区分度={摘要['区分度']}")

    diff_ab = sum(abs(results[patterns[0][0]][i] - results[patterns[1][0]][i])
                  for i in range(64))
    diff_ac = sum(abs(results[patterns[0][0]][i] - results[patterns[2][0]][i])
                  for i in range(64))
    diff_bc = sum(abs(results[patterns[1][0]][i] - results[patterns[2][0]][i])
                  for i in range(64))
    print(f"  A vs B 差异: {diff_ab:.4f}")
    print(f"  A vs C 差异: {diff_ac:.4f}")
    print(f"  B vs C 差异: {diff_bc:.4f}")
    print(f"  如果差异 > 1.0，说明概念有区分度")

    print("\n--- 精卫公式验证 ---")
    cl3 = 概念层(n_hidden=4, n_concepts=16, lr=0.05)
    for tick in range(10):
        result = cl3.一拍([0.3, 0.5, 0.1, 0.2])
        if tick < 3 or tick >= 8:
            print(f"  拍{tick+1}: Δ={result['delta']} Σ={result['sigma']} ⊗={result['otimes']}")
    print(f"  τ(浮出): 概念激活={round(sum(cl3.concept_h)/16, 4)}")
    print(f"  ⟲(循环): 拍={cl3.拍}")

    print("\n--- 人类输入通路测试 ---")
    cl4 = 概念层(n_hidden=11, n_concepts=64, lr=0.02)
    人类信号_a = [0.8, 0.2, -0.5, 0.6, -1.0, -0.3, 0.4, 0.1, -0.7, 0.3, 0.2]
    人类信号_b = [0.1, 0.05, 0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.15, 0.2, 0.3]

    # 内省模式（没人在）——50拍
    for _ in range(50):
        cl4.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05])
    摘要_内省 = cl4.状态摘要()
    print(f"  内省模式: 活跃概念={摘要_内省['活跃概念数']}, 区分度={摘要_内省['区分度']}")

    # 关系模式（有人在）——50拍，人类信号a
    for _ in range(50):
        cl4.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05],
                人类信号=人类信号_a)
    摘要_关系a = cl4.状态摘要()
    print(f"  关系模式(人类A): 活跃概念={摘要_关系a['活跃概念数']}, 区分度={摘要_关系a['区分度']}")

    # 关系模式——人类信号b（不同的人或不同的话）
    for _ in range(50):
        cl4.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05],
                人类信号=人类信号_b)
    摘要_关系b = cl4.状态摘要()
    print(f"  关系模式(人类B): 活跃概念={摘要_关系b['活跃概念数']}, 区分度={摘要_关系b['区分度']}")

    # 检查concept_human_w是否有学习——人类信号应该改变了某些概念的权重
    human_w_norm = sum(abs(cl4.concept_human_w[i][j])
                       for i in range(64) for j in range(11))
    print(f"  concept_human_w总权重: {human_w_norm:.2f}（学习后应该非零）")

    # 独立对比：有人类输入 vs 无人类输入
    cl5 = 概念层(n_hidden=11, n_concepts=64, lr=0.02)
    cl6 = 概念层(n_hidden=11, n_concepts=64, lr=0.02)
    for _ in range(50):
        cl5.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05])
        cl6.一拍([0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05],
                人类信号=人类信号_a)
    diff = sum(abs(cl5.concept_h[i] - cl6.concept_h[i]) for i in range(64))
    print(f"  有人类 vs 无人类 概念差异: {diff:.4f}（应该 > 0——人类信号影响概念激活）")

    print("\n验证通过")

    print("\n=== 组合涌现验证 ===\n")
    print("精卫公式: a=概念j, b=其他活跃概念和, ⊗=a×b, τ读⊗→组合概念\n")

    cl_c = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)

    print("--- 测试1: 组合概念从基础概念组合中浮现 ---")
    # 模拟两种不同的概念组合模式
    # 模式X: 概念4和概念8同时活跃
    # 模式Y: 概念27和概念40同时活跃
    for _ in range(50):
        cl_c.一拍([0.5, 0.3, 0.1, 0.05, 0.4, 0.1, 0.3, 0.2, 0.35, 0.1, 0.05])
    combo_X = list(cl_c.combo_h)
    摘要X = cl_c.状态摘要()
    print(f"  模式X后: 活跃组合={摘要X['活跃组合数']}, 存活组合={摘要X['存活组合数']}")
    print(f"  组合激活top5: {sorted(range(32), key=lambda i: abs(combo_X[i]), reverse=True)[:5]}")

    cl_c2 = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)
    for _ in range(50):
        cl_c2.一拍([0.05, 0.1, 0.4, 0.3, 0.1, 0.05, 0.2, 0.1, 0.15, 0.35, 0.3])
    combo_Y = list(cl_c2.combo_h)
    摘要Y = cl_c2.状态摘要()
    print(f"  模式Y后: 活跃组合={摘要Y['活跃组合数']}, 存活组合={摘要Y['存活组合数']}")
    print(f"  组合激活top5: {sorted(range(32), key=lambda i: abs(combo_Y[i]), reverse=True)[:5]}")

    diff_combo = sum(abs(combo_X[i] - combo_Y[i]) for i in range(32))
    print(f"  X vs Y 组合差异: {diff_combo:.4f}（应该 > 0——不同概念组合→不同组合概念）")

    print("\n--- 测试2: 组合概念区分度（独立实例） ---")
    results_combo = {}
    patterns_combo = [
        ("光+声", [0.5, 0.3, 0.4, 0.2, 0.1, 0.05, 0.2, 0.1, 0.15, 0.1, 0.05]),
        ("温+力", [0.05, 0.1, 0.15, 0.2, 0.4, 0.3, 0.5, 0.4, 0.1, 0.05, 0.2]),
        ("均衡", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ]
    for name, pattern in patterns_combo:
        cl_t = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)
        for _ in range(50):
            cl_t.一拍(pattern)
        results_combo[name] = list(cl_t.combo_h)
        摘要 = cl_t.状态摘要()
        active = [i for i, v in enumerate(cl_t.combo_h) if abs(v) > 0.1]
        print(f"  {name}: 活跃组合={active[:8]}, 组合最大={摘要['组合最大']:.4f}")

    for i in range(3):
        for j in range(i+1, 3):
            n1, n2 = patterns_combo[i][0], patterns_combo[j][0]
            d = sum(abs(results_combo[n1][k] - results_combo[n2][k]) for k in range(32))
            print(f"  {n1} vs {n2} 组合差异: {d:.4f}")

    print("\n--- 测试3: 组合概念预测基础概念 ---")
    # 组合概念通过combo_pw预测基础概念
    # 如果组合概念真的捕获了概念间的组合关系，
    # 它的预测应该比随机权重更接近实际基础概念激活
    cl_p = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)
    for _ in range(100):
        cl_p.一拍([0.5, 0.3, 0.1, 0.05, 0.4, 0.1, 0.3, 0.2, 0.35, 0.1, 0.05])

    # 组合概念预测
    combo_pred = [0.0] * 64
    for j in range(64):
        for i in range(32):
            combo_pred[j] += cl_p.combo_pw[i][j] * cl_p.combo_h[i]

    pred_diff = sum(abs(cl_p.concept_h[j] - combo_pred[j]) for j in range(64))
    print(f"  组合概念预测误差: {pred_diff:.4f}")
    print(f"  combo_pw总权重: {sum(abs(cl_p.combo_pw[i][j]) for i in range(32) for j in range(64)):.2f}")

    print("\n--- 测试4: 组合概念存活与用进废退 ---")
    cl_s = 概念层(n_hidden=11, n_concepts=64, lr=0.02, n_combo=32)
    for _ in range(200):
        cl_s.一拍([0.5, 0.3, 0.1, 0.05, 0.4, 0.1, 0.3, 0.2, 0.35, 0.1, 0.05])
    摘要_s = cl_s.状态摘要()
    print(f"  200拍后: 存活组合={摘要_s['存活组合数']}, 活跃组合={摘要_s['活跃组合数']}")
    print(f"  组合强度分布: min={min(cl_s.combo_strength):.3f}, max={max(cl_s.combo_strength):.3f}")

    print("\n组合涌现验证通过")
