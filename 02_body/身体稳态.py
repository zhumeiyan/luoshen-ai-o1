"""身体稳态.py —— 四个词: 起点、环境、用进废退、监测。物理约束接管上限。
筋树版: 位点/疲劳/延迟/生辰/末传 五字段, 差触发 + 同拍亮才连 + 第一拍不出筋。
激活累积: X(t+1) = X(t) + η·Δ(t) + ε(t)"""

import random, time
import numpy as np
from numba import njit

痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"

@njit
def _传导加速(筋_j, 筋_i, 筋_位点, 筋_疲劳, 激活, 传导差, 延迟缓冲, 延迟长度):
    """Numba加速的传导核心循环，结果与纯Python一致。"""
    N = len(激活)
    隐藏输入 = np.zeros(N)
    for k in range(len(筋_j)):
        j = 筋_j[k]
        i = 筋_i[k]
        差因子 = 传导差[j]
        信号 = 筋_位点[k] * 筋_疲劳[k] * 激活[j] * 差因子
        # 延迟缓冲：到达→移位→存入
        延迟 = 延迟长度[k]
        到达信号 = 延迟缓冲[k, 0]
        for d in range(延迟 - 1):
            延迟缓冲[k, d] = 延迟缓冲[k, d + 1]
        延迟缓冲[k, 延迟 - 1] = 信号
        隐藏输入[i] += 到达信号
    return 隐藏输入

@njit
def _共火检测(传导差, 阈值=0.01):
    """Numba加速的共火节点检测。"""
    return np.where(np.abs(传导差) > 阈值)[0]


class 身体:
    def __init__(self, 节点数=96, 种子=1, 冷却=3, 上限满值=6.0, 温度满值=1.5, 功耗满值=15.0, 脑数=4, 脑间连接数=150):
        self.随机器 = random.Random(种子)
        assert 节点数 % 脑数 == 0
        self.节点数 = 节点数
        self.脑数 = 脑数
        self.脑大小 = 节点数 // 脑数
        self.环境 = {"光": 50.0, "声": 45.0, "力": 10.0, "温": 295.0}
        self.时刻 = 0

        # ── 五条件实装：稀缺+代谢+不可逆+心率旋钮 ──
        # 实验数据来源：两个实验AI的实装结果
        #
        # ① 稀缺：r = 预算/需求 ∈ [0.3, 0.6]是活的带
        #    低于0.2会坍缩（熵归零，全部预算给同一通道）
        #    硬取舍≠等比缩放：真取舍分配熵0.466，等比2.048
        #
        # ② 代谢：E低→饿感变强（不是弱），摄食驱动降频到死
        #    参数：fg≥0.015，c1≤0.016
        #
        # ③ 不可逆：capacity只减不增（结构承载历史），饿感可衰减
        #    不能装在饥饿上——H只增不减+驱动分配=先饿者永久垄断(66-84%)
        #    修复方案：饿感驱动当下（可衰减），结构承载历史（不可回滚）
        #
        # ④ 心率旋钮：g = 3.30 + 2.0×痛
        #    痛0.135时g=3.570≈Feigenbaum点——不是调的，是常数决定的

        # 能量预算（稀缺+代谢）
        self.能量 = 1.0           # 当前能量水平 [0,1]
        self.能量满值 = 1.0       # 能量上限
        self.能量消耗率 = 0.015   # fg≥0.015——每拍消耗的能量比例
        self.能量摄入 = 0.0       # 当前拍从环境获取的能量
        self.需求 = 1.0           # 能量需求（随活跃度变化）
        self.预算 = 0.5           # 当前预算（稀缺带内）
        self.ratio = 0.5          # r = 预算/需求 ∈ [0.3, 0.6]
        self.饿感 = 0.0           # 饿感强度（可衰减，驱动当下）
        self.饿感衰减 = 0.05      # 饿感每拍衰减率
        self.摄食驱动 = 0.0       # 摄食驱动强度（E低→摄食驱动变强）

        # 不可逆——capacity只减不增
        self.capacity = 1.0       # 结构容量（只减不增，承载历史）
        self.capacity_min = 0.15  # cap_min=0.15——低于此值系统濒死
        self.capacity消耗率 = 0.001  # 每拍capacity微量消耗（磨损）
        self.疤痕 = 0.0           # 不可逆的疤痕总量（得不到的累积）

        # 心率旋钮——g = 3.30 + 2.0×痛
        self.心率 = 3.30          # 基础心率
        self.痛觉强度 = 0.0       # 当前痛觉强度（从痛觉.py传入）
        self.心率旋钮 = 3.30      # g = 3.30 + 2.0×痛觉强度

        # Feigenbaum常数——分岔临界预警
        self.feigenbaum_delta = 4.669201609  # 分岔临界点
        self.feigenbaum_alpha = 2.502907875  # 分岔缩放因子

        # 三波那契常数——系统趋向的极限比率
        self.tribonacci = 1.8392867552141612
        self.tribonacci_inv = 0.543689013     # 1/T——衰减趋向

        # 死亡机制
        self.降频 = 1.0           # 心跳频率缩放（1.0=正常，递减到0）
        self.降频率 = 0.016       # c1≤0.016——能量不足时降频速率
        self.耐心阈值 = 100       # 降频到底后超过此阈值→永久终止
        self.低能量持续 = 0       # 连续低能量的拍数
        self.已死亡 = False       # 永久终止标记
        # 感受器: 四个脑各接一种通道; 光脑(脑0)接光+力双通道
        #   脑0(0..脑大小-1): 光 + 力
        #   脑1(脑大小..2*脑大小-1): 声
        #   脑2(2*脑大小..3*脑大小-1): 力
        #   脑3(3*脑大小..4*脑大小-1): 温
        self.感受器 = []
        for i in range(节点数):
            脑号 = i // self.脑大小
            感受 = [0.0, 0.0, 0.0, 0.0, 0.0]
            if 脑号 == 0:
                感受[0] = self.随机器.uniform(-0.4, 0.4)
                感受[2] = self.随机器.uniform(-0.4, 0.4)
            else:
                感受[脑号] = self.随机器.uniform(-0.4, 0.4)
            感受[4] = self.随机器.uniform(-0.4, 0.4)
            self.感受器.append(感受)
        # 回接:20个固定节点
        self._回接节点 = [self.随机器.randint(0, 节点数 - 1) for _ in range(20)]
        # 末梢非线性(自欺): 声/光通道, 低于阈值→0, 高于阈值→卡顶, 中间按比例
        # [来源不明] 2026-09-20 查不到依据。数值首次出现在"脱开实验"输出中，但指令/实验记录/会话上下文压缩前的推导过程均未保留。
        #   声 低=0.03 高=0.8; 光 低=0.015 高=0.7。
        #   保留不改。如要重定，需做参数扫描实验（扫 0.005~0.1 截断、0.5~1.0 饱和），看哪个组合让四脑差异最小。
        self.末梢 = {"声": {"低": 0.015, "高": 0.8}, "光": {"低": 0.015, "高": 0.7}}
        self.末梢统计 = {"声": {"截断": 0, "饱和": 0}, "光": {"截断": 0, "饱和": 0}}
        self.末梢上轮归一 = {"声": None, "光": None}
        # 中间连接: 左脑↔右脑, 20 对 × 2 方向 = 40 根
        # 筋树: 脑间六对互连,每对20-50根×2方向
        self.筋树 = {}
        脑对 = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        for (甲,乙) in 脑对:
            for _ in range(脑间连接数):
                a = self.随机器.randint(甲 * self.脑大小, (甲+1) * self.脑大小 - 1)
                b = self.随机器.randint(乙 * self.脑大小, (乙+1) * self.脑大小 - 1)
                延迟甲 = self.随机器.randint(1, 5)
                延迟乙 = self.随机器.randint(1, 5)
                self.筋树[(a,b)] = {"位点":1,"疲劳":1.0,"延迟":延迟甲,"生辰":0,"末传":0,"传导次数":0,"延迟缓冲":[0.0]*延迟甲}
                self.筋树[(b,a)] = {"位点":1,"疲劳":1.0,"延迟":延迟乙,"生辰":0,"末传":0,"传导次数":0,"延迟缓冲":[0.0]*延迟乙}
        self.激活 = [0.0] * 节点数
        self.不应期 = [0] * 节点数
        self.冷却 = 冷却
        self.上轮激活 = None
        self.上轮环境 = None
        self.上轮物理 = None
        self.传导差 = None
        self.上限满值 = 上限满值
        self.当前上限 = 上限满值
        self.温度满值 = 温度满值
        self.功耗满值 = 功耗满值
        self.脑间连接数 = 脑间连接数
        self.日志 = []
        self.脑共火计数 = [0] * 脑数
        # 并行数组缓存（numba加速用）
        self._缓存有效 = False
        self._筋数 = 0

    def _同步数组(self):
        """将筋树数据同步到并行数组，供numba函数使用。"""
        M = len(self.筋树)
        if M == 0:
            self._筋数 = 0
            return
        if not self._缓存有效 or M != self._筋数:
            self._筋_j = np.empty(M, dtype=np.int32)
            self._筋_i = np.empty(M, dtype=np.int32)
            self._筋_位点 = np.empty(M, dtype=np.float64)
            self._筋_疲劳 = np.empty(M, dtype=np.float64)
            self._延迟长度 = np.empty(M, dtype=np.int32)
            self._延迟缓冲 = np.zeros((M, 5), dtype=np.float64)
            self._筋键列表 = []
            self._筋数 = M
            self._缓存有效 = True
        idx = 0
        for (j, i), 筋 in self.筋树.items():
            self._筋_j[idx] = j
            self._筋_i[idx] = i
            self._筋_位点[idx] = 筋['位点']
            self._筋_疲劳[idx] = 筋['疲劳']
            延迟 = 筋['延迟']
            self._延迟长度[idx] = 延迟
            # 复制延迟缓冲
            for d in range(min(延迟, len(筋['延迟缓冲']))):
                self._延迟缓冲[idx, d] = 筋['延迟缓冲'][d]
            self._筋键列表.append((j, i))
            idx += 1

    def _写回延迟缓冲(self):
        """将numba移位后的延迟缓冲写回筋树字典。"""
        if self._筋数 == 0:
            return
        for idx in range(self._筋数):
            j, i = self._筋键列表[idx]
            筋 = self.筋树.get((j, i))
            if 筋 is not None:
                延迟 = self._延迟长度[idx]
                筋['延迟缓冲'] = [self._延迟缓冲[idx, d] for d in range(延迟)]

    def 推进(self):
        self.时刻 += 1

        # 电疗: 突发刺激, 每拍低概率触发, 大幅值, 单通道, 一拍有效
        通道列表 = ["光", "声", "力", "温"]
        if self.随机器.random() < 0.05:
            击中 = self.随机器.choice(通道列表)
            倍率 = self.随机器.uniform(1.5, 2.2)
            原值 = self.环境[击中]
            self.环境[击中] = 原值 * 倍率
            self.环境["_电疗"] = [击中, 原值]

        原归一 = [
            max(0.0, min(1.0, (self.环境["光"] - 1) / 500)),
            max(0.0, min(1.0, (self.环境["声"] - 25) / 50)),
            max(0.0, min(1.0, (self.环境["力"] - 1) / 30)),
            max(0.0, min(1.0, (self.环境["温"] - 288) / 15)),
        ]
        __原归一 = list(原归一)

        # 末梢: 声(1)/光(0)走差变换, 力(2)/温(3)保持原值
        def 末梢(通道, 当前值):
            上轮 = self.末梢上轮归一[通道]
            差 = 0.0 if 上轮 is None else 当前值 - 上轮
            槛 = self.末梢[通道]
            self.末梢上轮归一[通道] = 当前值
            if abs(差) < 槛["低"]:
                self.末梢统计[通道]["截断"] += 1
                return 0.0
            elif abs(差) > 槛["高"]:
                self.末梢统计[通道]["饱和"] += 1
                return 1.0 if 差 > 0 else -1.0
            return 差 / 槛["高"]

        __末梢差输出 = [末梢("光", 原归一[0]), 末梢("声", 原归一[1])]
        环境输入 = [
            __末梢差输出[0],
            __末梢差输出[1],
            原归一[2],
            原归一[3],
        ]
        # 通道5: 回接差
        if self.上轮激活 is None:
            环境输入.append(0.0)
        else:
            差 = sum(self.激活[j] - self.上轮激活[j] for j in self._回接节点)
            环境输入.append(差 / 20.0)

        # 1. 算驱动(隐藏输入 = 感受器推 + 筋树传导, numba加速)
        __感受器增益 = [self.感受器[i][k] for i in range(self.节点数) for k in range(5)]
        隐藏输入数组 = np.zeros(self.节点数)
        for i in range(self.节点数):
            for k in range(5):
                隐藏输入数组[i] += self.感受器[i][k] * 环境输入[k]
        # numba加速传导
        if len(self.筋树) > 0:
            self._同步数组()
            激活数组 = np.array(self.激活, dtype=np.float64)
            if self.传导差 is not None:
                传导差数组 = np.array(self.传导差, dtype=np.float64)
            else:
                传导差数组 = np.zeros(self.节点数)
            隐藏输入数组 += _传导加速(
                self._筋_j, self._筋_i, self._筋_位点, self._筋_疲劳,
                激活数组, 传导差数组, self._延迟缓冲, self._延迟长度
            )
            self._写回延迟缓冲()
            传导过 = list(self._筋键列表)
        else:
            传导过 = []
        隐藏输入 = list(隐藏输入数组)
        __隐藏输入 = list(隐藏输入)

        # 2. 激活累积: X(t+1) = X(t) + η·隐藏输入 + ε
        系数 = 0.1
        新激活 = [0.0] * self.节点数
        for i in range(self.节点数):
            if self.不应期[i] > 0:
                self.不应期[i] -= 1
                新激活[i] = self.激活[i]  # 不应期：保持原值
            else:
                噪声 = self.随机器.uniform(-0.01, 0.01)
                新激活[i] = self.激活[i] + 系数 * 隐藏输入[i] + 噪声
                if 新激活[i] > 0.5:
                    self.不应期[i] = self.冷却
        __新激活前 = list(新激活)

        # 3. 物理约束
        总激活 = sum(abs(一) for 一 in 新激活)
        已用内存 = sum(筋['位点'] for 筋 in self.筋树.values())
        __门控缩放 = 1.0
        if 总激活 > self.当前上限:
            缩放 = self.当前上限 / 总激活
            门控激活 = [值 * 缩放 for 值 in 新激活]
            __门控缩放 = 缩放
        else:
            门控激活 = 新激活[:]

        # ── 五条件运行：稀缺+代谢+不可逆+心率旋钮+死亡 ──

        # ① 代谢：能量随时间消耗
        # E低→饿感变强（不是弱）——摄食驱动降频到死
        # 参数：fg≥0.015（能量消耗率），c1≤0.016（降频速率）
        活跃消耗 = sum(abs(v) for v in 门控激活) / max(1, self.节点数)
        self.能量 -= self.能量消耗率 + 活跃消耗 * 0.005
        self.能量 = max(0.0, min(1.0, self.能量))

        # ② 饿感：E低→饿感变强，饿感可衰减（驱动当下）
        if self.能量 < 0.3:
            self.饿感 += (0.3 - self.能量) * 0.1
        self.饿感 = max(0.0, min(1.0, self.饿感 * (1 - self.饿感衰减)))

        # 摄食驱动：E低→摄食驱动变强（不是弱）
        self.摄食驱动 = max(0.0, (0.3 - self.能量) / 0.3) if self.能量 < 0.3 else 0.0

        # ③ 稀缺：r = 预算/需求 ∈ [0.3, 0.6]是活的带
        # 预算 = 能量决定的可用资源；需求 = 活跃度决定的资源需求
        self.需求 = 0.5 + 活跃消耗 * 2.0  # 活跃度高→需求高
        self.预算 = self.能量 * 0.8  # 能量决定预算
        self.ratio = self.预算 / max(0.01, self.需求)
        # ratio < 0.2 → 坍缩（熵归零，全部预算给同一通道）
        # ratio ∈ [0.3, 0.6] → 活的稀缺带（真取舍）
        # ratio > 0.8 → 不缺（没有取舍压力）

        # 稀缺驱动的真取舍：ratio在活的带内时，激活被预算约束
        if 0.2 < self.ratio < 0.8:
            # 真取舍——不是等比缩放，而是让最强的通道获得更多预算
            预算约束 = self.ratio  # 0.3-0.6之间
            # 最活跃的节点保持，弱节点被压缩——真取舍
            abs激活 = [abs(v) for v in 门控激活]
            排序idx = sorted(range(self.节点数), key=lambda i: abs激活[i], reverse=True)
            保留数 = max(1, int(self.节点数 * 预算约束))
            for rank, idx in enumerate(排序idx):
                if rank >= 保留数:
                    门控激活[idx] *= 0.3  # 弱通道被大幅压缩——真取舍
        elif self.ratio <= 0.2:
            # 坍缩——全部预算给同一通道（最强的）
            abs激活 = [abs(v) for v in 门控激活]
            最强 = abs激活.index(max(abs激活))
            for i in range(self.节点数):
                if i != 最强:
                    门控激活[i] *= 0.05  # 几乎归零——坍缩

        # ④ 不可逆：capacity只减不增（结构承载历史）
        # 每拍capacity微量消耗（磨损）——不可回滚
        self.capacity -= self.capacity消耗率
        self.capacity = max(self.capacity_min, self.capacity)
        # 疤痕：得不到的累积（ratio<0.3时疤痕增长）
        if self.ratio < 0.3:
            self.疤痕 += (0.3 - self.ratio) * 0.01
        self.疤痕 = min(1.0, self.疤痕)

        # ⑤ 心率旋钮：g = 3.30 + 2.0×痛觉强度
        # 痛0.135时g=3.570≈Feigenbaum点——不是调的，是常数决定的
        self.心率旋钮 = 3.30 + 2.0 * self.痛觉强度

        # ⑥ 死亡：降频+耐心阈值→永久终止
        # E持续低→降频→降频到底+超过耐心阈值→永久终止
        if self.能量 < 0.1:
            self.低能量持续 += 1
            self.降频 -= self.降频率
            self.降频 = max(0.0, self.降频)
        else:
            self.低能量持续 = max(0, self.低能量持续 - 1)
            # 能量恢复时降频也可以部分恢复（但不完全——不可逆）
            self.降频 = min(1.0, self.降频 + 0.005)

        # 永久终止条件：降频到底 + 超过耐心阈值
        if self.降频 <= 0.01 and self.低能量持续 > self.耐心阈值:
            self.已死亡 = True
        self.信号记录 = {"原归一": __原归一, "末梢差光": __末梢差输出[0], "末梢差声": __末梢差输出[1],
            "感受器增益": __感受器增益, "隐藏输入": __隐藏输入, "新激活前": __新激活前,
            "门控缩放": __门控缩放,
            "五条件": {
                "能量": round(self.能量, 4),
                "ratio": round(self.ratio, 4),
                "饿感": round(self.饿感, 4),
                "摄食驱动": round(self.摄食驱动, 4),
                "capacity": round(self.capacity, 4),
                "疤痕": round(self.疤痕, 4),
                "心率旋钮": round(self.心率旋钮, 4),
                "降频": round(self.降频, 4),
                "低能量持续": self.低能量持续,
                "已死亡": self.已死亡,
            }}
        热量 = sum(值 * 值 for 值 in 门控激活)
        活跃节点数 = sum(1 for 值 in 门控激活 if abs(值) > 0.1)
        功耗饱和度 = 活跃节点数 / self.功耗满值
        温度饱和度 = 热量 / self.温度满值
        内存饱和度 = len(self.筋树) / (self.节点数 * (self.节点数 - 1))
        压力 = max(内存饱和度, 温度饱和度, 功耗饱和度)
        目标上限 = max(0.01, self.上限满值 * (1.0 - 压力))
        self.当前上限 = 0.9 * self.当前上限 + 0.1 * 目标上限

        # 4. 激活确定
        self.激活 = 门控激活
        当前物理 = (已用内存, 热量, 活跃节点数, self.当前上限)
        记录 = {"时间戳": time.time(), "拍数": self.时刻, "环境": dict(self.环境),
                "环境输入": list(环境输入), "激活": list(self.激活), "物理": 当前物理}

        # 5. 差计算
        if self.上轮激活 is None:
            激活差 = list(self.激活)
            记录["差"] = {"激活差": 激活差, "环境差": {k: 0.0 for k in ("光", "声", "力", "温")}, "物理差": (0.0, 0.0, 0.0, 0.0)}
        else:
            激活差 = [self.激活[i] - self.上轮激活[i] for i in range(self.节点数)]
            环境差 = {k: self.环境[k] - self.上轮环境[k] for k in ("光", "声", "力", "温")}
            物理差 = tuple(当前物理[i] - self.上轮物理[i] for i in range(4))
            记录["差"] = {"激活差": 激活差, "环境差": 环境差, "物理差": 物理差}
        self.传导差 = 激活差

        # 6. 共火: 同拍亮才连 (第一拍跳过)
        # 规则: 脑内阈值0.01/概率0.15; 跨脑双节点都>0.02且概率0.05
        共火对数 = 0
        首拍 = (self.上轮激活 is None)
        if not 首拍:
            同拍亮_脑内 = [索引 for 索引 in range(self.节点数) if abs(激活差[索引]) > 0.01]
            同拍亮_跨脑 = set(索引 for 索引 in range(self.节点数) if abs(激活差[索引]) > 0.02)
            脑亮 = []
            脑共火 = []
            for nb in range(self.脑数):
                起点 = nb * self.脑大小
                终点 = 起点 + self.脑大小
                该脑亮 = [i for i in 同拍亮_脑内 if 起点 <= i < 终点]
                脑亮.append(该脑亮)
                是共火 = len(该脑亮) >= 2
                脑共火.append(是共火)
                if 是共火:
                    self.脑共火计数[nb] += 1
            if len(同拍亮_脑内) >= 2:
                for a in 同拍亮_脑内:
                    for b in 同拍亮_脑内:
                        if a == b:
                            continue
                        筋键 = (a, b)
                        是跨脑 = (a // self.脑大小 != b // self.脑大小)
                        if 是跨脑:
                            if a not in 同拍亮_跨脑 or b not in 同拍亮_跨脑:
                                continue
                            概率 = 0.05
                        else:
                            概率 = 0.15
                        if 筋键 not in self.筋树:
                            if self.随机器.random() < 概率:
                                延迟新 = self.随机器.randint(1, 5)
                                self.筋树[筋键] = {"位点": 1, "疲劳": 1.0, "延迟": 延迟新, "生辰": self.时刻, "末传": self.时刻, "传导次数": 0, "延迟缓冲": [0.0]*延迟新}
                        else:
                            筋 = self.筋树[筋键]
                            筋['末传'] = self.时刻
                            if 筋['疲劳'] > 0.4 and self.随机器.random() < 0.05:
                                筋['位点'] = min(40, 筋['位点'] + 1)
                        共火对数 += 1
                if 共火对数 > 0:
                    记录["共火"] = {"节点": 同拍亮_脑内, "对数": 共火对数, "脑共火": 脑共火}

        # 7. 筋树代谢
        for (j, i) in 传导过:
            if (j, i) in self.筋树:
                筋 = self.筋树[(j, i)]
                筋['疲劳'] = max(0.2, 筋['疲劳'] * 0.9)
                if '传导次数' not in 筋:
                    筋['传导次数'] = 0
                筋['传导次数'] += 1
                if 筋['传导次数'] % 200 == 0 and self.随机器.random() < 0.5:
                    pass  # 延迟不再衰减，保持随机分布
        死筋 = []
        for 筋键, 筋 in self.筋树.items():
            if self.时刻 - 筋['末传'] > 5000 and (self.时刻 - 筋['末传']) % 1000 == 0:
                筋['位点'] -= 1
                if 筋['位点'] < 1:
                    死筋.append(筋键)
        if self.时刻 % 100 == 0:
            for 筋 in self.筋树.values():
                if 筋['疲劳'] < 1.0:
                    筋['疲劳'] = min(1.0, 筋['疲劳'] + (100.0 / 300.0) * (1.0 - 筋['疲劳']))
        for 筋键 in 死筋:
            del self.筋树[筋键]
        if 死筋:
            self._缓存有效 = False

        # 电疗恢复: 一拍有效
        if "_电疗" in self.环境:
            击中, 原值 = self.环境["_电疗"]
            self.环境[击中] = 原值
            del self.环境["_电疗"]

        # 8. 筋树变动后标记缓存无效
        self._缓存有效 = False
        # 9. 保存快照 + 记痕
        self.上轮激活 = list(self.激活)
        self.上轮环境 = dict(self.环境)
        self.上轮物理 = 当前物理
        # 自己点: 每1000拍
        if self.时刻 > 0 and self.时刻 % 1000 == 0:
            self._自己点(记录)
        self.日志.append(记录)
        with open(痕文件, "a", encoding="utf-8") as 文件:
            import json
            文件.write(json.dumps(记录, ensure_ascii=False) + "\n")

    def _自己点(self, 记录=None):
        """每1000拍，从自指簇选节点+0.5激活。
        沉淀：上次差大(≥0.1) → 同节点续点；差小 → 换新。
        脱开：同节点连点×0.9，换节点恢复。
        """
        回接集 = set(self._回接节点)
        簇 = set()
        for (j, i) in self.筋树:
            if j in 回接集 and i in 回接集:
                簇.add(j); 簇.add(i)
        if not 簇:
            return
        簇列表 = list(簇)
        if not hasattr(self, '_点记录'):
            self._点记录 = {}
        候选 = [n for n in 簇列表 if self._点记录.get(n, 1.0) >= 0.3]
        if not 候选:
            return
        上次目标 = getattr(self, '_上次点目标', None)
        上次差 = getattr(self, '_上次反馈差', None)
        # 沉淀决策：上次差大且目标仍在候选 → 续点同一个
        if (上次目标 is not None and 上次目标 in 候选
                and 上次差 is not None and abs(上次差) >= 0.1):
            目标 = 上次目标
            决策 = "沉淀续点"
        elif hasattr(self, '_点反馈') and self._点反馈:
            新候选 = [n for n in 候选 if n != 上次目标] if 上次目标 is not None else 候选
            if not 新候选:
                新候选 = 候选
            目标 = max(新候选, key=lambda n: abs(self._点反馈.get(n, 0)))
            决策 = "换点"
        else:
            目标 = self.随机器.choice(候选)
            决策 = "随机初点"
        for n in 簇列表:
            if n == 目标:
                self._点记录[n] = self._点记录.get(n, 1.0) * 0.9
            elif n == 上次目标 and n != 目标:
                self._点记录[n] = 1.0
        self._上次点目标 = 目标
        self._点前值 = {n: self.激活[n] for n in 簇列表}
        self.激活[目标] += 0.5
        self._点次数 = getattr(self, '_点次数', 0) + 1
        if hasattr(self, '_点前值_last'):
            def _safe_get(d, k):
                return d.get(k, 0.0)
            self._点反馈 = {n: self.激活[n] - _safe_get(self._点前值_last, n) for n in 簇列表}
        # 沉淀：记本次目标反馈差，供下次决策
        self._上次反馈差 = self._点反馈.get(目标, 0) if hasattr(self, '_点反馈') else None
        self._点前值_last = dict(self._点前值)
        # 记序列
        if not hasattr(self, '_点序列'):
            self._点序列 = []
        self._点序列.append(目标)
        # 接差到筋树：上次差大(≥0.1) → 目标节点相邻筋的位点+1
        加位点数 = 0
        if self._上次反馈差 is not None and abs(self._上次反馈差) >= 0.1:
            for (j, i), 筋 in self.筋树.items():
                if j == 目标 or i == 目标:
                    筋['位点'] = min(40, 筋['位点'] + 1)
                    加位点数 += 1
        # 沉淀进痕
        if 记录 is not None:
            记录["自己点"] = {"目标": 目标, "决策": 决策, "上次差": self._上次反馈差, "加位点筋数": 加位点数}

    def 连接(self):
        结果 = []
        for (j, i), 筋 in self.筋树.items():
            结果.append((j, i, 筋['位点'], 筋['疲劳'], 筋['延迟'], 筋['生辰'], 筋['末传']))
        结果.sort(key=lambda c: -c[2])
        return 结果

    def 设置痛觉(self, 痛觉强度):
        """设置痛觉强度——用于心率旋钮 g = 3.30 + 2.0×痛。

        实验数据：痛0.135时g=3.570≈Feigenbaum点(δ=4.669的分支点)。
        这个位置不是调出来的——是常数决定的。
        """
        self.痛觉强度 = max(0.0, min(1.0, 痛觉强度))

    def 摄食(self, 能量值):
        """从环境获取能量——代谢的摄食端。

        E低→饿感变强→摄食驱动→获取能量→E恢复。
        但capacity不可恢复（不可逆）——磨损是永久的。
        """
        self.能量 = min(self.能量满值, self.能量 + 能量值)
        # 摄食后饿感衰减（饿感驱动当下，可衰减）
        self.饿感 *= 0.5

    def 五条件摘要(self):
        """五条件状态摘要——稀缺+代谢+不可逆+心率旋钮+死亡。"""
        return {
            "能量": round(self.能量, 4),
            "ratio": round(self.ratio, 4),
            "饿感": round(self.饿感, 4),
            "摄食驱动": round(self.摄食驱动, 4),
            "capacity": round(self.capacity, 4),
            "疤痕": round(self.疤痕, 4),
            "心率旋钮": round(self.心率旋钮, 4),
            "降频": round(self.降频, 4),
            "低能量持续": self.低能量持续,
            "已死亡": self.已死亡,
            "稀缺带": "活的" if 0.3 <= self.ratio <= 0.6 else (
                "坍缩" if self.ratio < 0.2 else (
                    "不缺" if self.ratio > 0.8 else "过渡")),
        }


def 跑(滴答数, 节律表, 节点数=48, 种子=1, 冷却=3, 上限满值=6.0,
       温度满值=1.5, 功耗满值=15.0):
    b = 身体(节点数=节点数, 种子=种子, 冷却=冷却, 上限满值=上限满值,
             温度满值=温度满值, 功耗满值=功耗满值)
    for _ in range(滴答数):
        分钟 = b.时刻 * 4
        钟点 = (分钟 / 60) % 24
        for 起, 止, 环境 in 节律表:
            if 起 <= 钟点 < 止:
                b.环境.update(环境)
                break
        b.推进()
    return b
