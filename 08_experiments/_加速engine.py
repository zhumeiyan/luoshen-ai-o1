"""身体稳态.py —— V4 加速版。
numpy数组为主，每拍O(1)检查M，M不变时跳过重建。
无 _筋数=-1 hack。
"""
import random, time
import numpy as np
from numba import njit

@njit(error_model='python')
def _传导加速(筋_j, 筋_i, 筋_位点, 筋_疲劳, 激活, 传导差, 延迟缓冲, 延迟长度):
    N = len(激活)
    隐藏输入 = np.zeros(N)
    for k in range(len(筋_j)):
        j = 筋_j[k]
        i = 筋_i[k]
        差因子 = 传导差[j]
        信号 = 筋_位点[k] * 筋_疲劳[k] * 激活[j] * 差因子
        延迟 = 延迟长度[k]
        到达信号 = 延迟缓冲[k, 0]
        for d in range(延迟 - 1):
            延迟缓冲[k, d] = 延迟缓冲[k, d + 1]
        延迟缓冲[k, 延迟 - 1] = 信号
        隐藏输入[i] += 到达信号
    return 隐藏输入

@njit(error_model='python')
def _激活不应期更新(激活, 隐藏输入, 不应期, 系数, 噪声, 冷却):
    新激活 = np.empty_like(激活)
    for i in range(len(激活)):
        if 不应期[i] > 0:
            不应期[i] -= 1
            新激活[i] = 激活[i]
        else:
            新激活[i] = 激活[i] + 系数 * 隐藏输入[i] + 噪声[i]
            if 新激活[i] > 0.5:
                不应期[i] = 冷却
    return 新激活

class 身体:
    def __init__(self, 节点数=96, 种子=1, 冷却=3, 上限满值=6.0, 温度满值=1.5, 功耗满值=15.0, 脑数=4, 脑间连接数=150):
        self.随机器 = random.Random(种子)
        assert 节点数 % 脑数 == 0
        self.节点数 = 节点数
        self.脑数 = 脑数
        self.脑大小 = 节点数 // 脑数
        self.环境 = {"光": 50.0, "声": 45.0, "力": 10.0, "温": 295.0}
        self.时刻 = 0
        self.感受器 = []
        for i in range(节点数):
            脑号 = i // self.脑大小
            感受 = [0.0] * 5
            if 脑号 == 0:
                感受[0] = self.随机器.uniform(-0.4, 0.4)
                感受[2] = self.随机器.uniform(-0.4, 0.4)
            else:
                感受[脑号] = self.随机器.uniform(-0.4, 0.4)
            感受[4] = self.随机器.uniform(-0.4, 0.4)
            self.感受器.append(感受)
        self._感受器矩阵 = np.array(self.感受器, dtype=np.float64)
        self._回接节点 = [self.随机器.randint(0, 节点数 - 1) for _ in range(20)]
        self.末梢 = {"声": {"低": 0.03, "高": 0.8}, "光": {"低": 0.015, "高": 0.7}}
        self.末梢统计 = {"声": {"截断": 0, "饱和": 0}, "光": {"截断": 0, "饱和": 0}}
        self.末梢上轮归一 = {"声": None, "光": None}
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
        self.激活数组 = np.zeros(节点数, dtype=np.float64)
        self.不应期数组 = np.zeros(节点数, dtype=np.int32)
        self.不应期 = list(self.不应期数组)
        self.上轮激活数组 = None
        self.传导差数组 = None
        self.上轮环境 = None
        self.上轮物理 = None
        self.上限满值 = 上限满值
        self.当前上限 = 上限满值
        self.温度满值 = 温度满值
        self.功耗满值 = 功耗满值
        self.脑间连接数 = 脑间连接数
        self.冷却 = 冷却
        self.日志 = []
        self.脑共火计数 = [0] * 脑数
        self._筋数 = 0
        self._筋键列表缓存 = None
        self._筋索引映射 = {}
        self._need_rebuild = True
        self._筋_j = np.empty(0, dtype=np.int32)
        self._筋_i = np.empty(0, dtype=np.int32)
        self._筋_位点 = np.empty(0, dtype=np.float64)
        self._筋_疲劳 = np.empty(0, dtype=np.float64)
        self._延迟长度 = np.empty(0, dtype=np.int32)
        self._延迟缓冲 = np.zeros((0, 5), dtype=np.float64)
        self._生辰 = np.empty(0, dtype=np.int64)
        self._末传 = np.empty(0, dtype=np.int64)

    def _同步数组(self):
        """M变化时重建，保留existing entries的numpy衰减值"""
        M = len(self.筋树)
        if M == 0:
            self._筋数 = 0
            return
        if M == self._筋数 and not self._need_rebuild:
            return
        old_mapping = dict(self._筋索引映射) if self._筋索引映射 else {}
        old_num = self._筋数
        self._筋键列表缓存 = list(self.筋树.keys())
        self._筋索引映射 = {k: idx for idx, k in enumerate(self._筋键列表缓存)}
        realloc = (M != old_num or old_num <= 0 or self._筋_j.size != M)
        if realloc:
            new_筋_j = np.empty(M, dtype=np.int32)
            new_筋_i = np.empty(M, dtype=np.int32)
            new_位点 = np.empty(M, dtype=np.float64)
            new_疲劳 = np.empty(M, dtype=np.float64)
            new_延迟长度 = np.empty(M, dtype=np.int32)
            new_延迟缓冲 = np.zeros((M, 5), dtype=np.float64)
            new_生辰 = np.empty(M, dtype=np.int64)
            new_末传 = np.empty(M, dtype=np.int64)
            self._筋_j = new_筋_j
            self._筋_i = new_筋_i
            self._筋_位点 = new_位点
            self._筋_疲劳 = new_疲劳
            self._延迟长度 = new_延迟长度
            self._延迟缓冲 = new_延迟缓冲
            self._生辰 = new_生辰
            self._末传 = new_末传
        new_筋_j = self._筋_j
        new_筋_i = self._筋_i
        new_位点 = self._筋_位点
        new_疲劳 = self._筋_疲劳
        new_延迟长度 = self._延迟长度
        new_延迟缓冲 = self._延迟缓冲
        new_生辰 = self._生辰
        new_末传 = self._末传
        for idx, (j, i) in enumerate(self._筋键列表缓存):
            new_筋_j[idx] = j
            new_筋_i[idx] = i
        for idx, 键 in enumerate(self._筋键列表缓存):
            if 键 not in old_mapping or old_mapping[键] >= old_num:
                筋 = self.筋树[键]
                new_位点[idx] = 筋['位点']
                new_疲劳[idx] = 筋['疲劳']
                new_延迟长度[idx] = 筋['延迟']
                for d in range(筋['延迟']):
                    new_延迟缓冲[idx, d] = 筋['延迟缓冲'][d]
                new_生辰[idx] = 筋.get('生辰', 0)
                new_末传[idx] = 筋['末传']
            else:
                old_idx = old_mapping[键]
                new_疲劳[idx] = self._筋_疲劳[old_idx]
                new_位点[idx] = self._筋_位点[old_idx]
                new_末传[idx] = self._末传[old_idx]
                new_生辰[idx] = self._生辰[old_idx]
                new_延迟长度[idx] = self._延迟长度[old_idx]
                for d in range(min(5, self._延迟长度[old_idx])):
                    new_延迟缓冲[idx, d] = self._延迟缓冲[old_idx, d]
        self._筋数 = M
        self._need_rebuild = False

    def 推进(self):
        self.时刻 += 1
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

        环境输入 = [
            末梢("光", 原归一[0]),
            末梢("声", 原归一[1]),
            原归一[2],
            原归一[3],
        ]
        if self.上轮激活数组 is None:
            环境输入.append(0.0)
        else:
            回接索引 = np.array(self._回接节点, dtype=np.int32)
            差 = np.sum(self.激活数组[回接索引] - self.上轮激活数组[回接索引])
            环境输入.append(差 / 20.0)

        环境输入数组 = np.array(环境输入, dtype=np.float64)
        隐藏输入数组 = self._感受器矩阵 @ 环境输入数组
        if len(self.筋树) > 0:
            self._同步数组()
            if self.传导差数组 is None:
                传导差数组 = np.zeros(self.节点数)
            else:
                传导差数组 = self.传导差数组
            隐藏输入数组 += _传导加速(
                self._筋_j, self._筋_i, self._筋_位点, self._筋_疲劳,
                self.激活数组, 传导差数组, self._延迟缓冲, self._延迟长度
            )

        系数 = 0.1
        噪声 = np.random.uniform(-0.01, 0.01, self.节点数)
        新激活数组 = _激活不应期更新(
            self.激活数组, 隐藏输入数组, self.不应期数组, 系数, 噪声, self.冷却
        )

        总激活 = np.sum(np.abs(新激活数组))
        if self._筋数 > 0:
            已用内存 = float(np.sum(self._筋_位点))
        else:
            已用内存 = 0.0
        if 总激活 > self.当前上限:
            缩放 = self.当前上限 / 总激活
            门控激活数组 = 新激活数组 * 缩放
        else:
            门控激活数组 = 新激活数组.copy()
        热量 = np.sum(门控激活数组 * 门控激活数组)
        活跃节点数 = int(np.sum(np.abs(门控激活数组) > 0.1))
        功耗饱和度 = 活跃节点数 / self.功耗满值
        温度饱和度 = 热量 / self.温度满值
        内存饱和度 = len(self.筋树) / (self.节点数 * (self.节点数 - 1))
        压力 = max(内存饱和度, 温度饱和度, 功耗饱和度)
        目标上限 = max(0.01, self.上限满值 * (1.0 - 压力))
        self.当前上限 = 0.9 * self.当前上限 + 0.1 * 目标上限

        self.激活数组 = 门控激活数组
        当前物理 = (已用内存, 热量, 活跃节点数, self.当前上限)

        if self.上轮激活数组 is None:
            激活差数组 = self.激活数组.copy()
        else:
            激活差数组 = self.激活数组 - self.上轮激活数组
        self.传导差数组 = 激活差数组

        首拍 = (self.上轮激活数组 is None)
        if not 首拍:
            同拍亮_脑内 = [索引 for 索引 in range(self.节点数) if abs(激活差数组[索引]) > 0.01]
            同拍亮_跨脑 = set(索引 for 索引 in range(self.节点数) if abs(激活差数组[索引]) > 0.02)
            脑共火 = []
            for nb in range(self.脑数):
                该脑亮 = [i for i in 同拍亮_脑内 if nb * self.脑大小 <= i < (nb+1) * self.脑大小]
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
                            if 筋键 in self._筋索引映射:
                                idx = self._筋索引映射[筋键]
                                if idx < self._筋数 and idx < self._筋_位点.size:
                                    self._末传[idx] = self.时刻
                                    if self._筋_疲劳[idx] > 0.4 and self.随机器.random() < 0.05:
                                        self._筋_位点[idx] = min(40, self._筋_位点[idx] + 1)

        # 7. 疲劳衰减 (numpy)
        if self._筋数 > 0:
            self._筋_疲劳 *= 0.9
            np.maximum(self._筋_疲劳, 0.2, out=self._筋_疲劳)

        # 8. 死筋检测 (numpy)
        if self._筋数 > 0:
            时间差 = self.时刻 - self._末传
            死掩码 = (时间差 > 5000) & ((时间差 % 1000) == 0)
            if np.any(死掩码):
                self._筋_位点[死掩码] -= 1
                真死 = self._筋_位点 < 1
                if np.any(真死):
                    真死位置 = list(np.where(真死)[0])
                    for pos in 真死位置:
                        if pos < len(self._筋键列表缓存):
                            键 = self._筋键列表缓存[pos]
                            if 键 in self.筋树:
                                del self.筋树[键]
                    self._need_rebuild = True

        if self.时刻 % 100 == 0 and self._筋数 > 0:
            低疲劳 = self._筋_疲劳 < 1.0
            if np.any(低疲劳):
                恢复量 = (100.0 / 300.0) * (1.0 - self._筋_疲劳[低疲劳])
                self._筋_疲劳[低疲劳] = np.minimum(1.0, self._筋_疲劳[低疲劳] + 恢复量)

        if "_电疗" in self.环境:
            击中, 原值 = self.环境["_电疗"]
            self.环境[击中] = 原值
            del self.环境["_电疗"]

        self.上轮激活数组 = self.激活数组.copy()
        self.上轮环境 = dict(self.环境)
        self.上轮物理 = 当前物理
        if self.时刻 > 0 and self.时刻 % 1000 == 0:
            self._自己点()

    def _自己点(self, 记录=None):
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
        if (上次目标 is not None and 上次目标 in 候选
                and 上次差 is not None and abs(上次差) >= 0.1):
            目标 = 上次目标
        elif hasattr(self, '_点反馈') and self._点反馈:
            新候选 = [n for n in 候选 if n != 上次目标] if 上次目标 is not None else 候选
            if not 新候选:
                新候选 = 候选
            目标 = max(新候选, key=lambda n: abs(self._点反馈.get(n, 0)))
        else:
            目标 = self.随机器.choice(候选)
        for n in 簇列表:
            if n == 目标:
                self._点记录[n] = self._点记录.get(n, 1.0) * 0.9
            elif n == 上次目标 and n != 目标:
                self._点记录[n] = 1.0
        self._上次点目标 = 目标
        self._点前值 = {n: float(self.激活数组[n]) for n in 簇列表}
        self.激活数组[目标] += 0.5
        self._点次数 = getattr(self, '_点次数', 0) + 1
        if hasattr(self, '_点前值_last'):
            def _safe_get(d, k):
                return d.get(k, 0.0)
            self._点反馈 = {n: float(self.激活数组[n]) - _safe_get(self._点前值_last, n) for n in 簇列表}
        self._上次反馈差 = self._点反馈.get(目标, 0) if hasattr(self, '_点反馈') else None
        self._点前值_last = dict(self._点前值)
        if not hasattr(self, '_点序列'):
            self._点序列 = []
        self._点序列.append(目标)
        if self._上次反馈差 is not None and abs(self._上次反馈差) >= 0.1:
            for (j, i) in self.筋树:
                if j == 目标 or i == 目标:
                    if (j, i) in self._筋索引映射:
                        idx = self._筋索引映射[(j, i)]
                        if idx < self._筋数:
                            self._筋_位点[idx] = min(40, self._筋_位点[idx] + 1)

    def 连接(self):
        结果 = []
        for (j, i), 筋 in self.筋树.items():
            结果.append((j, i, 筋['位点'], 筋['疲劳'], 筋['延迟'], 筋['生辰'], 筋['末传']))
        结果.sort(key=lambda c: -c[2])
        return 结果

    @property
    def 激活(self):
        return list(self.激活数组)
