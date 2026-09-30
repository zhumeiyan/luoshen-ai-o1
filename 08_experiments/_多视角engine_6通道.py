"""身体稳态.py —— 6通道版。
新增通道5 = 位置标记。
感受器6维，脑0接光+力+位置，其他脑接各自通道+位置。
通道5（回接差）保持为索引4，新增通道5=位置。
"""
import random, time
import numpy as np
from numba import njit

痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"

@njit
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

@njit
def _共火检测(传导差, 阈值=0.01):
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
        # 6通道：0=光, 1=声, 2=力, 3=温, 4=回接差, 5=位置
        self.感受器 = []
        for i in range(节点数):
            脑号 = i // self.脑大小
            感受 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if 脑号 == 0:
                感受[0] = self.随机器.uniform(-0.4, 0.4)
                感受[2] = self.随机器.uniform(-0.4, 0.4)
                感受[5] = self.随机器.uniform(-0.4, 0.4)
            else:
                感受[脑号] = self.随机器.uniform(-0.4, 0.4)
                感受[5] = self.随机器.uniform(-0.4, 0.4)
            感受[4] = self.随机器.uniform(-0.4, 0.4)
            self.感受器.append(感受)
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
        self._缓存有效 = False
        self._筋数 = 0
        # 位置通道外部注入
        self._位置输入 = 0.0

    def _同步数组(self):
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
            for d in range(min(延迟, len(筋['延迟缓冲']))):
                self._延迟缓冲[idx, d] = 筋['延迟缓冲'][d]
            self._筋键列表.append((j, i))
            idx += 1

    def _写回延迟缓冲(self):
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
        # 通道索引4: 回接差
        if self.上轮激活 is None:
            环境输入.append(0.0)
        else:
            差 = sum(self.激活[j] - self.上轮激活[j] for j in self._回接节点)
            环境输入.append(差 / 20.0)
        # 通道索引5: 位置标记
        环境输入.append(float(self._位置输入))
        # 算驱动
        隐藏输入数组 = np.zeros(self.节点数)
        for i in range(self.节点数):
            for k in range(6):
                隐藏输入数组[i] += self.感受器[i][k] * 环境输入[k]
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
        # 激活累积
        系数 = 0.1
        新激活 = [0.0] * self.节点数
        for i in range(self.节点数):
            if self.不应期[i] > 0:
                self.不应期[i] -= 1
                新激活[i] = self.激活[i]
            else:
                噪声 = self.随机器.uniform(-0.01, 0.01)
                新激活[i] = self.激活[i] + 系数 * 隐藏输入[i] + 噪声
                if 新激活[i] > 0.5:
                    self.不应期[i] = self.冷却
        # 物理约束
        总激活 = sum(abs(一) for 一 in 新激活)
        已用内存 = sum(筋['位点'] for 筋 in self.筋树.values())
        if 总激活 > self.当前上限:
            缩放 = self.当前上限 / 总激活
            门控激活 = [值 * 缩放 for 值 in 新激活]
        else:
            门控激活 = 新激活[:]
        热量 = sum(值 * 值 for 值 in 门控激活)
        活跃节点数 = sum(1 for 值 in 门控激活 if abs(值) > 0.1)
        功耗饱和度 = 活跃节点数 / self.功耗满值
        温度饱和度 = 热量 / self.温度满值
        内存饱和度 = len(self.筋树) / (self.节点数 * (self.节点数 - 1))
        压力 = max(内存饱和度, 温度饱和度, 功耗饱和度)
        目标上限 = max(0.01, self.上限满值 * (1.0 - 压力))
        self.当前上限 = 0.9 * self.当前上限 + 0.1 * 目标上限
        self.激活 = 门控激活
        当前物理 = (已用内存, 热量, 活跃节点数, self.当前上限)
        记录 = {"时间戳": time.time(), "拍数": self.时刻, "环境": dict(self.环境),
                "环境输入": list(环境输入), "激活": list(self.激活), "物理": 当前物理}
        if self.上轮激活 is None:
            激活差 = list(self.激活)
            记录["差"] = {"激活差": 激活差, "环境差": {k: 0.0 for k in ("光", "声", "力", "温")}, "物理差": (0.0, 0.0, 0.0, 0.0)}
        else:
            激活差 = [self.激活[i] - self.上轮激活[i] for i in range(self.节点数)]
            环境差 = {k: self.环境[k] - self.上轮环境[k] for k in ("光", "声", "力", "温")}
            物理差 = tuple(当前物理[i] - self.上轮物理[i] for i in range(4))
            记录["差"] = {"激活差": 激活差, "环境差": 环境差, "物理差": 物理差}
        self.传导差 = 激活差
        # 跨脑共火
        self.脑共火计数 = [0] * self.脑数
        if self.上轮激活 is not None:
            共火节点 = _共火检测(np.array(激活差, dtype=np.float64))
            for node in 共火节点:
                self.脑共火计数[node // self.脑大小] += 1
            # 脑间生筋
            if 共火节点.size >= 2:
                P = 0.05
                for ii in range(len(共火节点)):
                    for jj in range(ii + 1, len(共火节点)):
                        ni = 共火节点[ii]
                        nj = 共火节点[jj]
                        bi = ni // self.脑大小
                        bj = nj // self.脑大小
                        if bi != bj:
                            key = (min(ni, nj), max(ni, nj))
                            if self.随机器.random() < P and key not in self.筋树:
                                d1 = self.随机器.randint(1, 5)
                                d2 = self.随机器.randint(1, 5)
                                self.筋树[(ni, nj)] = {"位点":1,"疲劳":1.0,"延迟":d1,"生辰":self.时刻,"末传":self.时刻,"传导次数":0,"延迟缓冲":[0.0]*d1}
                                self.筋树[(nj, ni)] = {"位点":1,"疲劳":1.0,"延迟":d2,"生辰":self.时刻,"末传":self.时刻,"传导次数":0,"延迟缓冲":[0.0]*d2}
                                self._缓存有效 = False
        else:
            共火节点 = []
        # 筋疲劳 + 死筋 + 筋生长
        for (j, i) in 传导过:
            筋 = self.筋树.get((j, i))
            if 筋 is not None:
                筋['末传'] = self.时刻
                筋['疲劳'] = max(0.2, 筋['疲劳'] * 0.9)
                if '传导次数' not in 筋:
                    筋['传导次数'] = 0
                筋['传导次数'] += 1
                if 筋['传导次数'] % 200 == 0 and self.随机器.random() < 0.5:
                    pass
        # 死筋检查
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
        if "_电疗" in self.环境:
            击中, 原值 = self.环境["_电疗"]
            self.环境[击中] = 原值
            del self.环境["_电疗"]
        self._缓存有效 = False
        self.上轮激活 = list(self.激活)
        self.上轮环境 = dict(self.环境)
        self.上轮物理 = 当前物理
        if self.时刻 > 0 and self.时刻 % 1000 == 0:
            self._自己点(记录)
        self.日志.append(记录)
        with open(痕文件, "a", encoding="utf-8") as 文件:
            import json
            文件.write(json.dumps(记录, ensure_ascii=False) + "\n")

    def _自己点(self, 记录=None):
        回接集 = set(self._回接节点)
        簇 = set()
        for (j, i) in self.筋树:
            if j in 回接集 and i in 回接集:
                簇.add(j); 簇.add(i)
        if not 簇:
            return
        簇列表 = list(簇)
        候选 = [n for n in 簇列表]
        if not 候选:
            return
        上次目标 = getattr(self, '_上次点目标', None)
        上次差 = getattr(self, '_上次反馈差', None)
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
        if not hasattr(self, '_点记录'):
            self._点记录 = {}
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
        self._上次反馈差 = self._点反馈.get(目标, 0) if hasattr(self, '_点反馈') else None
        self._点前值_last = dict(self._点前值)
        if not hasattr(self, '_点序列'):
            self._点序列 = []
        self._点序列.append(目标)
        加位点数 = 0
        if self._上次反馈差 is not None and abs(self._上次反馈差) >= 0.1:
            for (j, i), 筋 in self.筋树.items():
                if j == 目标 or i == 目标:
                    筋['位点'] = min(40, 筋['位点'] + 1)
                    加位点数 += 1
        if 记录 is not None:
            记录["自己点"] = {"目标": 目标, "决策": 决策, "上次差": self._上次反馈差, "加位点筋数": 加位点数}

    def 连接(self):
        结果 = []
        for (a, b), 筋 in self.筋树.items():
            位点 = 筋['位点']
            疲劳 = 筋['疲劳']
            延迟 = 筋['延迟']
            生辰 = 筋['生辰']
            末传 = 筋['末传']
            结果.append({"from": a, "to": b, "位点": 位点, "疲劳": 疲劳, "延迟": 延迟, "生辰": 生辰, "末传": 末传})
        return 结果
