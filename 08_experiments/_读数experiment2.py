"""读数实验2：burst周期/正池注入/筋连接拓扑"""
import sys
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py", "r", encoding="utf-8") as f:
    原源码 = f.read()

_old_init_end = '        self.脑共火计数 = [0] * 脑数\n'
_new_init_end = '        self.脑共火计数 = [0] * 脑数\n        self._正池 = 0.0\n        self._负池 = 0.0\n'
_old_通道5 = '        # 通道5: 回接差\n        if self.上轮激活 is None:\n            环境输入.append(0.0)\n        else:\n            差 = sum(self.激活[j] - self.上轮激活[j] for j in self._回接节点)\n            环境输入.append(差 / 20.0)'
_new_通道5 = '        # 通道5: 分流实验占位\n        环境输入.append(0.0)\n'
_old_隐藏 = '        隐藏输入 = list(隐藏输入数组)\n'
_new_隐藏 = '        隐藏输入 = list(隐藏输入数组)\n        # 分流实验：正池只加正感受器回接节点\n        if self.上轮激活 is not None and getattr(self, "_正池", 0.0) > 0:\n            for j in self._回接节点:\n                if self.感受器[j][4] > 0:\n                    隐藏输入[j] += self.感受器[j][4] * self._正池\n'
_old_推进头 = '        self.时刻 += 1\n'
_new_推进头 = '        self.时刻 += 1\n        # 分流实验：负池泄放\n        if hasattr(self, "_负池"):\n            self._负池 *= 0.99\n'
_old_差存储 = '        self.传导差 = 激活差\n'
_new_差存储 = '        self.传导差 = 激活差\n        # 分流实验：正负差分流累积\n        if self.上轮激活 is not None:\n            正 = 0.0\n            负 = 0.0\n            for j in self._回接节点:\n                差_j = 激活差[j]\n                if 差_j > 0:\n                    正 += 差_j\n                else:\n                    负 += abs(差_j)\n            self._正池 += 正\n            self._负池 += 负\n'

新源码 = 原源码.replace(_old_init_end, _new_init_end, 1)
新源码 = 新源码.replace(_old_通道5, _new_通道5, 1)
新源码 = 新源码.replace(_old_隐藏, _new_隐藏, 1)
新源码 = 新源码.replace(_old_推进头, _new_推进头, 1)
新源码 = 新源码.replace(_old_差存储, _new_差存储, 1)

with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py", "w", encoding="utf-8") as f:
    f.write(新源码)

import importlib
import 身体稳态
importlib.reload(身体稳态)

身体稳态.痕文件 = "/dev/null"

b = 身体稳态.身体(节点数=96, 种子=1, 上限满值=999999)
b.环境["光"] = 0.0
b.环境["声"] = 0.0
b.环境["力"] = 0.0
b.环境["温"] = 0.0
身体稳态.身体._自己点 = lambda self, *a, **k: None

回接列表 = b._回接节点
回接集 = set(回接列表)
正回接 = [j for j in 回接列表 if b.感受器[j][4] > 0]

信息全 = {}
for t in range(1, 20001):
    b.推进()
    # 每拍记录：时刻，各回接节点激活，正池
    信息 = {
        "正池": b._正池,
        "负池": b._负池,
        "激活": {j: b.激活[j] for j in 正回接}
    }
    信息全[t] = 信息

# 找burst头：节点62激活从baseline跳至5以上
burst头 = []
for t in range(2, 20001):
    if 信息全[t]["激活"][62] > 5 and 信息全[t-1]["激活"][62] < 1:
        burst头.append(t)

# 一、burst周期
print("= = = 一、burst周期与正池水位 = = =")
print(f"burst头总数 = {len(burst头)}")
print("所有burst头拍号：")
for bt in burst头:
    print(bt)

print()
print("相邻burst间隔：")
间隔列表 = []
for i in range(1, len(burst头)):
    间隔 = burst头[i] - burst头[i-1]
    间隔列表.append(间隔)
    print(f"burst{i}({burst头[i-1]}) -> burst{i+1}({burst头[i]}): {间隔}拍，正池={信息全[burst头[i-1]]['正池']:.2f}")

print()
# burst时正池增量 vs baseline增量
print("每次burst时正池涨多少：")
for i, bt in enumerate(burst头):
    if bt > 2:
        burst增量 = 信息全[bt]["正池"] - 信息全[bt-1]["正池"]
        baseline增量 = 信息全[bt-1]["正池"] - 信息全[bt-2]["正池"] if bt > 2 else 0
        倍率 = burst增量 / baseline增量 if baseline增量 > 0 else float('inf')
        print(f"拍{bt} 正池增量={burst增量:.2f} 前拍baseline增量={baseline增量:.2f} 倍率={倍率:.2f} 正池总={信息全[bt]['正池']:.2f}")

# 节点62/64 baseline期：取一个burst后的baseline拍（burst后第5拍）
print()
print("节点62/64在burst后baseline期（burst+5拍）的激活：")
for i, bt in enumerate(burst头[:15]):
    baseline_t = bt + 5
    if baseline_t in 信息全:
        print(f"拍{baseline_t}(burst@{bt}+5) 节点62={信息全[baseline_t]['激活'][62]:.8f} 节点64={信息全[baseline_t]['激活'][64]:.8f} 节点32={信息全[baseline_t]['激活'][32]:.8f}")

# 二、感受器[4] 详细值
print()
print("= = = 二、正池注入方式 = = =")
print("所有正感受器回接节点的 感受器[j][4] 值：")
for j in 正回接:
    print(f"节点{j}: 感受器[4]={b.感受器[j][4]:.6f}")

print()
print("正池379857乘以各节点感受器[4]：")
for j in 正回接:
    注入量 = 379857.71 * b.感受器[j][4]
    print(f"节点{j}: 感受器[4]={b.感受器[j][4]:.6f}, 正池*感受器[4]={注入量:.2f}")

# 三、筋连接拓扑
print()
print("= = = 三、节点62、64、5、17的筋连接 = = =")
for node in [62, 64, 5, 17]:
    print(f"--- 节点{node} ---")
    # 连出去的筋
    出筋 = [(j, i, 筋) for (j, i), 筋 in b.筋树.items() if j == node]
    入筋 = [(j, i, 筋) for (j, i), 筋 in b.筋树.items() if i == node]
    print(f"  出筋数={len(出筋)}，入筋数={len(入筋)}")
    print(f"  出筋详情（目标节点,位点,疲劳,跨脑否）：")
    for j, i, 筋 in 出筋:
        跨脑 = "跨脑" if (j//24 != i//24) else "脑内"
        回接标记 = "回接" if i in 回接集 else "非回接"
        print(f"    {j}->{i}: 位点={筋['位点']}, 疲劳={筋['疲劳']:.4f}, {跨脑}, 目标={回接标记}")
    # 筋位点合计
    位点合计 = sum(筋['位点'] for _, _, 筋 in 出筋)
    print(f"  出筋位点合计 = {位点合计}")

# 还原
with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py", "w", encoding="utf-8") as f:
    f.write(原源码)

print()
print("源码已还原")
