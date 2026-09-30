"""水位→话实验：正池水位触发输出，阈值80000"""
import sys
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")

with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py", "r", encoding="utf-8") as f:
    原源码 = f.read()

# 改1: __init__ 加池、阈值、水位跟踪
_old_init_end = '        self.脑共火计数 = [0] * 脑数\n'
_new_init_end = '        self.脑共火计数 = [0] * 脑数\n        self._正池 = 0.0\n        self._负池 = 0.0\n        self._正池泄放阈值 = 80000.0\n        self._上次水位档 = 0\n'

# 改2: 通道5占位
_old_通道5 = '        # 通道5: 回接差\n        if self.上轮激活 is None:\n            环境输入.append(0.0)\n        else:\n            差 = sum(self.激活[j] - self.上轮激活[j] for j in self._回接节点)\n            环境输入.append(差 / 20.0)'
_new_通道5 = '        # 通道5: 水位实验占位\n        环境输入.append(0.0)\n'

# 改3: 隐藏输入正池注入
_old_隐藏 = '        隐藏输入 = list(隐藏输入数组)\n'
_new_隐藏 = '        隐藏输入 = list(隐藏输入数组)\n        # 水位实验：正池只加正感受器回接节点\n        if self.上轮激活 is not None and getattr(self, "_正池", 0.0) > 0:\n            for j in self._回接节点:\n                if self.感受器[j][4] > 0:\n                    隐藏输入[j] += self.感受器[j][4] * self._正池\n'

# 改4: 负池泄放
_old_推进头 = '        self.时刻 += 1\n'
_new_推进头 = '        self.时刻 += 1\n        # 水位实验：负池泄放\n        if hasattr(self, "_负池"):\n            self._负池 *= 0.99\n'

# 改5: 分流累积 + 泄放 + 水位触发输出
_old_差存储 = '        self.传导差 = 激活差\n'
_new_差存储 = '        self.传导差 = 激活差\n        # 水位实验：正负差分流累积 + 泄放 + 水位触发\n        if self.上轮激活 is not None:\n            正 = 0.0\n            负 = 0.0\n            for j in self._回接节点:\n                差_j = 激活差[j]\n                if 差_j > 0:\n                    正 += 差_j\n                else:\n                    负 += abs(差_j)\n            self._正池 += 正\n            self._负池 += 负\n            # 泄放检查\n            已泄放 = False\n            if self._正池 >= self._正池泄放阈值:\n                泄前水位 = int(self._正池 // 20000)\n                self._正池 *= 0.5\n                已泄放 = True\n                print(f"第{self.时刻}拍 泄放，水位从{泄前水位}档落下")\n            # 水位档检查\n            水位档 = int(self._正池 // 20000)\n            if 水位档 > self._上次水位档:\n                print(f"第{self.时刻}拍 水位到{水位档}档")\n                self._上次水位档 = 水位档\n'

for name, old in [("init", _old_init_end), ("通道5", _old_通道5), ("隐藏", _old_隐藏), ("推进头", _old_推进头), ("差存储", _old_差存储)]:
    if old.strip() not in 原源码:
        print(f"FAIL: {name}")
        sys.exit(1)

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

回接集 = set(b._回接节点)
非回接 = [i for i in range(96) if i not in 回接集]

泄放拍 = []
for t in range(1, 20001):
    b.推进()
    if t % 2000 == 0:
        回接激活 = sum(abs(b.激活[j]) for j in 回接集) / len(回接集)
        非回接激活 = sum(abs(b.激活[i]) for i in 非回接) / len(非回接)
        有泄放 = "是" if t in 泄放拍 else "否"
        print(f"第{t}拍正池 = {b._正池:.6f}")
        print(f"第{t}拍正池有没有泄放 = {有泄放}")

with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py", "w", encoding="utf-8") as f:
    f.write(原源码)

print()
print("源码已还原")
