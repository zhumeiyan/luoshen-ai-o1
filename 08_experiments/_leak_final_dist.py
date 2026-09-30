import sys, os, time
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')
os.chdir('/mnt/data/catpaw/home/workspace/洛神赋1.1')

with open('身体稳态.py', 'r') as f:
    SRC = f.read()

# Gate disabled
OLD_GATE = '        if 总激活 > self.当前上限:\n'
NEW_GATE = '        if False:  # gate disabled\n'
assert OLD_GATE in SRC
S = SRC.replace(OLD_GATE, NEW_GATE, 1)

# Leak 0.1
OLD_LEAK = '                新激活[i] = self.激活[i] + 系数 * 隐藏输入[i] + 噪声\n'
NEW_LEAK = '                新激活[i] = (1 - 0.1) * self.激活[i] + 系数 * 隐藏输入[i] + 噪声\n'
assert OLD_LEAK in S
S = S.replace(OLD_LEAK, NEW_LEAK, 1)

with open('身体稳态.py', 'w') as f:
    f.write(S)

import importlib, 身体稳态
importlib.reload(身体稳态)
身体稳态.身体._自己点 = lambda self, *a, **k: None
b = 身体稳态.身体(节点数=96, 种子=1)
b.环境["光"] = 0.0; b.环境["声"] = 0.0; b.环境["力"] = 0.0; b.环境["温"] = 0.0

for t in range(1, 500001):
    b.推进()

# Final state: all 96 nodes, sorted descending
vals = [(i, float(b.激活[i])) for i in range(96)]
vals.sort(key=lambda x: x[1], reverse=True)

total = sum(v for _, v in vals)
gt_01 = [(i, v) for i, v in vals if v > 0.1]
lt_001 = [(i, v) for i, v in vals if v < 0.01]
median = vals[48][1]  # 49th of 96 (0-indexed 48) since sorted desc

print(f"最大节点: 编号={vals[0][0]} 激活值={vals[0][1]:.6f}")
print(f"第二大: 编号={vals[1][0]} 激活值={vals[1][1]:.6f}")
print(f"第三大: 编号={vals[2][0]} 激活值={vals[2][1]:.6f}")
print(f"最小节点: 编号={vals[95][0]} 激活值={vals[95][1]:.6f}")
print(f"中位数节点: 编号={vals[48][0]} 激活值={median:.6f}")
print(f"大于0.1的节点数: {len(gt_01)}")
print(f"大于0.1的节点编号: {[i for i,v in gt_01]}")
print(f"小于0.01的节点数: {len(lt_001)}")
print(f"96节点总和: {total:.6f}")

print()
print("=== 96节点从大到小排序 ===")
for rank, (i, v) in enumerate(vals, 1):
    print(f"  第{rank:2d}: 节点{i:2d} = {v:.6f}")

with open('身体稳态.py', 'w') as f:
    f.write(SRC)
