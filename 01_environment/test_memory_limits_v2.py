"""
test_memory_limits_v2.py — 修正版人类记忆压力测试
===============================================
修正问题：
1. 遗忘率太小（0.0001→0.01），加阈值截断（G<0.05归零）
2. 引入工作记忆限制（同时激活≤7±2概念）
3. 新旧知识重叠（真实干扰）
"""
import numpy as np
import random
from memory_system import HebbianMemory

class RealisticMemory(HebbianMemory):
    """更真实的记忆：有有效遗忘+工作记忆限制"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FORGETTING_RATE = 0.01  # 提高100倍
        self.FORGETTING_THRESHOLD = 0.05  # 低于此值直接归零
        self.WORKING_MEMORY_CAP = 7  # 工作记忆上限
    
    def experience(self, active_concepts, label=None):
        # 工作记忆限制：如果一次激活太多，只保留最强的7个
        if len(active_concepts) > self.WORKING_MEMORY_CAP:
            active_concepts = set(random.sample(list(active_concepts), self.WORKING_MEMORY_CAP))
        super().experience(active_concepts, label)
        
        # 有效遗忘：低于阈值直接归零
        self.G[np.abs(self.G) < self.FORGETTING_THRESHOLD] = 0

def test_capacity_v2(n_concepts=66, n_experiences=100, cluster_size=3):
    mem = RealisticMemory(dim=1024, n_concepts=n_concepts)
    experiences = []
    for _ in range(n_experiences):
        exp = frozenset(random.sample(range(n_concepts), cluster_size))
        experiences.append(exp)
        mem.experience(exp)
    
    correct = 0
    false_positive = 0
    for exp in experiences:
        cue = {list(exp)[0]}
        target = exp - cue
        recalled = mem.recall(cue)
        if target <= recalled:
            correct += 1
        false_positive += len(recalled - target - cue)
    
    return {
        "experiences": n_experiences,
        "recall_rate": correct / len(experiences),
        "false_positive": false_positive / (len(experiences) * n_concepts),
        "G_nonzero": np.count_nonzero(mem.G > 0),
        "G_mean": np.mean(np.abs(mem.G[mem.G != 0])) if np.any(mem.G != 0) else 0,
    }

def test_forgetting_v2(n_experiences=50):
    mem = RealisticMemory(dim=1024, n_concepts=66)
    experiences = []
    for _ in range(n_experiences):
        exp = frozenset(random.sample(range(66), 3))
        experiences.append(exp)
        mem.experience(exp)
    
    intervals = [0, 10, 50, 100, 200, 500]
    results = []
    
    for interval in intervals:
        mem_copy = RealisticMemory(dim=1024, n_concepts=66)
        mem_copy.G = mem.G.copy()
        
        for _ in range(interval):
            mem_copy.G *= (1 - mem_copy.FORGETTING_RATE)
            mem_copy.G[np.abs(mem_copy.G) < mem_copy.FORGETTING_THRESHOLD] = 0
        
        correct = 0
        for exp in experiences:
            cue = {list(exp)[0]}
            target = exp - cue
            recalled = mem_copy.recall(cue)
            if target <= recalled:
                correct += 1
        
        results.append({
            "interval": interval,
            "recall_rate": correct / n_experiences,
            "G_nonzero": np.count_nonzero(mem_copy.G > 0),
        })
    
    return results

def test_interference_v2(n_old=30, n_new=30):
    """真实干扰：新旧知识共享一半概念"""
    mem = RealisticMemory(dim=1024, n_concepts=66)
    
    # 旧知识：概念0-20为主
    old_experiences = []
    for _ in range(n_old):
        exp = frozenset(random.sample(range(0, 25), 3))
        old_experiences.append(exp)
        mem.experience(exp)
    
    # 新知识：概念15-40（与旧知识重叠10个概念）
    new_experiences = []
    for _ in range(n_new):
        exp = frozenset(random.sample(range(15, 45), 3))
        new_experiences.append(exp)
        mem.experience(exp)
    
    old_correct = sum(1 for exp in old_experiences 
                      if (exp - {list(exp)[0]}) <= mem.recall({list(exp)[0]}))
    new_correct = sum(1 for exp in new_experiences 
                      if (exp - {list(exp)[0]}) <= mem.recall({list(exp)[0]}))
    
    return {"old_recall": old_correct / n_old, "new_recall": new_correct / n_new}

if __name__ == "__main__":
    print("=" * 60)
    print("洛神记忆系统 — 修正版人类记忆压力测试")
    print("=" * 60)
    
    print("\n=== T1 容量上限（真实遗忘+工作记忆限制） ===")
    for cap in [10, 50, 100, 200, 500, 1000]:
        s = test_capacity_v2(n_experiences=cap)
        print(f"  经历{cap:4d}: 回忆率={s['recall_rate']:.1%}, "
              f"假阳性={s['false_positive']:.3f}, "
              f"G连接={s['G_nonzero']:4d}, "
              f"G平均强度={s['G_mean']:.2f}")
    
    print("\n=== T2 遗忘曲线（有效遗忘） ===")
    for r in test_forgetting_v2():
        print(f"  遗忘迭代{r['interval']:4d}: 回忆率={r['recall_rate']:.1%}, "
              f"G连接={r['G_nonzero']}")
    
    print("\n=== T3 干扰测试（新旧知识重叠15个概念） ===")
    intr = test_interference_v2()
    print(f"  旧知识回忆率: {intr['old_recall']:.1%}")
    print(f"  新知识回忆率: {intr['new_recall']:.1%}")
    
    print("\n" + "=" * 60)
