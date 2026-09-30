"""
long_term_memory.py — 洛神长期记忆系统
======================================
从可遗忘的 G 矩阵（海马体）到不可逆的 G_hard（皮层）。

人类长期记忆的生物基础：
- 蛋白质合成（CREB基因表达）
- 新突触结构长出
- 髓鞘包裹加速传导

洛神对应实现：
- G_fast：分钟级，高频衰减（工作记忆）
- G_slow：小时级，低频衰减（短期记忆）
- G_hard：永久，只增不减（长期记忆，需固化）
"""
import numpy as np
import random
import time

class LongTermMemory:
    """三层记忆 + 固化机制 + 睡眠回放"""
    
    def __init__(self, dim=1024, n_concepts=66):
        self.dim = dim
        self.n_concepts = n_concepts
        
        self.G_fast = np.zeros((dim, dim))   # 工作记忆
        self.G_slow = np.zeros((dim, dim))   # 短期记忆
        self.G_hard = np.zeros((dim, dim))   # 长期记忆
        
        self.DECAY_FAST = 0.1
        self.DECAY_SLOW = 0.01
        self.DECAY_HARD = 0.0
        
        self.LEARN_RATE = 0.1
        self.CONSOLIDATE_THRESHOLD = 0.5
        self.CONSOLIDATE_MIN_ACTIVATIONS = 3
        
        self.activation_count = np.zeros((dim, dim), dtype=np.int32)
        self.snapshots = []
    
    def experience(self, active_concepts, label=None):
        """经历：激活概念 → 强化G_fast"""
        active_list = list(active_concepts)
        for i in active_list:
            for j in active_list:
                if i != j:
                    idx_i = i % self.dim
                    idx_j = j % self.dim
                    self.G_fast[idx_i, idx_j] += self.LEARN_RATE
                    self.G_fast[idx_i, idx_j] = np.clip(self.G_fast[idx_i, idx_j], 0, 1.0)
                    self.activation_count[idx_i, idx_j] += 1
        
        if label:
            self.snapshots.append({"time": time.time(), "label": label, "concepts": sorted(active_concepts)})
    
    def consolidate(self, mode="threshold"):
        """固化：G_fast → G_slow → G_hard"""
        if mode == "threshold":
            strong = np.where(self.G_fast > self.CONSOLIDATE_THRESHOLD)
            for i, j in zip(strong[0], strong[1]):
                if self.activation_count[i, j] >= self.CONSOLIDATE_MIN_ACTIVATIONS:
                    self.G_slow[i, j] += self.G_fast[i, j] * 0.5
                    self.G_slow[i, j] = np.clip(self.G_slow[i, j], 0, 1.0)
                    
                    if self.activation_count[i, j] >= self.CONSOLIDATE_MIN_ACTIVATIONS * 3:
                        self.G_hard[i, j] += self.G_slow[i, j] * 0.3
                        self.G_hard[i, j] = np.clip(self.G_hard[i, j], 0, 1.0)
                    
                    self.G_fast[i, j] = 0
        
        elif mode == "sleep":
            for layer, target in [(self.G_fast, self.G_slow), (self.G_slow, self.G_hard)]:
                strong = np.where(layer > 0.3)
                for i, j in zip(strong[0], strong[1]):
                    if self.activation_count[i, j] >= self.CONSOLIDATE_MIN_ACTIVATIONS:
                        target[i, j] += layer[i, j] * 0.5
                        target[i, j] = np.clip(target[i, j], 0, 1.0)
                        layer[i, j] *= 0.5
    
    def sleep(self, n_replays=10):
        """睡眠：回放 + 固化 + 清理"""
        if self.snapshots and n_replays > 0:
            for _ in range(min(n_replays, len(self.snapshots))):
                snap = random.choice(self.snapshots)
                self.experience(set(snap["concepts"]), label="replay_" + snap.get("label", ""))
        
        self.consolidate(mode="sleep")
        self.G_fast *= (1 - self.DECAY_FAST)
        self.G_slow *= (1 - self.DECAY_SLOW)
    
    def recall(self, cue_concepts):
        """主动回忆：三层G合并"""
        recalled = set(cue_concepts)
        for c in cue_concepts:
            idx = c % self.dim
            total_G = self.G_fast[idx, :self.n_concepts] + \
                      self.G_slow[idx, :self.n_concepts] + \
                      self.G_hard[idx, :self.n_concepts] * 2
            threshold = np.max(total_G) * 0.3 if np.max(total_G) > 0 else 0.01
            strong = np.where(total_G > threshold)[0]
            recalled.update(strong)
        return recalled
    
    def memory_stats(self):
        return {
            "G_fast_nonzero": np.count_nonzero(self.G_fast > 0),
            "G_slow_nonzero": np.count_nonzero(self.G_slow > 0),
            "G_hard_nonzero": np.count_nonzero(self.G_hard > 0),
            "snapshots": len(self.snapshots),
        }


if __name__ == "__main__":
    print("=" * 60)
    print("洛神长期记忆系统测试")
    print("=" * 60)
    
    mem = LongTermMemory(dim=1024, n_concepts=66)
    
    print("\n=== T1 学习与固化 ===")
    for _ in range(5):
        mem.experience({17, 29, 58}, label="乌鸦喝水")
    
    print(f"  学习5次后: fast={mem.memory_stats()['G_fast_nonzero']}, "
          f"slow={mem.memory_stats()['G_slow_nonzero']}, "
          f"hard={mem.memory_stats()['G_hard_nonzero']}")
    
    recalled_before = mem.recall({17})
    print(f"  固化前回忆: {recalled_before}")
    
    mem.consolidate()
    print(f"  固化后: fast={mem.memory_stats()['G_fast_nonzero']}, "
          f"slow={mem.memory_stats()['G_slow_nonzero']}, "
          f"hard={mem.memory_stats()['G_hard_nonzero']}")
    
    recalled_after = mem.recall({17})
    print(f"  固化后回忆: {recalled_after}")
    
    print("\n=== T2 睡眠测试 ===")
    mem2 = LongTermMemory(dim=1024, n_concepts=66)
    for _ in range(10):
        mem2.experience({17, 29, 58}, label="乌鸦喝水")
    
    print(f"  睡眠前: {mem2.memory_stats()}")
    mem2.sleep(n_replays=5)
    print(f"  睡眠后: {mem2.memory_stats()}")
    
    print("\n=== T3 遗忘抵抗 ===")
    mem3 = LongTermMemory(dim=1024, n_concepts=66)
    for _ in range(10):
        mem3.experience({17, 29, 58}, label="乌鸦喝水")
    mem3.sleep(n_replays=5)
    
    for _ in range(100):
        mem3.G_fast *= (1 - mem3.DECAY_FAST)
        mem3.G_slow *= (1 - mem3.DECAY_SLOW)
    
    recalled = mem3.recall({17})
    print(f"  100次遗忘后回忆: {recalled}")
    print(f"  hard连接数: {mem3.memory_stats()['G_hard_nonzero']}")
    assert 29 in recalled and 58 in recalled, "长期记忆应抵抗遗忘"
    print("  [PASS] 长期记忆抵抗遗忘成功")
    
    print("\n" + "=" * 60)
    print("长期记忆系统测试全部通过 ✓")
    print("=" * 60)
