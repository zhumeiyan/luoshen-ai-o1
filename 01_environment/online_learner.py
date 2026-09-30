"""
online_learner.py — 运行时增量学习
==================================
洛神遇到新文本 → 自动学习 → 固化记忆

闭环流程：
1. 文本输入 → TextBridge分词
2. 未知词 → ExpandableConceptSpace自动注册
3. 词间共现 → HierarchicalRNetwork建立关系
4. 经历积累 → LongTermMemory固化到hard
"""
import numpy as np
from expandable_concepts import ExpandableConceptSpace
from text_bridge import TextBridge
from long_term_memory import LongTermMemory
from hierarchical_r_network import HierarchicalRNetwork

class OnlineLearner:
    """运行时增量学习器：读文本，学新概念，建关系，固记忆"""
    
    def __init__(self):
        self.cs = ExpandableConceptSpace()
        self.bridge = TextBridge(self.cs)
        self.memory = LongTermMemory(dim=1024, n_concepts=1024)
        self.rnet = HierarchicalRNetwork()
        self.experience_count = 0
    
    def read_text(self, text, story_label=None):
        """
        读一段文本，自动学习：
        1. 分词
        2. 注册新概念
        3. 建立共现关系
        4. 作为经历存入记忆
        """
        dE, tokens = self.bridge.text_to_potential(text)
        
        # 获取所有词的ID
        concept_ids = set()
        for token in tokens:
            cid = self.cs.get_id(token)
            concept_ids.add(cid)
        
        # 建立共现关系（窗口内共现的词建立R网边）
        window_size = 3
        for i in range(len(tokens)):
            for j in range(i+1, min(i+window_size, len(tokens))):
                if tokens[i] != tokens[j]:
                    self.rnet.add_concept_rel(tokens[i], tokens[j], "共现", w=0.5)
        
        # 存入记忆（经历）
        self.memory.experience(concept_ids, label=story_label or f"exp_{self.experience_count}")
        self.experience_count += 1
        
        return {
            "tokens": tokens,
            "concept_ids": concept_ids,
            "new_concepts": len([t for t in tokens if t not in self.cs.concepts]),
        }
    
    def sleep_and_consolidate(self):
        """睡眠：回放 + 固化"""
        self.memory.sleep(n_replays=min(10, self.experience_count))
    
    def recall_from_text(self, cue_text):
        """给定文本线索，回忆相关概念"""
        _, tokens = self.bridge.text_to_potential(cue_text)
        cue_ids = {self.cs.get_id(t) for t in tokens}
        return self.memory.recall(cue_ids)
    
    def stats(self):
        return {
            "concepts": self.cs.stats()["total_concepts"],
            "experiences": self.experience_count,
            "memory": self.memory.memory_stats(),
        }


if __name__ == "__main__":
    print("=" * 60)
    print("运行时增量学习测试")
    print("=" * 60)
    
    learner = OnlineLearner()
    
    # T1: 读新故事
    print("\n=== T1 读新故事 ===")
    result1 = learner.read_text("乌鸦口渴了看见瓶子里有水", story_label="乌鸦喝水")
    print(f"  输入: '乌鸦口渴了看见瓶子里有水'")
    print(f"  分词: {result1['tokens']}")
    print(f"  概念数: {learner.stats()['concepts']}")
    
    # T2: 再读一段，有重叠概念
    print("\n=== T2 再读新故事（有重叠） ===")
    result2 = learner.read_text("猴子也口渴了想要喝水", story_label="猴子喝水")
    print(f"  输入: '猴子也口渴了想要喝水'")
    print(f"  分词: {result2['tokens']}")
    print(f"  概念数: {learner.stats()['concepts']}")
    
    # T3: 睡眠固化
    print("\n=== T3 睡眠固化 ===")
    before = learner.stats()["memory"]
    print(f"  固化前: {before}")
    learner.sleep_and_consolidate()
    after = learner.stats()["memory"]
    print(f"  固化后: {after}")
    
    # T4: 线索回忆
    print("\n=== T4 线索回忆 ===")
    recalled = learner.recall_from_text("乌鸦")
    print(f"  线索: '乌鸦'")
    print(f"  想起的概念ID: {recalled}")
    
    # 解释
    id2name = {v["id"]: k for k, v in learner.cs.concepts.items()}
    names = [id2name.get(i, f"ID{i}") for i in recalled if i in id2name]
    print(f"  想起的概念名: {names}")
    
    print("\n" + "=" * 60)
    print("运行时增量学习测试完成 ✓")
    print("=" * 60)
