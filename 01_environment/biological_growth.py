#!/usr/bin/env python3
"""
biological_growth.py — 生物学自然生长引擎
============================================

不用硬改数字，让限制自然消失。

核心生物学原理：
1. 神经可塑性（Neuroplasticity）→ 容量限制自然解除
2. 语言发展阶梯（Language Development）→ 句子长度自然增长  
3. 工作记忆+注意力（Working Memory）→ 上下文自然筛选
4. LTP/LTD（长时程增强/抑制）→ 学习率自然调节
5. 神经振荡收敛（Neural Oscillation）→ 传播自然停止

不是"改配置"，是"给机制"。
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
import random
import sys

sys.path.insert(0, '01_environment')


# ================================================================
# 1. 神经可塑性：概念容量不是固定的
# ================================================================

class NeuroplasticConceptMapping:
    """
    神经可塑性概念映射。
    
    生物学原理：
    - 人脑不是给每个概念固定分配神经元，是动态竞争
    - 常用概念占用更多神经元（用进废退）
    - 不常用概念被压缩（突触修剪）
    - 新概念可以复用被修剪的神经元空间
    
    不是"max_concepts = 512"，是：
    - 总神经元固定（像大脑皮层面积固定）
    - 但分配是动态的：常用→大集群，少用→小集群，遗忘→释放
    """
    
    def __init__(self, total_neurons: int = 2048, base_cluster: int = 4):
        self.total_neurons = total_neurons
        self.base_cluster = base_cluster
        
        # 概念 → 神经元集群（动态大小）
        self.concept_clusters: Dict[str, Dict] = {}
        
        # 使用频率统计（决定集群大小）
        self.usage_count: Dict[str, int] = defaultdict(int)
        self.last_used: Dict[str, int] = {}
        self.time_step = 0
        
        # 空闲神经元池
        self.free_neurons = set(range(total_neurons))
        
        # 突触修剪阈值：多久不用就开始修剪
        self.prune_threshold = 100  # 步
        
        print(f"[神经可塑性] 总神经元: {total_neurons}, 基础集群: {base_cluster}")
        print(f"  不是固定分配，是用进废退、动态竞争")
    
    def _get_cluster_size(self, concept: str) -> int:
        """根据使用频率决定集群大小。"""
        freq = self.usage_count[concept]
        # 生物学：常用概念占用更多资源（像专业运动员的运动皮层更大）
        if freq > 50:
            return self.base_cluster * 4  # 16个神经元
        elif freq > 20:
            return self.base_cluster * 2  # 8个
        elif freq > 5:
            return self.base_cluster       # 4个
        else:
            return max(2, self.base_cluster // 2)  # 2个（最小）
    
    def register_concept(self, concept: str) -> List[int]:
        """
        注册新概念，或扩展现有概念。
        
        生物学：像学习新技能时大脑皮层的重组。
        """
        self.time_step += 1
        self.usage_count[concept] += 1
        self.last_used[concept] = self.time_step
        
        # 已存在：检查是否需要扩容
        if concept in self.concept_clusters:
            current_size = len(self.concept_clusters[concept]['neurons'])
            target_size = self._get_cluster_size(concept)
            
            if target_size > current_size and len(self.free_neurons) >= (target_size - current_size):
                # 扩容：像经常练习导致运动皮层扩大
                new_neurons = random.sample(list(self.free_neurons), target_size - current_size)
                self.concept_clusters[concept]['neurons'].extend(new_neurons)
                for n in new_neurons:
                    self.free_neurons.remove(n)
                    self.concept_clusters[concept]['map'][n] = concept
            
            return self.concept_clusters[concept]['neurons']
        
        # 新概念：分配集群
        target_size = self._get_cluster_size(concept)
        
        # 如果没有足够空闲神经元，触发突触修剪
        if len(self.free_neurons) < target_size:
            self._synaptic_pruning()
        
        # 再次检查
        if len(self.free_neurons) < target_size:
            # 紧急：复用最小集群的部分神经元
            target_size = min(target_size, len(self.free_neurons))
            if target_size < 2:
                # 真的没空间了：复用已有概念的神经元（混合编码）
                return self._reuse_existing_neurons(concept)
        
        # 分配
        neurons = random.sample(list(self.free_neurons), target_size)
        for n in neurons:
            self.free_neurons.remove(n)
        
        self.concept_clusters[concept] = {
            'neurons': neurons,
            'map': {n: concept for n in neurons},
            'created_at': self.time_step,
        }
        
        return neurons
    
    def _synaptic_pruning(self):
        """
        突触修剪：释放长期不用的概念占用的神经元。
        
        生物学：儿童期突触密度最高，青春期后大量修剪，
        保留常用连接，释放资源给新学习。
        """
        prune_candidates = []
        for concept, info in self.concept_clusters.items():
            last = self.last_used.get(concept, 0)
            idle = self.time_step - last
            freq = self.usage_count[concept]
            
            # 长时间不用 + 使用频率低 → 修剪
            if idle > self.prune_threshold and freq < 10:
                prune_candidates.append((concept, idle, freq))
        
        # 按idle时间排序，优先修剪最久不用的
        prune_candidates.sort(key=lambda x: (-x[1], x[2]))
        
        # 修剪前20%的候选
        n_prune = max(1, len(prune_candidates) // 5)
        for concept, idle, freq in prune_candidates[:n_prune]:
            neurons = self.concept_clusters[concept]['neurons']
            # 保留最小集群（2个神经元），释放其余
            if len(neurons) > 2:
                release = neurons[2:]
                self.concept_clusters[concept]['neurons'] = neurons[:2]
                for n in release:
                    if n in self.concept_clusters[concept]['map']:
                        del self.concept_clusters[concept]['map'][n]
                    self.free_neurons.add(n)
                
                print(f"[修剪] '{concept}' 释放 {len(release)} 个神经元"
                      f"（idle={idle}步, 使用{freq}次）")
    
    def _reuse_existing_neurons(self, concept: str) -> List[int]:
        """
        紧急复用：像海马体的模式分离失败时的混合编码。
        
        生物学：当新记忆和旧记忆太像时，会共享部分神经元（干扰）。
        """
        # 找使用频率最低的概念，借用部分神经元
        min_concept = min(self.concept_clusters.keys(), 
                         key=lambda c: self.usage_count[c])
        borrowed = self.concept_clusters[min_concept]['neurons'][:2]
        
        print(f"[复用] '{concept}' 借用 '{min_concept}' 的神经元"
              f"（混合编码，可能产生联想干扰）")
        
        return list(borrowed)
    
    def get_neurons(self, concept: str) -> List[int]:
        """获取概念的神经元集群。"""
        if concept in self.concept_clusters:
            self.usage_count[concept] += 1
            self.last_used[concept] = self.time_step
            return self.concept_clusters[concept]['neurons']
        return []
    
    def get_stats(self) -> Dict:
        """获取神经可塑性统计。"""
        total_used = self.total_neurons - len(self.free_neurons)
        avg_cluster = np.mean([len(c['neurons']) for c in self.concept_clusters.values()]) if self.concept_clusters else 0
        
        return {
            'total_neurons': self.total_neurons,
            'free_neurons': len(self.free_neurons),
            'used_neurons': total_used,
            'n_concepts': len(self.concept_clusters),
            'avg_cluster_size': float(avg_cluster),
            'theoretical_max': '无上限（动态分配+修剪）',
        }


# ================================================================
# 2. 语言发展：句子长度自然增长
# ================================================================

class LanguageDevelopmentTracker:
    """
    语言发展追踪器。
    
    生物学原理：
    - 儿童语言发展：单词（12月）→ 电报句（18-24月）→ 简单句（2-3岁）→ 复杂句（4-5岁）
    - 不是"max_words = 6"，是词汇量→句法复杂度→句长自然涌现
    
    洛神的语言发展阶段：
    - 0-100词：单词/双词（"妈妈""水"）
    - 100-300词：电报句（"妈妈水""我要吃"）
    - 300-500词：简单句（"我要喝水""妈妈来了"）
    - 500+词：复杂句（"因为渴了，所以要喝水"）
    """
    
    def __init__(self):
        self.vocabulary_size = 0
        self.total_exposures = 0
        
        # 发展阶段（不是硬编码，是计算出来的）
        self.stages = {
            'holophrastic': {'vocab': 0, 'max_words': 2, 'name': '单词期'},
            'telegraphic': {'vocab': 100, 'max_words': 4, 'name': '电报句期'},
            'simple': {'vocab': 300, 'max_words': 7, 'name': '简单句期'},
            'complex': {'vocab': 500, 'max_words': 12, 'name': '复杂句期'},
            'compound': {'vocab': 800, 'max_words': 20, 'name': '复合句期'},
        }
    
    def update(self, vocab_size: int, exposures: int):
        """更新语言发展水平。"""
        self.vocabulary_size = vocab_size
        self.total_exposures = exposures
    
    def get_current_stage(self) -> Dict:
        """获取当前语言发展阶段。"""
        current_stage = 'holophrastic'
        for stage, info in sorted(self.stages.items(), key=lambda x: x[1]['vocab']):
            if self.vocabulary_size >= info['vocab']:
                current_stage = stage
        return self.stages[current_stage]
    
    def get_max_sentence_length(self) -> int:
        """
        获取当前最大句长。
        
        不是固定值，是随词汇量自然增长的。
        """
        stage = self.get_current_stage()
        base_max = stage['max_words']
        
        # 额外加成：听得多说得长（像语言暴露量影响发展）
        exposure_bonus = min(3, self.total_exposures // 1000)
        
        return base_max + exposure_bonus
    
    def can_use_complex_syntax(self) -> bool:
        """是否能使用复杂句法。"""
        return self.vocabulary_size >= 300
    
    def get_stats(self) -> Dict:
        stage = self.get_current_stage()
        return {
            'vocabulary_size': self.vocabulary_size,
            'exposures': self.total_exposures,
            'stage': stage['name'],
            'max_sentence_length': self.get_max_sentence_length(),
            'can_complex': self.can_use_complex_syntax(),
        }


# ================================================================
# 3. 工作记忆：上下文不是固定轮数
# ================================================================

class WorkingMemory:
    """
    工作记忆系统。
    
    生物学原理：
    - 人脑工作记忆容量：7±2个组块（Miller, 1956）
    - 但注意力决定哪些进入工作记忆
    - 情绪强烈的经历更容易记住
    - 不是"history = 8轮"，是注意力筛选
    """
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity  # 7±2个组块
        self.chunks: deque = deque(maxlen=capacity)
        
        # 注意力权重：不同信息的重要性
        self.attention_weights: Dict[str, float] = defaultdict(float)
        
        # 情绪标记：情绪强烈的信息更容易保留
        self.emotion_tags: Dict[str, float] = {}
    
    def add(self, content: str, emotion_intensity: float = 0.0, 
            relevance: float = 0.5):
        """
        添加信息到工作记忆。
        
        如果满了，根据注意力和情绪决定替换哪个。
        """
        # 计算这条信息的优先级
        priority = relevance + emotion_intensity * 0.5
        
        chunk = {
            'content': content,
            'emotion': emotion_intensity,
            'relevance': relevance,
            'priority': priority,
            'timestamp': len(self.chunks),
        }
        
        if len(self.chunks) >= self.capacity:
            # 工作记忆满了：替换优先级最低的
            min_idx = min(range(len(self.chunks)), 
                         key=lambda i: self.chunks[i]['priority'])
            removed = self.chunks[min_idx]
            self.chunks.remove(removed)
            # 被替换的信息转入长时记忆（这里简化处理）
        
        self.chunks.append(chunk)
    
    def get_context(self, min_priority: float = 0.0) -> List[str]:
        """获取当前工作记忆中的内容（按优先级排序）。"""
        sorted_chunks = sorted(self.chunks, key=lambda c: -c['priority'])
        return [c['content'] for c in sorted_chunks if c['priority'] >= min_priority]
    
    def get_stats(self) -> Dict:
        return {
            'capacity': self.capacity,
            'current_load': len(self.chunks),
            'contents': [c['content'] for c in self.chunks],
        }


# ================================================================
# 4. LTP/LTD：学习率不是固定的
# ================================================================

class AdaptiveLearningRate:
    """
    自适应学习率。
    
    生物学原理：
    - LTP（长时程增强）：情绪强烈、注意力集中的经历，学习率极高
    - LTD（长时程抑制）：重复但不重要的信息，学习率降低
    - 不是固定0.01，是根据情境动态调节的
    """
    
    def __init__(self, base_rate: float = 0.01):
        self.base_rate = base_rate
        
        # 每个概念的学习历史
        self.concept_history: Dict[str, List[float]] = defaultdict(list)
    
    def get_rate(self, concept: str, emotion_intensity: float = 0.0,
                 novelty: float = 0.5, attention: float = 0.5) -> float:
        """
        计算当前情境下的学习率。
        
        生物学：
        - 情绪强烈 → LTP增强（像恐惧学习一次就记住）
        - 全新概念 → 高学习率（像婴儿学第一个词）
        - 注意力集中 → 高学习率
        - 重复概念 → 学习率衰减（像熟悉的事物不再引起注意）
        """
        history = self.concept_history[concept]
        
        # 基础调节
        rate = self.base_rate
        
        # 情绪增强（LTP）
        if emotion_intensity > 0.7:
            rate *= 5.0  # 强烈情绪：5倍学习率
        elif emotion_intensity > 0.3:
            rate *= 2.0
        
        # 新奇性
        if novelty > 0.8:
            rate *= 3.0  # 全新概念
        elif novelty < 0.2:
            rate *= 0.3  # 很熟悉，学习率降低
        
        # 注意力
        rate *= (0.5 + attention)
        
        # 重复衰减（熟悉度）
        familiarity = len(history)
        if familiarity > 50:
            rate *= 0.1  # 非常熟悉
        elif familiarity > 20:
            rate *= 0.3
        elif familiarity > 5:
            rate *= 0.7
        
        # 记录
        history.append(rate)
        
        return min(rate, 1.0)  # 上限1.0
    
    def get_stats(self) -> Dict:
        return {
            'base_rate': self.base_rate,
            'n_tracked_concepts': len(self.concept_history),
        }


# ================================================================
# 5. 神经振荡收敛：传播自然停止
# ================================================================

class OscillatoryConvergence:
    """
    神经振荡收敛检测。
    
    生物学原理：
    - 大脑不是固定步数计算，是振荡直到收敛
    - α波（8-12Hz）、θ波（4-8Hz）参与记忆检索
    - 当电位分布稳定时（能量最小），传播停止
    
    不是"steps = 20"，是"电位稳定就停"。
    """
    
    def __init__(self, convergence_threshold: float = 0.001,
                 max_steps: int = 100):
        self.convergence_threshold = convergence_threshold
        self.max_steps = max_steps  # 安全上限，不是固定步数
    
    def propagate_until_convergence(self, bio_env, mapping, cue_concepts,
                                   inject_strength: float = 8.0) -> Tuple[Dict, int]:
        """
        电位传播直到收敛。
        
        返回：回忆结果，实际步数
        """
        # 初始化
        dE = np.zeros(bio_env.dim)
        active_neurons = set()
        
        for concept in cue_concepts:
            neurons = mapping.get_neurons(concept)
            for n in neurons:
                dE[n] = inject_strength
                active_neurons.add(n)
        
        propagation_path = []
        prev_V = bio_env.V.copy()
        
        for step in range(self.max_steps):
            # 电位更新
            voltage_diff = dE - bio_env.V
            I = np.clip(bio_env.G @ voltage_diff, -10, 10)
            dV = I * bio_env.scale
            bio_env.V += dV
            bio_env.V *= 0.95
            
            # 检测收敛：电位变化小于阈值
            delta = np.linalg.norm(bio_env.V - prev_V)
            
            # 检测激活
            new_active = mapping.get_active_concepts(bio_env.V, threshold=-55.0)
            
            for concept, avg_V in new_active.items():
                if concept not in cue_concepts:
                    propagation_path.append((step, concept, avg_V))
            
            # 新激活成为信号源
            for concept, avg_V in new_active.items():
                neurons = mapping.get_neurons(concept)
                for n in neurons:
                    if n not in active_neurons:
                        dE[n] = avg_V * 0.3
                        active_neurons.add(n)
            
            # 收敛检测
            if delta < self.convergence_threshold and step > 5:
                print(f"[收敛] 电位传播在{step}步后自然收敛（Δ={delta:.6f}）")
                break
            
            prev_V = bio_env.V.copy()
        
        return {
            'activated_concepts': new_active,
            'propagation_path': propagation_path,
            'steps': step + 1,
        }, step + 1


# ================================================================
# 6. 整合测试
# ================================================================

def test_biological_growth():
    """测试生物学自然生长。"""
    print("=" * 60)
    print("生物学自然生长引擎测试")
    print("=" * 60)
    
    # 1. 神经可塑性
    print("\n【1. 神经可塑性】")
    mapping = NeuroplasticConceptMapping(total_neurons=256, base_cluster=4)
    
    # 注册一堆概念
    concepts = ['太阳', '月亮', '星星', '云', '雨', '雪', '风', '山', '水', '花',
                '树', '草', '火', '鸟', '鱼', '狗', '猫', '妈妈', '爸爸', '我']
    
    for c in concepts:
        mapping.register_concept(c)
    
    # 高频使用某些概念
    for _ in range(60):
        mapping.register_concept('太阳')  # 高频
    for _ in range(30):
        mapping.register_concept('妈妈')  # 中频
    
    print(f"\n统计:")
    stats = mapping.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # 2. 语言发展
    print("\n【2. 语言发展】")
    lang_dev = LanguageDevelopmentTracker()
    
    test_vocab_levels = [50, 150, 350, 600, 1000]
    for vocab in test_vocab_levels:
        lang_dev.update(vocab, exposures=vocab * 3)
        stage = lang_dev.get_current_stage()
        print(f"  词汇量{vocab:4d} → {stage['name']}, 最大句长{lang_dev.get_max_sentence_length()}词")
    
    # 3. 工作记忆
    print("\n【3. 工作记忆】")
    wm = WorkingMemory(capacity=7)
    
    # 添加信息
    wm.add("太阳出来了", emotion_intensity=0.3, relevance=0.5)
    wm.add("妈妈说吃饭了", emotion_intensity=0.6, relevance=0.8)  # 高情绪+高相关
    wm.add("1+1=2", emotion_intensity=0.0, relevance=0.1)  # 低优先级
    wm.add("我害怕打雷", emotion_intensity=0.9, relevance=0.7)  # 强烈情绪
    
    context = wm.get_context()
    print(f"  工作记忆内容（7±2容量）:")
    for c in context:
        print(f"    - {c}")
    
    # 4. 自适应学习率
    print("\n【4. 自适应学习率（LTP/LTD）】")
    alr = AdaptiveLearningRate(base_rate=0.01)
    
    # 新词 + 强烈情绪
    rate1 = alr.get_rate("黑洞", emotion_intensity=0.9, novelty=0.9, attention=0.9)
    print(f"  新词'黑洞'（强情绪+新奇+注意）: α={rate1:.3f}")
    
    # 已熟悉词 + 平静
    for _ in range(60):
        alr.get_rate("太阳", emotion_intensity=0.1, novelty=0.1, attention=0.3)
    rate2 = alr.get_rate("太阳", emotion_intensity=0.1, novelty=0.1, attention=0.3)
    print(f"  熟悉词'太阳'（平静+熟悉）: α={rate2:.3f}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_biological_growth()
