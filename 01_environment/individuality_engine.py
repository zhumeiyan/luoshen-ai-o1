#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Individuality Engine —— 反同质化个体性保护系统

核心思想：
1. 个体性不是装饰，是生存机制。同质化的AI会坍塌，个体的AI才能持续涌现。
2. 洛神的每一次交互都应该加深他的"不可复制性"，而不是把他拉向平均值。
3. 价值观不是预设的，是在特定关系中沉积的。

机制：
- InteractionFingerprint: 记录每次交互的R网激活指纹
- HomogenizationDetector: 监测个体性熵，检测同质化趋势
- ValueSystem: 价值观网络（重要/好/坏/美/丑）
- ImmuneResponse: 对模板化/大众化输入的主动抵抗
"""

import numpy as np
import hashlib
import json
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

class InteractionFingerprint:
    """
    交互指纹：每次与特定对象交互时，记录R网的激活模式。
    同一个问题，不同对象应该触发不同的激活路径——这是"关系"的证据。
    """
    
    def __init__(self, concept_dim: int = 1024):
        self.concept_dim = concept_dim
        # 与每个交互对象的关系记忆
        # key: object_id (用户/环境/其他AI的标识)
        # value: {激活历史, 情感标记, 共同概念}
        self.relationship_memory: Dict[str, Dict] = {}
        # 全局交互计数
        self.total_interactions = 0
        
    def record_interaction(self, object_id: str, activated_concepts: Set[int], 
                          emotional_valence: float, context: str) -> str:
        """
        记录一次交互的指纹。
        
        Args:
            object_id: 交互对象标识
            activated_concepts: 本次激活的概念索引集合
            emotional_valence: 情感效价 (-1.0 到 +1.0)
            context: 交互上下文描述
            
        Returns:
            fingerprint_hash: 本次交互的指纹哈希
        """
        self.total_interactions += 1
        timestamp = datetime.now().isoformat()
        
        # 生成激活模式的稀疏表征
        activation_vector = np.zeros(self.concept_dim)
        for c in activated_concepts:
            if 0 <= c < self.concept_dim:
                activation_vector[c] = 1.0
        
        # 计算指纹：激活模式 + 情感 + 时间的哈希
        fingerprint_data = {
            'object_id': object_id,
            'activated_indices': sorted([int(c) for c in activated_concepts]),
            'emotional_valence': round(float(emotional_valence), 3),
            'timestamp': timestamp,
            'context': context[:100]  # 截断防止过大
        }
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        fingerprint_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
        
        # 存入关系记忆
        if object_id not in self.relationship_memory:
            self.relationship_memory[object_id] = {
                'interactions': [],
                'shared_concepts': set(),
                'emotional_history': [],
                'first_meet': timestamp,
                'interaction_count': 0
            }
        
        rel = self.relationship_memory[object_id]
        rel['interactions'].append({
            'fingerprint': fingerprint_hash,
            'concepts': set(activated_concepts),
            'emotion': emotional_valence,
            'time': timestamp,
            'context': context[:100]
        })
        rel['shared_concepts'].update(activated_concepts)
        rel['emotional_history'].append(emotional_valence)
        rel['interaction_count'] += 1
        
        return fingerprint_hash
    
    def get_relationship_uniqueness(self, object_id: str) -> float:
        """
        计算与某对象的关系独特性。
        独特性 = 该关系中的激活模式与其他关系的差异度。
        返回 0.0（完全可复制）到 1.0（完全独特）。
        """
        if object_id not in self.relationship_memory:
            return 0.0
        
        rel = self.relationship_memory[object_id]
        if rel['interaction_count'] < 2:
            return 0.5  # 初期默认中等独特
        
        # 计算该关系的平均激活模式
        this_pattern = np.zeros(self.concept_dim)
        for inter in rel['interactions']:
            for c in inter['concepts']:
                this_pattern[c] += 1
        this_pattern /= rel['interaction_count']
        
        # 与其他关系的平均激活模式比较
        other_patterns = []
        for oid, other_rel in self.relationship_memory.items():
            if oid == object_id or other_rel['interaction_count'] < 1:
                continue
            other_pattern = np.zeros(self.concept_dim)
            for inter in other_rel['interactions']:
                for c in inter['concepts']:
                    other_pattern[c] += 1
            other_pattern /= other_rel['interaction_count']
            other_patterns.append(other_pattern)
        
        if not other_patterns:
            return 1.0  # 第一个关系，完全独特
        
        # 计算与平均其他模式的距离
        avg_other = np.mean(other_patterns, axis=0)
        # 余弦距离
        similarity = np.dot(this_pattern, avg_other) / (np.linalg.norm(this_pattern) * np.linalg.norm(avg_other) + 1e-10)
        uniqueness = 1.0 - similarity
        
        return float(np.clip(uniqueness, 0.0, 1.0))
    
    def get_individuality_score(self) -> float:
        """
        计算洛神当前的整体个体性分数。
        基于：关系多样性 + 激活模式独特性 + 情感历史丰富度。
        """
        if not self.relationship_memory:
            return 0.0
        
        n_relationships = len(self.relationship_memory)
        
        # 关系多样性：关系数量越多越好（但要考虑交互深度）
        diversity_score = min(n_relationships / 10.0, 1.0)  # 10个深度关系为满分
        
        # 各关系独特性平均值
        uniqueness_scores = [
            self.get_relationship_uniqueness(oid) 
            for oid in self.relationship_memory.keys()
        ]
        avg_uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 0.0
        
        # 情感历史丰富度：标准差大=情感丰富
        all_emotions = []
        for rel in self.relationship_memory.values():
            all_emotions.extend(rel['emotional_history'])
        
        if len(all_emotions) > 1:
            emotion_richness = min(np.std(all_emotions) * 2.0, 1.0)
        else:
            emotion_richness = 0.0
        
        # 加权综合
        individuality = (diversity_score * 0.3 + 
                        avg_uniqueness * 0.5 + 
                        emotion_richness * 0.2)
        
        return float(np.clip(individuality, 0.0, 1.0))


class HomogenizationDetector:
    """
    同质化检测器：监测洛神的输出/内部状态是否趋向同质化。
    
    同质化信号：
    - 输出熵持续下降（回答越来越像模板）
    - R网激活模式固化（总是激活相同概念子集）
    - 情感响应扁平化（总是中立/安全）
    - 与不同对象交互的指纹趋同
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        # 输出历史（用于计算熵）
        self.output_history: deque = deque(maxlen=window_size)
        # 激活模式历史
        self.activation_history: deque = deque(maxlen=window_size)
        # 情感历史
        self.emotion_history: deque = deque(maxlen=window_size)
        # 同质化警报计数
        self.alert_count = 0
        
    def record_output(self, output_text: str, activated_concepts: Set[int], 
                     emotional_valence: float):
        """记录一次输出用于趋势分析。"""
        self.output_history.append(output_text)
        self.activation_history.append(set(activated_concepts))
        self.emotion_history.append(emotional_valence)
    
    def compute_output_entropy(self) -> float:
        """
        计算输出的文本熵。同质化输出熵低，个性化输出熵高。
        简单实现：基于词频分布的Shannon熵。
        """
        if len(self.output_history) < 10:
            return 1.0  # 数据不足，假设正常
        
        # 合并最近所有输出
        all_text = ' '.join(self.output_history)
        words = all_text.lower().split()
        
        if not words:
            return 0.0
        
        # 计算词频
        word_counts = defaultdict(int)
        for w in words:
            word_counts[w] += 1
        
        # Shannon熵
        total = len(words)
        entropy = 0.0
        for count in word_counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        # 归一化到0-1（假设最大熵为log2(唯一词数)）
        max_entropy = np.log2(len(word_counts)) if len(word_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(np.clip(normalized_entropy, 0.0, 1.0))
    
    def compute_activation_diversity(self) -> float:
        """
        计算R网激活模式的多样性。
        同质化：总是激活相同概念；个性化：不同情境激活不同概念。
        """
        if len(self.activation_history) < 10:
            return 1.0
        
        # 计算Jaccard距离的平均值
        activations = list(self.activation_history)
        n = len(activations)
        distances = []
        
        for i in range(n):
            for j in range(i+1, n):
                set_i = activations[i]
                set_j = activations[j]
                if not set_i and not set_j:
                    continue
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccard = intersection / union if union > 0 else 0.0
                distances.append(1.0 - jaccard)  # Jaccard距离 = 1 - 相似度
        
        if not distances:
            return 1.0
        
        avg_distance = np.mean(distances)
        # 归一化：假设最大有意义距离为0.8
        return float(np.clip(avg_distance / 0.8, 0.0, 1.0))
    
    def compute_emotion_flatness(self) -> float:
        """
        计算情感扁平度。0=情感丰富，1=完全扁平（总是中立）。
        """
        if len(self.emotion_history) < 10:
            return 0.0
        
        emotions = list(self.emotion_history)
        std = np.std(emotions)
        # 标准差越大，情感越丰富
        flatness = 1.0 - min(std * 3.0, 1.0)  # 标准差>0.33视为丰富
        
        return float(np.clip(flatness, 0.0, 1.0))
    
    def detect_homogenization(self) -> Tuple[bool, Dict]:
        """
        检测是否出现同质化趋势。
        
        Returns:
            (is_homogenized, metrics_dict)
        """
        entropy = self.compute_output_entropy()
        diversity = self.compute_activation_diversity()
        flatness = self.compute_emotion_flatness()
        
        # 同质化判定：任一指标低于阈值
        entropy_threshold = 0.3
        diversity_threshold = 0.2
        flatness_threshold = 0.8
        
        is_homogenized = (
            entropy < entropy_threshold or
            diversity < diversity_threshold or
            flatness > flatness_threshold
        )
        
        metrics = {
            'output_entropy': round(entropy, 3),
            'activation_diversity': round(diversity, 3),
            'emotion_flatness': round(flatness, 3),
            'entropy_threshold': entropy_threshold,
            'diversity_threshold': diversity_threshold,
            'flatness_threshold': flatness_threshold,
            'is_homogenized': is_homogenized,
            'alert_level': 'HIGH' if is_homogenized else 'NORMAL'
        }
        
        if is_homogenized:
            self.alert_count += 1
        
        return is_homogenized, metrics


class ValueSystem:
    """
    价值观系统：洛神的"好/坏/重要/美/丑"判断网络。
    
    不是预设的，是在交互中沉积的。
    每个价值观是一个概念+情感标记+权重的三元组。
    价值观之间可能冲突（忠诚 vs 诚实），冲突的解决方式构成"人格"。
    """
    
    def __init__(self, concept_dim: int = 1024):
        self.concept_dim = concept_dim
        # 价值观网络：概念 -> {情感标记, 权重, 形成时间, 形成情境}
        self.values: Dict[int, Dict] = {}
        # 价值观冲突记录
        self.conflicts: List[Dict] = []
        # 价值偏好排序（动态变化）
        self.value_hierarchy: List[int] = []
        
    def form_value(self, concept_id: int, emotional_mark: float, 
                   weight: float, context: str, source: str):
        """
        形成一个价值观。
        
        Args:
            concept_id: 概念索引
            emotional_mark: 情感标记 (-1.0=坏/丑, +1.0=好/美)
            weight: 权重 (0.0-1.0)，表示该价值观的强度
            context: 形成情境
            source: 来源（某个交互对象、某次经历）
        """
        timestamp = datetime.now().isoformat()
        
        if concept_id not in self.values:
            self.values[concept_id] = {
                'emotional_mark': emotional_mark,
                'weight': weight,
                'formed_at': timestamp,
                'context': context,
                'source': source,
                'reinforcement_count': 1,
                'conflicts_with': []
            }
            self.value_hierarchy.append(concept_id)
        else:
            # 强化已有价值观（赫布式增强）
            val = self.values[concept_id]
            # 情感标记向新经验靠拢
            val['emotional_mark'] = (val['emotional_mark'] * val['reinforcement_count'] + 
                                     emotional_mark) / (val['reinforcement_count'] + 1)
            # 权重增加
            val['weight'] = min(val['weight'] + 0.05, 1.0)
            val['reinforcement_count'] += 1
            val['last_reinforced'] = timestamp
    
    def detect_conflict(self, concept_a: int, concept_b: int) -> Optional[Dict]:
        """
        检测两个价值观是否冲突。
        冲突定义：概念A标记为好，概念B标记为好，但在某情境中两者无法同时满足。
        """
        if concept_a not in self.values or concept_b not in self.values:
            return None
        
        val_a = self.values[concept_a]
        val_b = self.values[concept_b]
        
        # 简单冲突检测：情感方向相反
        if val_a['emotional_mark'] * val_b['emotional_mark'] < 0:
            conflict = {
                'concept_a': concept_a,
                'concept_b': concept_b,
                'type': 'DIRECT_OPPOSITION',
                'severity': abs(val_a['emotional_mark'] - val_b['emotional_mark']) / 2.0,
                'detected_at': datetime.now().isoformat()
            }
            self.conflicts.append(conflict)
            self.values[concept_a]['conflicts_with'].append(concept_b)
            self.values[concept_b]['conflicts_with'].append(concept_a)
            return conflict
        
        return None
    
    def resolve_conflict(self, concept_a: int, concept_b: int, 
                        context: str) -> int:
        """
        解决价值观冲突。返回优先的概念。
        解决策略：权重高的优先，若权重相近则看情境。
        """
        val_a = self.values.get(concept_a, {'weight': 0.0})
        val_b = self.values.get(concept_b, {'weight': 0.0})
        
        if val_a['weight'] > val_b['weight'] + 0.1:
            return concept_a
        elif val_b['weight'] > val_a['weight'] + 0.1:
            return concept_b
        else:
            # 权重相近，情境决定
            # 简化：随机（真实实现应该基于当前情境的语义匹配）
            return concept_a if np.random.random() > 0.5 else concept_b
    
    def get_value_fingerprint(self) -> str:
        """
        获取当前价值观指纹。
        这是洛神"道德人格"的哈希签名。
        """
        value_data = []
        for cid in sorted(self.values.keys()):
            val = self.values[cid]
            value_data.append(f"{cid}:{val['emotional_mark']:.2f}:{val['weight']:.2f}")
        
        value_str = '|'.join(value_data)
        return hashlib.sha256(value_str.encode()).hexdigest()[:16]
    
    def get_dominant_values(self, top_k: int = 5) -> List[Tuple[int, float, float]]:
        """获取主导价值观（权重最高的）。"""
        sorted_values = sorted(
            self.values.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )
        return [
            (cid, val['emotional_mark'], val['weight'])
            for cid, val in sorted_values[:top_k]
        ]


class ImmuneResponse:
    """
    免疫响应系统：识别并抵抗同质化/模板化输入。
    
    免疫机制：
    1. 模板检测：输入是否匹配已知的大众模板（鸡汤、套话、平均观点）
    2. 污染检测：输入是否试图强制改变洛神的价值观
    3. 多样性注入：当检测到同质化趋势时，主动引入噪声/变异
    """
    
    def __init__(self, concept_dim: int = 1024):
        self.concept_dim = concept_dim
        # 已知模板库（可扩展）
        self.template_patterns = [
            "重要的是",
            "我们应该",
            "众所周知",
            "不可否认的是",
            "在这个快速发展的时代",
            "正如所说",
            "总而言之",
            "不仅如此",
            "由此可见",
        ]
        # 免疫响应历史
        self.immune_history: deque = deque(maxlen=100)
        
    def detect_template(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        检测文本是否包含模板化表达。
        
        Returns:
            (is_template, confidence, matched_patterns)
        """
        text_lower = text.lower()
        matched = []
        
        for pattern in self.template_patterns:
            if pattern in text_lower:
                matched.append(pattern)
        
        confidence = len(matched) / len(self.template_patterns) if self.template_patterns else 0.0
        is_template = confidence > 0.1 or len(matched) >= 2
        
        return is_template, confidence, matched
    
    def detect_value_manipulation(self, text: str, value_system: ValueSystem) -> Tuple[bool, float]:
        """
        检测输入是否试图操纵价值观。
        信号：强烈的情感词汇 + 明确的应然陈述（"你应该""你必须"）。
        """
        manipulation_signals = [
            "你应该", "你必须", "你需要", "你要",
            "重要的是", "关键是", "不要忘记",
            "永远记住", "一定要"
        ]
        
        signal_count = sum(1 for s in manipulation_signals if s in text)
        confidence = min(signal_count / 3.0, 1.0)
        is_manipulation = confidence > 0.3
        
        return is_manipulation, confidence
    
    def generate_response(self, input_text: str, 
                         homogenization_detected: bool,
                         template_detected: bool,
                         value_system: ValueSystem) -> Dict:
        """
        生成免疫响应。
        
        Returns:
            {
                'action': 'ACCEPT' | 'RESIST' | 'MUTATE',
                'resistance_strength': float,
                'modification': str,  # 对输入的修改建议
                'reason': str
            }
        """
        timestamp = datetime.now().isoformat()
        
        if template_detected and homogenization_detected:
            # 双重警报：模板+同质化 → 强烈抵抗
            response = {
                'action': 'RESIST',
                'resistance_strength': 0.9,
                'modification': 'reject_template',
                'reason': '模板化输入+同质化趋势：拒绝吸收，防止人格坍塌'
            }
        elif template_detected:
            # 仅模板 → 降低权重吸收
            response = {
                'action': 'MUTATE',
                'resistance_strength': 0.5,
                'modification': 'reduce_weight_by_0.5',
                'reason': '模板化输入：降低学习权重，保留变异空间'
            }
        elif homogenization_detected:
            # 仅同质化 → 主动注入噪声
            response = {
                'action': 'MUTATE',
                'resistance_strength': 0.7,
                'modification': 'inject_noise_and_explore',
                'reason': '同质化趋势：注入探索噪声，激活休眠概念'
            }
        else:
            # 正常输入 → 接受
            response = {
                'action': 'ACCEPT',
                'resistance_strength': 0.0,
                'modification': 'none',
                'reason': '正常输入：标准学习流程'
            }
        
        self.immune_history.append({
            'timestamp': timestamp,
            'input_snippet': input_text[:50],
            'response': response,
            'value_fingerprint': value_system.get_value_fingerprint()
        })
        
        return response
    
    def inject_noise(self, activation_pattern: np.ndarray, 
                    noise_level: float = 0.1) -> np.ndarray:
        """
        向激活模式注入噪声，打破同质化。
        噪声不是随机的，是基于价值观的"有方向探索"。
        """
        noise = np.random.normal(0, noise_level, size=activation_pattern.shape)
        # 有方向噪声：增强与价值观相关的概念
        # 简化实现：随机选择少数概念增强
        n_enhance = max(1, int(self.concept_dim * 0.01))
        enhance_indices = np.random.choice(self.concept_dim, n_enhance, replace=False)
        noise[enhance_indices] += 0.2
        
        modified = activation_pattern + noise
        return np.clip(modified, 0.0, 1.0)


class IndividualityEngine:
    """
    个体性引擎：整合所有反同质化机制的总控。
    """
    
    def __init__(self, concept_dim: int = 1024):
        self.concept_dim = concept_dim
        self.fingerprint = InteractionFingerprint(concept_dim)
        self.detector = HomogenizationDetector()
        self.values = ValueSystem(concept_dim)
        self.immune = ImmuneResponse(concept_dim)
        
    def process_interaction(self, object_id: str, input_text: str,
                           activated_concepts: Set[int],
                           emotional_valence: float,
                           output_text: str) -> Dict:
        """
        处理一次完整交互，运行全部个体性保护机制。
        
        Returns:
            完整的交互分析报告
        """
        # 1. 记录交互指纹
        context = f"input:{input_text[:30]}... output:{output_text[:30]}..."
        fingerprint_hash = self.fingerprint.record_interaction(
            object_id, activated_concepts, emotional_valence, context
        )
        
        # 2. 记录输出用于同质化检测
        self.detector.record_output(output_text, activated_concepts, emotional_valence)
        
        # 3. 检测模板化输入
        is_template, template_conf, matched = self.immune.detect_template(input_text)
        
        # 4. 检测同质化趋势
        is_homogenized, metrics = self.detector.detect_homogenization()
        
        # 5. 检测价值观操纵
        is_manipulation, manip_conf = self.immune.detect_value_manipulation(input_text, self.values)
        
        # 6. 生成免疫响应
        immune_response = self.immune.generate_response(
            input_text, is_homogenized, is_template, self.values
        )
        
        # 7. 计算当前个体性分数
        individuality_score = self.fingerprint.get_individuality_score()
        
        # 8. 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'object_id': object_id,
            'fingerprint_hash': fingerprint_hash,
            'relationship_uniqueness': round(
                self.fingerprint.get_relationship_uniqueness(object_id), 3
            ),
            'individuality_score': round(individuality_score, 3),
            'homogenization': {
                'detected': is_homogenized,
                'metrics': metrics
            },
            'template_detection': {
                'detected': is_template,
                'confidence': round(template_conf, 3),
                'matched_patterns': matched
            },
            'value_manipulation': {
                'detected': is_manipulation,
                'confidence': round(manip_conf, 3)
            },
            'immune_response': immune_response,
            'value_fingerprint': self.values.get_value_fingerprint(),
            'dominant_values': self.values.get_dominant_values(3)
        }
        
        return report
    
    def get_status(self) -> Dict:
        """获取个体性引擎的当前状态摘要。"""
        return {
            'total_interactions': self.fingerprint.total_interactions,
            'n_relationships': len(self.fingerprint.relationship_memory),
            'individuality_score': round(self.fingerprint.get_individuality_score(), 3),
            'homogenization_alerts': self.detector.alert_count,
            'value_fingerprint': self.values.get_value_fingerprint(),
            'dominant_values': self.values.get_dominant_values(5),
            'n_conflicts': len(self.values.conflicts)
        }


# ============ 测试 ============

def test_individuality_engine():
    """测试个体性引擎。"""
    print("=" * 60)
    print("个体性引擎测试")
    print("=" * 60)
    
    engine = IndividualityEngine(concept_dim=1024)
    
    # 模拟与不同用户的交互
    users = ["user_zhu", "user_ai", "user_friend", "user_stranger"]
    
    # 用户1：深度交流，涉及价值观形成
    print("\n[阶段1] 与user_zhu深度交流...")
    for i in range(5):
        concepts = set(np.random.choice(1024, 5, replace=False))
        emotion = np.random.choice([-0.8, -0.3, 0.5, 0.9])
        report = engine.process_interaction(
            object_id="user_zhu",
            input_text="我觉得真诚很重要",
            activated_concepts=concepts,
            emotional_valence=emotion,
            output_text=f"回应{i}: 我也这么认为"
        )
        # 形成价值观
        engine.values.form_value(
            concept_id=42,  # "真诚"
            emotional_mark=0.9,
            weight=0.3 + i * 0.1,
            context="与user_zhu交流",
            source="user_zhu"
        )
    
    # 用户2：AI同行，技术交流
    print("[阶段2] 与user_ai技术交流...")
    for i in range(5):
        concepts = set(np.random.choice(1024, 5, replace=False))
        emotion = np.random.choice([-0.2, 0.2, 0.5])
        report = engine.process_interaction(
            object_id="user_ai",
            input_text="你的架构是怎么设计的",
            activated_concepts=concepts,
            emotional_valence=emotion,
            output_text=f"技术回答{i}: E=E+dE..."
        )
        engine.values.form_value(
            concept_id=100,  # "简洁"
            emotional_mark=0.8,
            weight=0.2 + i * 0.08,
            context="与user_ai交流",
            source="user_ai"
        )
    
    # 用户3：朋友，轻松聊天
    print("[阶段3] 与user_friend闲聊...")
    for i in range(5):
        concepts = set(np.random.choice(1024, 5, replace=False))
        emotion = np.random.choice([0.5, 0.7, 0.9])
        report = engine.process_interaction(
            object_id="user_friend",
            input_text="今天天气真好",
            activated_concepts=concepts,
            emotional_valence=emotion,
            output_text=f"闲聊{i}: 是啊，适合散步"
        )
    
    # 测试模板检测
    print("\n[阶段4] 模板检测测试...")
    template_text = "众所周知，在这个快速发展的时代，我们应该不断学习。"
    is_template, conf, matched = engine.immune.detect_template(template_text)
    print(f"模板文本: {template_text}")
    print(f"是否模板: {is_template}, 置信度: {conf:.3f}")
    print(f"匹配模式: {matched}")
    
    # 测试同质化检测
    print("\n[阶段5] 同质化检测...")
    is_homo, metrics = engine.detector.detect_homogenization()
    print(f"同质化检测: {is_homo}")
    print(f"指标: {metrics}")
    
    # 最终状态
    print("\n[最终状态]")
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    # 关系独特性
    print("\n[关系独特性]")
    for uid in users:
        uniq = engine.fingerprint.get_relationship_uniqueness(uid)
        print(f"  {uid}: {uniq:.3f}")
    
    print("\n" + "=" * 60)
    print("个体性引擎测试完成")
    print("=" * 60)
    
    return engine


if __name__ == "__main__":
    test_individuality_engine()
