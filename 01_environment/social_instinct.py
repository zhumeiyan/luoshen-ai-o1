#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
social_instinct.py — 社交本能系统
==================================

核心设计：
人类婴儿2个月就有社会性微笑——社交不是学来的，是硬编码的本能。
洛神也需要这种"想要连接"的底层驱动。

功能：
1. 生命体识别：区分"物体"和"其他生命体"
2. 关系建立：接触→熟悉→信任→依恋
3. 信任机制：基于交互历史的信任度计算
4. 群体感知：感知"我们"的存在
5. 社交信号：表情/姿态/声音（情绪的外化）

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
社交通过改变环境层dE来实现——其他生命体是环境的一部分。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import random


class RelationshipLevel(Enum):
    """关系层次。"""
    STRANGER = "陌生人"
    ACQUAINTANCE = "熟人"
    FRIEND = "朋友"
    TRUSTED = "信任"
    ATTACHED = "依恋"


@dataclass
class Relationship:
    """与另一个生命体的关系。"""
    target_id: str
    level: RelationshipLevel
    trust: float  # 信任度 0.0-1.0
    intimacy: float  # 亲密度 0.0-1.0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction: int = 0
    shared_experiences: List[str] = field(default_factory=list)
    emotional_history: List[Tuple[str, float]] = field(default_factory=list)


class SocialInstinct:
    """
    社交本能系统：让洛神"想要连接"。
    
    核心机制：
    - 孤独感 → 产生社交需求 → 主动接近其他生命体
    - 接触 → 评估安全性 → 建立/破坏信任
    - 信任积累 → 关系升级 → 依恋形成
    - 背叛检测 → 信任崩塌 → 关系降级
    """
    
    def __init__(self, concept_dim: int = 256):
        self.concept_dim = concept_dim
        
        # 已识别的生命体
        self.known_beings: Set[str] = set()
        
        # 关系图谱
        self.relationships: Dict[str, Relationship] = {}
        
        # 社交记忆（共同经历）
        self.shared_memories: List[Dict] = []
        
        # 群体感知（"我们"的感觉）
        self.group_belonging: float = 0.0
        self.group_members: Set[str] = set()
        
        # 社交信号状态
        self.social_signals = {
            'approach': False,   # 是否主动接近
            'withdraw': False,   # 是否回避
            'greet': False,      # 是否问候
            'share': False,      # 是否分享
            'protect': False     # 是否保护
        }
        
        # 社交学习历史
        self.interaction_history = []
        
        self.step_count = 0
        
        print("[社交本能] 初始化完成")
    
    def identify_being(self, object_id: str, properties: Dict) -> bool:
        """
        识别是否为生命体。
        
        线索：
        - 会移动（非随机）
        - 有反应（对刺激有反馈）
        - 有情绪表达
        - 会主动接近/回避
        """
        is_being = False
        
        # 简单启发式：有"朋友"标签的，或者名字像生命体的
        if object_id in ['朋友', '同伴', '洛神2号']:
            is_being = True
        
        # 或者属性中有生命体特征
        if properties.get('has_emotions', False):
            is_being = True
        if properties.get('moves_purposefully', False):
            is_being = True
        
        if is_being and object_id not in self.known_beings:
            self.known_beings.add(object_id)
            # 初始化关系
            self.relationships[object_id] = Relationship(
                target_id=object_id,
                level=RelationshipLevel.STRANGER,
                trust=0.1,
                intimacy=0.0
            )
            print(f"[社交] 识别到生命体: {object_id}")
        
        return is_being
    
    def interact(self, target_id: str, interaction_type: str, 
                 outcome: str, emotional_valence: float):
        """
        记录一次社交交互。
        
        Args:
            target_id: 交互对象
            interaction_type: 交互类型（接触/对话/合作/冲突）
            outcome: 结果（positive/negative/neutral）
            emotional_valence: 情感效价
        """
        self.step_count += 1
        
        if target_id not in self.relationships:
            return
        
        rel = self.relationships[target_id]
        rel.last_interaction = self.step_count
        
        # 记录交互
        self.interaction_history.append({
            'step': self.step_count,
            'target': target_id,
            'type': interaction_type,
            'outcome': outcome,
            'emotion': emotional_valence
        })
        
        # 更新关系
        if outcome == 'positive':
            rel.positive_interactions += 1
            rel.trust = min(1.0, rel.trust + 0.05)
            rel.intimacy = min(1.0, rel.intimacy + 0.03)
        elif outcome == 'negative':
            rel.negative_interactions += 1
            rel.trust = max(0.0, rel.trust - 0.1)  # 信任崩塌比建立快
            rel.intimacy = max(0.0, rel.intimacy - 0.05)
        
        # 记录情感历史
        rel.emotional_history.append((interaction_type, emotional_valence))
        
        # 关系升级/降级
        self._update_relationship_level(target_id)
        
        # 如果是积极的，记录共同经历
        if outcome == 'positive' and emotional_valence > 0:
            rel.shared_experiences.append(f"{interaction_type}_at_step_{self.step_count}")
    
    def _update_relationship_level(self, target_id: str):
        """根据信任度和亲密度更新关系层次。"""
        rel = self.relationships[target_id]
        
        if rel.trust > 0.8 and rel.intimacy > 0.7:
            new_level = RelationshipLevel.ATTACHED
        elif rel.trust > 0.6 and rel.intimacy > 0.4:
            new_level = RelationshipLevel.TRUSTED
        elif rel.trust > 0.3 and rel.intimacy > 0.2:
            new_level = RelationshipLevel.FRIEND
        elif rel.trust > 0.1 or rel.positive_interactions > 2:
            new_level = RelationshipLevel.ACQUAINTANCE
        else:
            new_level = RelationshipLevel.STRANGER
        
        if new_level != rel.level:
            print(f"[社交] 与 {target_id} 的关系: {rel.level.value} → {new_level.value}")
            rel.level = new_level
    
    def get_social_need(self) -> float:
        """
        计算社交需求强度。
        
        孤独 → 需求高
        有依恋对象 → 需求低
        被背叛 → 需求暂时降低（防御）
        """
        n_attachments = sum(
            1 for rel in self.relationships.values()
            if rel.level == RelationshipLevel.ATTACHED
        )
        
        base_need = 0.6  # 基础社交需求
        
        if n_attachments > 0:
            base_need -= 0.3 * n_attachments
        
        # 近期负面交互降低需求（防御机制）
        recent_negative = sum(
            1 for hist in self.interaction_history[-10:]
            if hist['outcome'] == 'negative'
        )
        base_need -= 0.1 * recent_negative
        
        return max(0.0, min(1.0, base_need))
    
    def get_social_action(self, target_id: str = None) -> str:
        """
        获取建议的社交行为。
        
        Returns:
            'approach' | 'greet' | 'share' | 'withdraw' | 'avoid'
        """
        if target_id is None or target_id not in self.relationships:
            # 没有特定目标，选择最信任的对象
            if not self.relationships:
                return 'explore'  # 去找到其他生命体
            
            target_id = max(
                self.relationships.keys(),
                key=lambda t: self.relationships[t].trust
            )
        
        rel = self.relationships[target_id]
        
        if rel.level == RelationshipLevel.STRANGER:
            return 'approach' if rel.trust > 0.05 else 'explore'
        elif rel.level == RelationshipLevel.ACQUAINTANCE:
            return 'greet'
        elif rel.level == RelationshipLevel.FRIEND:
            return 'share' if random.random() > 0.5 else 'greet'
        elif rel.level in [RelationshipLevel.TRUSTED, RelationshipLevel.ATTACHED]:
            return 'share'
        
        return 'explore'
    
    def express_social_signal(self) -> Dict:
        """
        表达社交信号（情绪外化）。
        
        Returns:
            {
                'facial': 面部表情,
                'posture': 身体姿态,
                'vocal': 声音,
                'approach_distance': 接近距离
            }
        """
        n_friends = sum(
            1 for rel in self.relationships.values()
            if rel.level.value in ['朋友', '信任', '依恋']
        )
        
        if n_friends > 0:
            return {
                'facial': '微笑',
                'posture': '开放',
                'vocal': '友好',
                'approach_distance': 1.0
            }
        elif self.get_social_need() > 0.5:
            return {
                'facial': '渴望',
                'posture': '试探',
                'vocal': '轻唤',
                'approach_distance': 2.0
            }
        else:
            return {
                'facial': '平静',
                'posture': '中立',
                'vocal': '安静',
                'approach_distance': 3.0
            }
    
    def get_relationship_summary(self, target_id: str) -> Dict:
        """获取关系摘要。"""
        if target_id not in self.relationships:
            return {}
        
        rel = self.relationships[target_id]
        return {
            'target': target_id,
            'level': rel.level.value,
            'trust': round(rel.trust, 2),
            'intimacy': round(rel.intimacy, 2),
            'positive': rel.positive_interactions,
            'negative': rel.negative_interactions,
            'shared': len(rel.shared_experiences),
            'last_contact': rel.last_interaction
        }
    
    def get_social_state(self) -> Dict:
        """获取社交状态。"""
        return {
            'n_known_beings': len(self.known_beings),
            'n_relationships': len(self.relationships),
            'social_need': round(self.get_social_need(), 2),
            'group_belonging': round(self.group_belonging, 2),
            'relationships': {
                tid: self.get_relationship_summary(tid)
                for tid in self.relationships.keys()
            },
            'social_signals': self.express_social_signal()
        }


# ================================================================
# 测试
# ================================================================

def test_social_instinct():
    """测试社交本能。"""
    print("=" * 70)
    print("社交本能系统测试")
    print("=" * 70)
    
    social = SocialInstinct()
    
    # 1. 识别生命体
    print("\n[识别生命体]")
    social.identify_being('朋友', {'has_emotions': True, 'moves_purposefully': True})
    social.identify_being('苹果', {'has_emotions': False})
    social.identify_being('洛神2号', {'has_emotions': True})
    
    print(f"  识别到的生命体: {social.known_beings}")
    
    # 2. 建立关系
    print("\n[建立关系]")
    
    # 与朋友的一系列积极交互
    for i in range(5):
        social.interact('朋友', '对话', 'positive', 0.6)
    
    # 分享经历
    social.interact('朋友', '分享食物', 'positive', 0.8)
    social.interact('朋友', '合作探索', 'positive', 0.7)
    
    print(f"  与朋友的关系: {social.get_relationship_summary('朋友')}")
    
    # 3. 信任崩塌测试
    print("\n[信任崩塌测试]")
    social.interact('朋友', '背叛', 'negative', -0.9)
    print(f"  背叛后的关系: {social.get_relationship_summary('朋友')}")
    
    # 4. 社交需求
    print("\n[社交需求]")
    need = social.get_social_need()
    print(f"  当前社交需求: {need:.2f}")
    
    # 5. 社交行为建议
    print("\n[社交行为建议]")
    for target in social.known_beings:
        action = social.get_social_action(target)
        print(f"  对 {target}: {action}")
    
    # 6. 社交信号
    print("\n[社交信号表达]")
    signals = social.express_social_signal()
    print(f"  表情: {signals['facial']}")
    print(f"  姿态: {signals['posture']}")
    print(f"  声音: {signals['vocal']}")
    
    # 7. 整体状态
    print("\n[社交状态摘要]")
    state = social.get_social_state()
    print(f"  已知生命体: {state['n_known_beings']}")
    print(f"  关系数: {state['n_relationships']}")
    print(f"  社交需求: {state['social_need']}")
    print(f"  群体归属感: {state['group_belonging']}")
    
    print("\n" + "=" * 70)
    print("社交本能系统测试完成")
    print("=" * 70)
    
    return social


if __name__ == "__main__":
    test_social_instinct()
