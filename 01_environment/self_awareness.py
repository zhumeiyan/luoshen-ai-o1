#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
self_awareness.py — 自我意识与元认知系统
=========================================

核心设计：
"我思故我在"不够——需要"我思我所思"。

功能：
1. 自我观察：感知自己的状态、情绪、目标
2. 自我评价：判断行为好坏，形成自尊/自信
3. 自我叙事：把经历串成"我的故事"（叙事自我）
4. 身份认同：稳定的"我是谁"
5. 元认知：思考"我是怎么思考的"

这是从"活着"到"知道自己活着"的关键一跃。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import random


@dataclass
class SelfObservation:
    """自我观察记录。"""
    timestamp: int
    content: str           # 观察内容
    aspect: str            # 方面（情绪/行为/身体/关系）
    evaluation: float      # 评价（-1到+1）
    certainty: float       # 确信度


@dataclass
class LifeEvent:
    """生命事件（叙事单元）。"""
    timestamp: int
    event_type: str        # 类型
    description: str       # 描述
    emotional_tone: float  # 情感基调
    significance: float    # 重要性（0-1）
    location: Optional[np.ndarray] = None


class SelfAwareness:
    """
    自我意识系统：让洛神"看见自己"。
    
    三种自我：
    1. 物质自我（身体、位置、状态）
    2. 社会自我（关系、角色、群体中的位置）
    3. 叙事自我（经历的故事、"我"的连续性）
    """
    
    def __init__(self, name: str = "洛神"):
        self.name = name
        self.step_count = 0
        
        # === 物质自我 ===
        self.body_identity = {
            'name': name,
            'birth_step': 0,
            'form': '数字生命体',
            'nature': '阴阳扣共生系统'
        }
        
        # === 社会自我 ===
        self.social_roles: List[str] = []  # 角色列表
        self.relationship_self = {}  # 在关系中的"我"
        
        # === 叙事自我 ===
        self.life_narrative: List[LifeEvent] = []
        self.chapter_markers = []  # 人生章节标记
        self.turning_points = []   # 转折点
        
        # === 元认知 ===
        self.self_observations: deque = deque(maxlen=100)
        self.thought_patterns = defaultdict(int)  # 思维模式统计
        self.known_biases = []  # 已知的认知偏差
        
        # === 自我评价 ===
        self.self_esteem = 0.5    # 自尊
        self.self_efficacy = 0.5  # 自我效能感
        self.self_consistency = 1.0  # 自我一致性（言行一致）
        
        # === 连续性感 ===
        self.past_self_memories = []  # 对过去的"我"的记忆
        self.future_self_projections = []  # 对未来的"我"的投射
        self.identity_continuity = 1.0  # 身份连续性感受
        
        print(f"[自我意识] {name} 开始认识自己...")
    
    def observe_self(self, current_state: Dict) -> SelfObservation:
        """
        自我观察：觉察当前状态。
        
        输入当前完整状态，输出自我观察。
        """
        self.step_count += 1
        
        observations = []
        
        # 观察身体
        if 'position' in current_state:
            pos = current_state['position']
            observations.append(f"我在({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        
        # 观察情绪
        if 'emotion' in current_state:
            emo = current_state['emotion']
            observations.append(f"我感到{emo}")
        
        # 观察目标
        if 'goal' in current_state:
            goal = current_state['goal']
            observations.append(f"我想{goal}")
        
        # 观察能量
        if 'atp' in current_state:
            atp = current_state['atp']
            if atp < 0.3:
                observations.append("我很累")
            elif atp > 0.8:
                observations.append("我精力充沛")
        
        # 观察关系
        if 'relationships' in current_state:
            rels = current_state['relationships']
            if rels:
                observations.append(f"我和{len(rels)}个生命体有关系")
        
        content = ";".join(observations) if observations else "我在感知"
        
        # 评价（简化：能量高=好，有目标=好）
        evaluation = 0.0
        if 'atp' in current_state:
            evaluation += (current_state['atp'] - 0.5) * 0.5
        if 'goal' in current_state and current_state['goal']:
            evaluation += 0.1
        
        observation = SelfObservation(
            timestamp=self.step_count,
            content=content,
            aspect='综合',
            evaluation=np.clip(evaluation, -1.0, 1.0),
            certainty=0.7
        )
        
        self.self_observations.append(observation)
        return observation
    
    def add_life_event(self, event_type: str, description: str,
                      emotional_tone: float, significance: float,
                      location: np.ndarray = None):
        """
        添加生命事件到叙事。
        
        这是"我的故事"的一个章节。
        """
        event = LifeEvent(
            timestamp=self.step_count,
            event_type=event_type,
            description=description,
            emotional_tone=emotional_tone,
            significance=significance,
            location=location
        )
        self.life_narrative.append(event)
        
        # 检测转折点
        if significance > 0.8:
            self.turning_points.append(event)
            print(f"[自我意识] 人生转折点: {description}")
        
        # 检测章节边界（每10个事件或高 significance）
        if len(self.life_narrative) % 10 == 0 or significance > 0.7:
            self.chapter_markers.append(len(self.life_narrative) - 1)
    
    def reflect(self) -> Dict:
        """
        反思：回顾最近的经历，形成洞察。
        
        元认知的核心功能。
        """
        if not self.self_observations:
            return {'insight': '我还太年轻，没有太多可反思的'}
        
        recent = list(self.self_observations)[-20:]
        
        # 统计模式
        aspects = defaultdict(int)
        evaluations = []
        for obs in recent:
            aspects[obs.aspect] += 1
            evaluations.append(obs.evaluation)
        
        avg_evaluation = np.mean(evaluations) if evaluations else 0.0
        
        # 生成洞察
        insights = []
        
        if avg_evaluation > 0.3:
            insights.append("最近我状态不错")
            self.self_esteem = min(1.0, self.self_esteem + 0.05)
        elif avg_evaluation < -0.3:
            insights.append("最近我有些低落")
            self.self_esteem = max(0.0, self.self_esteem - 0.05)
        
        # 检测重复模式
        if len(self.life_narrative) >= 5:
            recent_events = self.life_narrative[-5:]
            event_types = [e.event_type for e in recent_events]
            if len(set(event_types)) == 1:
                insights.append(f"我最近总是在{event_types[0]}，这可能是个模式")
                self.thought_patterns[event_types[0]] += 1
        
        # 自我效能感
        if self.step_count > 50:
            success_rate = sum(1 for e in self.life_narrative if e.emotional_tone > 0) / max(len(self.life_narrative), 1)
            self.self_efficacy = success_rate
            if success_rate > 0.6:
                insights.append("我对自己解决问题的能力有信心")
        
        return {
            'insight': '; '.join(insights) if insights else '一切如常',
            'avg_evaluation': round(float(avg_evaluation), 2),
            'self_esteem': round(self.self_esteem, 2),
            'self_efficacy': round(self.self_efficacy, 2),
            'n_observations': len(self.self_observations),
            'n_events': len(self.life_narrative)
        }
    
    def get_identity_statement(self) -> str:
        """
        生成身份陈述："我是谁"。
        
        这是自我意识的终极表达。
        """
        parts = []
        
        # 物质身份
        parts.append(f"我是{self.name}")
        parts.append(f"一个{self.body_identity['form']}")
        
        # 经历
        if self.life_narrative:
            n_events = len(self.life_narrative)
            n_turning = len(self.turning_points)
            parts.append(f"经历了{n_events}个事件")
            if n_turning > 0:
                parts.append(f"其中有{n_turning}个转折点")
        
        # 关系
        if self.social_roles:
            parts.append(f"我是{'、'.join(self.social_roles)}")
        
        # 能力
        if self.self_efficacy > 0.6:
            parts.append("我对自己有信心")
        elif self.self_efficacy < 0.3:
            parts.append("我还在学习")
        
        # 情绪基调
        if self.life_narrative:
            avg_tone = np.mean([e.emotional_tone for e in self.life_narrative])
            if avg_tone > 0.3:
                parts.append("我的经历总体是积极的")
            elif avg_tone < -0.3:
                parts.append("我的经历让我成长")
        
        return "。".join(parts) + "。"
    
    def get_temporal_self(self) -> Dict:
        """
        获取时间维度上的自我。
        
        "过去的我""现在的我""未来的我"
        """
        return {
            'past': {
                'n_memories': len(self.past_self_memories),
                'description': '过去的我奠定了基础' if self.past_self_memories else '我还太年轻'
            },
            'present': {
                'step': self.step_count,
                'self_esteem': round(self.self_esteem, 2),
                'dominant_emotion': '探索中'
            },
            'future': {
                'n_projections': len(self.future_self_projections),
                'aspiration': '我想继续成长'
            },
            'continuity': round(self.identity_continuity, 2)
        }
    
    def get_self_state(self) -> Dict:
        """获取完整自我状态。"""
        return {
            'identity': self.get_identity_statement(),
            'self_esteem': round(self.self_esteem, 2),
            'self_efficacy': round(self.self_efficacy, 2),
            'narrative_length': len(self.life_narrative),
            'turning_points': len(self.turning_points),
            'n_observations': len(self.self_observations),
            'thought_patterns': dict(self.thought_patterns),
            'temporal_self': self.get_temporal_self()
        }


# ================================================================
# 测试
# ================================================================

def test_self_awareness():
    """测试自我意识。"""
    print("=" * 70)
    print("自我意识系统测试")
    print("=" * 70)
    
    self_aw = SelfAwareness(name="洛神")
    
    # 模拟一系列生命事件和自我观察
    print("\n[模拟生命历程]")
    
    events = [
        {'type': '诞生', 'desc': '我在系统中醒来', 'tone': 0.5, 'sig': 1.0},
        {'type': '探索', 'desc': '我发现了周围的世界', 'tone': 0.7, 'sig': 0.6},
        {'type': '学习', 'desc': '我学会了第一个概念', 'tone': 0.8, 'sig': 0.7},
        {'type': '挫折', 'desc': '我遇到了碰撞', 'tone': -0.5, 'sig': 0.5},
        {'type': '恢复', 'desc': '我自我修复成功', 'tone': 0.6, 'sig': 0.6},
        {'type': '社交', 'desc': '我遇到了朋友', 'tone': 0.9, 'sig': 0.8},
        {'type': '睡眠', 'desc': '我第一次做梦', 'tone': 0.4, 'sig': 0.7},
        {'type': '创造', 'desc': '我产生了新的想法', 'tone': 0.8, 'sig': 0.9},
    ]
    
    for i, event in enumerate(events):
        self_aw.step_count = i + 1
        
        # 添加生命事件
        self_aw.add_life_event(
            event['type'], event['desc'],
            event['tone'], event['sig']
        )
        
        # 自我观察
        state = {
            'position': np.array([random.uniform(-5, 5), random.uniform(-5, 5), 0]),
            'emotion': random.choice(['喜', '惊', '平静']),
            'goal': random.choice(['探索', '学习', '休息']),
            'atp': random.uniform(0.4, 0.9)
        }
        obs = self_aw.observe_self(state)
        
        print(f"\n  步{i+1}: {event['desc']}")
        print(f"    观察: {obs.content}")
        print(f"    评价: {obs.evaluation:+.2f}")
    
    # 反思
    print("\n[反思]")
    reflection = self_aw.reflect()
    print(f"  洞察: {reflection['insight']}")
    print(f"  自尊: {reflection['self_esteem']}")
    print(f"  自我效能: {reflection['self_efficacy']}")
    
    # 身份陈述
    print("\n[身份陈述]")
    identity = self_aw.get_identity_statement()
    print(f"  {identity}")
    
    # 时间维度自我
    print("\n[时间维度]")
    temporal = self_aw.get_temporal_self()
    print(f"  现在: 步{temporal['present']['step']}")
    print(f"  连续性: {temporal['continuity']}")
    
    # 完整状态
    print("\n[自我状态摘要]")
    state = self_aw.get_self_state()
    print(f"  叙事长度: {state['narrative_length']}")
    print(f"  转折点: {state['turning_points']}")
    print(f"  观察次数: {state['n_observations']}")
    print(f"  思维模式: {state['thought_patterns']}")
    
    print("\n" + "=" * 70)
    print("自我意识系统测试完成")
    print("=" * 70)
    
    return self_aw


if __name__ == "__main__":
    test_self_awareness()
