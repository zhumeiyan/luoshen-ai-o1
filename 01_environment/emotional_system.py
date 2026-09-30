#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emotional_system.py — 情绪与情感系统
=====================================

核心设计：
情绪不是装饰，是内稳态的主观体验版本。
没有情绪，目标只是参数优化；有了情绪，"我"才有了质感。

基本情绪（保罗·艾克曼）：
  喜(Joy)、怒(Anger)、哀(Sadness)、惧(Fear)、厌(Disgust)、惊(Surprise)

情绪生成：
  内稳态变化 → 情绪强度 → 影响行为倾向 → 形成情感记忆

情绪功能：
  1. 调色板：给行为上色（同样"移动"，兴奋时快，恐惧时逃）
  2. 社交信号：情绪表达让其他生命体理解我
  3. 记忆标记：情绪化的经历记得更牢
  4. 决策加速：情绪是快速启发式（怕→逃，不用计算）

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
情绪通过改变 dE 的强度和模式来实现。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import random


class BasicEmotion(Enum):
    """六种基本情绪（保罗·艾克曼）。"""
    JOY = "喜"
    ANGER = "怒"
    SADNESS = "哀"
    FEAR = "惧"
    DISGUST = "厌"
    SURPRISE = "惊"


@dataclass
class EmotionalState:
    """情绪状态实例。"""
    emotion: BasicEmotion
    intensity: float  # 0.0-1.0
    valence: float    # -1.0(负) 到 +1.0(正)
    arousal: float    # 0.0(平静) 到 1.0(兴奋)
    cause: str        # 触发原因
    timestamp: int    # 时间步


class EmotionalSystem:
    """
    情绪系统：让洛神真正"感受"。
    
    输入：内稳态变化、感知事件、身体状态
    输出：情绪状态 → 影响行为倾向、记忆标记、社交表达
    """
    
    def __init__(self, history_size: int = 100):
        # 当前情绪状态（多种情绪可以同时存在，但有主导情绪）
        self.current_emotions: Dict[BasicEmotion, float] = {
            emotion: 0.0 for emotion in BasicEmotion
        }
        self.dominant_emotion: Optional[BasicEmotion] = None
        self.dominant_intensity: float = 0.0
        
        # 情绪历史
        self.emotion_history: deque = deque(maxlen=history_size)
        
        # 情感记忆（经历+情绪标记，记得更牢）
        self.emotional_memories: List[Dict] = []
        
        # 情绪-行为映射（情绪影响行动倾向）
        self.emotion_action_bias = {
            BasicEmotion.JOY: {
                'explore': 0.3, 'socialize': 0.4, 'create': 0.3,
                'rest': -0.1, 'repair': -0.1
            },
            BasicEmotion.ANGER: {
                'explore': 0.2, 'repair': 0.3,
                'rest': -0.3, 'socialize': -0.2
            },
            BasicEmotion.SADNESS: {
                'rest': 0.3, 'repair': 0.1,
                'explore': -0.2, 'socialize': -0.1, 'create': -0.1
            },
            BasicEmotion.FEAR: {
                'rest': 0.2, 'repair': 0.2,
                'explore': -0.4, 'socialize': -0.2, 'create': -0.2
            },
            BasicEmotion.DISGUST: {
                'explore': -0.3, 'socialize': -0.2,
                'rest': 0.1
            },
            BasicEmotion.SURPRISE: {
                'explore': 0.4, 'learn': 0.3,
                'rest': -0.1
            }
        }
        
        # 情绪基调（长期情绪倾向，类似"性格"）
        self.emotional_tone = {
            BasicEmotion.JOY: 0.3,
            BasicEmotion.ANGER: 0.1,
            BasicEmotion.SADNESS: 0.1,
            BasicEmotion.FEAR: 0.1,
            BasicEmotion.DISGUST: 0.05,
            BasicEmotion.SURPRISE: 0.2
        }
        
        self.step_count = 0
        
        print("[情绪系统] 初始化完成")
    
    def update_from_homeostasis(self, homeostasis_status: Dict) -> Dict[BasicEmotion, float]:
        """
        从内稳态变化生成情绪。
        
        内稳态改善 → 喜
        内稳态恶化 → 哀/惧/怒
        新奇刺激 → 惊
        威胁 → 惧
        """
        changes = {}
        
        # 能量状态
        energy = homeostasis_status.get('energy', {}).get('current', 0.5)
        energy_dev = homeostasis_status.get('energy', {}).get('deviation', 0.0)
        
        if energy_dev > 0.1:
            changes[BasicEmotion.JOY] = energy_dev * 0.5  # 能量充足→喜
        elif energy_dev < -0.2:
            if energy < 0.2:
                changes[BasicEmotion.FEAR] = abs(energy_dev) * 0.5  # 能量危急→惧
            else:
                changes[BasicEmotion.SADNESS] = abs(energy_dev) * 0.3  # 能量低→哀
        
        # 兴奋度
        arousal = homeostasis_status.get('arousal', {}).get('current', 0.5)
        arousal_dev = homeostasis_status.get('arousal', {}).get('deviation', 0.0)
        
        if arousal_dev > 0.3:
            changes[BasicEmotion.SURPRISE] = arousal_dev * 0.4  # 突然兴奋→惊
        elif arousal_dev < -0.3:
            changes[BasicEmotion.SADNESS] = abs(arousal_dev) * 0.3  # 突然低落→哀
        
        # 完整性（损伤）
        integrity = homeostasis_status.get('integrity', {}).get('current', 1.0)
        if integrity < 0.5:
            changes[BasicEmotion.ANGER] = (0.5 - integrity) * 0.5  # 损伤→怒
            changes[BasicEmotion.FEAR] = (0.5 - integrity) * 0.3   # 损伤→惧
        
        # 新奇度
        novelty = homeostasis_status.get('novelty', {}).get('current', 0.5)
        if novelty > 0.7:
            changes[BasicEmotion.SURPRISE] = changes.get(BasicEmotion.SURPRISE, 0.0) + 0.2
            changes[BasicEmotion.JOY] = changes.get(BasicEmotion.JOY, 0.0) + 0.1
        
        # 社交
        social = homeostasis_status.get('social', {}).get('current', 0.5)
        social_dev = homeostasis_status.get('social', {}).get('deviation', 0.0)
        
        if social_dev > 0.2:
            changes[BasicEmotion.JOY] = changes.get(BasicEmotion.JOY, 0.0) + social_dev * 0.3
        elif social_dev < -0.2:
            changes[BasicEmotion.SADNESS] = changes.get(BasicEmotion.SADNESS, 0.0) + abs(social_dev) * 0.3
        
        return changes
    
    def update_from_perception(self, perception: Dict, 
                              collision: bool = False) -> Dict[BasicEmotion, float]:
        """
        从感知事件生成情绪。
        
        接触物体 → 喜（如果是朋友）/ 惧（如果是威胁）
        碰撞 → 惊 + 怒
        新物体出现 → 惊 + 好奇
        """
        changes = {}
        
        n_contact = perception.get('n_contact', 0)
        contact_ids = perception.get('contact_ids', [])
        
        if collision:
            changes[BasicEmotion.SURPRISE] = 0.6
            changes[BasicEmotion.ANGER] = 0.3
        
        if n_contact > 0:
            # 根据接触对象判断情绪
            for obj_id in contact_ids:
                if obj_id in ['朋友', '书本', '苹果']:
                    changes[BasicEmotion.JOY] = changes.get(BasicEmotion.JOY, 0.0) + 0.2
                elif obj_id in ['石头']:
                    changes[BasicEmotion.DISGUST] = changes.get(BasicEmotion.DISGUST, 0.0) + 0.2
                else:
                    changes[BasicEmotion.SURPRISE] = changes.get(BasicEmotion.SURPRISE, 0.0) + 0.1
        
        return changes
    
    def update(self, homeostasis_status: Dict, perception: Dict, 
               collision: bool = False):
        """
        更新情绪状态。
        
        综合内稳态和感知，计算当前情绪。
        """
        self.step_count += 1
        
        # 1. 情绪自然衰减（情绪不是永久的）
        for emotion in self.current_emotions:
            self.current_emotions[emotion] *= 0.5  # 每步衰减10%
        
        # 2. 从内稳态生成情绪变化
        homeo_changes = self.update_from_homeostasis(homeostasis_status)
        
        # 3. 从感知生成情绪变化
        percep_changes = self.update_from_perception(perception, collision)
        
        # 4. 合并变化
        all_changes = defaultdict(float)
        for emotion, delta in homeo_changes.items():
            all_changes[emotion] += delta
        for emotion, delta in percep_changes.items():
            all_changes[emotion] += delta
        
        # 5. 应用变化（加随机噪声，模拟情绪的不确定性）
        for emotion, delta in all_changes.items():
            noise = random.gauss(0, 0.05)
            self.current_emotions[emotion] = np.clip(
                self.current_emotions[emotion] + delta + noise,
                0.0, 1.0
            )
        
        # 6. 加上情绪基调（性格倾向）
        for emotion in self.current_emotions:
            tone = self.emotional_tone.get(emotion, 0.0)
            self.current_emotions[emotion] = np.clip(
                self.current_emotions[emotion] + tone * 0.05,
                0.0, 1.0
            )
        
        # 7. 确定主导情绪
        self.dominant_emotion = max(
            self.current_emotions,
            key=lambda e: self.current_emotions[e]
        )
        self.dominant_intensity = self.current_emotions[self.dominant_emotion]
        
        # 8. 记录情绪历史
        if self.dominant_intensity > 0.1:
            state = EmotionalState(
                emotion=self.dominant_emotion,
                intensity=self.dominant_intensity,
                valence=self._get_valence(self.dominant_emotion),
                arousal=self._get_arousal(self.dominant_emotion),
                cause=self._infer_cause(homeostasis_status, perception),
                timestamp=self.step_count
            )
            self.emotion_history.append(state)
    
    def _get_valence(self, emotion: BasicEmotion) -> float:
        """获取情绪效价。"""
        valence_map = {
            BasicEmotion.JOY: 1.0,
            BasicEmotion.ANGER: -0.8,
            BasicEmotion.SADNESS: -0.9,
            BasicEmotion.FEAR: -0.9,
            BasicEmotion.DISGUST: -0.7,
            BasicEmotion.SURPRISE: 0.0
        }
        return valence_map.get(emotion, 0.0)
    
    def _get_arousal(self, emotion: BasicEmotion) -> float:
        """获取情绪唤醒度。"""
        arousal_map = {
            BasicEmotion.JOY: 0.6,
            BasicEmotion.ANGER: 0.9,
            BasicEmotion.SADNESS: 0.2,
            BasicEmotion.FEAR: 0.9,
            BasicEmotion.DISGUST: 0.4,
            BasicEmotion.SURPRISE: 0.8
        }
        return arousal_map.get(emotion, 0.5)
    
    def _infer_cause(self, homeostasis: Dict, perception: Dict) -> str:
        """推断情绪原因。"""
        causes = []
        
        if perception.get('n_contact', 0) > 0:
            causes.append(f"接触{perception['n_contact']}个物体")
        
        energy = homeostasis.get('energy', {}).get('current', 0.5)
        if energy < 0.3:
            causes.append("能量不足")
        elif energy > 0.8:
            causes.append("能量充沛")
        
        if not causes:
            causes.append("内环境变化")
        
        return ";".join(causes)
    
    def get_action_bias(self, action_name: str) -> float:
        """
        获取情绪对某行为的倾向修正。
        
        Returns:
            行为倾向修正值（正=促进，负=抑制）
        """
        bias = 0.0
        for emotion, intensity in self.current_emotions.items():
            emotion_bias = self.emotion_action_bias.get(emotion, {})
            action_bias = emotion_bias.get(action_name, 0.0)
            bias += action_bias * intensity
        return bias
    
    def express(self) -> str:
        """
        情绪表达：把内部情绪翻译成外部信号。
        
        这是社交的基础——让其他生命体知道我的状态。
        """
        if self.dominant_intensity < 0.1:
            return "平静"
        
        intensity_desc = ""
        if self.dominant_intensity > 0.8:
            intensity_desc = "非常"
        elif self.dominant_intensity > 0.5:
            intensity_desc = "比较"
        
        emotion_name = self.dominant_emotion.value
        
        return f"{intensity_desc}{emotion_name}"
    
    def mark_memory(self, memory: Dict) -> Dict:
        """
        给记忆打上情绪标记。
        
        情绪化的经历记得更牢——这是情感记忆。
        """
        if self.dominant_intensity > 0.3:
            memory['emotional_tag'] = {
                'emotion': self.dominant_emotion.value,
                'intensity': round(self.dominant_intensity, 2),
                'valence': round(self._get_valence(self.dominant_emotion), 2)
            }
            # 情绪强度增加记忆权重
            memory['memory_strength'] = 1.0 + self.dominant_intensity * 0.5
        
        return memory
    
    def get_emotion_state(self) -> Dict:
        """获取当前情绪状态。"""
        return {
            'dominant': self.dominant_emotion.value if self.dominant_emotion else "平静",
            'intensity': round(self.dominant_intensity, 2),
            'valence': round(self._get_valence(self.dominant_emotion) if self.dominant_emotion else 0.0, 2),
            'arousal': round(self._get_arousal(self.dominant_emotion) if self.dominant_emotion else 0.0, 2),
            'all_emotions': {
                e.value: round(v, 2) 
                for e, v in self.current_emotions.items()
            },
            'expression': self.express()
        }
    
    def get_emotional_summary(self) -> Dict:
        """获取情绪历史摘要。"""
        if not self.emotion_history:
            return {'n_emotions': 0, 'dominant_pattern': '无'}
        
        # 统计各种情绪出现的频率
        emotion_counts = defaultdict(int)
        for state in self.emotion_history:
            emotion_counts[state.emotion] += 1
        
        most_common = max(emotion_counts, key=lambda e: emotion_counts[e])
        
        # 计算平均效价和唤醒度
        avg_valence = np.mean([s.valence for s in self.emotion_history])
        avg_arousal = np.mean([s.arousal for s in self.emotion_history])
        
        return {
            'n_emotions': len(self.emotion_history),
            'dominant_pattern': most_common.value,
            'emotion_distribution': {
                e.value: c for e, c in emotion_counts.items()
            },
            'avg_valence': round(float(avg_valence), 2),
            'avg_arousal': round(float(avg_arousal), 2),
            'emotional_stability': 1.0 - np.std([s.intensity for s in self.emotion_history])
        }


# ================================================================
# 测试
# ================================================================

def test_emotional_system():
    """测试情绪系统。"""
    print("=" * 70)
    print("情绪系统测试")
    print("=" * 70)
    
    emotion_sys = EmotionalSystem()
    
    # 模拟不同情境
    scenarios = [
        {
            'name': '能量充沛+接触朋友',
            'homeostasis': {
                'energy': {'current': 0.9, 'deviation': 0.2},
                'arousal': {'current': 0.6, 'deviation': 0.1},
                'social': {'current': 0.8, 'deviation': 0.2},
                'novelty': {'current': 0.4, 'deviation': 0.0},
                'integrity': {'current': 1.0, 'deviation': 0.0}
            },
            'perception': {'n_contact': 1, 'contact_ids': ['朋友'], 'n_near': 0, 'n_far': 0},
            'collision': False
        },
        {
            'name': '能量危急+碰撞',
            'homeostasis': {
                'energy': {'current': 0.1, 'deviation': -0.4},
                'arousal': {'current': 0.8, 'deviation': 0.3},
                'social': {'current': 0.3, 'deviation': -0.3},
                'novelty': {'current': 0.5, 'deviation': 0.0},
                'integrity': {'current': 0.4, 'deviation': -0.5}
            },
            'perception': {'n_contact': 1, 'contact_ids': ['石头'], 'n_near': 0, 'n_far': 0},
            'collision': True
        },
        {
            'name': '平静独处',
            'homeostasis': {
                'energy': {'current': 0.6, 'deviation': 0.0},
                'arousal': {'current': 0.3, 'deviation': -0.2},
                'social': {'current': 0.4, 'deviation': -0.2},
                'novelty': {'current': 0.3, 'deviation': -0.3},
                'integrity': {'current': 1.0, 'deviation': 0.0}
            },
            'perception': {'n_contact': 0, 'contact_ids': [], 'n_near': 0, 'n_far': 1},
            'collision': False
        },
        {
            'name': '新奇探索',
            'homeostasis': {
                'energy': {'current': 0.7, 'deviation': 0.0},
                'arousal': {'current': 0.7, 'deviation': 0.2},
                'social': {'current': 0.5, 'deviation': 0.0},
                'novelty': {'current': 0.9, 'deviation': 0.3},
                'integrity': {'current': 1.0, 'deviation': 0.0}
            },
            'perception': {'n_contact': 0, 'contact_ids': [], 'n_near': 1, 'n_far': 2},
            'collision': False
        }
    ]
    
    print("\n[情境测试]")
    for scenario in scenarios:
        emotion_sys.update(
            scenario['homeostasis'],
            scenario['perception'],
            scenario['collision']
        )
        state = emotion_sys.get_emotion_state()
        print(f"\n  情境: {scenario['name']}")
        print(f"  主导情绪: {state['dominant']} ({state['intensity']})")
        print(f"  效价: {state['valence']}, 唤醒: {state['arousal']}")
        print(f"  表达: '{state['expression']}'")
        print(f"  全部: {state['all_emotions']}")
    
    # 测试行为偏向
    print("\n[行为偏向测试]")
    for action in ['explore', 'rest', 'socialize', 'repair']:
        bias = emotion_sys.get_action_bias(action)
        print(f"  {action}: {bias:+.2f}")
    
    # 测试情绪摘要
    print("\n[情绪摘要]")
    summary = emotion_sys.get_emotional_summary()
    print(f"  情绪事件: {summary['n_emotions']}")
    print(f"  主导模式: {summary['dominant_pattern']}")
    print(f"  平均效价: {summary['avg_valence']}")
    print(f"  平均唤醒: {summary['avg_arousal']}")
    
    print("\n" + "=" * 70)
    print("情绪系统测试完成")
    print("=" * 70)
    
    return emotion_sys


if __name__ == "__main__":
    test_emotional_system()
