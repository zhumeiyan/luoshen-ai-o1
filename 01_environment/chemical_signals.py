#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chemical_signals.py — 化学信号与神经递质系统
=============================================

核心设计：
真实神经系统不是单一电信号，是多通道的：
- 多巴胺(Dopamine)：奖励/动机/学习信号
- 血清素(Serotonin)：稳定/满足/情绪调节
- 去甲肾上腺素(NE)：警觉/应激/注意
- GABA：抑制（镇静、减缓）
- 谷氨酸(Glutamate)：兴奋（增强、加速）
- 内啡肽(Endorphin)：镇痛/愉悦

关键洞察：
同一组神经元，在不同化学环境下，行为完全不同。
化学信号是"氛围"，不是"指令"。

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
化学通过改变 f(E) 和 g(E) 的增益来实现。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import random


class Neurotransmitter(Enum):
    """神经递质类型。"""
    DOPAMINE = "多巴胺"      # 奖励/动机/学习
    SEROTONIN = "血清素"     # 稳定/满足/情绪
    NOREPINEPHRINE = "去甲肾上腺素"  # 警觉/应激/注意
    GABA = "GABA"           # 抑制
    GLUTAMATE = "谷氨酸"     # 兴奋
    ENDORPHIN = "内啡肽"     # 镇痛/愉悦


class ChemicalState:
    """
    化学状态：神经递质浓度分布。
    
    不是单一值，是每个神经元集群的局部浓度。
    """
    
    def __init__(self, dim: int = 256):
        self.dim = dim
        
        # 每种递质的浓度（0.0-1.0）
        self.concentrations = {
            nt: np.zeros(dim) for nt in Neurotransmitter
        }
        
        # 基准浓度（每个人的"性格"不同）
        self.baseline = {
            Neurotransmitter.DOPAMINE: 0.3,
            Neurotransmitter.SEROTONIN: 0.4,
            Neurotransmitter.NOREPINEPHRINE: 0.2,
            Neurotransmitter.GABA: 0.3,
            Neurotransmitter.GLUTAMATE: 0.4,
            Neurotransmitter.ENDORPHIN: 0.1
        }
        
        # 初始化基准
        for nt, base in self.baseline.items():
            self.concentrations[nt] += base
        
        # 衰减率
        self.decay_rates = {
            Neurotransmitter.DOPAMINE: 0.1,
            Neurotransmitter.SEROTONIN: 0.05,
            Neurotransmitter.NOREPINEPHRINE: 0.15,
            Neurotransmitter.GABA: 0.08,
            Neurotransmitter.GLUTAMATE: 0.1,
            Neurotransmitter.ENDORPHIN: 0.2
        }
        
        self.step_count = 0
        
        print("[化学信号] 神经递质系统初始化完成")
    
    def release(self, transmitter: Neurotransmitter, 
                target_indices: np.ndarray, amount: float):
        """
        在特定神经元集群释放神经递质。
        """
        for idx in target_indices:
            if 0 <= idx < self.dim:
                self.concentrations[transmitter][idx] = min(
                    1.0, self.concentrations[transmitter][idx] + amount
                )
    
    def global_release(self, transmitter: Neurotransmitter, amount: float):
        """全局释放（类似激素广播）。"""
        self.concentrations[transmitter] = np.clip(
            self.concentrations[transmitter] + amount,
            0.0, 1.0
        )
    
    def decay(self):
        """递质自然衰减（再摄取）。"""
        for nt in Neurotransmitter:
            self.concentrations[nt] = np.clip(
                self.concentrations[nt] - self.decay_rates[nt],
                self.baseline[nt] * 0.5,  # 不低于基准的一半
                1.0
            )
    
    def get_modulation(self) -> Dict[str, float]:
        """
        获取当前化学环境对神经活动的调制效应。
        
        Returns:
            {
                'excitation_gain': 兴奋增益,
                'inhibition_gain': 抑制增益,
                'learning_rate': 学习率调制,
                'motivation': 动机水平,
                'stress': 应激水平
            }
        """
        # 平均浓度
        avg = {nt: float(np.mean(self.concentrations[nt])) for nt in Neurotransmitter}
        
        # 调制效应
        excitation = avg[Neurotransmitter.GLUTAMATE] - avg[Neurotransmitter.GABA] * 0.5
        inhibition = avg[Neurotransmitter.GABA] - avg[Neurotransmitter.GLUTAMATE] * 0.3
        learning = avg[Neurotransmitter.DOPAMINE] * 0.5 + avg[Neurotransmitter.NOREPINEPHRINE] * 0.3
        motivation = avg[Neurotransmitter.DOPAMINE] * 0.7 + avg[Neurotransmitter.NOREPINEPHRINE] * 0.3
        stress = avg[Neurotransmitter.NOREPINEPHRINE] - avg[Neurotransmitter.SEROTONIN] * 0.5
        
        return {
            'excitation_gain': round(np.clip(excitation, 0.0, 2.0), 2),
            'inhibition_gain': round(np.clip(inhibition, 0.0, 2.0), 2),
            'learning_rate': round(np.clip(learning, 0.1, 2.0), 2),
            'motivation': round(np.clip(motivation, 0.0, 1.0), 2),
            'stress': round(np.clip(stress, -1.0, 1.0), 2),
            'pleasure': round(avg[Neurotransmitter.DOPAMINE] + avg[Neurotransmitter.ENDORPHIN], 2),
            'calmness': round(avg[Neurotransmitter.SEROTONIN] + avg[Neurotransmitter.GABA], 2)
        }
    
    def trigger_reward(self, strength: float = 0.5):
        """触发奖励回路：多巴胺释放。"""
        self.global_release(Neurotransmitter.DOPAMINE, strength)
        self.global_release(Neurotransmitter.ENDORPHIN, strength * 0.3)
        print(f"[化学] 奖励! 多巴胺+{strength:.2f}")
    
    def trigger_stress(self, strength: float = 0.5):
        """触发应激反应：去甲肾上腺素释放。"""
        self.global_release(Neurotransmitter.NOREPINEPHRINE, strength)
        self.global_release(Neurotransmitter.GLUTAMATE, strength * 0.3)
        print(f"[化学] 应激! 去甲肾上腺素+{strength:.2f}")
    
    def trigger_relax(self, strength: float = 0.3):
        """触发放松：血清素和GABA释放。"""
        self.global_release(Neurotransmitter.SEROTONIN, strength)
        self.global_release(Neurotransmitter.GABA, strength * 0.5)
        print(f"[化学] 放松... 血清素+{strength:.2f}")
    
    def get_state(self) -> Dict:
        """获取化学状态摘要。"""
        return {
            'concentrations': {
                nt.value: round(float(np.mean(self.concentrations[nt])), 2)
                for nt in Neurotransmitter
            },
            'modulation': self.get_modulation()
        }


class ChemicalEnvironment:
    """
    化学环境：整合化学信号与生命事件的完整系统。
    """
    
    def __init__(self, dim: int = 256):
        self.chemistry = ChemicalState(dim)
        self.event_history = []
        self.step_count = 0
        
        print("[化学环境] 初始化完成")
    
    def process_event(self, event_type: str, outcome: str, 
                     intensity: float = 0.5):
        """
        处理生命事件，触发相应的化学响应。
        
        事件 → 化学变化 → 行为倾向改变
        """
        self.step_count += 1
        
        if outcome == 'positive':
            if event_type in ['学习', '创造', '社交']:
                self.chemistry.trigger_reward(intensity)
            elif event_type == '休息':
                self.chemistry.trigger_relax(intensity)
        
        elif outcome == 'negative':
            if event_type in ['碰撞', '损伤', '威胁']:
                self.chemistry.trigger_stress(intensity)
            elif event_type == '失败':
                self.chemistry.global_release(
                    Neurotransmitter.SEROTONIN, -0.1  # 血清素降低
                )
        
        elif outcome == 'novel':
            # 新奇 → 多巴胺+去甲肾上腺素
            self.chemistry.global_release(Neurotransmitter.DOPAMINE, intensity * 0.3)
            self.chemistry.global_release(Neurotransmitter.NOREPINEPHRINE, intensity * 0.2)
        
        # 记录
        self.event_history.append({
            'step': self.step_count,
            'event': event_type,
            'outcome': outcome,
            'intensity': intensity
        })
    
    def step(self):
        """每步化学衰减。"""
        self.chemistry.decay()
    
    def get_chemical_mood(self) -> str:
        """
        获取当前化学"心情"。
        
        不是情绪，是化学环境的整体感受。
        """
        mod = self.chemistry.get_modulation()
        
        if mod['pleasure'] > 0.7:
            return "愉悦"
        elif mod['stress'] > 0.5:
            return "紧张"
        elif mod['calmness'] > 0.7:
            return "平静"
        elif mod['motivation'] > 0.7:
            return " motivated"
        elif mod['stress'] < -0.3:
            return "迟钝"
        else:
            return "一般"


# ================================================================
# 测试
# ================================================================

def test_chemical_signals():
    """测试化学信号系统。"""
    print("=" * 70)
    print("化学信号系统测试")
    print("=" * 70)
    
    chem = ChemicalEnvironment(dim=256)
    
    # 测试不同事件触发的化学响应
    print("\n[事件响应测试]")
    
    events = [
        ('学习', 'positive', 0.6),
        ('碰撞', 'negative', 0.8),
        ('社交', 'positive', 0.7),
        ('休息', 'positive', 0.4),
        ('威胁', 'negative', 0.9),
        ('探索', 'novel', 0.5),
    ]
    
    for event, outcome, intensity in events:
        print(f"\n  事件: {event} ({outcome}, 强度{intensity})")
        chem.process_event(event, outcome, intensity)
        state = chem.chemistry.get_state()
        print(f"  多巴胺: {state['concentrations']['多巴胺']:.2f}")
        print(f"  血清素: {state['concentrations']['血清素']:.2f}")
        print(f"  去甲肾上腺素: {state['concentrations']['去甲肾上腺素']:.2f}")
        print(f"  心情: {chem.get_chemical_mood()}")
        
        # 衰减一步
        chem.step()
    
    # 测试调制效应
    print("\n[调制效应]")
    mod = chem.chemistry.get_modulation()
    print(f"  兴奋增益: {mod['excitation_gain']}")
    print(f"  抑制增益: {mod['inhibition_gain']}")
    print(f"  学习率: {mod['learning_rate']}")
    print(f"  动机: {mod['motivation']}")
    print(f"  应激: {mod['stress']}")
    print(f"  愉悦: {mod['pleasure']}")
    print(f"  平静: {mod['calmness']}")
    
    print("\n" + "=" * 70)
    print("化学信号系统测试完成")
    print("=" * 70)
    
    return chem


if __name__ == "__main__":
    test_chemical_signals()
