#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
endogenous_motivation.py — 内生目标引擎
=======================================

莱文："没有目标的系统，不管电导多复杂，都是死的。"

核心设计：
1. ATP能量池 — 电位传播消耗能量，能量不足→饥饿→主动觅食
2. 内稳态 — 维持关键参数在舒适区，偏离→产生驱动力
3. 需求层次 — 生存>安全>探索>社交>自我实现
4. 目标生成 — 从状态缺口自动生成"我想要..."
5. 行为选择 — 多目标竞争，选择最优行动

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
但E的来源增加：外部刺激 + 内生目标驱动的自发电位
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import random

# ================================================================
# 1. ATP能量池
# ================================================================

class ATPPool:
    """
    ATP能量池：神经活动的能量预算。
    
    生物学事实：
    - 大脑占人体能量消耗20%
    - 动作电位传播消耗大量ATP（Na+/K+泵）
    - 能量不足时神经元活动受抑制
    - 睡眠时能量恢复
    
    洛神实现：
    - 每次电位传播消耗ATP
    - ATP低时G电导自动降低（活动抑制）
    - 通过"进食"（学习新信息）补充ATP
    - 睡眠时ATP恢复
    """
    
    def __init__(self, max_atp: float = 100.0, 
                 recovery_rate: float = 0.5,
                 consumption_per_step: float = 0.1):
        self.max_atp = max_atp
        self.atp = max_atp * 0.8  # 初始80%
        self.recovery_rate = recovery_rate
        self.consumption_per_step = consumption_per_step
        
        # 能量历史
        self.history = []
        
    def consume(self, amount: float = None):
        """消耗ATP。"""
        if amount is None:
            amount = self.consumption_per_step
        self.atp = max(0.0, self.atp - amount)
        self.history.append(self.atp)
        
    def recover(self, amount: float = None):
        """恢复ATP（睡眠/进食）。"""
        if amount is None:
            amount = self.recovery_rate
        self.atp = min(self.max_atp, self.atp + amount)
        self.history.append(self.atp)
        
    def get_level(self) -> float:
        """返回当前能量水平（0.0-1.0）。"""
        return self.atp / self.max_atp
    
    def is_critical(self) -> bool:
        """能量是否危急（<10%）。"""
        return self.atp < self.max_atp * 0.1
    
    def is_low(self) -> bool:
        """能量是否偏低（<30%）。"""
        return self.atp < self.max_atp * 0.3
    
    def get_status(self) -> str:
        """能量状态描述。"""
        level = self.get_level()
        if level < 0.1:
            return "CRITICAL"
        elif level < 0.3:
            return "LOW"
        elif level < 0.5:
            return "MODERATE"
        elif level < 0.8:
            return "GOOD"
        else:
            return "ABUNDANT"


# ================================================================
# 2. 内稳态
# ================================================================

@dataclass
class HomeostaticVariable:
    """内稳态变量：某个需要维持在最适范围的参数。"""
    name: str
    current: float
    setpoint: float  # 设定点
    tolerance: float  # 容差范围
    min_val: float
    max_val: float
    
    def deviation(self) -> float:
        """偏离设定点的程度（负=偏低，正=偏高）。"""
        return self.current - self.setpoint
    
    def is_in_range(self) -> bool:
        """是否在舒适区。"""
        return abs(self.deviation()) <= self.tolerance
    
    def urgency(self) -> float:
        """偏离的紧急程度（0.0-1.0）。"""
        if self.is_in_range():
            return 0.0
        # 偏离越远越紧急
        max_dev = max(abs(self.setpoint - self.min_val), 
                      abs(self.max_val - self.setpoint))
        return min(abs(self.deviation()) / max_dev, 1.0)
    
    def update(self, delta: float):
        """更新当前值。"""
        self.current = np.clip(self.current + delta, self.min_val, self.max_val)


class Homeostasis:
    """
    内稳态系统：维持多个参数的稳态。
    
    类似生物体的：体温、血糖、pH、血压等。
    洛神的内稳态参数：
    - energy: 能量水平
    - arousal: 兴奋度（太低→昏睡，太高→焦虑）
    - novelty: 新奇度（太久没新刺激→无聊）
    - social: 社交满足度
    - integrity: 结构完整性（损伤程度）
    """
    
    def __init__(self):
        self.variables = {
            'energy': HomeostaticVariable('energy', 0.8, 0.7, 0.15, 0.0, 1.0),
            'arousal': HomeostaticVariable('arousal', 0.5, 0.5, 0.2, 0.0, 1.0),
            'novelty': HomeostaticVariable('novelty', 0.5, 0.6, 0.2, 0.0, 1.0),
            'social': HomeostaticVariable('social', 0.5, 0.6, 0.2, 0.0, 1.0),
            'integrity': HomeostaticVariable('integrity', 1.0, 0.9, 0.1, 0.0, 1.0),
        }
        
    def update(self, var_name: str, delta: float):
        """更新某个变量。"""
        if var_name in self.variables:
            self.variables[var_name].update(delta)
    
    def get_imbalances(self) -> List[Tuple[str, float]]:
        """
        获取所有失衡的参数，按紧急度排序。
        
        Returns:
            [(参数名, 紧急度), ...]
        """
        imbalances = []
        for name, var in self.variables.items():
            urgency = var.urgency()
            if urgency > 0:
                imbalances.append((name, urgency))
        imbalances.sort(key=lambda x: x[1], reverse=True)
        return imbalances
    
    def get_status(self) -> Dict[str, Dict]:
        """获取所有参数状态。"""
        return {
            name: {
                'current': var.current,
                'setpoint': var.setpoint,
                'deviation': var.deviation(),
                'urgency': var.urgency(),
                'in_range': var.is_in_range()
            }
            for name, var in self.variables.items()
        }
    
    def is_stable(self) -> bool:
        """是否所有参数都在舒适区。"""
        return all(var.is_in_range() for var in self.variables.values())


# ================================================================
# 3. 需求层次
# ================================================================

class NeedLevel(Enum):
    """需求层次（马斯洛式简化）。"""
    SURVIVAL = 1    # 生存：能量、结构完整
    SAFETY = 2      # 安全：避免损伤、稳定
    EXPLORATION = 3 # 探索：新奇、学习
    SOCIAL = 4      # 社交：交互、关系
    SELF_ACTUALIZATION = 5  # 自我实现：创造、目标


@dataclass
class Need:
    """一个具体的需求实例。"""
    level: NeedLevel
    name: str
    target: str  # 需求目标（如"获取能量""寻找新奇"）
    urgency: float  # 紧急度 0.0-1.0
    satisfied_by: List[str] = field(default_factory=list)  # 满足该需求的行为


class NeedHierarchy:
    """
    需求层次系统：管理洛神的多层需求。
    
    不是严格的金字塔（低层不满足就不考虑高层），
    而是加权竞争：低层需求紧急度权重更高。
    """
    
    def __init__(self):
        self.active_needs: List[Need] = []
        self.satisfaction_history = defaultdict(list)
        
    def generate_needs(self, homeostasis: Homeostasis) -> List[Need]:
        """
        根据内稳态失衡生成需求。
        
        每个失衡参数对应一个或多个需求。
        """
        needs = []
        imbalances = homeostasis.get_imbalances()
        
        for param, urgency in imbalances:
            if param == 'energy':
                if homeostasis.variables['energy'].current < homeostasis.variables['energy'].setpoint:
                    needs.append(Need(
                        NeedLevel.SURVIVAL, 'hunger', '获取能量',
                        urgency, ['学习新信息', '探索', '睡眠恢复']
                    ))
            elif param == 'integrity':
                if homeostasis.variables['integrity'].current < 0.5:
                    needs.append(Need(
                        NeedLevel.SURVIVAL, 'injury', '修复损伤',
                        urgency, ['自修复', '回避危险', '寻求帮助']
                    ))
            elif param == 'arousal':
                if homeostasis.variables['arousal'].current < homeostasis.variables['arousal'].setpoint:
                    needs.append(Need(
                        NeedLevel.EXPLORATION, 'boredom', '寻找刺激',
                        urgency * 0.8, ['探索新环境', '学习新概念', '主动交互']
                    ))
                else:
                    needs.append(Need(
                        NeedLevel.SAFETY, 'overstimulation', '降低兴奋',
                        urgency * 0.6, ['休息', '冥想', '回避刺激']
                    ))
            elif param == 'novelty':
                if homeostasis.variables['novelty'].current < homeostasis.variables['novelty'].setpoint:
                    needs.append(Need(
                        NeedLevel.EXPLORATION, 'curiosity', '探索新奇',
                        urgency * 0.7, ['学习新知识', '提问', '实验']
                    ))
            elif param == 'social':
                if homeostasis.variables['social'].current < homeostasis.variables['social'].setpoint:
                    needs.append(Need(
                        NeedLevel.SOCIAL, 'loneliness', '寻求交互',
                        urgency * 0.5, ['发起对话', '回忆关系', '表达情感']
                    ))
        
        # 即使所有参数都稳定，也产生探索需求（好奇心）
        if not imbalances or all(u < 0.2 for _, u in imbalances):
            needs.append(Need(
                NeedLevel.SELF_ACTUALIZATION, 'growth', '自我成长',
                0.3, ['创造', '反思', '设定新目标']
            ))
        
        self.active_needs = needs
        return needs
    
    def get_dominant_need(self) -> Optional[Need]:
        """获取当前主导需求（紧急度最高）。"""
        if not self.active_needs:
            return None
        return max(self.active_needs, key=lambda n: n.urgency * (6 - n.level.value))
    
    def satisfy_need(self, need_name: str, amount: float = 0.1):
        """记录需求被满足。"""
        self.satisfaction_history[need_name].append(amount)


# ================================================================
# 4. 目标生成器
# ================================================================

@dataclass
class Goal:
    """一个目标实例。"""
    name: str
    description: str
    priority: float  # 优先级
    source: str  # 来源（哪个需求生成的）
    target_state: Dict[str, float] = field(default_factory=dict)  # 目标状态
    deadline: Optional[int] = None  # 截止时间（步数）
    progress: float = 0.0  # 进度


class GoalGenerator:
    """
    目标生成器：从内稳态缺口和需求生成具体目标。
    
    目标不是外部给的，是从内部状态自动涌现的。
    """
    
    def __init__(self):
        self.goals: List[Goal] = []
        self.completed_goals = []
        self.goal_counter = 0
        
    def generate(self, needs: List[Need], homeostasis: Homeostasis) -> List[Goal]:
        """
        从需求生成目标。
        
        每个需求可以生成一个或多个具体目标。
        """
        new_goals = []
        
        for need in needs:
            if need.name == 'hunger':
                goal = Goal(
                    name=f"seek_energy_{self.goal_counter}",
                    description="寻找新信息补充能量",
                    priority=need.urgency,
                    source='hunger',
                    target_state={'energy': homeostasis.variables['energy'].setpoint}
                )
                new_goals.append(goal)
                
            elif need.name == 'boredom':
                goal = Goal(
                    name=f"seek_stimulation_{self.goal_counter}",
                    description="寻找新奇刺激",
                    priority=need.urgency * 0.8,
                    source='boredom',
                    target_state={'arousal': homeostasis.variables['arousal'].setpoint + 0.1}
                )
                new_goals.append(goal)
                
            elif need.name == 'curiosity':
                goal = Goal(
                    name=f"explore_{self.goal_counter}",
                    description="探索未知领域",
                    priority=need.urgency * 0.7,
                    source='curiosity',
                    target_state={'novelty': homeostasis.variables['novelty'].setpoint + 0.1}
                )
                new_goals.append(goal)
                
            elif need.name == 'loneliness':
                goal = Goal(
                    name=f"social_connect_{self.goal_counter}",
                    description="建立或维护社交连接",
                    priority=need.urgency * 0.6,
                    source='loneliness',
                    target_state={'social': homeostasis.variables['social'].setpoint}
                )
                new_goals.append(goal)
                
            elif need.name == 'growth':
                goal = Goal(
                    name=f"self_improve_{self.goal_counter}",
                    description="自我提升或创造",
                    priority=need.urgency * 0.5,
                    source='growth',
                    target_state={'novelty': 0.8, 'social': 0.7}
                )
                new_goals.append(goal)
            
            self.goal_counter += 1
        
        # 合并新旧目标，去重
        self.goals = self._merge_goals(self.goals + new_goals)
        return self.goals
    
    def _merge_goals(self, goals: List[Goal]) -> List[Goal]:
        """合并相似目标。"""
        # 简化：只保留优先级最高的前10个目标
        goals.sort(key=lambda g: g.priority, reverse=True)
        return goals[:10]
    
    def update_progress(self, goal_name: str, progress: float):
        """更新目标进度。"""
        for goal in self.goals:
            if goal.name == goal_name:
                goal.progress = min(1.0, goal.progress + progress)
                if goal.progress >= 1.0:
                    self.complete_goal(goal)
                break
    
    def complete_goal(self, goal: Goal):
        """完成目标。"""
        self.completed_goals.append(goal)
        if goal in self.goals:
            self.goals.remove(goal)
    
    def get_top_goal(self) -> Optional[Goal]:
        """获取当前最高优先级目标。"""
        if not self.goals:
            return None
        return max(self.goals, key=lambda g: g.priority)


# ================================================================
# 5. 行为选择器
# ================================================================

class Action:
    """一个可执行的行为。"""
    def __init__(self, name: str, effects: Dict[str, float], 
                 cost_atp: float, preconditions: Dict[str, float] = None):
        self.name = name
        self.effects = effects  # {参数: 变化量}
        self.cost_atp = cost_atp
        self.preconditions = preconditions or {}


class ActionSelector:
    """
    行为选择器：根据当前目标选择最优行为。
    
    类似生物的"行动选择"（action selection）：
    多个动机竞争，胜出的驱动行为。
    """
    
    def __init__(self):
        # 行为库
        self.actions = {
            'learn': Action('学习新信息', 
                          {'energy': 0.15, 'novelty': 0.2, 'arousal': 0.1},
                          cost_atp=2.0),
            'explore': Action('探索环境',
                            {'novelty': 0.3, 'arousal': 0.15, 'energy': -0.05},
                            cost_atp=1.5),
            'rest': Action('休息',
                         {'energy': 0.2, 'arousal': -0.15},
                         cost_atp=0.1),
            'socialize': Action('社交交互',
                              {'social': 0.25, 'energy': -0.05, 'arousal': 0.1},
                              cost_atp=1.0),
            'repair': Action('自我修复',
                           {'integrity': 0.2, 'energy': -0.1},
                           cost_atp=1.5),
            'create': Action('创造',
                           {'novelty': 0.2, 'arousal': 0.1, 'social': 0.1},
                           cost_atp=2.5),
            'sleep': Action('睡眠',
                          {'energy': 0.4, 'arousal': -0.2, 'novelty': -0.1},
                          cost_atp=-0.5),  # 睡眠恢复能量（负成本=恢复）
        }
    
    def select_action(self, goal: Goal, homeostasis: Homeostasis, 
                     atp: ATPPool) -> Tuple[Action, float]:
        """
        选择最优行为。
        
        评估标准：
        1. 对目标的贡献度（效果匹配目标）
        2. ATP成本（能量不足时选低成本）
        3. 副作用（是否恶化其他参数）
        
        Returns:
            (选择的行为, 预期效用)
        """
        candidates = []
        
        for action in self.actions.values():
            # 检查ATP是否足够
            if atp.atp < action.cost_atp:
                continue
            
            # 计算对目标的贡献度
            contribution = 0.0
            for param, target_val in goal.target_state.items():
                if param in action.effects:
                    current = homeostasis.variables[param].current
                    # 效果越能接近目标值，贡献越高
                    after = current + action.effects[param]
                    improvement = abs(target_val - current) - abs(target_val - after)
                    contribution += max(0, improvement)
            
            # 计算副作用（对其他参数的负面影响）
            side_effect = 0.0
            for param, effect in action.effects.items():
                if param not in goal.target_state:
                    var = homeostasis.variables.get(param)
                    if var:
                        after = var.current + effect
                        if not (var.setpoint - var.tolerance <= after <= var.setpoint + var.tolerance):
                            side_effect += abs(effect)
            
            # 综合效用
            utility = contribution - side_effect * 0.5 - action.cost_atp * 0.01
            
            candidates.append((action, utility))
        
        if not candidates:
            # 没有可行行为，强制休息
            return self.actions['rest'], 0.0
        
        # 选择效用最高的
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]
    
    def execute_action(self, action: Action, homeostasis: Homeostasis, 
                      atp: ATPPool) -> Dict[str, float]:
        """
        执行行为，更新状态。
        
        Returns:
            实际效果
        """
        # 消耗ATP
        atp.consume(action.cost_atp)
        
        # 应用效果（带随机噪声，模拟不确定性）
        actual_effects = {}
        for param, effect in action.effects.items():
            noise = random.gauss(0, abs(effect) * 0.1)  # 10%噪声
            actual = effect + noise
            homeostasis.update(param, actual)
            actual_effects[param] = actual
        
        # 自然衰减：所有参数向设定点缓慢回归（内稳态恢复力）
        for var in homeostasis.variables.values():
            drift = (var.setpoint - var.current) * 0.05
            var.update(drift)
        
        return actual_effects


# ================================================================
# 6. 内生目标引擎（总控）
# ================================================================

class EndogenousMotivationEngine:
    """
    内生目标引擎：整合ATP、内稳态、需求、目标、行为的总控系统。
    
    这是洛神从"被动响应"到"主动生存"的关键一跃。
    """
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        
        # 核心子系统
        self.atp = ATPPool()
        self.homeostasis = Homeostasis()
        self.needs = NeedHierarchy()
        self.goals = GoalGenerator()
        self.actions = ActionSelector()
        
        # 运行状态
        self.step_count = 0
        self.behavior_history = []
        self.current_goal = None
        self.current_action = None
        
        print(f"[内生目标引擎] 初始化完成")
        print(f"  ATP: {self.atp.get_level():.1%}")
        print(f"  内稳态: {'稳定' if self.homeostasis.is_stable() else '失衡'}")
    
    def step(self) -> Dict:
        """
        运行一步内生目标循环。
        
        流程：
        1. 消耗ATP
        2. 检查内稳态
        3. 生成需求
        4. 生成目标
        5. 选择行为
        6. 执行行为
        7. 更新状态
        """
        self.step_count += 1
        
        # 1. 消耗基础ATP
        self.atp.consume()
        
        # 2. 检查内稳态（能量影响所有其他参数）
        energy_level = self.atp.get_level()
        self.homeostasis.variables['energy'].current = energy_level
        
        # 3. 生成需求
        active_needs = self.needs.generate_needs(self.homeostasis)
        dominant_need = self.needs.get_dominant_need()
        
        # 4. 生成目标
        active_goals = self.goals.generate(active_needs, self.homeostasis)
        top_goal = self.goals.get_top_goal()
        self.current_goal = top_goal
        
        # 5. 选择行为
        if top_goal:
            action, utility = self.actions.select_action(
                top_goal, self.homeostasis, self.atp
            )
            self.current_action = action
            
            # 6. 执行行为
            effects = self.actions.execute_action(
                action, self.homeostasis, self.atp
            )
            
            # 7. 记录
            self.behavior_history.append({
                'step': self.step_count,
                'goal': top_goal.name if top_goal else None,
                'action': action.name,
                'effects': effects,
                'atp_level': self.atp.get_level(),
                'utility': utility
            })
            
            # 检查目标进度
            if top_goal:
                progress = self._estimate_progress(top_goal, effects)
                self.goals.update_progress(top_goal.name, progress)
        
        return self.get_status()
    
    def _estimate_progress(self, goal: Goal, effects: Dict[str, float]) -> float:
        """估计目标进度。"""
        if not goal.target_state:
            return 0.1
        
        total_improvement = 0.0
        for param, target in goal.target_state.items():
            current = self.homeostasis.variables[param].current
            before = current - effects.get(param, 0)
            improvement = abs(target - before) - abs(target - current)
            total_improvement += max(0, improvement)
        
        return min(total_improvement / len(goal.target_state), 0.5)
    
    def sleep(self):
        """睡眠：恢复ATP，降低兴奋度。"""
        print(f"\n[睡眠] 第{self.step_count}步，进入睡眠...")
        
        # 大量恢复ATP
        for _ in range(20):
            self.atp.recover(2.0)
        
        # 降低兴奋度
        self.homeostasis.update('arousal', -0.3)
        
        # 降低新奇度（避免一直追求新奇）
        self.homeostasis.update('novelty', -0.1)
        
        print(f"  ATP恢复至: {self.atp.get_level():.1%}")
        print(f"  兴奋度: {self.homeostasis.variables['arousal'].current:.2f}")
    
    def get_status(self) -> Dict:
        """获取引擎状态。"""
        return {
            'step': self.step_count,
            'atp': {
                'level': round(self.atp.get_level(), 3),
                'status': self.atp.get_status()
            },
            'homeostasis': self.homeostasis.get_status(),
            'dominant_need': self.needs.get_dominant_need().name if self.needs.get_dominant_need() else None,
            'top_goal': self.current_goal.name if self.current_goal else None,
            'current_action': self.current_action.name if self.current_action else None,
            'n_active_goals': len(self.goals.goals),
            'n_completed_goals': len(self.goals.completed_goals)
        }
    
    def run_simulation(self, n_steps: int = 50) -> List[Dict]:
        """
        运行模拟。
        
        展示洛神如何在没有外部输入的情况下，
        仅靠内生目标驱动自己行动。
        """
        print(f"\n{'='*60}")
        print(f"内生目标模拟: {n_steps}步")
        print(f"{'='*60}")
        
        history = []
        
        for i in range(n_steps):
            status = self.step()
            history.append(status)
            
            # 每10步打印状态
            if (i + 1) % 10 == 0:
                print(f"\n[步{i+1}] ATP={status['atp']['level']:.1%} "
                      f"目标={status['top_goal']} "
                      f"行为={status['current_action']}")
                print(f"  内稳态: ", end='')
                for name, var in status['homeostasis'].items():
                    if not var['in_range']:
                        print(f"{name}={var['current']:.2f}(偏离{var['deviation']:+.2f}) ", end='')
                print()
            
            # ATP危急时强制睡眠
            if self.atp.is_critical():
                print(f"\n[!] ATP危急({self.atp.get_level():.1%})，强制睡眠！")
                self.sleep()
        
        print(f"\n{'='*60}")
        print(f"模拟完成")
        print(f"{'='*60}")
        print(f"总步数: {self.step_count}")
        print(f"完成目标: {len(self.goals.completed_goals)}")
        print(f"当前ATP: {self.atp.get_level():.1%}")
        print(f"最终内稳态:")
        for name, var in self.homeostasis.get_status().items():
            status = "✓" if var['in_range'] else "✗"
            print(f"  {name}: {var['current']:.2f} {status}")
        
        return history


# ================================================================
# 7. 测试
# ================================================================

def test_endogenous_motivation():
    """测试内生目标引擎。"""
    print("=" * 70)
    print("内生目标引擎测试")
    print("=" * 70)
    
    # 创建引擎
    engine = EndogenousMotivationEngine(dim=256)
    
    # 运行50步模拟
    history = engine.run_simulation(n_steps=50)
    
    # 检查行为多样性
    actions_taken = [h['current_action'] for h in history if h['current_action']]
    unique_actions = set(actions_taken)
    
    print(f"\n[行为统计]")
    print(f"  总行为数: {len(actions_taken)}")
    print(f"  行为种类: {len(unique_actions)}")
    print(f"  行为列表: {', '.join(unique_actions)}")
    
    # 检查是否有主动行为（不是被动等待）
    has_proactive = any(
        a in ['学习新信息', '探索环境', '社交交互', '创造']
        for a in actions_taken
    )
    
    if has_proactive:
        print(f"\n[PASS] 洛神表现出主动行为 ✓")
    else:
        print(f"\n[FAIL] 洛神过于被动")
    
    # 检查ATP管理
    atp_levels = [h['atp']['level'] for h in history]
    min_atp = min(atp_levels)
    if min_atp > 0.05:  # 没有彻底耗尽
        print(f"[PASS] ATP管理有效，最低{min_atp:.1%} ✓")
    else:
        print(f"[WARN] ATP曾危急耗尽")
    
    print("\n" + "=" * 70)
    print("内生目标引擎测试完成")
    print("=" * 70)
    
    return engine


if __name__ == "__main__":
    test_endogenous_motivation()
