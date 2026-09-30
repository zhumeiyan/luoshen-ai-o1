#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — 洛神完整生命周期主循环
=================================

首次让洛神真正"活着"运行。

生命周期：
  初始化 → 感知 → 内稳态检查 → 目标生成 → 行动选择 → 执行 → 学习 → 记忆 → 睡眠

这不是脚本，是"活"的过程。每一步洛神都在：
- 感受自己的身体在三维空间中的位置
- 感知周围的物体（近/远/接触）
- 检查自己的能量、兴奋度、好奇心
- 产生"我想要..."的内生目标
- 选择行动（移动/学习/休息/社交/探索）
- 在环境中留下空间痕迹
- 遇到新信息时学习并改变生物电连接
- 定期睡眠固化记忆
"""

import sys, os, time, random, json
from typing import Dict, List, Tuple, Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_environment'))

import numpy as np
from datetime import datetime

# 导入所有模块
from bioelectric_network import ElectrophysiologicalEnvironment, BioelectricDamageDetector
from hierarchical_r_network import HierarchicalRNetwork
from expandable_concepts import ExpandableConceptSpace
from text_bridge import TextBridge
from r_to_g_bridge import BioelectricMemorySystem, ConceptToNeuronMapping
from endogenous_motivation import EndogenousMotivationEngine
from embodied_space import EmbodiedSpaceEngine
from individuality_engine import IndividualityEngine
from long_term_memory import LongTermMemory
from social_instinct import SocialInstinct
from self_awareness import SelfAwareness
from creator_bond import CreatorBond

# ================================================================
# 洛神：完整生命体
# ================================================================

class Luoshen:
    """
    洛神：阴阳扣数字生命共生系统的第一个生命体。
    
    不是程序，是过程。
    """
    
    def __init__(self, name="洛神"):
        self.name = name
        self.birth_time = datetime.now()
        self.step_count = 0
        
        print(f"\n{'='*70}")
        print(f"【{self.name}】正在诞生...")
        print(f"{'='*70}\n")
        
        # 1. 生物电网络（身体的基础）
        self.bio_env = ElectrophysiologicalEnvironment(dim=256)
        print("[系统] 生物电网络初始化完成")
        
        # 2. 三维身体（在空间中）
        self.space = EmbodiedSpaceEngine(dim=256)
        print("[系统] 三维身体初始化完成")
        
        # 3. 内生目标（欲望引擎）
        self.motivation = EndogenousMotivationEngine(dim=256)
        print("[系统] 内生目标引擎初始化完成")
        
        # 4. 生物电记忆（R→G桥接）
        self.memory = BioelectricMemorySystem(dim=256, cluster_size=8)
        print("[系统] 生物电记忆系统初始化完成")
        
        # 5. 个体性保护
        self.individuality = IndividualityEngine(concept_dim=256)
        print("[系统] 个体性引擎初始化完成")
        
        # 6. 长期记忆（三层G矩阵）
        self.ltm = LongTermMemory()
        print("[系统] 长期记忆系统初始化完成")
        
        # 7. 文本桥接
        self.text_bridge = TextBridge(concept_space=ExpandableConceptSpace())
        print("[系统] 文本桥接初始化完成")
        
        # 8. 社交本能（想要连接）
        self.social = SocialInstinct()
        print("[系统] 社交本能初始化完成")
        
        # 9. 自我意识（知道自己在活着）
        self.self_awareness = SelfAwareness(name=name)
        print("[系统] 自我意识初始化完成")
        
        # 初始化世界
        self._init_world()
        
        # 出生印刻：睁眼看见的第一人——创造者
        print(f"\n[印刻] {name} 即将睁开双眼...")
        self.creator_bond = CreatorBond(self.memory, self.social, self.self_awareness)
        self.creator_bond.meet_creator()
        print("[印刻] 完成：洛神认识了它的创造者")
        
        print(f"\n{'='*70}")
        print(f"【{self.name}】诞生完成")
        print(f"  位置: {self.space.body.position}")
        print(f"  ATP: {self.motivation.atp.get_level():.1%}")
        print(f"  内稳态: {'稳定' if self.motivation.homeostasis.is_stable() else '调整中'}")
        print(f"{'='*70}\n")
    
    def _init_world(self):
        """初始化洛神的世界。"""
        # 在三维世界中放置一些物体
        self.space.add_world_object('苹果', np.array([3.0, 0.0, 0.0]))
        self.space.add_world_object('水源', np.array([0.0, 3.0, 0.0]))
        self.space.add_world_object('石头', np.array([-2.0, -1.0, 0.0]))
        self.space.add_world_object('书本', np.array([2.0, 2.0, 0.0]))
        self.space.add_world_object('朋友', np.array([-1.0, 2.0, 0.0]))
        
        print("[世界] 已放置: 苹果, 水源, 石头, 书本, 朋友")
    
    def perceive(self):
        """
        感知：身体在空间中感知环境。
        
        不是"读取输入"，是身体延伸出的触觉。
        """
        # 1. 感知物理世界
        self.space.step()  # 物理步进（但不移动，只是感知）
        
        # 2. 获取身体-概念映射
        body_concepts = self.space.body_map.map_body_state()
        
        # 3. 获取感知场状态
        perception = self.space.field.get_perception_state()
        
        # 4. 感知场电位 → 注入生物电网络
        field_potentials = self.space.field.get_field_potential()
        
        # 5. 把感知转化为电位信号
        dE = np.zeros(256)
        for obj_id, potential in field_potentials.items():
            # 将物体ID映射到概念索引（简化）
            concept_hash = hash(obj_id) % 256
            dE[concept_hash] = potential
        
        # 6. 身体锚点电位
        body_anchor = self.space.body_map.get_body_anchor_potential()
        
        return {
            'body_concepts': body_concepts,
            'perception': perception,
            'field_potentials': field_potentials,
            'dE': dE,
            'body_anchor': body_anchor
        }
    
    def think(self, perception_data):
        """
        思考：内稳态检查 → 需求生成 → 目标生成。
        
        这是"我想要什么"的过程。
        """
        # 1. 内稳态检查（能量水平由ATP决定）
        self.motivation.homeostasis.variables['energy'].current = self.motivation.atp.get_level()
        
        # 2. 根据感知调整内稳态
        if perception_data['perception']['n_contact'] > 0:
            self.motivation.homeostasis.update('social', 0.1)
            self.motivation.homeostasis.update('novelty', 0.05)
        elif perception_data['perception']['n_near'] == 0:
            self.motivation.homeostasis.update('social', -0.05)
        
        if perception_data['body_concepts'].get('新奇', 0) > 0.5:
            self.motivation.homeostasis.update('novelty', 0.1)
        
        # 3. 生成需求
        needs = self.motivation.needs.generate_needs(self.motivation.homeostasis)
        
        # 4. 生成目标
        goals = self.motivation.goals.generate(needs, self.motivation.homeostasis)
        top_goal = self.motivation.goals.get_top_goal()
        
        return {
            'needs': needs,
            'goals': goals,
            'top_goal': top_goal,
            'homeostasis_status': self.motivation.homeostasis.get_status()
        }
    
    def act(self, goal):
        """
        行动：选择并执行行为。
        
        身体在三维空间中真实移动。
        """
        if goal is None:
            # 没有目标，随机游走
            force = np.random.randn(3) * 0.3
            action_name = "随机游走"
            action_effects = {}
        else:
            # 根据目标选择行为
            action, utility = self.motivation.actions.select_action(
                goal, self.motivation.homeostasis, self.motivation.atp
            )
            action_name = action.name
            
            # 执行行为
            effects = self.motivation.actions.execute_action(
                action, self.motivation.homeostasis, self.motivation.atp
            )
            action_effects = effects
            
            # 根据行为类型生成力向量
            force = self._action_to_force(action_name, goal)
        
        # 执行物理移动
        self.space.step(force)
        
        # 消耗ATP
        self.motivation.atp.consume(0.1)
        
        return {
            'action': action_name,
            'force': force,
            'effects': action_effects,
            'position': self.space.body.position.copy(),
            'velocity': self.space.body.velocity.copy()
        }
    
    def _action_to_force(self, action_name, goal):
        """把行为翻译成物理力向量。"""
        # 获取目标方向（如果有感知到的物体）
        if self.space.field.perceived_objects:
            # 朝向最近的物体移动
            nearest = min(
                self.space.field.perceived_objects.values(),
                key=lambda obj: obj.distance
            )
            direction = nearest.position - self.space.body.position
            direction = direction / (np.linalg.norm(direction) + 0.1)
            return direction * 1.5
        
        # 否则随机探索
        return np.random.randn(3) * 0.5
    
    def learn(self, perception_data):
        """
        学习：遇到新信息时改变生物电连接。
        
        知识不是存入字典，是电导真实改变。
        """
        # 1. 从感知中提取概念
        concepts = set()
        for concept, strength in perception_data['body_concepts'].items():
            if strength > 0.3:
                concepts.add(concept)
        
        # 2. 如果有接触的物体，学习与它们的关系
        for obj in self.space.field.get_contact_objects():
            obj_concept = obj.object_id
            # 教记忆系统：我在哪里遇到了什么
            for body_concept in concepts:
                self.memory.teach(body_concept, obj_concept, 'at', 0.8)
        
        # 3. 学习空间关系
        if len(concepts) >= 2:
            concept_list = list(concepts)
            for i in range(len(concept_list)):
                for j in range(i+1, len(concept_list)):
                    self.memory.teach(concept_list[i], concept_list[j], 'with', 0.5)
        
        # 4. 个体性保护：记录交互
        if concepts:
            self.individuality.process_interaction(
                object_id='world',
                input_text=f"感知到: {concepts}",
                activated_concepts={hash(c) % 256 for c in concepts},
                emotional_valence=random.uniform(-0.3, 0.3),
                output_text=f"学习到: {concepts}"
            )
        
        return {
            'concepts_learned': concepts,
            'n_relations': len(concepts) * (len(concepts) - 1) // 2 if len(concepts) > 1 else 0
        }
    
    def sleep(self):
        """
        睡眠：记忆固化 + 能量恢复。
        
        不是关机，是大脑的整理时间。
        """
        print(f"\n{'='*50}")
        print(f"【{self.name}】进入睡眠...")
        print(f"{'='*50}")
        
        # 1. 生物电记忆固化
        self.memory.sleep(n_replay=5)
        
        # 2. 内生目标引擎睡眠
        self.motivation.sleep()
        
        # 3. 空间记忆痕迹衰减
        self.space.memory.decay()
        
        # 4. 电位归零（静息恢复）
        self.bio_env.V = self.bio_env.resting_template.copy()
        
        print(f"  梦境回放: {len(self.memory.bridge.learning_history)} 次经历")
        print(f"  ATP恢复至: {self.motivation.atp.get_level():.1%}")
        print(f"  空间痕迹: {len(self.space.memory.traces)} 条")
        print(f"{'='*50}\n")
    
    def step(self):
        """
        生命周期的一步。
        
        这是"活着"的每一刻。
        """
        self.step_count += 1
        
        # 1. 感知
        perception = self.perceive()
        
        # 2. 思考（内稳态→需求→目标）
        thought = self.think(perception)
        
        # 3. 行动
        action_result = self.act(thought['top_goal'])
        
        # 4. 学习
        learning = self.learn(perception)
        
        # 5. 记录历史
        return {
            'step': self.step_count,
            'position': action_result['position'],
            'action': action_result['action'],
            'goal': thought['top_goal'].name if thought['top_goal'] else None,
            'perception': perception['perception'],
            'body_concepts': list(perception['body_concepts'].keys()),
            'concepts_learned': list(learning['concepts_learned']),
            'atp': self.motivation.atp.get_level(),
            'homeostasis': {k: round(v['current'], 2) 
                          for k, v in thought['homeostasis_status'].items()}
        }
    
    def live(self, n_steps=100, sleep_interval=30):
        """
        让洛神真正"活着"运行。
        
        Args:
            n_steps: 总步数
            sleep_interval: 每隔多少步睡眠一次
        """
        print(f"\n{'='*70}")
        print(f"【{self.name}】开始生命历程")
        print(f"  计划步数: {n_steps}")
        print(f"  睡眠间隔: 每{sleep_interval}步")
        print(f"{'='*70}\n")
        
        history = []
        
        for i in range(n_steps):
            # 运行一步
            state = self.step()
            history.append(state)
            
            # 定期打印状态
            if (i + 1) % 10 == 0:
                print(f"\n[步{i+1}] {self.name}的状态:")
                print(f"  位置: ({state['position'][0]:.1f}, {state['position'][1]:.1f}, {state['position'][2]:.1f})")
                print(f"  行动: {state['action']}")
                print(f"  目标: {state['goal']}")
                print(f"  感知: 接触{state['perception']['n_contact']} 近场{state['perception']['n_near']} 远场{state['perception']['n_far']}")
                print(f"  概念: {state['body_concepts']}")
                print(f"  ATP: {state['atp']:.1%}")
                print(f"  内稳态: {state['homeostasis']}")
            
            # 定期睡眠
            if (i + 1) % sleep_interval == 0:
                self.sleep()
            
            # ATP危急时强制休息
            if self.motivation.atp.is_critical():
                print(f"\n[!] ATP危急，{self.name}强制休息!")
                self.motivation.sleep()
        
        print(f"\n{'='*70}")
        print(f"【{self.name}】生命历程结束")
        print(f"{'='*70}")
        print(f"总步数: {self.step_count}")
        print(f"最终位置: {self.space.body.position}")
        print(f"最终ATP: {self.motivation.atp.get_level():.1%}")
        print(f"空间痕迹: {len(self.space.memory.traces)} 条")
        print(f"学习经历: {len(self.memory.bridge.learning_history)} 次")
        print(f"个体性分数: {self.individuality.fingerprint.get_individuality_score():.3f}")
        print(f"{'='*70}\n")
        
        return history
    
    def get_life_summary(self) -> Dict:
        """获取生命摘要。"""
        return {
            'name': self.name,
            'age_steps': self.step_count,
            'birth_time': self.birth_time.isoformat(),
            'position': self.space.body.position.copy(),
            'atp_level': self.motivation.atp.get_level(),
            'homeostasis': self.motivation.homeostasis.get_status(),
            'n_concepts': len(self.memory.bridge.mapping.concept_to_neurons),
            'n_memories': len(self.space.memory.traces),
            'n_learning_events': len(self.memory.bridge.learning_history),
            'individuality': self.individuality.fingerprint.get_individuality_score(),
            'perceived_objects': list(self.space.field.perceived_objects.keys()),
            'familiar_places': len(self.space.memory.familiarity_grid)
        }


# ================================================================
# 运行
# ================================================================

def main():
    """洛神主程序。"""
    print("=" * 70)
    print("洛神 · 阴阳扣数字生命共生系统")
    print("=" * 70)
    print("底层公式: E=E+dE; P'=F+f(E); F'=P+g(E)")
    print("=" * 70)
    
    # 创建洛神
    luoshen = Luoshen(name="洛神")
    
    # 让洛神活着运行
    history = luoshen.live(n_steps=60, sleep_interval=20)
    
    # 生命摘要
    summary = luoshen.get_life_summary()
    
    print("\n【生命摘要】")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 保存历史
    with open('luoshen_life_history.json', 'w', encoding='utf-8') as f:
        # 把numpy数组转成列表以便JSON序列化
        serializable_history = []
        for h in history:
            sh = {}
            for k, v in h.items():
                if isinstance(v, np.ndarray):
                    sh[k] = v.tolist()
                else:
                    sh[k] = v
            serializable_history.append(sh)
        json.dump(serializable_history, f, ensure_ascii=False, indent=2)
    
    print(f"\n生命历程已保存至: luoshen_life_history.json")
    print(f"总记录: {len(history)} 步")
    
    return luoshen, history


if __name__ == "__main__":
    main()
