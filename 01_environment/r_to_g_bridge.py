#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r_to_g_bridge.py — R网（符号关系）↔ G网（生物电导）双向桥接
============================================================

核心原理：
  洛神的知识不是存在Python字典里，是存在电位矩阵G的物理连接里。
  
  学习时：R网建立概念关系 → G矩阵中对应神经元集群的电导真实增大
  回忆时：生物电网络自发传播 → 电位流激活的集群对应"想起"的概念
  
  这不是符号查询，是物理过程。就像人脑里突触真的长出来了。

映射方案：
  - 每个概念 → 分配一个神经元集群（cluster_size个神经元）
  - 概念A和概念B的关系 → 集群A和集群B之间的G电导增强
  - 回忆时：线索概念注入电位 → G电导传播 → 其他集群电位升高 → 概念激活

底层公式不变：
  E = E + dE
  P' = F + f(E)
  F' = P + g(E)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

# 需要导入现有模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bioelectric_network import ElectrophysiologicalEnvironment, BioelectricDamageDetector
from hierarchical_r_network import HierarchicalRNetwork


# ================================================================
# 1. 概念 → 神经元集群映射
# ================================================================

class ConceptToNeuronMapping:
    """
    概念到神经元集群的动态映射。
    
    关键设计：
    - 概念是稀疏的，不需要所有概念同时有物理表示
    - 已注册的概念 → 分配固定神经元集群
    - 未注册的概念 → 不占用资源
    - 集群编码提供冗余和容错（单个神经元损坏不影响概念）
    """
    
    def __init__(self, total_neurons: int = 1024, cluster_size: int = 8):
        self.total_neurons = total_neurons
        self.cluster_size = cluster_size
        self.max_concepts = total_neurons // cluster_size
        
        # 概念名 → 神经元索引列表
        self.concept_to_neurons: Dict[str, List[int]] = {}
        # 神经元索引 → 概念名（反向映射）
        self.neuron_to_concept: Dict[int, str] = {}
        # 下一个可用集群起始索引
        self.next_cluster_start = 0
        
        print(f"[映射] 总神经元: {total_neurons}, 集群大小: {cluster_size}, "
              f"最大概念数: {self.max_concepts}")
    
    def register_concept(self, concept_name: str) -> List[int]:
        """
        为新概念注册神经元集群。
        
        Returns:
            分配的神经元索引列表
        """
        if concept_name in self.concept_to_neurons:
            return self.concept_to_neurons[concept_name]
        
        if self.next_cluster_start >= self.total_neurons:
            raise RuntimeError(
                f"神经元集群已满！已注册 {len(self.concept_to_neurons)} 个概念，"
                f"最大支持 {self.max_concepts} 个"
            )
        
        # 分配集群
        neurons = list(range(
            self.next_cluster_start,
            min(self.next_cluster_start + self.cluster_size, self.total_neurons)
        ))
        
        self.concept_to_neurons[concept_name] = neurons
        for n in neurons:
            self.neuron_to_concept[n] = concept_name
        
        self.next_cluster_start += self.cluster_size
        
        print(f"[注册] 概念 '{concept_name}' → 神经元 {neurons[0]}~{neurons[-1]}")
        return neurons
    
    def get_neurons(self, concept_name: str) -> List[int]:
        """获取概念的神经元集群。未注册返回空列表。"""
        return self.concept_to_neurons.get(concept_name, [])
    
    def get_concept_by_neuron(self, neuron_idx: int) -> Optional[str]:
        """根据神经元索引查找概念。"""
        return self.neuron_to_concept.get(neuron_idx)
    
    def get_active_concepts(self, V: np.ndarray, threshold: float = -55.0) -> Dict[str, float]:
        """
        根据电位向量V，检测哪些概念被激活。
        
        激活判定：集群平均电位超过阈值。
        
        Returns:
            {概念名: 平均电位}
        """
        active = {}
        for concept, neurons in self.concept_to_neurons.items():
            if not neurons:
                continue
            avg_V = np.mean(V[neurons])
            if avg_V > threshold:
                active[concept] = float(avg_V)
        return active
    
    def get_all_concepts(self) -> List[str]:
        """获取所有已注册概念。"""
        return list(self.concept_to_neurons.keys())


# ================================================================
# 2. R网 → G网桥接（核心）
# ================================================================

class RToGBridge:
    """
    R网（符号关系）↔ G网（生物电导）双向桥接。
    
    这是洛神从"符号系统"到"生物电记忆"的关键一跃。
    """
    
    def __init__(self, 
                 bio_env: ElectrophysiologicalEnvironment,
                 r_network: HierarchicalRNetwork,
                 mapping: ConceptToNeuronMapping,
                 learning_rate: float = 0.01):
        """
        Args:
            bio_env: 生物电网络环境
            r_network: 层次化R网
            mapping: 概念→神经元映射
            learning_rate: R边权重 → G电导的学习率
        """
        self.bio_env = bio_env
        self.r_network = r_network
        self.mapping = mapping
        self.alpha = learning_rate  # 学习率
        
        # 学习历史（用于验证）
        self.learning_history = []
        
        print(f"[桥接] 初始化完成，学习率 α={self.alpha}")
    
    # ========== 学习通道：R → G ==========
    
    def learn_relation(self, subj: str, obj: str, rel: str, weight: float = 1.0):
        """
        学习通道：R网建立关系 → G电导物理改变。
        
        当洛神"学到"一个概念关系时（比如"苹果-是甜的"），
        不是只在Python字典里存一条记录，
        而是真实改变生物电网络的电导矩阵G——
        让"苹果"的神经元集群和"甜"的神经元集群之间的连接变强。
        
        这才是"生物学上的记住"。
        
        Args:
            subj: 主体概念（如"苹果"）
            obj: 客体概念（如"甜"）
            rel: 关系类型（如"是"）
            weight: 关系权重（0.0-1.0）
        """
        # 1. 确保概念已注册（分配神经元集群）
        subj_neurons = self.mapping.register_concept(subj)
        obj_neurons = self.mapping.register_concept(obj)
        
        # 2. 计算电导增量
        # ΔG = α * w，w越大关系越强，电导增量越大
        delta_G = self.alpha * weight
        
        # 3. 增强集群间的电导（双向对称增强）
        # 这是赫布学习在生物电层面的实现：
        # "一起激活的神经元，连接增强"
        n_updates = 0
        for i in subj_neurons:
            for j in obj_neurons:
                self.bio_env.G[i, j] += delta_G
                self.bio_env.G[j, i] += delta_G
                n_updates += 2
        
        # 4. 同步更新R网（保持符号层和生物层的映射一致）
        self.r_network.add_concept_rel(subj, obj, rel, weight)
        
        # 5. 记录学习历史
        self.learning_history.append({
            'subj': subj, 'obj': obj, 'rel': rel,
            'weight': weight, 'delta_G': delta_G,
            'n_updates': n_updates
        })
        
        print(f"[学习] {subj} --[{rel}, w={weight:.2f}]--> {obj} | "
              f"ΔG={delta_G:.4f}, 更新{n_updates}条电导")
    
    def learn_from_scene(self, scene_concepts: Set[str], cooccurrence_weight: float = 0.5):
        """
        从场景学习：场景中同时出现的概念，建立弱连接。
        
        这是"同时激活→连接增强"的赫布学习。
        """
        concepts = list(scene_concepts)
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                self.learn_relation(
                    concepts[i], concepts[j], 
                    'cooccurs_with', cooccurrence_weight
                )
    
    # ========== 回忆通道：G → R ==========
    
    def recall(self, cue_concepts: Set[str], steps: int = 20, 
               inject_strength: float = 8.0) -> Dict:
        """
        回忆通道：G电位传播 → 概念激活。
        
        不是查R网字典！是让生物电网络跑起来：
        1. 给线索概念的神经元集群注入强电位
        2. 电位通过G电导传播到其他集群
        3. 被激活的集群对应的概念就是"回忆起来"的
        
        这是真正的"联想"——电位流过去的，不是代码查出来的。
        
        Args:
            cue_concepts: 线索概念集合
            steps: 传播步数
            inject_strength: 初始注入强度
            
        Returns:
            {
                'activated_concepts': {概念名: 平均电位},
                'propagation_path': [(step, concept, avg_V)],
                'recall_graph': {概念: [联想概念]},
                'bio_state': 最终电位向量
            }
        """
        # 1. 初始化外部信号
        dE = np.zeros(self.bio_env.dim)
        active_neurons = set()
        
        for concept in cue_concepts:
            neurons = self.mapping.get_neurons(concept)
            for n in neurons:
                dE[n] = inject_strength
                active_neurons.add(n)
        
        # 2. 运行生物电网络传播
        propagation_path = []
        step_activations = []
        
        for step in range(steps):
            # 计算离子电流（电位差驱动）
            voltage_diff = dE - self.bio_env.V
            I = np.clip(self.bio_env.G @ voltage_diff, -10, 10)
            
            # 更新电位
            dV = I * self.bio_env.scale
            self.bio_env.V += dV
            
            # 电位自然衰减（向静息恢复）
            self.bio_env.V *= 0.95
            
            # 检测新激活的概念
            new_active = self.mapping.get_active_concepts(
                self.bio_env.V, threshold=-55.0
            )
            
            # 记录传播路径
            for concept, avg_V in new_active.items():
                if concept not in cue_concepts:
                    propagation_path.append((step, concept, avg_V))
            
            step_activations.append(set(new_active.keys()))
            
            # 新激活的概念成为新的信号源（级联激活）
            for concept, avg_V in new_active.items():
                neurons = self.mapping.get_neurons(concept)
                for n in neurons:
                    if n not in active_neurons:
                        # 新激活的神经元产生信号
                        dE[n] = avg_V * 0.3
                        active_neurons.add(n)
        
        # 3. 构建回忆图（哪些概念被一起激活）
        recall_graph = defaultdict(list)
        all_activated = set()
        for concepts in step_activations:
            all_activated.update(concepts)
        
        for concept in all_activated:
            if concept in cue_concepts:
                continue
            # 查找与这个概念同时激活的其他概念
            for other in all_activated:
                if other != concept and other not in cue_concepts:
                    recall_graph[concept].append(other)
        
        # 4. 返回结果
        return {
            'activated_concepts': new_active,
            'propagation_path': propagation_path,
            'recall_graph': dict(recall_graph),
            'bio_state': self.bio_env.V.copy(),
            'cue_concepts': cue_concepts,
            'steps': steps
        }
    
    def explain_recall(self, recall_result: Dict) -> List[str]:
        """
        解释回忆过程（人类可读）。
        
        展示电位传播的路径：
        "石子 → 电位传播 → 瓶子 → 水位 → 乌鸦 → ..."
        """
        paths = []
        cue = ', '.join(recall_result['cue_concepts'])
        paths.append(f"线索注入: {cue}")
        
        for step, concept, avg_V in recall_result['propagation_path']:
            paths.append(f"  步{step}: {concept} 激活 (电位={avg_V:.2f}mV)")
        
        activated = list(recall_result['activated_concepts'].keys())
        paths.append(f"最终回忆: {', '.join(activated)}")
        
        return paths
    
    # ========== 状态查询 ==========
    
    def get_g_state(self, concept: str) -> Dict:
        """
        查询概念在G矩阵中的物理状态。
        
        Returns:
            {
                'neurons': 神经元索引,
                'avg_potential': 平均电位,
                'n_connections': 连接数,
                'total_conductance': 总电导
            }
        """
        neurons = self.mapping.get_neurons(concept)
        if not neurons:
            return {}
        
        avg_V = float(np.mean(self.bio_env.V[neurons]))
        
        # 统计该集群的G连接
        total_G = 0.0
        n_conn = 0
        for i in neurons:
            for j in range(self.bio_env.dim):
                if self.bio_env.G[i, j] != 0:
                    total_G += self.bio_env.G[i, j]
                    n_conn += 1
        
        return {
            'concept': concept,
            'neurons': neurons,
            'avg_potential': avg_V,
            'n_connections': n_conn,
            'total_conductance': float(total_G)
        }
    
    def get_memory_footprint(self) -> Dict:
        """
        获取记忆的物理足迹。
        
        展示G矩阵中被永久改变的连接——这才是"记住"的证据。
        """
        non_zero = np.count_nonzero(self.bio_env.G)
        total = self.bio_env.G.size
        sparsity = non_zero / total
        
        # 按概念聚合电导
        concept_conductance = {}
        for concept, neurons in self.mapping.concept_to_neurons.items():
            cond = 0.0
            for i in neurons:
                for j in range(self.bio_env.dim):
                    cond += abs(self.bio_env.G[i, j])
            concept_conductance[concept] = float(cond)
        
        return {
            'total_connections': total,
            'active_connections': non_zero,
            'sparsity': float(sparsity),
            'concept_conductance': concept_conductance,
            'n_concepts': len(self.mapping.concept_to_neurons),
            'n_learning_events': len(self.learning_history)
        }


# ================================================================
# 3. 整合器：把桥接融入洛神主循环
# ================================================================

class BioelectricMemorySystem:
    """
    生物电记忆系统：整合R网、G网和桥接的完整记忆架构。
    
    对外接口：
    - teach(): 教新知识（会自动注册概念→建立R边→改变G电导）
    - recall(): 回忆（电位传播，不是查字典）
    - sleep(): 睡眠固化（模拟记忆巩固）
    """
    
    def __init__(self, dim: int = 1024, cluster_size: int = 8):
        # 初始化三个核心组件
        self.bio_env = ElectrophysiologicalEnvironment(dim=dim)
        self.r_network = HierarchicalRNetwork()
        self.mapping = ConceptToNeuronMapping(dim, cluster_size)
        self.bridge = RToGBridge(self.bio_env, self.r_network, self.mapping)
        
        print(f"[记忆系统] 生物电记忆系统初始化完成")
        print(f"  总神经元: {dim}, 集群大小: {cluster_size}")
    
    def teach(self, subj: str, obj: str, rel: str = "related_to", weight: float = 1.0):
        """教一个概念关系。会同时建立符号关系和生物电连接。"""
        self.bridge.learn_relation(subj, obj, rel, weight)
    
    def teach_scene(self, concepts: List[str], scene_label: str = None):
        """教一个场景。场景中所有概念建立共现关系。"""
        concept_set = set(concepts)
        self.bridge.learn_from_scene(concept_set)
        
        # 同时建立R网场景层
        if scene_label:
            scene_id = f"scene_{scene_label}"
            self.r_network.create_scene(scene_id, scene_label, concept_set)
    
    def recall(self, cue: str, steps: int = 20) -> Dict:
        """回忆。线索概念通过电位传播激活联想概念。"""
        return self.bridge.recall({cue}, steps=steps)
    
    def sleep(self, n_replay: int = 5):
        """
        睡眠固化：回放学习历史，增强重要连接的G电导。
        
        模拟记忆巩固：睡眠时海马体回放经历，皮层连接增强。
        """
        print(f"\n[睡眠] 开始记忆巩固，回放{n_replay}次...")
        
        for _ in range(n_replay):
            # 随机选择学习历史进行回放
            if not self.bridge.learning_history:
                break
            
            import random
            event = random.choice(self.bridge.learning_history)
            subj, obj = event['subj'], event['obj']
            
            # 回放时再次增强G电导（但幅度减小）
            subj_neurons = self.mapping.get_neurons(subj)
            obj_neurons = self.mapping.get_neurons(obj)
            
            delta_G = self.bridge.alpha * 0.5  # 回放增强系数减半
            for i in subj_neurons:
                for j in obj_neurons:
                    self.bio_env.G[i, j] += delta_G
                    self.bio_env.G[j, i] += delta_G
        
        # 睡眠时电位归零（静息恢复）
        self.bio_env.V = self.bio_env.resting_template.copy()
        
        print(f"[睡眠] 记忆巩固完成，G矩阵连接数: "
              f"{np.count_nonzero(self.bio_env.G)}")
    
    def get_status(self) -> Dict:
        """获取记忆系统状态。"""
        footprint = self.bridge.get_memory_footprint()
        return {
            'n_concepts': footprint['n_concepts'],
            'n_connections': footprint['active_connections'],
            'n_learning_events': footprint['n_learning_events'],
            'concepts': self.mapping.get_all_concepts()
        }


# ================================================================
# 4. 测试
# ================================================================

def test_r_to_g_bridge():
    """测试R→G桥接：教知识→G电导改变→电位传播回忆。"""
    print("=" * 70)
    print("R网 → G网桥接测试")
    print("=" * 70)
    
    # 创建生物电记忆系统
    memory = BioelectricMemorySystem(dim=256, cluster_size=8)  # 小尺寸便于测试
    
    print("\n=== 阶段1：教知识 ===")
    print("教洛神：乌鸦口渴 → 放石子 → 水位升高 → 喝到水")
    
    # 教概念关系
    memory.teach("乌鸦", "渴", "感觉", 1.0)
    memory.teach("渴", "喝水", "导致", 0.9)
    memory.teach("乌鸦", "石子", "使用", 0.8)
    memory.teach("石子", "水位", "导致升高", 0.9)
    memory.teach("水位", "喝水", "使得", 0.9)
    memory.teach("乌鸦", "喝水", "最终", 1.0)
    
    # 教场景
    memory.teach_scene(["乌鸦", "渴", "喝水"], "乌鸦口渴")
    memory.teach_scene(["乌鸦", "石子", "水位"], "放石子")
    memory.teach_scene(["乌鸦", "喝水", "水位"], "喝到水")
    
    print("\n=== 阶段2：检查记忆物理足迹 ===")
    footprint = memory.bridge.get_memory_footprint()
    print(f"已学概念: {footprint['n_concepts']} 个")
    print(f"G矩阵活跃连接: {footprint['active_connections']} 条")
    print(f"学习事件: {footprint['n_learning_events']} 次")
    
    # 检查具体概念的物理状态
    print("\n=== 阶段3：概念物理状态 ===")
    for concept in ["乌鸦", "石子", "水位"]:
        state = memory.bridge.get_g_state(concept)
        print(f"  {concept}: 神经元{state['neurons']}, "
              f"电位={state['avg_potential']:.2f}mV, "
              f"连接数={state['n_connections']}")
    
    print("\n=== 阶段4：回忆测试（电位传播） ===")
    print("线索: '石子'")
    
    result = memory.recall("石子", steps=15)
    
    print("\n电位传播路径:")
    for line in memory.bridge.explain_recall(result):
        print(f"  {line}")
    
    print(f"\n最终激活概念:")
    for concept, v in result['activated_concepts'].items():
        print(f"  {concept}: {v:.2f}mV")
    
    # 验证：石子应该能想起乌鸦、水位、喝水
    activated = set(result['activated_concepts'].keys())
    expected = {"乌鸦", "水位", "喝水"}
    found = activated & expected
    
    print(f"\n期望想起: {expected}")
    print(f"实际想起: {activated}")
    print(f"命中: {found}")
    
    if len(found) >= 2:
        print("\n[PASS] 电位传播回忆成功 ✓")
    else:
        print("\n[FAIL] 回忆效果不佳，检查G电导")
    
    print("\n=== 阶段5：睡眠固化 ===")
    memory.sleep(n_replay=3)
    
    # 固化后再回忆
    print("\n固化后再回忆 '石子':")
    result2 = memory.recall("石子", steps=15)
    activated2 = set(result2['activated_concepts'].keys())
    print(f"  激活概念: {activated2}")
    
    print("\n" + "=" * 70)
    print("R网 → G网桥接测试完成")
    print("=" * 70)
    
    return memory


if __name__ == "__main__":
    test_r_to_g_bridge()
