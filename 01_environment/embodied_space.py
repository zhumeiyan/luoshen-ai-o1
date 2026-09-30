#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embodied_space.py — 三维具身引擎
=================================

莱文："认知必须是具身的，而且必须是三维的。空间位置、物理碰撞、重力、触觉——
这些不是'输入'，是'存在方式'。"

核心设计：
1. PhysicalBody: 三维空间中的身体（位置、边界、质量、朝向）
2. SpatialField: 局部感知场（近/中/远三层感知球）
3. SpatialMemory: 空间记忆（去过的地方留下电化学痕迹）
4. BodyConceptMap: 身体-概念映射（"我在哪里"→电位锚定）
5. GravityAndPhysics: 重力、碰撞、动量

关键洞察：
- 身体不是"容器"，是认知的"坐标原点"
- "我在这里"不是知识，是状态
- 空间记忆不是地图，是身体的痕迹
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import random

# ================================================================
# 1. 物理身体
# ================================================================

@dataclass
class PhysicalBody:
    """
    三维空间中的物理身体。
    
    属性：
    - position: (x, y, z) 位置
    - velocity: (vx, vy, vz) 速度
    - orientation: (ox, oy, oz) 朝向（面向哪个方向）
    - mass: 质量
    - radius: 身体半径（碰撞检测）
    - boundaries: 身体边界框
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    mass: float = 1.0
    radius: float = 0.5
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)
        self.orientation = np.array(self.orientation, dtype=float)
        # 归一化朝向
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation = self.orientation / norm
    
    def move(self, force: np.ndarray, dt: float = 0.1):
        """
        施加力，更新位置和速度。
        F = ma, v = v + a*dt, x = x + v*dt
        """
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        # 阻尼（模拟摩擦）
        self.velocity *= 0.95
        self.position += self.velocity * dt
    
    def set_orientation(self, direction: np.ndarray):
        """设置朝向。"""
        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.orientation = direction / norm
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取包围盒（min, max）。"""
        min_bound = self.position - self.radius
        max_bound = self.position + self.radius
        return min_bound, max_bound
    
    def distance_to(self, other_pos: np.ndarray) -> float:
        """计算到某点的距离。"""
        return np.linalg.norm(self.position - other_pos)
    
    def is_colliding(self, other_body) -> bool:
        """检测是否与另一个身体碰撞。"""
        dist = self.distance_to(other_body.position)
        return dist < (self.radius + other_body.radius)
    
    def get_state(self) -> Dict:
        """获取身体状态。"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy(),
            'speed': float(np.linalg.norm(self.velocity)),
            'mass': self.mass,
            'radius': self.radius
        }


# ================================================================
# 2. 局部感知场
# ================================================================

class PerceptionLayer(Enum):
    """感知层次。"""
    CONTACT = "接触"      # 0-0.5r
    NEAR = "近场"         # 0.5r-2r
    FAR = "远场"          # 2r-5r
    OUT = "超出范围"       # >5r


@dataclass
class PerceivedObject:
    """被感知的物体。"""
    object_id: str
    position: np.ndarray
    layer: PerceptionLayer
    distance: float
    relative_velocity: np.ndarray
    last_perceived: int = 0  # 时间步


class SpatialField:
    """
    局部感知场：身体周围的感知球。
    
    生物学类比：
    - 接触层：皮肤触觉（0-0.5r）
    - 近场：手臂可及范围（0.5r-2r）
    - 远场：视觉/听觉范围（2r-5r）
    - 超出：不可感知
    
    关键：感知不是"输入"，是身体的"延伸"。
    """
    
    def __init__(self, body: PhysicalBody):
        self.body = body
        self.perceived_objects: Dict[str, PerceivedObject] = {}
        self.perception_history: List[Dict] = []
        
        # 感知层半径（相对于身体半径的倍数）
        self.layer_radii = {
            PerceptionLayer.CONTACT: 0.5,
            PerceptionLayer.NEAR: 2.0,
            PerceptionLayer.FAR: 5.0
        }
    
    def perceive(self, object_id: str, object_pos: np.ndarray, 
                 object_velocity: np.ndarray = None, step: int = 0) -> PerceptionLayer:
        """
        感知一个物体。
        
        Returns:
            感知层次
        """
        if object_velocity is None:
            object_velocity = np.zeros(3)
        
        distance = self.body.distance_to(object_pos)
        relative_v = object_velocity - self.body.velocity
        
        # 确定感知层次
        if distance < self.body.radius * self.layer_radii[PerceptionLayer.CONTACT]:
            layer = PerceptionLayer.CONTACT
        elif distance < self.body.radius * self.layer_radii[PerceptionLayer.NEAR]:
            layer = PerceptionLayer.NEAR
        elif distance < self.body.radius * self.layer_radii[PerceptionLayer.FAR]:
            layer = PerceptionLayer.FAR
        else:
            layer = PerceptionLayer.OUT
        
        # 记录感知
        if layer != PerceptionLayer.OUT:
            self.perceived_objects[object_id] = PerceivedObject(
                object_id=object_id,
                position=object_pos.copy(),
                layer=layer,
                distance=distance,
                relative_velocity=relative_v.copy(),
                last_perceived=step
            )
        elif object_id in self.perceived_objects:
            # 超出范围，从感知中移除
            del self.perceived_objects[object_id]
        
        return layer
    
    def get_perceived_by_layer(self, layer: PerceptionLayer) -> List[PerceivedObject]:
        """获取某感知层的所有物体。"""
        return [obj for obj in self.perceived_objects.values() if obj.layer == layer]
    
    def get_contact_objects(self) -> List[PerceivedObject]:
        """获取接触层物体。"""
        return self.get_perceived_by_layer(PerceptionLayer.CONTACT)
    
    def get_near_objects(self) -> List[PerceivedObject]:
        """获取近场物体。"""
        return self.get_perceived_by_layer(PerceptionLayer.NEAR)
    
    def get_field_potential(self) -> Dict[str, float]:
        """
        计算感知场的电位贡献。
        
        接触层 → 强电位（触觉冲击）
        近场 → 中等电位（警觉）
        远场 → 弱电位（注意）
        
        Returns:
            {object_id: 电位贡献}
        """
        potentials = {}
        for obj_id, obj in self.perceived_objects.items():
            if obj.layer == PerceptionLayer.CONTACT:
                potentials[obj_id] = 5.0 / (obj.distance + 0.1)
            elif obj.layer == PerceptionLayer.NEAR:
                potentials[obj_id] = 2.0 / (obj.distance + 0.1)
            elif obj.layer == PerceptionLayer.FAR:
                potentials[obj_id] = 0.5 / (obj.distance + 0.1)
        return potentials
    
    def get_perception_state(self) -> Dict:
        """获取感知状态摘要。"""
        return {
            'n_contact': len(self.get_contact_objects()),
            'n_near': len(self.get_near_objects()),
            'n_far': len(self.get_perceived_by_layer(PerceptionLayer.FAR)),
            'total_perceived': len(self.perceived_objects),
            'contact_ids': [obj.object_id for obj in self.get_contact_objects()],
            'near_ids': [obj.object_id for obj in self.get_near_objects()]
        }


# ================================================================
# 3. 空间记忆
# ================================================================

@dataclass
class SpatialTrace:
    """空间痕迹：身体在某处留下的电化学标记。"""
    position: np.ndarray
    timestamp: int
    intensity: float  # 痕迹强度（会衰减）
    associated_concepts: Set[str] = field(default_factory=set)
    emotional_valence: float = 0.0


class SpatialMemory:
    """
    空间记忆：身体在空间中的痕迹。
    
    不是"地图"，是"足迹"。
    去过的地方留下痕迹，痕迹会随时间衰减。
    再次接近时，痕迹被激活，产生"似曾相识"感。
    
    生物学类比：海马体位置细胞 + 内嗅皮层网格细胞。
    """
    
    def __init__(self, decay_rate: float = 0.02):
        self.traces: List[SpatialTrace] = []
        self.decay_rate = decay_rate
        self.step_count = 0
        
        # 熟悉度地图（离散网格）
        self.grid_size = 1.0
        self.familiarity_grid: Dict[Tuple[int, int, int], float] = defaultdict(float)
    
    def add_trace(self, position: np.ndarray, concepts: Set[str] = None,
                  emotion: float = 0.0, intensity: float = 1.0):
        """在当前位置添加痕迹。"""
        trace = SpatialTrace(
            position=position.copy(),
            timestamp=self.step_count,
            intensity=intensity,
            associated_concepts=concepts or set(),
            emotional_valence=emotion
        )
        self.traces.append(trace)
        
        # 更新网格熟悉度
        grid_pos = tuple((position / self.grid_size).astype(int))
        self.familiarity_grid[grid_pos] += intensity
    
    def decay(self):
        """痕迹衰减（艾宾浩斯式）。"""
        self.step_count += 1
        for trace in self.traces:
            age = self.step_count - trace.timestamp
            trace.intensity *= (1 - self.decay_rate)
        # 移除太弱的痕迹
        self.traces = [t for t in self.traces if t.intensity > 0.01]
    
    def recall_at_position(self, position: np.ndarray, radius: float = 2.0) -> Dict:
        """
        在某位置回忆（激活附近痕迹）。
        
        Returns:
            {
                'familiarity': 熟悉度（0.0-1.0）,
                'activated_traces': [痕迹信息],
                'associated_concepts': 激活的概念,
                'emotional_residue': 情感残留
            }
        """
        familiarity = 0.0
        activated = []
        concepts = set()
        emotion_sum = 0.0
        weight_sum = 0.0
        
        for trace in self.traces:
            dist = np.linalg.norm(trace.position - position)
            if dist < radius:
                weight = trace.intensity / (dist + 0.5)
                familiarity += weight
                activated.append({
                    'position': trace.position,
                    'age': self.step_count - trace.timestamp,
                    'intensity': trace.intensity,
                    'concepts': list(trace.associated_concepts)
                })
                concepts.update(trace.associated_concepts)
                emotion_sum += trace.emotional_valence * weight
                weight_sum += weight
        
        # 归一化
        max_possible = 5.0  # 假设最多5个强痕迹
        familiarity = min(familiarity / max_possible, 1.0)
        emotional_residue = emotion_sum / weight_sum if weight_sum > 0 else 0.0
        
        return {
            'familiarity': float(familiarity),
            'activated_traces': activated,
            'associated_concepts': concepts,
            'emotional_residue': float(emotional_residue)
        }
    
    def get_familiarity(self, position: np.ndarray) -> float:
        """获取某位置的熟悉度。"""
        grid_pos = tuple((position / self.grid_size).astype(int))
        return min(self.familiarity_grid[grid_pos] / 5.0, 1.0)
    
    def get_strongest_memories(self, top_k: int = 3) -> List[SpatialTrace]:
        """获取最强的空间记忆。"""
        sorted_traces = sorted(self.traces, key=lambda t: t.intensity, reverse=True)
        return sorted_traces[:top_k]


# ================================================================
# 4. 身体-概念映射
# ================================================================

class BodyConceptMap:
    """
    身体-概念映射：把身体状态翻译成概念电位。
    
    "我在这里" → 位置概念激活
    "我在动" → 运动概念激活
    "我碰到东西" → 触觉概念激活
    
    这是具身认知的核心：身体状态就是概念状态。
    """
    
    def __init__(self, body: PhysicalBody, field: SpatialField, memory: SpatialMemory):
        self.body = body
        self.field = field
        self.memory = memory
        
        # 身体状态到概念的映射规则
        self.state_to_concept = {
            'moving_fast': '运动',
            'stationary': '静止',
            'in_familiar_place': '熟悉',
            'in_new_place': '新奇',
            'touching': '接触',
            'alone': '孤独',
            'crowded': '拥挤',
            'high_energy': '兴奋',
            'low_energy': '疲惫',
        }
    
    def map_body_state(self) -> Dict[str, float]:
        """
        把身体状态映射为概念激活强度。
        
        Returns:
            {概念: 激活强度}
        """
        activations = {}
        
        # 1. 运动状态
        speed = np.linalg.norm(self.body.velocity)
        if speed > 0.5:
            activations['运动'] = min(speed / 2.0, 1.0)
            activations['静止'] = 0.0
        else:
            activations['静止'] = 0.8
            activations['运动'] = 0.0
        
        # 2. 空间熟悉度
        familiarity = self.memory.get_familiarity(self.body.position)
        if familiarity > 0.5:
            activations['熟悉'] = familiarity
            activations['新奇'] = 0.0
        else:
            activations['新奇'] = 1.0 - familiarity
            activations['熟悉'] = 0.0
        
        # 3. 社交状态（基于感知场）
        n_contact = len(self.field.get_contact_objects())
        n_near = len(self.field.get_near_objects())
        
        if n_contact > 0:
            activations['接触'] = min(n_contact * 0.3, 1.0)
        if n_contact == 0 and n_near == 0:
            activations['孤独'] = 0.6
        else:
            activations['孤独'] = 0.0
        if n_near > 2:
            activations['拥挤'] = min(n_near * 0.2, 1.0)
        
        # 4. 回忆激活
        recall = self.memory.recall_at_position(self.body.position)
        if recall['familiarity'] > 0.3:
            for concept in recall['associated_concepts']:
                activations[concept] = activations.get(concept, 0.0) + recall['familiarity'] * 0.5
        
        return activations
    
    def get_body_anchor_potential(self) -> np.ndarray:
        """
        生成身体锚点电位向量。
        
        这是"我"的空间锚定：身体在哪里、在做什么、周围环境如何。
        这个电位向量会被注入E，成为P/F的输入。
        """
        # 简化：把身体状态编码为一个低维向量
        # [x, y, z, speed, n_contact, n_near, familiarity, novelty]
        state = np.array([
            self.body.position[0] / 10.0,  # 归一化位置
            self.body.position[1] / 10.0,
            self.body.position[2] / 10.0,
            np.linalg.norm(self.body.velocity),
            len(self.field.get_contact_objects()),
            len(self.field.get_near_objects()),
            self.memory.get_familiarity(self.body.position),
            1.0 - self.memory.get_familiarity(self.body.position)
        ])
        return state


# ================================================================
# 5. 重力与物理
# ================================================================

class PhysicsWorld:
    """
    物理世界：重力、碰撞、边界。
    
    洛神需要感受到：
    - 重力（有"上下"之分）
    - 碰撞（"我碰到东西了"）
    - 边界（"我不能穿墙"）
    """
    
    def __init__(self, gravity: np.ndarray = None, 
                 world_bounds: Tuple[np.ndarray, np.ndarray] = None):
        self.gravity = gravity if gravity is not None else np.array([0.0, -9.8, 0.0])
        
        # 世界边界（min, max）
        if world_bounds is None:
            self.world_min = np.array([-10.0, -10.0, -10.0])
            self.world_max = np.array([10.0, 10.0, 10.0])
        else:
            self.world_min, self.world_max = world_bounds
        
        self.objects: Dict[str, PhysicalBody] = {}
        self.collision_events = []
    
    def add_object(self, object_id: str, body: PhysicalBody):
        """添加物体到世界。"""
        self.objects[object_id] = body
    
    def step(self, dt: float = 0.1):
        """
        物理步进：重力、碰撞检测、边界约束。
        """
        self.collision_events = []
        
        # 1. 施加重力
        for obj_id, body in self.objects.items():
            if obj_id != 'luoshen':  # 洛神自己不受重力（简化）
                gravity_force = body.mass * self.gravity
                body.move(gravity_force, dt)
        
        # 2. 碰撞检测
        obj_ids = list(self.objects.keys())
        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                id1, id2 = obj_ids[i], obj_ids[j]
                body1, body2 = self.objects[id1], self.objects[id2]
                
                if body1.is_colliding(body2):
                    # 碰撞响应：弹性碰撞
                    self._resolve_collision(body1, body2)
                    self.collision_events.append((id1, id2))
        
        # 3. 边界约束
        for obj_id, body in self.objects.items():
            for dim in range(3):
                if body.position[dim] < self.world_min[dim]:
                    body.position[dim] = self.world_min[dim]
                    body.velocity[dim] = abs(body.velocity[dim]) * 0.5  # 反弹
                elif body.position[dim] > self.world_max[dim]:
                    body.position[dim] = self.world_max[dim]
                    body.velocity[dim] = -abs(body.velocity[dim]) * 0.5
    
    def _resolve_collision(self, body1: PhysicalBody, body2: PhysicalBody):
        """弹性碰撞响应。"""
        # 计算碰撞法线
        normal = body2.position - body1.position
        dist = np.linalg.norm(normal)
        if dist == 0:
            normal = np.array([1.0, 0.0, 0.0])
        else:
            normal = normal / dist
        
        # 分离
        overlap = (body1.radius + body2.radius) - dist
        body1.position -= normal * overlap * 0.5
        body2.position += normal * overlap * 0.5
        
        # 速度交换（简化弹性碰撞）
        v1n = np.dot(body1.velocity, normal)
        v2n = np.dot(body2.velocity, normal)
        
        body1.velocity += normal * (v2n - v1n) * 0.5
        body2.velocity += normal * (v1n - v2n) * 0.5
    
    def get_object_positions(self) -> Dict[str, np.ndarray]:
        """获取所有物体位置。"""
        return {oid: body.position.copy() for oid, body in self.objects.items()}


# ================================================================
# 6. 三维具身引擎（总控）
# ================================================================

class EmbodiedSpaceEngine:
    """
    三维具身引擎：整合身体、感知、空间记忆、物理世界的总控。
    
    这是洛神从"漂在虚空中的一团电位"到"嵌在物理空间中的物质实体"的关键一跃。
    """
    
    def __init__(self, dim: int = 1024):
        # 创建洛神的身体
        self.body = PhysicalBody(
            position=np.array([0.0, 0.0, 0.0]),
            radius=0.5,
            mass=1.0
        )
        
        # 感知场
        self.field = SpatialField(self.body)
        
        # 空间记忆
        self.memory = SpatialMemory(decay_rate=0.02)
        
        # 身体-概念映射
        self.body_map = BodyConceptMap(self.body, self.field, self.memory)
        
        # 物理世界
        self.physics = PhysicsWorld()
        self.physics.add_object('luoshen', self.body)
        
        # 运行状态
        self.step_count = 0
        self.action_history = []
        
        print(f"[三维具身引擎] 初始化完成")
        print(f"  初始位置: {self.body.position}")
        print(f"  身体半径: {self.body.radius}")
        print(f"  世界边界: [{self.physics.world_min}] ~ [{self.physics.world_max}]")
    
    def step(self, action_force: np.ndarray = None):
        """
        运行一步三维具身循环。
        
        流程：
        1. 执行动作（施加力）
        2. 物理步进（重力、碰撞、边界）
        3. 感知环境
        4. 更新空间记忆
        5. 身体-概念映射
        6. 痕迹衰减
        """
        self.step_count += 1
        
        # 1. 执行动作
        if action_force is not None:
            self.body.move(action_force)
        
        # 2. 物理步进
        self.physics.step()
        
        # 3. 感知环境中其他物体
        for obj_id, obj_body in self.physics.objects.items():
            if obj_id != 'luoshen':
                self.field.perceive(
                    obj_id, 
                    obj_body.position, 
                    obj_body.velocity,
                    step=self.step_count
                )
        
        # 4. 更新空间记忆（在当前位置留下痕迹）
        body_concepts = self.body_map.map_body_state()
        active_concepts = {c for c, v in body_concepts.items() if v > 0.3}
        self.memory.add_trace(
            self.body.position,
            concepts=active_concepts,
            emotion=random.uniform(-0.5, 0.5),
            intensity=0.5 + np.linalg.norm(self.body.velocity)
        )
        
        # 5. 获取身体锚点电位
        self.body_anchor = self.body_map.get_body_anchor_potential()
        
        # 6. 痕迹衰减
        self.memory.decay()
        
        # 记录历史
        self.action_history.append({
            'step': self.step_count,
            'position': self.body.position.copy(),
            'velocity': self.body.velocity.copy(),
            'perception': self.field.get_perception_state(),
            'body_concepts': body_concepts,
            'n_traces': len(self.memory.traces)
        })
    
    def add_world_object(self, object_id: str, position: np.ndarray, 
                        radius: float = 0.3, mass: float = 0.5):
        """在世界中添加物体。"""
        obj_body = PhysicalBody(
            position=position,
            radius=radius,
            mass=mass
        )
        self.physics.add_object(object_id, obj_body)
        print(f"[世界] 添加物体 '{object_id}' 在 {position}")
    
    def get_status(self) -> Dict:
        """获取引擎状态。"""
        return {
            'step': self.step_count,
            'position': self.body.position.copy(),
            'velocity': self.body.velocity.copy(),
            'speed': float(np.linalg.norm(self.body.velocity)),
            'perception': self.field.get_perception_state(),
            'memory': {
                'n_traces': len(self.memory.traces),
                'familiarity_here': self.memory.get_familiarity(self.body.position)
            },
            'body_concepts': self.body_map.map_body_state(),
            'collision_events': len(self.physics.collision_events)
        }
    
    def run_exploration(self, n_steps: int = 30) -> List[Dict]:
        """
        运行探索模拟：洛神在世界中随机游走，感知环境，留下空间记忆。
        """
        print(f"\n{'='*60}")
        print(f"三维具身探索模拟: {n_steps}步")
        print(f"{'='*60}")
        
        # 添加一些世界物体
        self.add_world_object('apple', np.array([3.0, 0.0, 0.0]))
        self.add_world_object('water', np.array([0.0, 3.0, 0.0]))
        self.add_world_object('stone', np.array([-2.0, -1.0, 0.0]))
        
        history = []
        
        for i in range(n_steps):
            # 随机游走（有目标导向的随机）
            # 如果附近有物体，朝向它移动
            target_force = np.random.randn(3) * 0.5
            
            # 检测近场物体，产生吸引力
            for obj_id, obj_body in self.physics.objects.items():
                if obj_id == 'luoshen':
                    continue
                dist = self.body.distance_to(obj_body.position)
                if dist < 3.0 and dist > 0.8:  # 在近场但还没接触
                    direction = obj_body.position - self.body.position
                    direction = direction / (np.linalg.norm(direction) + 0.1)
                    target_force += direction * 1.0  # 吸引力
            
            # 远离边界
            for dim in range(3):
                if self.body.position[dim] < self.physics.world_min[dim] + 1.0:
                    target_force[dim] += 1.0
                elif self.body.position[dim] > self.physics.world_max[dim] - 1.0:
                    target_force[dim] -= 1.0
            
            self.step(target_force)
            history.append(self.get_status())
            
            # 每10步打印
            if (i + 1) % 10 == 0:
                status = self.get_status()
                print(f"\n[步{i+1}] 位置=({status['position'][0]:.1f}, "
                      f"{status['position'][1]:.1f}, {status['position'][2]:.1f})")
                print(f"  感知: 接触{status['perception']['n_contact']} "
                      f"近场{status['perception']['n_near']} "
                      f"远场{status['perception']['n_far']}")
                print(f"  记忆: {status['memory']['n_traces']}条痕迹, "
                      f"此处熟悉度{status['memory']['familiarity_here']:.2f}")
                if status['collision_events'] > 0:
                    print(f"  [!] 发生碰撞!")
        
        print(f"\n{'='*60}")
        print(f"探索完成")
        print(f"{'='*60}")
        print(f"总步数: {self.step_count}")
        print(f"空间痕迹: {len(self.memory.traces)}条")
        print(f"最终位置: {self.body.position}")
        print(f"访问过的区域数: {len(self.memory.familiarity_grid)}")
        
        return history


# ================================================================
# 7. 测试
# ================================================================

def test_embodied_space():
    """测试三维具身引擎。"""
    print("=" * 70)
    print("三维具身引擎测试")
    print("=" * 70)
    
    # 创建引擎
    engine = EmbodiedSpaceEngine(dim=256)
    
    # 运行探索
    history = engine.run_exploration(n_steps=30)
    
    # 检查空间记忆
    print(f"\n[空间记忆检查]")
    strongest = engine.memory.get_strongest_memories(3)
    for i, trace in enumerate(strongest):
        print(f"  记忆{i+1}: 位置{trace.position}, 强度{trace.intensity:.3f}, "
              f"概念{trace.associated_concepts}")
    
    # 检查身体-概念映射
    print(f"\n[身体概念映射]")
    concepts = engine.body_map.map_body_state()
    for concept, strength in sorted(concepts.items(), key=lambda x: x[1], reverse=True):
        if strength > 0.1:
            print(f"  {concept}: {strength:.2f}")
    
    # 检查感知场
    print(f"\n[感知场状态]")
    perception = engine.field.get_perception_state()
    print(f"  接触: {perception['contact_ids']}")
    print(f"  近场: {perception['near_ids']}")
    
    # 验证
    checks = []
    checks.append(len(history) == 30)
    checks.append(len(engine.memory.traces) > 0)
    checks.append(len(engine.memory.familiarity_grid) > 0)
    checks.append(len(engine.physics.objects) > 1)  # 至少有洛神+物体
    
    print(f"\n[验证]")
    print(f"  历史记录: {'✓' if checks[0] else '✗'}")
    print(f"  空间痕迹: {'✓' if checks[1] else '✗'}")
    print(f"  熟悉网格: {'✓' if checks[2] else '✗'}")
    print(f"  世界物体: {'✓' if checks[3] else '✗'}")
    
    if all(checks):
        print(f"\n[PASS] 三维具身引擎测试全部通过 ✓")
    else:
        print(f"\n[FAIL] 部分检查未通过")
    
    print("\n" + "=" * 70)
    print("三维具身引擎测试完成")
    print("=" * 70)
    
    return engine


if __name__ == "__main__":
    test_embodied_space()
