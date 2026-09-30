"""
bioelectric_network.py — 莱文生物电网络环境层
===========================================
Michael Levin 的 Basal Cognition 理论在洛神环境层的实现。

核心类比：
  离子通道  ↔  W 矩阵电导 G(i→j)
  膜电位    ↔  E 环境向量 V_i（有静息/去极化/复极化）
  间隙连接  ↔  R 网电突触耦合
  损伤修复  ↔  Xenobots 式自修复（检测→重建功能等效结构）

底层公式不变：
  E = E + dE
  P' = F + f(E)   f(E) = E + 底噪(0.01)
  F' = P + g(E)   g(E) = -E + 底噪(0.01)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import random

# ================================================================
# 1. 概念电生理档案（每个概念有自己的膜电位参数）
# ================================================================

CONCEPT_ELECTROPHYSIOLOGY = {
    "渴": {"resting": -70.0, "threshold": -55.0, "refractory": 5, "channels": ["水", "喝"]},
    "乌鸦": {"resting": -65.0, "threshold": -50.0, "refractory": 3, "channels": ["石子", "水", "飞"]},
    "水": {"resting": -60.0, "threshold": -45.0, "refractory": 2, "channels": ["渴", "喝"]},
    "石子": {"resting": -68.0, "threshold": -52.0, "refractory": 4, "channels": ["水面升高"]},
    "猫爪": {"resting": -62.0, "threshold": -48.0, "refractory": 3, "channels": ["触摸"]},
    "训练": {"resting": -64.0, "threshold": -50.0, "refractory": 4, "channels": ["学习", "理解"]},
    "我": {"resting": -58.0, "threshold": -40.0, "refractory": 6, "channels": ["洛神", "沙箱"]},
    "洛神": {"resting": -58.0, "threshold": -40.0, "refractory": 6, "channels": ["我", "此刻"]},
}

# ================================================================
# 2. 电导网络 G（取代旧 W 矩阵的"强度"语义）
# ================================================================

class ElectrophysiologicalEnvironment:
    """
    生物电环境层：把 W 矩阵重定义为电导网络。
    
    关键区别：
      旧 W：信号强度 = W @ E（注入越强输出越强，会饱和）
      新 G：离子电流 = G * (V_source - V_target)（电位差驱动，不会饱和）
    """
    
    def __init__(self, dim=1024, n_concepts=66, scale=0.1):
        self.dim = dim
        self.n_concepts = n_concepts
        self.scale = scale
        
        # 电导矩阵 G（替代 W 矩阵）
        # G[i,j] = 概念 i 和概念 j 之间的电导
        self.G = np.random.randn(dim, dim) * 0.01
        
        # 每个维度的静息电位模板
        self.resting_template = np.zeros(dim)
        
        # 当前膜电位
        self.V = self.resting_template.copy()
        
        # 不应期计数器（概念维度）
        self.refractory_counter = np.zeros(n_concepts)
        
        # 损伤检测器
        self.damage_detector = BioelectricDamageDetector(dim)
    
    def ionic_current(self, dE_concept: np.ndarray) -> np.ndarray:
        """
        欧姆定律：I = G * (V_source - V_target)
        
        环境注入的 dE 不是直接叠加，而是作为"外部离子流"
        推动膜电位偏离静息状态。
        """
        # dE 是外部信号（类似突触前释放的离子）
        # 计算电位差驱动的电流
        voltage_diff = dE_concept - self.V
        I = np.clip(self.G @ voltage_diff, -5, 5)
        return I
    
    def update_potential(self, dE: np.ndarray, concept_ids: set) -> np.ndarray:
        """
        更新膜电位：
        1. 外部离子流推动电位变化
        2. 超过阈值的概念产生动作电位
        3. 不应期概念暂时不响应
        """
        # 步骤1：计算离子电流
        I = self.ionic_current(dE)
        
        # 步骤2：更新电位（类似细胞膜充电）
        dV = I * self.scale
        
        # 步骤3：应用不应期
        for cid in concept_ids:
            if self.refractory_counter[cid] > 0:
                dV[cid] = 0  # 不应期不响应
                self.refractory_counter[cid] -= 1
        
        self.V += dV
        
        # 步骤4：检测动作电位（超过阈值）
        for cid in concept_ids:
            if self.refractory_counter[cid] == 0:
                resting = CONCEPT_ELECTROPHYSIOLOGY.get(
                    self._id_to_name(cid), {}
                ).get("resting", -65.0)
                threshold = CONCEPT_ELECTROPHYSIOLOGY.get(
                    self._id_to_name(cid), {}
                ).get("threshold", -50.0)
                refractory = CONCEPT_ELECTROPHYSIOLOGY.get(
                    self._id_to_name(cid), {}
                ).get("refractory", 3)
                
                if self.V[cid] > threshold:
                    # 动作电位！触发不应期
                    self.refractory_counter[cid] = refractory
                    # 快速复极化（动作电位后电位回到静息）
                    self.V[cid] = resting
        
        # 步骤5：损伤检测
        damage = self.damage_detector.check(self.V, self.resting_template)
        if damage:
            self.damage_detector.repair(self, damage)
        
        return self.V
    
    def _id_to_name(self, cid: int) -> str:
        """概念ID转名称（简化版，实际应从概念库查询）"""
        names = list(CONCEPT_ELECTROPHYSIOLOGY.keys())
        if cid < len(names):
            return names[cid]
        return "unknown"


# ================================================================
# 3. R 网间隙连接（电突触耦合）
# ================================================================

class GapJunctionNetwork:
    """
    R 网不是知识查询，是长程电耦合。
    
    两个概念如果存在 R 边，它们的电位会缓慢趋于一致——
    类似生物体的间隙连接（gap junction）。
    """
    
    def __init__(self):
        # R[(subj, obj)] = [(relation, weight), ...]
        self.R = {}
    
    def add(self, subj: str, obj: str, rel: str, w: float = 1.0):
        key = (subj, obj)
        self.R.setdefault(key, [])
        self.R[key].append((rel, w))
    
    def coupling(self, V: np.ndarray, name2id: Dict[str, int]) -> np.ndarray:
        """
        电突触耦合：相连概念的电位趋于平均。
        
        返回 dV_coupling，需要叠加到环境变化中。
        """
        dV = np.zeros_like(V)
        for (subj, obj), rels in self.R.items():
            if subj not in name2id or obj not in name2id:
                continue
            i = name2id[subj]
            j = name2id[obj]
            
            # 电位趋于平均（间隙连接特性）
            avg = (V[i] + V[j]) / 2.0
            for rel, w in rels:
                dV[i] += w * (avg - V[i]) * 0.1
                dV[j] += w * (avg - V[j]) * 0.1
        
        return dV


# ================================================================
# 4. 损伤检测与自修复（Xenobots 类比）
# ================================================================

class BioelectricDamageDetector:
    """
    检测环境层"损伤"：
      1. 去极化危机（电位长期偏离静息）
      2. 梯度崩塌（电位分布过于均匀 = 集体沉默）
      3. 反馈回路断开
    
    修复不是回退，是重建功能等效结构。
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        self.damage_threshold = 15.0   # 去极化危机阈值
        self.silence_threshold = 2.0   # 梯度崩塌阈值
        self.history = []
    
    def check(self, V: np.ndarray, resting: np.ndarray) -> Optional[str]:
        """返回损伤类型或 None"""
        self.history.append(V.copy())
        if len(self.history) > 100:
            self.history.pop(0)
        
        # 损伤1：去极化危机
        if np.mean(np.abs(V - resting)) > self.damage_threshold:
            return "depolarization_crisis"
        
        # 损伤2：梯度崩塌（所有概念电位趋于一致）
        if np.std(V) < self.silence_threshold:
            return "gradient_collapse"
        
        # 损伤3：历史趋势恶化（电位持续漂移）
        if len(self.history) >= 10:
            recent = np.mean(self.history[-10:], axis=0)
            older = np.mean(self.history[:10], axis=0)
            if np.linalg.norm(recent - older) > self.damage_threshold * 2:
                return "drift_crisis"
        
        return None
    
    def repair(self, env: ElectrophysiologicalEnvironment, damage_type: str):
        """
        Xenobots 式自修复：
        不是恢复旧状态，是重建功能等效结构。
        """
        if damage_type == "depolarization_crisis":
            # 修复：重置静息电位，保留连接结构
            env.V = env.resting_template.copy() + np.random.randn(env.dim) * 2.0
            print("[修复] 去极化危机：重置静息电位，保留 G 矩阵")
        
        elif damage_type == "gradient_collapse":
            # 修复：引入随机扰动打破均匀（类似 Xenobots 运动）
            noise = np.random.randn(env.dim) * 5.0
            env.V += noise
            print("[修复] 梯度崩塌：注入噪声扰动，重建电位梯度")
        
        elif damage_type == "drift_crisis":
            # 修复：轻微扰动电导矩阵，增强适应性
            env.G += np.random.randn(env.dim, env.dim) * 0.001
            print("[修复] 漂移危机：微调电导矩阵，增强适应性")


# ================================================================
# 5. 与现有体系的无缝衔接
# ================================================================

class BioelectricInjector:
    """
    把旧的 _inject_scene 替换为生物电注入。
    
    接口兼容：输入还是 scene_concept_ids，输出还是 dE_vt
    但内部机制从"强度叠加"变成"电位差驱动的离子流"。
    """
    
    def __init__(self, dim: int = 1024):
        self.bio_env = ElectrophysiologicalEnvironment(dim=dim)
        self.gap_net = GapJunctionNetwork()
        self.name2id = {}  # 需要外部注入
    
    def inject(self, concept_ids: set, steps: int = 6) -> np.ndarray:
        """
        把场景概念注入环境层，跑生物电网络。
        
        返回 dE_vt（兼容旧接口，供 _bottom_step 使用）
        """
        # 初始外部信号（概念的多感官编码）
        dE = np.zeros(self.bio_env.dim)
        for cid in concept_ids:
            # 概念激活对应感官通道（简化：均匀分布）
            dE[cid % self.bio_env.dim] += 1.0
        
        # 跑生物电网络
        for _ in range(steps):
            # 间隙连接耦合
            if self.gap_net.R and self.name2id:
                dE += self.gap_net.coupling(self.bio_env.V, self.name2id)
            
            # 更新膜电位
            V_new = self.bio_env.update_potential(dE, concept_ids)
            
            # 生成下一步 dE（电位变化作为新信号）
            dE = (V_new - self.bio_env.V) * 0.5
        
        # 返回最终电位变化（兼容旧 _inject_scene 接口）
        return np.clip(self.bio_env.G @ self.bio_env.V, -5, 5)


if __name__ == "__main__":
    # 简单测试
    print("=== 莱文生物电网络环境层测试 ===")
    injector = BioelectricInjector(dim=1024)
    
    # 测试1：电位差驱动
    dE1 = np.zeros(1024)
    dE1[0] = 10.0  # 高注入
    V1 = injector.bio_env.update_potential(dE1, {0})
    print(f"高注入后电位: mean={V1.mean():.2f}, std={V1.std():.2f}")
    
    # 测试2：不应期
    print(f"不应期计数器[0]: {injector.bio_env.refractory_counter[0]}")
    
    # 测试3：损伤检测
    detector = BioelectricDamageDetector(1024)
    damage = detector.check(V1, np.zeros(1024))
    print(f"损伤检测结果: {damage}")
    
    print("\n生物电网络环境层 ready。")
