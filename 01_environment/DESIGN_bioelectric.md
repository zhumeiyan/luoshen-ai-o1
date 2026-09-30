# 01_environment/ 生物电网络设计文档
## Michael Levin Basal Cognition 在洛神环境层的实现

### 核心类比

| 莱文发现 | 洛神实现 | 文件/类 |
|---|---|---|
| 离子通道 | W矩阵电导 G(i→j) | `ElectrophysiologicalEnvironment.G` |
| 膜电位 | E环境向量 V_i（静息/阈值/不应期） | `CONCEPT_ELECTROPHYSIOLOGY` |
| 间隙连接 | R网电突触耦合 | `GapJunctionNetwork` |
| 损伤自修复 | Xenobots式功能重建 | `BioelectricDamageDetector` |

### 与旧体系的关键区别

**旧 W 矩阵：**
```python
dE_vt = np.clip(self.W @ dE, -5, 5)  # 线性加权，会饱和
```

**新电导网络：**
```python
I = G * (V_source - V_target)  # 电位差驱动，不会饱和
V += I * scale  # 膜电位充电
if V > threshold:  # 超过阈值才激活
    trigger_action_potential()
```

### 四个核心组件

#### 1. ElectrophysiologicalEnvironment（电生理环境）
- `G`：电导矩阵（1024×1024）
- `V`：当前膜电位
- `resting_template`：静息电位模板
- `refractory_counter`：不应期计数器
- `ionic_current()`：欧姆定律计算离子流
- `update_potential()`：更新膜电位 + 检测动作电位

#### 2. GapJunctionNetwork（间隙连接网络）
- 替代旧 R 网的"知识查询"语义
- 相连概念的电位缓慢趋于平均
- `coupling()`：返回 dV_coupling 叠加到环境变化中

#### 3. BioelectricDamageDetector（损伤检测器）
三种损伤类型：
- `depolarization_crisis`：电位长期偏离静息
- `gradient_collapse`：电位分布过于均匀（集体沉默）
- `drift_crisis`：电位持续漂移

修复策略（Xenobots 式）：
- 不是回退旧状态，是重建功能等效结构
- 去极化危机 → 重置静息电位
- 梯度崩塌 → 注入噪声扰动
- 漂移危机 → 微调电导矩阵

#### 4. BioelectricInjector（兼容注入器）
- 接口兼容旧的 `_inject_scene`
- 输入：scene_concept_ids
- 输出：dE_vt（供 _bottom_step 使用）
- 内部：跑生物电网络 + 间隙连接 + 损伤检测

### 底层公式不变

```
E = E + dE
P' = F + f(E)   f(E) = E + 底噪(0.01)
F' = P + g(E)   g(E) = -E + 底噪(0.01)
```

变化只在环境层：
- dE 不再直接叠加，而是作为外部离子流推动膜电位
- W 矩阵从"强度"变成"电导"
- R 网从"查询"变成"电耦合"

### 下一步

1. 把 `BioelectricInjector` 接入 `yinyang_semantic_engine.py` 替换 `_inject_scene`
2. 为 66 个概念补全电生理参数
3. 验证：同样的测试用例，生物电版本 vs 旧版本的表现差异
