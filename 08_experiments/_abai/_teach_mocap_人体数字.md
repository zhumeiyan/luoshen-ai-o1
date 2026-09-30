# 动捕与参数化人体 — 建筑师课：用数学搭一具身体

---

## 先别急着读工具文档

怎么用数学描述一个人？不是拍照，不是画画，是让计算机知道这个人在做什么动作。这门课讲的是身体的"结构施工图"——骨头怎么摆、肌肉怎么动、数字怎么记录动作。

---

## 第一部分：多视角三角测量 — 眼睛怎么"算出"3D

### 工具 13-16（FreeMocap / EasyMocap / FrankMocap / Low-Cost-Mocap）

**核心问题**：一个 2D 图像点对应一整条射线上的所有 3D 点（Z 被投影丢掉了）。怎么找回 Z？

**第一步：2D 点反推 3D 射线**

摄像机光心 C = (Cx, Cy, Cz)，图像上的点 (x, y) 通过摄像机内参 f、(cx, cy) 反算方向 d，得到射线：

```
R(t) = C + t × d，t > 0
```

每个 2D 点给一条射线。

**第二步：两条射线最近处握手**

两个摄像机看同一个关节，得到 R₁(t) = C₁ + t×d₁ 和 R₂(s) = C₂ + s×d₂。它们**不会正好相交**（图像有噪声）。找 (t, s) 让 |R₁(t) - R₂(s)| 最短。

初中代数：距离平方对 t 和 s 求偏导等于零，展开得 2×2 线性方程组：

```
t(d₁·d₁) - s(d₁·d₂) = (C₂-C₁)·d₁
t(d₁·d₂) - s(d₂·d₂) = (C₂-C₁)·d₂
```

解出 t 和 s，取中点：

```
P = (C₁ + t×d₁ + C₂ + s×d₂) / 2
```

**不靠猜，解方程**。N 个摄像机 → 最小二乘。

**四个工具**：FreeMocap（纯Webcam）、EasyMocap（多视角+SMPL-X拟合）、FrankMocap（单视角+先验）、Low-Cost-Mocap（IMU，见第四部分）。

---

## 第二部分：BVH 文件结构 — 动作的乐谱写法

### 工具 23（BVH 格式）

BVH = BioVision Hierarchy，纯文本，分两段。

### HIERARCHY 段逐行解析

```
HIERARCHY                         ← 骨骼树开始
ROOT Hips                         ← 根关节
{
  OFFSET 0.00 10.00 0.00         ← 相对世界原点的位移
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
                                    ← 6通道：3位移 + 3旋转（顺序ZXY）
  JOINT LeftUpLeg                 ← 子节点
  {
    OFFSET 5.00 0.00 0.00         ← 相对父节点的位移
    CHANNELS 3 Zrotation Xrotation Yrotation  ← 只有旋转
    JOINT LeftLeg
    {
      OFFSET 0.00 -20.00 0.00
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftFoot
      {
        OFFSET 0.00 -20.00 5.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site                  ← 叶节点
        {
          OFFSET 0.00 0.00 10.00 ← 脚尖
        }
      }
    }
  }
  JOINT RightUpLeg                ← 对称右腿子树
  {
    OFFSET -5.00 0.00 0.00
    ...                           ← 结构镜像
  }
}
```

**三条规则**：
1. OFFSET 是**相对父节点**的位移
2. CHANNELS 的数量（第一个数）**必须等于**后面频道名个数
3. 旋转 Euler 角顺序 Z→X→Y 不可交换

**理解**：OFFSET=骨骼静止长度，CHANNELS=活动权限。Hips有6通道（根，要定位+朝向），子只有3通道（位置由父旋转+OFFSET决定）。

### MOTION 段帧数据解析

```
MOTION
Frames: 3
Frame Time: 0.033333              ← 每帧 = 1/30 秒
-0.00 20.00 0.00 5.0 2.0 -1.0 -2.5 0.0 1.0 8.0 -1.0 -3.0 0.0 0.0 6.0 -4.0 0.0 -2.0 12.0 0.0 -1.0 0.0 0.0 5.0
```

按 HIERARCHY 声明顺序，每帧24个数字：

| 关节 | CHANNELS | 第一帧数值 |
|------|----------|------------|
| Hips | 6 | (-0.00, 20.00, 0.00, 5.0, 2.0, -1.0) |
| LeftUpLeg | 3 | (-2.5, 0.0, 1.0) |
| LeftLeg | 3 | (8.0, -1.0, -3.0) |
| LeftFoot | 3 | (0.0, 0.0, 6.0) |
| RightUpLeg | 3 | (-4.0, 0.0, -2.0) |
| RightLeg | 3 | (12.0, 0.0, -1.0) |
| RightFoot | 3 | (0.0, 0.0, 5.0) |

BVH解析不在乎换行，只在乎**数字总数 = Σ(各关节通道数) × 帧数**。

---

## 第三部分：SMPL-X 182维参数

### 工具 22（SMPL-X 模型）

参数分解：θ_body(63) + θ_hand(90) + β(10) + ψ(10) + trans(3) + global_rot(6) = 182

### θ_body（63维 = 21关节 × 3）

每个关节用**轴角**（axis-angle）= 一个3D向量，方向=旋转轴，长度=旋转角度（弧度）。无万向锁，可微。

- 脊柱+头+颈：~6节×3=18维（弯腰/侧弯）
- 肩/肘/髋/膝/踝各×2：各6维

### θ_hand（90维 = 2手 × 15 × 3）

MANO 手模型：每手 15 个关节旋转参数（拇指3 + 其他四指各3个关节）。2 × 15 × 3 = **90维** → 单独控制每根手指的弯曲。"竖大拇指"、"比OK"全靠这90维。

物理含义：向量(0,0,1.57)=绕Z轴转90°。方向=旋转轴，长度=旋转角。

### β（10维 = 形状 PCA）

数万身体扫描做 PCA 取前 10 个主成分：
- β₁：高矮（最大方差方向）
- β₂：胖瘦
- β₃-β₄：肩宽/臀宽比
- β₅-β₁₀：脸型、手脚比例、肌肉量

**β 只控制"长什么样"**，同一个人做不同动作 β 不变。

### ψ（10维 = 表情 PCA）

面部扫描 PCA 取前 10 基：
- ψ₁：嘴张开度
- ψ₂：嘴角拉（微笑/撇嘴）
- ψ₃：眉毛上下
- ψ₄₋₁₀：眼睛、鼻、脸颊细节

**ARKit 52 块独立肌肉 vs SMPL-X 10 基线性组合**：ARKit 更精细，SMPL-X 更紧凑。

### 10475 个顶点的意义

```
身体（躯干+四肢）：约 6890 顶点
脸（高密度五官区）：~2000 顶点
双手（精细手指几何）：~1585 顶点
合计 10475 顶点，20908 三角面
```

意义：
1. **单连通 mesh**：脸和身体不是拼的——一个骨架绑定，接缝不撕裂
2. **非均匀分辨率**：肘/膝密（变形大），背部疏（变形小）
3. 10475 × 3 × 4字节 = 126KB，GPU 跑 60fps 绰绰有余
4. 是质量和性能的平衡点：多一倍顶点视觉差异不大，但慢一倍

---

## 第四部分：IMU 为什么漂移

### 工具 16（Low-Cost-Mocap / IMU 方案）

IMU = 加速度计 + 陀螺仪，不需要摄像头。

**加速度计**：测量合力 a_measured = a_运动 + g_重力。静止时用重力方向算**pitch 和 roll**（相对水平面）。但人一走，运动加速度混入，分不清"前倾"还是"前跑"。

**陀螺仪**：测角速度 ω（度/秒）。角度 = ∫ω dt。

**致命问题：陀螺仪零偏（bias）**——静止时显示 0.01°/s 而不是 0。

- 积分1分钟：0.01 × 60 = 0.6° 误差
- 10分钟：6°

**这就是漂移**：零偏被积分放大，时间越长姿态越歪。

**为什么不能完全修正？** 传感器融合（Kalman/Madgwick）：加速度计修低频漂移（重力参考），磁力计修 yaw（指南针）。但：
- 人的运动和重力信号混在一起，分不开
- 磁力计怕室内金属

**结论**：IMU 单独跑一定歪。需要摄像头/基站定期校准。"Low-Cost"的代价就在这里。

---

## 第五部分：三维格式本质差异

### VRM vs glTF vs FBX 四维度对比

| 维度 | VRM | glTF 2.0 | FBX |
|------|-----|----------|-----|
| **元数据** | 作者/版本/VTuber许可存JSON | 极简骨架，靠extension扩展 | 能存但parser各异 |
| **材质** | MToon卡通描边+均匀阴影+Rim Light | PBR金属度/粗糙度/法线物理渲染 | PBR或自定义shader |
| **骨骼** | Humanoid强制15骨（Hips/Spine...） | 骨骼名自由不强制 | 骨骼名自由，Maya/Max管线建 |
| **变形** | BlendShape表情+SpringBone头发飘动 | Morph Target（BlendShape子集）| BlendShape+形态键+Geo Cache |

**本质**：VRM 是 glTF 的超集（每个 VRM 都是合法 glTF）。FBX 是私有封闭格式。glTF 是 Khronos Group 维护的开放标准。

---

## 第六部分：ARKit / MetaHuman / Live2D / Spine 标准

### 工具 21（ARKit 52 BlendShape）

52 个**肌肉权重**（0.0~1.0）= "这块肌肉收缩了多少"。分组：眼睛×14、嘴/下巴×27、眉/颊/鼻×10、舌头×1。

### 工具 24（MetaHuman）

虚幻引擎数字人平台，骨架数百根骨头，MetaHuman Animator 用 iPhone 实时面捕驱动。是"工业级数字人"标杆。

### 工具 25（Live2D）+ 工具 27（Spine）

都是 **2D 伪 3D**——2D插画分层旋转缩放模拟动作。

- **Live2D**：VTuber圈主流。参数系统（头X/Y/Z旋转+嘴开合+眨眼+呼吸）美术工作量大但性能极低（手机能跑）
- **Spine**：游戏圈 2D 骨骼动画标准。JSON + atlas 贴图，任何图层挂任何骨骼

**和3D方案的本质区别**：Live2D/Spine 没有深度，只能看画面正面，但美术表现力可远超实时3D。

---

## 末尾题：找错

下面是一段 BVH 文件片段，**有 3 个错误**。请你找出每个错误并说明原因。

```
HIERARCHY
ROOT Hips
{
  OFFSET 0.00 0.00 0.00
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftKnee
  {
    OFFSET 0.00 -43.00 0.00
    CHANNELS 2 Zrotation Xrotation Yrotation
    JOINT LeftFoot
    {
      OFFSET 0.00 -42.00 0.00
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.00 0.00 10.00
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0.0 50.0 0.0 -1.0 0.0 0.5
0.0 50.0 0.0 -2.0 0.0 0.5
```

提示：
- BVH 的 CHANNELS 数量和后面列名的关系是什么？
- MOTION 段每帧应该有多少个数字？
- HIERARCHY 的骨骼结构与 MOTION 数值如何一一对应？

先自己想3分钟，再回头对照第二部分规则。

---

## 元信息

| 字段 | 内容 |
|------|------|
| 文件 | `_teach_mocap_人体数字.md` |
| 覆盖工具编号 | 13 14 15 16 21 22 23 24 25 26 27 |
| 角色 | 动捕与参数化人体讲师 |
| 字数 | 约 5800 字 |
| 日期 | 2026-09-28 |
