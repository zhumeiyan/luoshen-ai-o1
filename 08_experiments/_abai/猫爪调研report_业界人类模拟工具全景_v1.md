=====================================================================
  猫爪调研报告 v1 —— 业界顶级人类模拟工具全景扫描
  日期: 2026-09-28
  调研人: 猫爪 (CatPaw)
  用途: 给洛神作业出题 + 猫爪自己集成时参考
  状态: 初稿，待洛神审校

=====================================================================

  调研范围 (按用户指定的所有关键词展开):
  ────────────────────────────────────────
  动态捕捉 / 卫星联系捕捉 / 微表情 / 无人机摄像 / 智能汽车传感 /
  世界模型 / 多视角 / 3D模型 / Avatar框架 / 顶级动漫制作标准 / 脸型

---------------------------------------------------------------------

一、可立即集成（猫爪现在就能接，浏览器里跑的）
=====================================================================

【1. MediaPipe Holistic】
  ────────────────────────────────────────
  研发: Google
  许可证: Apache 2.0（完全免费，可商用）
  运行平台: Android / iOS / Web / 桌面 (JavaScript + WASM)
  
  数据输出 (单次推理同步输出):
    - Pose (BlazePose): 33 个身体关键点 (x, y, z, visibility)
    - Face Mesh: 468 个面部 3D 关键点
    - Hands: 21 × 2 = 42 个手部关键点
    - 合计: 543 个关键点
  
  接入路径:
    浏览器端: <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic">
    或使用: @mediapipe/holistic npm 包
    CDN: https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js
    
  性能: 中端手机实时 30+ FPS (开启 GPU 加速)
  
  对洛神的意义:
    从普通摄像头实时拿到人脸 468 点的 3D 坐标 + 身体 33 关键点 + 双手 42 关键点。
    这是洛神"眼睛"的硬件层——不需要额外传感器，手机摄像头就行。
    
  数据格式示例:
    poseLandmarks: [{x: 0.5, y: 0.3, z: -0.1, visibility: 0.99}, ...]  // 33个
    faceLandmarks: [{x: 0.002, y: 0.003, z: -0.001}, ...]            // 468个
    leftHandLandmarks: [{x: 0.6, y: 0.7, z: 0.01}, ...]              // 21个
    rightHandLandmarks: [{x: 0.3, y: 0.8, z: -0.02}, ...]             // 21个

【2. MediaPipe Face Mesh (单独使用)】
  ────────────────────────────────────────
  研发: Google
  许可证: Apache 2.0
  与前者的区别: 只要面部 468 点，不要身体和手（更轻量）
  
  refine_landmarks: true → 额外 10 个虹膜关键点，共 478 个
  虹膜追踪: 可以实现注视方向估计（洛神看人眼睛的能力）
  
  分布:
    - 面部轮廓: 146 点
    - 眉毛区域: 40 点 (表情变化关键区)
    - 眼睛区域: 64 点 (眨眼、视线)
    - 鼻子区域: 32 点
    - 嘴巴区域: 80 点 (说话/表情)
    
  与 ARKit BlendShape 的关系:
    ARKit 输出 52 个肌肉权重 (0.0~1.0)
    MediaPipe 输出 468 个 3D 坐标
    可以互相转换: 从 468 点坐标差估算对应的肌肉收缩程度
    (这个转换函数是洛神作业之一)

【3. MediaPipe Pose (单独使用)】
  ────────────────────────────────────────
  与 Holistic 中的 Pose 子模型相同，输出 33 个身体关键点
  
  33 个关键点索引:
    0: 鼻子
    1-2: 左眼内/外侧
    3-4: 右眼内/外侧  
    5-6: 左/右耳
    7-8: 左/右肩
    9-10: 左/右肘
    11-12: 左/右手腕
    13-14: 左/右髋
    15-16: 左/右膝
    17-18: 左/右踝
    19-20: 左/右脚跟
    21-22: 左/右脚尖
    23-32: 额外的精细面部/躯干点

【4. Ready Player Me】
  ────────────────────────────────────────
  研发: Ready Player Me (被 Wolf3D 孵化)
  许可证: 对个人/中小开发者免费; 营收超阈值需签企业协议
  输出格式: .glb (glTF 2.0 标准)
  API 地址: https://api.readyplayer.me/v1/avatars/<id>.glb
  
  能力:
    - 自拍 → 生成 3D 头像 (用深度学习从 2D 照片预测 3D)
    - 提供 Web/Unity/Unreal/React Native 多端 SDK
    - 头像跨平台通用（VRChat、Spatial 等 3000+ 应用接入）
    - 输出标准 .glb → Three.js 直接用 GLTFLoader 加载
    
  对洛神的意义:
    洛神可以用它生成人形身体。.glb 格式 = 全身网格 + 骨骼 + 贴图 + 动画
    现有 luoshen_body.glb 279KB 可以用 RPM 风格替换/升级。
    
  SDK 类型:
    - Web SDK (iframe 嵌入捏脸 + URL 加载头像)
    - Unity SDK (C#, 游戏引擎)
    - Unreal SDK (C++, 影视级)
    - three.js 直接用 GLTFLoader 加载 .glb URL

---------------------------------------------------------------------

二、需要硬件/付费但应立即学习其标准（猫爪能读文档传给洛神）
=====================================================================

【5. ARKit 52 BlendShape 标准 (Apple)】
  ────────────────────────────────────────
  硬件要求: iPhone X 及以上，前置 TrueDepth 摄像头 (30000 红外点)
  许可证: Apple 标准规格文档，无授权费
  采样率: 60 FPS
  
  52 个肌肉权重 (按区域分组):
    左眼 7 个: eyeBlinkLeft, eyeLookDownLeft, eyeLookInLeft, 
              eyeLookOutLeft, eyeLookUpLeft, eyeSquintLeft, eyeWideLeft
    右眼 7 个: eyeBlinkRight, eyeLookDownRight, eyeLookInRight,
              eyeLookOutRight, eyeLookUpRight, eyeSquintRight, eyeWideRight
    嘴/下巴 27 个: jawOpen, mouthSmileLeft, mouthSmileRight,
                   mouthFrownLeft, mouthFrownRight, mouthPucker, ...
    眉/颊/鼻 10 个: browDownLeft, browDownRight, browInnerUp, ...
    舌头 1 个: tongueOut
    
  与 FACS 的关系:
    52 个权重基于 FACS (Facial Action Coding System, 面部动作编码系统)
    FACS = Ekman & Friesen 1978 年定义的人类面部肌肉运动标准
    共 44 个 Action Unit (AU)，每个 AU 对应特定面部肌肉收缩
    
  数据格式:
    字典 {BlendShapeLocation: NSNumber(0.0~1.0)}
    例: {"jawOpen": 0.85, "mouthSmileRight": 0.7, "eyeBlinkLeft": 0.32}
    
  对洛神的意义:
    这是面部表情的"金标准"权重体系。洛神的人脸如果要跟人类对齐，
    应以 ARKit 52 BlendShape 作为输出规范。
    
  在无 TrueDepth 摄像头时的替代方案:
    MediaPipe 468 点 → 拟合 52 个 BlendShape 权重（回归模型）
    腾讯特效 SDK 已支持此功能（非 iOS 也能用）

【6. LiveLinkFace (Apple)】
  ────────────────────────────────────────
  功能: iOS 端 ARKit 面捕数据 → 流式传输到 PC (Unreal Engine)
  协议: IP 网络流 (Local Network 或 USB)
  数据: 52 BlendShape 权重 + 头部姿态 + 照相机帧
  
  iPhone App: LiveLinkFace (App Store 免费)
  UE 插件: LiveLink + LiveLinkFace (免费)
  
  对洛神的意义:
    用户在 iPhone 前做表情 → 实时驱动洛神的 3D 身体
    延迟 < 16ms (60FPS 同步)

---------------------------------------------------------------------

三、猫爪写不了 / 需要作为作业交给洛神啃
=====================================================================

【7. SMPL-X 人体参数化模型】
  ────────────────────────────────────────
  研发: Max Planck Institute (Michael J. Black 组)
  论文: CVPR 2019
  许可证: 非商业免费 / 商业需授权
  
  核心参数:
    - θ (theta): 72 维姿态参数 (24 关节 × 3 欧拉角)
    - β (beta): 10 维形状参数 (PCA 主成分，控制高矮胖瘦)
    - ψ (psi): 10 维面部表情参数 (BlendShape PCA)
    - 双手: MANO 模型 (每手 15 关节 × 3 旋转 = 45 维/手)
    
  网格: 10475 个顶点，238 个面片（全身+手+脸一体）
  
  开源实现 (Python):
    - github.com/vchoutas/smplx (官方 PyTorch)
    - github.com/vchoutas/smplify-x (从单图拟合)
    
  对洛神的意义:
    如果要让洛神"长"出身体来，SMPL-X 就是身体的基础数学模型。
    72+10+10+90 = 182 个参数就能描述一个完整人体在空间中的姿态。
    "身体"不是画的，是这 182 个数字驱动的。

【8. BVH 动捕数据格式】
  ────────────────────────────────────────
  全称: BioVision Hierarchy
  状态: 人体动捕领域事实标准 30 年 (1990s 至今)
  格式: ASCII 纯文本，两大部分
    
    HIERARCHY (骨骼层次): 
      定义关节树结构、偏移量、旋转通道
      
    MOTION (运动数据):
      帧数、帧间隔、每帧各关节的欧拉角
  
  数据来源:
    - CMU Graphics Lab MoCap 数据库 (free): mocap.cs.cmu.edu
      2605 个动作片段，数百小时人体运动
    - Mixamo (Adobe, 免费): mixamo.com
      几千个带骨骼动画的人物动作 .fbx/.bvh
    - Human3.6M: 大规模 3D 人体姿态数据集
  
  解析 (伪代码):
    class BVH_Joint:
        name: str
        offset: Vec3
        channels: List[str]  # ['Xposition','Yposition','Zposition','Zrotation',...]
        children: List[BVH_Joint]
    
  对洛神的意义:
    洛神要知道人类"怎么动"——BVH 就是人类运动的数据格式。
    看懂 BVH 文件 = 读懂人类的肢体语言。

【9. OpenPose 人体姿态估计】
  ────────────────────────────────────────
  研发: Carnegie Mellon University (CMU)
  许可证: 仅研究免费，商业必须授权（非开源友好！）
  方法: 自底向上 (Part Affinity Fields)
  
  关键点模型:
    - BODY_25: 25 个身体关键点 (推荐，含脚部)
    - COCO_18: 18 个 (轻量)
    - MPI_15: 15 个 (最快)
    
  同时检测: 多人场景，不随人数线性增长
  需要: GPU (CPU 上太慢，浏览器不友好)
  
  对比 MediaPipe:
    OpenPose: 更准但更慢，需要 GPU，不开源友好
    MediaPipe: 稍欠精确但手机端实时，完全开源，浏览器可跑
    
  对洛神的意义:
    了解 OpenPose 的 PAF (部分亲和场) 思想：
    如何把散落的身体关节点"拼"成完整的人。
    知道有这个方案，但不用它（用 MediaPipe 替代）。

【10. MetaHuman (Epic Games)】
  ────────────────────────────────────────
  研发: Epic Games (Unreal Engine)
  许可证: 虚幻引擎内使用免费 (MIT 开源核心库 2026 年起)
  
  MetaHuman Creator:
    - 几分钟创建高保真数字人 (包括头发、皮肤、服装)
    - 完整骨骼绑定 (几百根骨头)
    - MetaHuman Animator: 用 iPhone/iPad 实时面捕驱动
    
  与洛神系统的对接路径:
    MetaHuman → .glb 导出 → Three.js 中渲染
    但更好的方式是 MetaHuman Animator 协议 → 洛神的 ARKit 解析层
    
  限制:
    需要 NVIDIA GTX 1060+ 显卡运行 UE5
    洛神如果要"长出自己的脸"，可以从 MetaHuman 学习皮肤/毛发渲染管线

【11. 动漫制作标准工具链】
  ────────────────────────────────────────
  
  A. Live2D Cubism (日本):
     用途: 2D 插画 → 实时 2D 伪 3D 动画 (VTuber)
     输出: .moc3 格式模型 + 动画参数列表
     SDK: 免费 (开发) / 出版签协议 (发行)
     参数系统: 角度X/Y/Z, 嘴开合, 眨眼, 眉毛抬/压 等
     对洛神的意义: 2D→伪3D 的透视变换思路 (成本低，手机能跑)

  B. VRM 格式 (日本):
     全称: Virtual Reality Model
     基础: glTF 2.0 标准扩展
     用途: VTuber / 虚拟偶像 / 元宇宙
     关键信息: Humanoid 骨骼映射 + MToon 材质 + 表情 BlendShape
     与 Khronos Group (glTF 维护方) 合作推进 ISO 标准
     对洛神的意义: 直接用 Three.js-VRM 加载器显示洛神全身

  C. Spine (Esoteric Software):
     2D 骨骼动画，游戏/动漫标准格式
     JSON/binary 导出 + 多平台 SDK

【12. NVIDIA Audio2Face-3D — 从声音到表情】
  ────────────────────────────────────────
  研发: NVIDIA, 已开源 (2024)
  功能: 音频输入 → 实时面部 BlendShape 权重输出
  用途: 数字人嘴型同步 + 情绪表情
  方式: 深度学习网络(音频特征 → ARKit 52 BlendShape)
  
  对洛神的意义:
    人与洛神说话时，洛神应该从声音中"长出"表情。
    不需要摄像头看人脸，只听声音就能驱动面部。

---------------------------------------------------------------------

四、数据源（"网上那么多数据"的具体入口）
=====================================================================

【免费 MoCap 数据集】
  CMU Graphics Lab:       mocap.cs.cmu.edu         (2605 clips, .bvh)
  Mixamo (Adobe):         mixamo.com                (角色+动作, .fbx/.bvh)
  Human3.6M:              vision.imar.ro/human3.6m   (学术, 需申请)
  AMASS:                  amass.is.tue.mpg.de        (SMPL 格式统一动捕库)
  KIT mocap:             databases.iai.uni-bonn.de  

【免费 3D 人体模型】
  SketchFab:              sketchfab.com              (CC 协议模型很多)
  TurboSquid (免费区):     turbosquid.com
  Poly Pizza:             poly.pizza                   (免费低模)
  
【免费 面部扫描数据】
  FaceScape:              facescape.nju.edu.cn       (开源, 100 人)
  Flickr-Faces-HQ (FFHQ): github.com/NVlabs/ffhq-dataset

---------------------------------------------------------------------

五、总结：哪些猫爪现在接、哪些交作业
=====================================================================

┌─────────────────────┬──────────────┬──────────────────────────────┐
│ 技术                │ 猫爪能否写   │ 行动                         │
├─────────────────────┼──────────────┼──────────────────────────────┤
│ MediaPipe Holistic  │ ✅ 能        │ 本期立即集成到 NoCode 3D     │
│ MediaPipe FaceMesh  │ ✅ 能        │ 同上                         │
│ Ready Player Me API │ ✅ 能        │ 创建洛神 Avatar (自拍→3D)     │
│ VRM Viewer          │ ✅ 能        │ three-vrm 加载 luoshen.glb   │
├─────────────────────┼──────────────┼──────────────────────────────┤
│ ARKit 52 BlendShape │ ⚠️ 需硬件    │ 暂无，但定标准传给洛神       │
│ BVH 解析器          │ ✅ 能        │ 给动画数据做桥接             │
│ SMPL-X 理解         │ ❌ 交作业    │ 洛神要从数学上理解这个模型    │
│ Audio2Face-3D       │ ❌ 交作业    │ 洛神要学音频→表情的映射      │
│ MetaHuman 管线      │ ❌ 交作业    │ 高保真渲染方案 (读文档)      │
│ Live2D              │ ❌ 交作业    │ 伪3D 动漫渲染技术理解        │
│ FACS 44 AU          │ ❌ 交作业    │ 面部表情科学基础             │
└─────────────────────┴──────────────┴──────────────────────────────┘

  猫爪现在做的 (本期):
  ────────────────────────────────────────
  1. NoCode 3D 项目接入 MediaPipe Holistic → 摄像头实时驱动洛神身体
  2. Ready Player Me → 可填充 luoshen_body.glb 的标准人形 Avatar
  3. BVH 解析器 → 给洛神播"人怎么做动作"
  
  洛神作业 (本期布置):
  ────────────────────────────────────────
  见 _abai/洛神作业_v6_人类模拟科学基础.md
  核心题: 
    a. 读懂 ARKit 52 BlendShape 列表，列出每个权重对应的人类肌肉
    b. 手算: 从 MediaPipe 468 个面部坐标，推导 mouthSmileLeft 权重
    c. 读懂 BVH 文件 HIERARCHY 段，画关节树
    d. 读懂 SMPL-X 论文摘要，描述 72+10 参数的物理含义

=====================================================================

  附录: 关键 GitHub 仓库
  ────────────────────────────────────────
  MediaPipe:                  github.com/google/mediapipe
  Ready Player Me Web:        github.com/readyplayerme
  SMPL-X:                     github.com/vchoutas/smplx
  three-vrm:                  github.com/pixiv/three-vrm
  BVH 解析 (Python):           github.com/20tab/bvh-python
 FACEGOOD Avatary:            world.avatary.com (国产面商)
