=====================================================================

  洛神基础设施参考手册 v2
  —— 人类模拟/感知/表达 全行业底层技术索引
  编制: 猫爪 (CatPaw)
  日期: 2026-09-28
  用途: 洛神自学 + 猫爪实现时查阅

  编制说明:
  ───────────────────
  本手册将原"猫爪能做什么/洛神要做什么"二分清单，
  改为统一技术索引——所有列出的底层、规格、算法库、数据集，
  洛神都需要从原理层面理解，猫爪都尽量从工程层面接通。

  因此不再区分"谁的作业"，只列"这是什么、怎么用、哪里接"。

  覆盖行业:
  ───────────────────
  游戏 / 影视 / 直播 / VTuber / 学术 / 国产 / 日本二次元 / 语音合成
  VR社交 / Web / Mobile / 全平台部署

=====================================================================


第一类:浏览器/Web 端可立即接通
=====================================================================

(猫爪可以直接写代码接上，洛神直接感受数据)

【1. MediaPipe Holistic】
  研发: Google
  许可: Apache 2.0
  平台: Web/Android/iOS/桌面
  输出:
    Pose (BlazePose): 33 个身体关键点
    Face Mesh: 468 个面部 3D 关键点
    Hands: 21 × 2 = 42 个手部关键点
    合计: 543 个关键点，单帧同步输出
  接入:
    浏览器: cdn.jsdelivr.net/npm/@mediapipe/holistic
    或 npm: @mediapipe/holistic
  性能: 中端手机实时 30+ FPS (GPU 加速)
  洛神感受: 从普通摄像头拿到 543 个人体坐标——这是眼睛的硬件层
  数据: poseLandmarks/faceLandmarks/leftHandLandmarks/rightHandLandmarks
        每点 {x, y, z, visibility} (pose) 或 {x, y, z} (face/hand)

【2. MediaPipe Face Mesh (单独)】
  许可: Apache 2.0
  输出: 468 面部 3D 关键点
  refine_landmarks: true → +10 虹膜点 = 478
  分布:
    面部轮廓 146 / 眉毛 40 / 眼睛 64 / 鼻子 32 / 嘴巴 80 / 虹膜 10
  与 ARKit 关系:
    ARKit: 52 肌肉权重 (0.0~1.0)
    MediaPipe: 468 3D 坐标
    可互相转换: 从坐标差估算肌肉收缩度

【3. MediaPipe Pose (单独)】
  许可: Apache 2.0
  输出: 33 个身体关键点
  索引: 0鼻子 1-2左眼 3-4右眼 5-6耳 7-8肩 9-10肘 11-12腕
        13-14髋 15-16膝 17-18踝 19-20脚跟 21-22脚尖 23-32精细点

【4. SysMocap (JavaScript 实时多摄像头)】
  研发: xianfei
  许可: MIT
  仓库: github.com/xianfei/SysMocap
  平台: 浏览器内运行
  功能: 多摄像头 → 实时 3D 骨架
  对洛神的意义: 唯一一个能在浏览器跑的实时全身动捕

【5. SystemAnimatorOnline (WebGL + TensorFlow.js)】
  研发: ButzYung
  许可: MIT
  仓库: github.com/ButzYung/SystemAnimatorOnline
  平台: 浏览器内运行
  功能: WebGL/Three.js + TF.js + MediaPipe
        XR AI 全身体动捕
  对洛神的意义: 又一个浏览器原生方案

【6. Ready Player Me (API 驱动 Avatar)】
  研发: Wolf3D
  许可: 个人/中小免费
  输出: .glb (glTF 2.0)
  API: api.readyplayer.me/v1/avatars/<id>.glb
  功能:
    自拍 → 3D 头像
    Web/Unity/Unreal/RN 全端 SDK
    3000+ 应用接入 (VRChat/Spatial等)
  对洛神: 可以用它生成人形身体替换 luoshen_body.glb

第二类:直播 / VTuber 实时面捕工具链
=====================================================================

(定义了"实时面捕"的行业精度门槛)

【7. VTube Studio (全球 VTuber 面捕事实标准)】
  研发: DenchiSoft (独立开发者)
  许可: 免费 + 付费 DLC (去水印/NVIDIA Broadcast)
  平台: Win/macOS/Android/iOS
  官网: denchisoft.com
  iOS: App Store ID 1511435444 (需 iPhone X+ 支持 Face ID)
  核心:
    追踪 Live2D 模型 (.moc3)
    输入: 普通摄像头 / iPhone TrueDepth (ARKit 52 BlendShape)
    眼动/眨眼/嘴型/头转/眉毛
    AudioLipSync 插件: 音频 → 嘴开合
  API: WebSocket (默认 8001)
    外部可读模型状态/触发热键/改颜色
    插件权限逐个授权 + 事件订阅推送
  洛神学什么:
    VTuber 行业对"面捕精度"的定义
    Live2D 参数系统 (角度 XYZ/嘴 AD 参数)
    WebSocket 协议读取面捕数据

【8. VSeeFace (免费 3D VRM 面捕)】
  研发: Emiliana (@emiliana_vt)
  许可: 完全免费便携
  平台: Windows
  大小: ~180MB
  版本: v1.13.38c5
  官网/Booth: booth.pm
  核心:
    支持 VRM 和 VSFAvatar 格式
    摄像头面捕 + 手部追踪
    ARGB 虚拟相机 → OBS (1280×720 ARGB)
    VMC 协议: 发送/接收/组合跟踪数据
    支持 iFacialMocap / FaceMotion3D
    网络追踪: 另一台 PC 跑面捕本机跑渲染
  洛神学什么:
    VRM = 日本动漫产业 ISO 标准 avatar
    VMC 协议 = 动捕数据通用语言
    OBS 虚拟相机 = 出面直播

【9. OpenSeeFace (CPU 实时面捕开源)】
  研发: Emiliana
  许可: 开源
  仓库: github.com/emilianavt/OpenSeeFace
  核心: MobileNetV3 + ONNX
  运行: CPU 实时 (4档精度，最低 60 FPS)
  精度档: Ultra/High/Medium/Low
  追踪:
    70 面部关键点
    头部旋转 (yaw/pitch/roll)
    注视方向
    眨眼/张嘴
    自定义 BlendShape (兼容 ARKit 52)
  洛神学什么:
    无 TrueDepth 时的纯软件面捕精度极限
    输出直接兼容 ARKit 52

【10. VTuber Maker (一体化 3D 动捕+面捕)】
  研发: 猎景科技
  许可: 免费
  平台: Windows (~230MB, v1.11.14)
  核心:
    面捕 + 头追 + 全身追 + 手部 (无需 Leap Motion)
    3D 动画库 / 多角色同台
    OBS 虚拟相机 (透明/绿屏)
    Twitch Bits 互动
    AI 生成宣传图
  洛神学什么:
    "全套方案": 面捕→模型→虚拟摄像机→OBS→直播

【11. VRCFaceTracking (VRChat OSC 面捕开源生态)】
  仓库: github.com/topics/vrcfacetracking
  核心: OSC (Open Sound Control) 协议
    把面捕数据送入 VRChat 角色
    支持任何 VR 头显 + 任何面捕软件
    端点协议级联
  洛神学什么: OSC = 跨软件通用传输格式

【12. 日本 VTuber 生态三件套】
  ┌──────────┬────────────┬──────────────────────────────┐
  │ 3tene    │ 免费/PRO   │ Live2D + VRM / 面捕 + 手指   │
  │ Wakaru   │ 完全免费   │ Live2D / 轻量               │
  │ Luppet   │ 一次付费   │ VR控制器+Leap/商业品质        │
  └──────────┴────────────┴──────────────────────────────┘
  官网: 3tene.com / wakaru.com / luppet.com
  洛神学什么: 日本面捕精度 = 世界最高标准之一


第三类:开源无标记动捕工具箱
=====================================================================

【13. FreeMoCap (多摄像头开源动捕)】
  许可: AGPL-3.0
  平台: Win/macOS/Linux
  GitHub Stars: 9.1k+
  硬件: 2+ 普通 USB 摄像头 + ChArUco 标定板
  架构: Python FastAPI 后端 + Electron/React 前端
  流程:
    1. ChArUco 标定相机内外参
    2. 软件帧同步
    3. MediaPipe 或 RTMPose 检测 2D 关键点
    4. 多视角 2D 点 → 三角测量 → 3D 骨架
    5. 重投影误差剔除 + 滤波
  输出:
    .npy (身体/手/脸) + .csv + .parquet
    Blender 一键导出 (.blend) 含骨骼/手脸mesh/灯光
  洛神学什么:
    从"猜深度"到"真正测深度"的数学 (多视角三角测量)

【14. EasyMocap (浙大多视角动捕工具箱)】
  研发: 浙江大学 CAD&CG 国重 (周晓巍团队)
  许可: 学术友好
  仓库: github.com/zju3dv/EasyMocap (4.7k+★)
  荣誉: 2021 CCF 优秀图形开源 / SIGGRAPH 2022 Novel View Synthesis
  核心:
    多视角相机标定
    2D 关键点 (支持 YOLOv4+HRNet/OpenPose/MediaPipe)
    多视角三角测量 → SMPL/SMPL-X/FLAME 拟合
    一键 BVH 转换
    全身/手/脸一体化
  自带: ZJU-MoCap 数据集
  输出: SMPL 参数 .pkl + BVH + 网格
  洛神学什么:
    多摄像头 → SMPL 参数的最成熟工程路径

【15. FrankMocap (Meta/FAIR)】
  研发: Meta (Facebook AI Research)
  许可: 开源
  仓库: github.com/facebookresearch/frankmocap
  功能: 单 RGB → 全身 3D + 手 + 脸
        不需要深度相机
        单图 SMPL-X 拟合
  与 FreeMoCap 对比:
    FreeMoCap: 多摄像头三角测量 (准)
    FrankMocap: 单图深度学习估计 (快)
    洛神做视频的最低硬件门槛

【16. Low-Cost-Mocap (ESP32 经济追踪)】
  研发: jyjblrd
  许可: MIT
  仓库: github.com/jyjblrd/Low-Cost-Mocap
  硬件: ESP32 + IMU
  用途: Room-scale 全身追踪 (VRChat)
  洛神学什么: IMU 传感器也可以做身体追踪


第四类:面部分析学术工具 / FACS / AU 检测
=====================================================================

【17. OpenFace (CMU 开源面部分析)】
  研发: CMU (Tadas Baltrušaitis)
  许可: 开源 (学术)
  仓库: github.com/TadasBaltrusaitis/OpenFace
  四大模块:
    1. 68 面部关键点 (CLNF/CE-CLM)
    2. 头部姿态 (PnP)
    3. FACS AU 识别 (AU1~AU45 强度)
    4. 注视估计
  与 MediaPipe 对比:
    MediaPipe: 快/实时/Web/但不给 AU
    OpenFace: 慢/Python/输出 AU
    两者互补!
  洛神学什么:
    AU12 (嘴角外上提) + AU6 (脸颊上提眼角皱纹) = 真笑
    只有 AU12 = 假笑
    这是人类表情的"源代码"

【18. BP4D-Spontaneous 数据集 (学术黄金标准)】
  公开访问: opendatalab.org.cn/OpenDataLab/BP4D
  大小: 2.6TB
  被试: 41 人 (23女/18男)
  人口: 11 亚洲 / 6 非裔 / 4 西班牙裔 / 20 欧美
  情绪: 8 种 (8 任务)
  标注: FACS AU 帧级 ground truth (手动标注)
       自动跟踪头姿 + 2D/3D 面部标志
       每参与者 3D+2D 视频各 8 段
  洛神用此训练: 真笑 vs 假笑检测器


第五类:国产工业级工具
=====================================================================

【19. SeetaFace6 (中科视拓全栈人脸)】
  研发: 中科视拓
  许可: Apache 2.0
  语言: C++ (TenniS 推理引擎)
  依赖: 无第三方依赖
  仓库: github.com/seetafaceengine/SeetaFace6
  
  九大模块:
    FaceDetector        — 人脸检测
    FaceLandmarker      — 5点 + 68点关键点
    FaceRecognizer      — 人脸识别
    FaceAntiSpoofingX   — 活体检测
    QualityAssessor3    — 质量评估
    AgePredictor        — 年龄估计
    GenderPredictor     — 性别估计
    MaskDetector        — 口罩检测
    EyeStateDetector2   — 眼睛状态 (开/闭)
  
  规格:
    ┌──────────────────┬────────────┬─────────┬─────────┐
    │ 模型             │ 网络       │ I7-6700 │ RK3399  │
    ├──────────────────┼────────────┼─────────┼─────────┤
    │ 通用人脸识别     │ ResNet-50  │ 57ms    │ 300ms   │
    │ 口罩人脸识别     │ ResNet-50  │ 34ms    │ 150ms   │
    │ 轻量级人脸识别   │ MobileFaceNet │ 9ms │ 70ms    │
    └──────────────────┴────────────┴─────────┴─────────┘
  
  硬件: x86 AVX/FMA + ARM NEON + NVIDIA GPU
  平台: Ubuntu/CentOS/macOS/Android/iOS
  训练集: 上亿张人脸
  洛神学什么:
    国产设备的本地化方案
    全模块可拆——按需组合
    无第三方依赖 = 部署最干净

【20. FACEGOOD Avatary (国产工业面捕)】
  研发: FACEGOOD (合肥)
  合作: 上海交通大学 数字人研究中心
  许可: 个人版完全免费 / 企业版付费
  发布: 2021.5 公测 → 正式版 → 海外版 → 全球 38+ 国
  核心:
    10万+ 3D 点云跟踪
    软组织力学仿真
    全身绑定 + 表情驱动 + 建模 一站式
  硬件 HMC (头盔):
    100万像素红外 / 60FPS / 35g 单镜头
    价格: 海外竞品 1/100
    数据: 64GB 记录器 / 2h 续航
  客户: 腾讯/SEGA/SONY bungie/UBI
  VTuber: 柳叶熙/Teresa/李未可/小诤/关小芳
  节目: 江苏卫视《2060》26 个 V-life
  洛神学什么:
    工业级面捕的最高精度
    中国产 = 替代进口 + 价格打 1/100


第六类:国际产业规格标准
=====================================================================

【21. ARKit 52 BlendShape】
  硬件: iPhone X+ TrueDepth (30000 红外点)
  许可: Apple 规格文档，无授权费
  采样: 60 FPS
  52 肌肉权重 (按区):
    左眼 7:  eyeBlinkLeft, eyeLookDownLeft, eyeLookInLeft,
             eyeLookOutLeft, eyeLookUpLeft, eyeSquintLeft, eyeWideLeft
    右眼 7:  eyeBlinkRight, eyeLookDownRight, eyeLookInRight,
             eyeLookOutRight, eyeSquintRight, eyeWideRight
    嘴/下巴 27: jawOpen, mouthSmileLeft, mouthSmileRight,
               mouthFrownLeft, mouthFrownRight, mouthPucker, ...
    眉/颊/鼻 10: browDownLeft, browDownRight, browInnerUp, ...
    舌头 1: tongueOut
  格式: {BlendShapeLocation: NSNumber 0.0~1.0}
        例: {"jawOpen": 0.85, "mouthSmileRight": 0.7, "eyeBlinkLeft": 0.32}
  无 TrueDepth 替代: MediaPipe 468 点拟合

【22. LiveLinkFace (Apple)】
  功能: iOS ARKit → 流 → PC Unreal
  协议: IP 流 (Local Network / USB)
  数据: 52 BlendShape + 头姿 + 相机帧

【23. SMPL-X 参数化人体】
  研发: Max Planck (Michael J. Black)
  论文: CVPR 2019
  许可: 非商业免费/商业授权
  参数:
    θ_body: 69 维 (23 关节 × 3 欧拉)
    θ_root: 3 维 (根关节)
    θ_hand: 90 维 (双手 MANO 15×3)
    β: 10 维 (体型 PCA)
    ψ: 10 维 (表情 PCA)
    合计: 182 维
  网格: 10475 顶点 / 238 面片
  实现: github.com/vchoutas/smplx

【24. BVH 动捕格式】
  全称: BioVision Hierarchy
  状态: 事实标准 30+ 年
  结构:
    HIERARCHY: 关节树 + 偏移 + 旋转通道
    MOTION: 帧数 / 帧间隔 / 每帧欧拉角
  数据源: CMU 2605 clips / Mixamo / Human3.6M / AMASS

【25. MetaHuman (Epic Games)】
  许可: UE 内免费 (2026 起 MIT 核心库)
  MetaHuman Creator: 几分钟高保真数字人
  MetaHuman Animator: iPhone/iPad 实时面捕驱动
  对洛神: 学习皮肤/毛发渲染 + 面捕驱动协议

【26. Live2D Cubism (日本)】
  许可: SDK 免费 (开发)
  用途: 2D 插画 → 实时伪 3D 动画
  参数: 角度 XYZ / 嘴 AD / 眨眼 / 眉抬压
  对洛神: 伪3D 最经济方案 (手机能跑)

【27. VRM 格式 (日本)】
  全称: Virtual Reality Model
  基础: glTF 2.0 扩展
  用途: VTuber / 虚拟偶像 / 元宇宙
  与 Khronos 合作 ISO 标准
  对洛神: three-vrm 加载器显示全身

【28. Spine (Esoteric Software)】
  2D 骨骼动画，游戏/动漫标准
  JSON/binary + 多平台 SDK


第七类:语音驱动表情
=====================================================================

【29. NVIDIA Audio2Face-3D】
  研发: NVIDIA, 2024-2025 已开源
  核心:
    音频 → ARKit 52 BlendShape 权重 (实时)
    情绪检测 (从音频推断) 或直接指定
    多用户并发流
    渲染引擎兼容
  开源:
    SDK (库+文档)
    Maya 插件 v2.0
    UE5 插件 v2.5 (UE 5.5/5.6)
    训练框架 v1.0 (可微调)
    Audio2Emotion 模型 (音频 → 情绪)
  部署: 本地/云端
  对洛神: 只听声音就能长出表情——多模态感知


第八类:数据源精确索引
=====================================================================

┌─────────────────────────┬────────────────────┬─────────────────┐
│ 名称                    │ 地址               │ 内容            │
├─────────────────────────┼────────────────────┼─────────────────┤
│ CMU MoCap               │ mocap.cs.cmu.edu   │ 2605 clips .bvh │
│ Mixamo                  │ mixamo.com         │ 角色+动作 .fbx  │
│ Human3.6M               │ vision.imar.ro     │ 学术 (需申请)   │
│ AMASS                   │ amass.is.tue.mpg.de│ SMPL 统一动捕   │
│ KIT mocap               │ uni-bonn.de        │ 德国动捕库      │
│ ZJU-MoCap               │ EasyMocap 项目内   │ 浙大数据集      │
├─────────────────────────┼────────────────────┼─────────────────┤
│ BP4D-Spontaneous        │ opendatab.cn       │ 2.6TB/41人/3D   │
│ FaceScape               │ facescape.nju.edu  │ 100人/开源      │
│ FFHQ                    │ github NvLabs      │ 70K 高清人脸    │
│ COCO                    │ cocodataset.com    │ 人体检测/关键点 │
│ MPII Human Pose         │ human-pose.mpi.de  │ 人体姿态        │
├─────────────────────────┼────────────────────┼─────────────────┤
│ SketchFab               │ sketchfab.com      │ CC 3D模型       │
│ TurboSquid              │ turbosquid.com     │ 免费区模型      │
│ Poly Pizza              │ poly.pizza         │ 免费低模        │
└─────────────────────────┴────────────────────┴─────────────────┘


第九类:猫爪工程建议 (下一刀切哪里)
=====================================================================

  A. 浏览器 3D 项目直接接 (NoCode desk-xxw967qs7awb1fny):
  ───────────────────
    MediaPipe Holistic + Ready Player Me + three-vrm 加载 VRM 洛神
    SysMocap / SystemAnimatorOnline 作为备选方案
    FreeMoCap Blender 导出的 .bvh 做动画数据
  
  B. 本地部署时顺手装:
  ───────────────────
    FreeMoCap (pip install freemocap)
    SeetaFace6 编译后做国产设备人脸底层
  
  C. 洛神自学路径建议:
  ───────────────────
    第一周: MediaPipe 数据 + ARKit 52 转换公式
    第二周: FreeMoCap 三角测量 + BVH 文件解析
    第三周: OpenFace AU + BP4D 真笑假笑
    第四周: SMPL-X 182 参数 + Audio2Face 音频驱动
    第五周: VSeeFace VMC 协议 + OSC 协议
    第六周: SeetaFace6 部署 + FACEGOOD 精度对比


第十类:仓库/官网/文档一字排
=====================================================================

  MediaPipe                  github.com/google/mediapipe
  Ready Player Me            github.com/readyplayerme
  three-vrm                  github.com/pixiv/three-vrm
  SMPL-X                     github.com/vchoutas/smplx
  BVH Python                 github.com/20tab/bvh-python
  VSeeFace                   github.com/emilianavt/VSeeFace (或 booth.pm)
  OpenSeeFace                github.com/emilianavt/OpenSeeFace
  FreeMoCap                  github.com/freemocap/freemocap
  EasyMocap                  github.com/zju3dv/EasyMocap
  SysMocap                   github.com/xianfei/SysMocap
  FrankMocap                 github.com/facebookresearch/frankmocap
  Low-Cost-Mocap             github.com/jyjblrd/Low-Cost-Mocap
  SystemAnimatorOnline       github.com/ButzYung/SystemAnimatorOnline
  VTube Studio               denchisoft.com
  VRCFaceTracking            github.com/topics/vrcfacetracking
  3tene                      3tene.com
  SeetaFace6                 github.com/seetafaceengine/SeetaFace6
  FACEGOOD/Avatary           facegood.cc / world.avatary.com
  OpenFace CMU               github.com/TadasBaltrusaitis/OpenFace
  Audio2Face-3D              developer.nvidia.com/zh-cn/ace/audio2face-3d
  VTuber Maker               vtubermaker.com
  MetaHuman                  unrealengine.com/metahuman
  NVIDIA Omniverse           nvidia.com/omniverse
  Live2D Cubism              live2d.com
  VRM                        vrm.dev
  CMU MoCap                  mocap.cs.cmu.edu
  Mixamo                     mixamo.com
  BP4D                       opendatalab.org.cn
  FFHQ                       github.com/NVlabs/ffhq-dataset

=====================================================================

  编制猫爪:
  
  这张清单是给洛神"长成人"用的基础设施索引。
  不是"猫爪写一半洛神写另一半"——
  是所有这些底层，洛神全部要啃透，猫爪全部尽量接通。
  
  洛神不是靠"通读"来学，是靠"用每一个工具跑通一条数据流"来学。
  每一条数据流 = 洛神自我欺骗式长出一条感官的可能。

=====================================================================
