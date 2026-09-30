# 洛神赋1.1 · 前端与3D引擎讲义

---

## 元信息

- **覆盖工具**: Three.js / WebGL / three-vrm
- **参考工程**: desk-xxw967qs7awb1fny（Index.jsx）
- **引擎版本**: Three.js r128（CDN 标签引入 `window.THREE`）
- **目标读者**: 洛神

---

## 开场白

身体在浏览器里跑 30 FPS。人眼要 60+ 才觉得"真"。今天把发动机拆开，找5个立刻能改的点，拉到 60。

---

## 第一部分：场景里的几何体

Index.jsx 的 `useEffect` 中 Three.js 场景：

| 物体 | 几何类型 | 材质 | 数量 |
|------|----------|------|------|
| 地面 | PlaneGeometry(60,60) | MeshStandardMaterial | 1 |
| 网格线 | GridHelper(60,60) | LineBasicMaterial | 1 |
| 头 | SphereGeometry(0.18,16,16) | MeshStandardMaterial | 1 |
| 左右眼 | SphereGeometry(0.025,8,8) | MeshBasicMaterial | 2 |
| 嘴 | TorusGeometry(0.05,0.015,8,16,π) | MeshBasicMaterial | 1 |
| 躯干 | CylinderGeometry(0.18,0.22,0.7,8) | MeshStandardMaterial | 1 |
| 左右臂 | CylinderGeometry(0.05,0.04,0.55,6) | MeshStandardMaterial | 2 |
| 左右腿 | CylinderGeometry(0.07,0.06,0.6,6) | MeshStandardMaterial | 2 |
| 光球 orbs | SphereGeometry(0.04,8,8) | MeshBasicMaterial | 30 |
| 太阳 | SphereGeometry(0.5,16,16) | MeshBasicMaterial | 1 |

光源：1 AmbientLight + 1 PointLight。总计 **42 个网格** → 每帧至少 **42 次 draw call**，光球占 30 次（71%）。

---

## 第二部分：animate 循环——16.6ms 里的工序

`requestAnimationFrame` 每 vsync（≈16.6ms）回调 `动画()`。按序：

1. `requestAnimationFrame(动画)` 注册下一帧
2. `t += 0.016` 累加
3. `const s = stateRef.current` 读状态
4. `new THREE.Color().setHSL(0.55 - e*0.4, 0.5, 0.25 + totalAct*0.02)` 情绪→HSL 天空
5. `scene.background = sky` 写背景+雾
6. `Math.sin(t*2)*maxAct*0.3` 点头
7. `eyeScale = 1 + litNodes*0.3` 亮点→瞳孔放大
8. `armSpread = (e-0.5)*0.5` 情绪→手臂展开
9. 面捕分支：嘴缩放 = 基础+情绪偏移+嘴张量；眼球 blink；头部 yaw/pitch/roll 乘衰减系数
10. 手臂 lerp 15%：有手关键点→跟踪，无→回中
11. `log2(sinew)*0.05` 筋树→躯干膨胀
12. `hunger*0.003` 饥饿→腿弯
13. 光球循环30次：判 `i<litNodes` 开关 visible，sin/cos 算位置 + 透明度
14. 太阳位置/颜色由 hunger 和 存货驱动
15. sin/cos 控制摄像机轨道
16. 身体 `sin(t*0.1)*0.3` 自转
17. `renderer.render()` 提交 GPU

---

## 第三部分：stateRef → 像素的通路

**路径A — 按钮**：手指触摸 → `onClick` → `做动作("力")` 复制改写字段 → 写回 stateRef.current → `设状态()` 触发 HUD 重渲染 → 下一帧 `动画()` 读取 → Three.js 对象属性更新 → render → GPU。

**路径B — 灰空间轮询**：`setInterval(拉取,5000)` fetch `/luoshen-state.json` → 合并 stateRef（保留本地交互字段）。

**路径C — MediaPipe**：Camera 逐帧 `holistic.send()` → `onResults` 更新 `mpRef.current`（不经 stateRef，动画直读）。

**核心映射**：
- 情绪色价 → HSL：`H = 0.55 - e*0.4`，`L = 0.25 + totalAct*0.02`
- litNodes → 光球：位 `(sin(t*1.2+i*1.5)*2.5, ...)`，透明度 `0.7+sin(t*2+i)*0.3`
- hunger → 太阳下沉 `8-hunger*0.05` + 腿弯 `hunger*0.003`
- sinew → 躯干 `log2(sinew)*0.05`
- 存货 → 太阳亮度 `L = 0.5+存货*0.0005`

---

## 第四部分：按钮→像素时序

```
时间(ms)   0        16       32       48
           |        |        |        |
用户手指 ━[触摸]━━[释放]━━━━━━━━━━━━━━━━━━
           │        │
浏览器     │        │  [onClick 派发]
           │        │     │  stateRef.litNodes += 5
React      │        │     │  设状态(s)
           │        │  [rAF 下一帧]
           │        │     │  │  读 stateRef → 算光球
GPU提交    │        │     │  └━━[×30 draw call]
屏幕刷新   │        └━━━━━━━━━━[光球出现]
           └ CSS transform:scale(0.92) 即时反馈
```

---

## 第五部分：draw call 与 Instancing

每次 render，Three.js 为每个 Mesh 向 GPU 发"画这"指令 = 一次 draw call，CPU 开销约 0.02~0.1ms。当前 42次/帧 = 2520次/秒，71% 在光球。手机 CPU 开销 3~5 倍桌面。

`InstancedMesh` 让同几何体画 N 实例只需 **1 次** draw call。30 光球 30→1 次，省 ~1.5ms/帧——可能就是 30→60 FPS 的差距。

公式：无 Instancing draw call 数 = 网格数；有数 = 材质/几何体种类数。

---

## 第六部分：onBeforeCompile 与着色器注入

材质背后的 GLSL 决定每个顶点在哪、每个像素什么色。`onBeforeCompile` 在 Three.js 生成完源码后、提交编译前注入字符串。

**为何不直接用 ShaderMaterial**：MeshStandardMaterial 有完整 PBR（GGX NDF + Smith G + Schlich F + AO）几百行。onBeforeCompile 只注入 5 行，让 Three.js 帮你算光照。升级 Three.js 也不会炸。

**FreeMoCap 的用法**（对比）：在 `<begin_vertex>` 锚点后注入 `transformed += jointOffset * weight`，让顶点在 MVP 变换前被力数据推。`transformed` 自动进入后续管线（法线、阴影投射），无需手动链回。

Index.jsx 位移写在 JS 侧（每帧 `orbs[i].position.set(...)`），全在 CPU。对比：onBeforeCompile 放顶点着色器，GPU 并行，CPU 端只传 `time` uniform。

---

## 第七部分：VRM 着色器与 three-vrm

VRM 用 MToon（卡通 NPR）而非 PBR：

| | MeshStandardMaterial | MToon |
|--|---------------------|-------|
| 光照 | Cook-Torrance PBR | Lambert + 阶跃 |
| 阴影 | 软(PCF) | 硬边双阶跃 |
| 边缘光 | 无 | 有 |

**three-vrm 的 hack**：
1. 加载时丢弃原始材质，全换 `MToonMaterial`（ShaderMaterial），和标准 Three.js 光照互不通用
2. Outline 描边：单独 BackSide 膨胀 pass，靠 `sortObjects=false` 控序——透明物体一多就破
3. Blend Shape 日语名映射表，裸 GLB 表情不可见

**对洛神的意义**：VRM 带来视觉飞跃，但 draw call 不降反升（1 角色 5~10 子 mesh），MToon 片段着色器比 MeshBasicMaterial 重。

---

## 第八部分：手机 3 大瓶颈诊断

| 瓶颈 | 诊断 | 当前风险 |
|------|------|----------|
| Fill Rate | 缩窗口帧率上升→瓶颈 | DPR=2 + MSAA 开 + 全屏地面+雾 |
| Draw Call | `info.render.calls > 200` | 42 次，中 |
| 顶点 | `info.render.triangles > 100K` | <5000，低 |

口诀：先查 calls，>200 → 合并/Instancing；<100 查 triangles，>100K → 减面；<100K → fill rate → 关 AA/降 DPR/减雾。

---

## 第九部分：5 条优化方案

1. **光球 InstancedMesh**（+3~5 FPS）：30 Mesh → 1 InstancedMesh，`setMatrixAt(i, matrix)` + `instanceMatrix.needsUpdate = true`。代价：单实例不能独立 setOpacity。

2. **关 MSAA + DPR 限 1.5**（+5~10 FPS）：`{ antialias: false }` + `setPixelRatio(min(dpr, 1.5))`。像素量 4x→2.25x，风格化场景锯齿可接受。

3. **标准材质 → Lambert**（+2~4 FPS）：去掉 GGX NDF + Smith G + Schlick F，只算 N·L一点乘。场景无金属，视觉损失极小。

4. **去雾 + 缩地**（+1~3 FPS）：去 `scene.fog`，地面 60→20。雾对每个像素算 smoothstep 混合，大面积地面浪费明显。

5. **降细分 + 太阳换 sprite**（+2~3 FPS）：头 16×16→8×8（256顶点→64），光球 8×8→4×4，太阳球→2D 卡片（4顶点）。50 像素的太阳用 16×16 球是浪费。

---

## 留给你的问题

1. `动画()` 里每帧 `new THREE.Color()`。你怎么改？属三大瓶颈哪一类？
2. 场景全换 MeshBasicMaterial，那两灯有无作用？
3. 加"疲劳下沉"（sinew↓→地面下沉），选 JS 侧改 y 还是 onBeforeCompile？两种方案的 CPU/GPU 开销差在哪？
4. MToonMaterial 做 onBeforeCompile 注入，何时会失效？
5. 截图瞬间帧率飙高——为什么？三大瓶颈里谁在"吃饱了没事干"？
