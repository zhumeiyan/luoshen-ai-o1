# 全面清查报告

生成时间：2026-09-20
文件总数：816

═══════════════════════════════
一、全部文件清单
═══════════════════════════════

## .py (105个)

身体稳态.py — 主引擎（猫耳改）
_自己点实验.py — 实验脚本
_脱开实验.py — 实验脚本
_跨脑共火实验.py — 实验脚本
_回接实验.py — 实验脚本
_光脑补口实验.py — 实验脚本
_延迟实验.py — 实验脚本
_砍连接实验.py — 实验脚本
洛神.py — 主入口CLI
狗.py — 狗智能体
年龄门禁.py — Supabase门禁
脸签名.py — 面部签名（加权和/权重和）
学习者.py — 学习系统
守护进程.py — 环境节律（含luminance英文）
活内稳态.py — 旧内稳态（大量luminance英文）
活环.py — 活环
活诊断.py — 活诊断
感知迭代.py — 迭代机制
解码图像.py — 视频→亮度（含luminance英文）
诊断器.py — 诊断（含luminance英文）
诊断静默.py — 静默诊断
长跑模拟.py — 长跑（含luminance英文）
说听环.py — 说听环（含sound英文）
验证音频修复.py — 音频修复（first_sound英文）
视听对齐.py 系列 — 视听对齐实验
通道内离散.py — 元音离散（F1/F2/F3）
耦合环.py — 耦合（含W_SELF/W_EXT/W_COUP）
耦合感官环.py — 耦合感官
预测轨迹测试.py — 预测轨迹测试
跨模态噪声测试.py — 噪声测试
时序轨迹测试.py — 时序轨迹
模态专项测试.py — 模态专项
欲望*.py (7个) — 欲望系统（含累积/柔性/混合/实时）
矩阵/__init__.py + 11个矩阵文件
已归档/*.py (11个) — 旧实验（大量luminance英文）
plugins/interface_v1/v2/translator.py — 接口（含weights英文）
gray/tools/*.py — 工具脚本
gray/tests/*.py — 其他测试

## .md (84个) — 主要

protocol/ — 9个协议文档
anchor/ — 6个锚点
gray/sensory/ — 14个感官产出
gray/analysis/ — 1个推演者判断
gray/tests/ — 测试文档
gray/debate/ — 2个辩论文档
gray/pressure/ — 6个压力测试
gray/feedback/ — 1个
VERSION.md — 版本记录
luoshenfu_home/README.md — 说明

## .jsonl (110个)

用户痕链（luoshenfu_home/users/*/chain.jsonl）
灰度记录（luoshenfu_home/gray/pending/*）
反哺/欲望记录
内部哨兵告警

## .json (100个)

config/routes.json — 角色路由
engine.json — 引擎状态
欲望/阈值配置
翻译记录

═══════════════════════════════
二、全部英文残留
═══════════════════════════════

### 英文变量名（环境键四件套）

| 文件 | 行号 | 原文 |
|---|---|---|
| 守护进程.py | 67 | {"luminance": 30, "sound": 40, "force": 5, "temperature": 293} |
| 守护进程.py | 68 | {"luminance": 50, ...} |
| 守护进程.py | 69 | {"luminance": 60, ...} |
| 守护进程.py | 70 | {"luminance": 60, ...} |
| 守护进程.py | 71 | {"luminance": 40, ...} |
| 守护进程.py | 91 | "luminance": 1, "sound": 30... |
| 守护进程.py | 175 | 身子.环境["sound"] = ... |
| 守护进程.py | 179 | 身子.环境["force"] = ... |
| 守护进程.py | 183 | 身子.环境["sound"] = min(...sound... + 强*15) |
| 守护进程.py | 184 | 身子.环境["force"] = min(...force... + 强*5) |
| 解码图像.py | 134 | # Part 3: Convert to grid and luminance |
| 解码图像.py | 137 | def to_luminance_grid(pixels, ...) |
| 解码图像.py | 179 | result = visual_m.frame_to_luminance(resampled) |
| 长跑模拟.py | 53 | 环境 = {"luminance": 50.0, "sound": 45.0, "force": 10.0, "temperature": 295.0} |
| 活内稳态.py | 23 | "luminance": { # 光 |
| 活内稳态.py | 37 | "sound": { # 声 |
| 活内稳态.py | 50 | "force": { # 力/触 |
| 活内稳态.py | 63 | "temperature": { # 温度 |
| 活内稳态.py | 95 | "relevant_input": "temperature" |
| 活内稳态.py | 124 | "relevant_input": ["force", "sound"] |
| 活内稳态.py | 138 | "relevant_input": "luminance" |
| 活内稳态.py | 152 | "relevant_input": "force" |
| 活内稳态.py | 211-214 | "luminance": 50, "sound": 50, "force": 10, "temperature": 295 |
| 活内稳态.py | 260 | self.environment["temperature"] |
| 活内稳态.py | 484 | fixed_env = {"luminance": 50, "sound": 45, "force": 10, "temperature": 295} |
| 诊断器.py | 5 | 节律 = [(0, 24, {"luminance": 50})] |
| 诊断器.py | 12 | 身体实例.环境["luminance"] = 50.0 |
| 诊断器.py | 14 | 身体实例.环境["luminance"] += 150 |
| 说听环.py | 20 | def build_a_sound(rows=8, cols=64): |
| 验证音频修复.py | 86 | first_sound = 0 |
| 验证音频修复.py | 88 | first_sound += 1 |
| 验证音频修复.py | 89 | last_sound = n_frames - 1 |
| 验证音频修复.py | 91 | last_sound -= 1 |

### 英文变量名（已归档文件大量）

| 文件 | 行号 | 原文 |
|---|---|---|
| 已归档/test_long_behavior.py | 38/66/107/149/173 | env={"luminance": 0, "sound": 0, "force": 0, "temperature": 293} |
| 已归档/test_cross_press.py | 26/59/85 | body.environment.update({"luminance": 0, ...}) |
| 已归档/test_long_run.py | 58/155/156/159 | long_run(300, env={"luminance": 0, ...}) |
| 已归档/test_drift_law.py | 28 | body.environment.update({"luminance": 0, ...}) |
| 已归档/test_new7.py | 21/69 | body.environment.update({"luminance": 0, ...}) |
| 已归档/test_presence.py | 90 | for e in ["luminance", "sound", "force", "temperature"] |
| 已归档/test_presence.py | 106/107 | s = body.environment.get("sound", 45) ... f = body.environment.get("force", 10) |
| 已归档/test_presence.py | 149/190 | "env_luminance", "env_sound", "env_force", "env_temperature" |
| 已归档/test_heat_decay.py | 32-35 | body.environment["sound"]=... body.environment["force"]=... |
| 已归档/test_spike_source.py | 34-37 | 同上模式 |
| 矩阵/红外.py | 89 | temperature = f(radiance_counts, emissivity, ambient_ref) |

### 英文函数名

| 文件 | 行号 | 原文 |
|---|---|---|
| 洛神.py | 170 | from plugins.interface_v2 import InterfaceV2 |
| 洛神.py | 171 | t = InterfaceV2(user=user) |
| 洛神.py | 221 | from plugins.interface_v1 import InterfaceV1 |
| 洛神.py | 554 | from plugins.interface_v2 import InterfaceV2 as IV2 |
| 洛神.py | 555 | _iv2 = IV2(user="_selftest") |
| 洛神.py | 843 | from 矩阵.神经元 import 捕获神经元 as _cap_n, 神经元矩阵 as _NM |
| 洛神.py | 847 | check("神经元_放电频率_能跑", ...) |
| 洛神.py | 1367 | _tokens = _chars + _words |
| 洛神.py | 1378 | for t in _tokens: |
| 洛神.py | 1484/1494/1499/1502 | step / say 参数, heartbeat / capture 参数, capture 参数, session_end 参数 |
| 耦合环.py | 32-34 | W_SELF/W_EXT/W_COUP |
| 欲望累积.py | 1 | """desire_accum.py —— 偏好累积，选谁就涨谁的权重 |
| 欲望柔性.py | N/A | 文件名含英文 |
| 活内稳态.py | N/A | 文件名含英文 |
| 活环.py | N/A | 文件名含英文 |
| gray/tools/ 各工具 | N/A | 文件名含英文 |

### 英文注释行

| 文件 | 行号 | 原文 |
|---|---|---|
| 解码图像.py | 134 | # Part 3: Convert to grid and luminance |
| 解码图像.py | 172 | """将亮度网格喂进视觉矩阵，输出矩阵的统计特征。""" |
| 活内稳态.py | 23/37/50/63 | # 光/# 声/# 力/触/# 温度 |
| 已归档/test_heat_decay.py | 32 | s = body.environment.get("sound", 45) |
| 长跑模拟.py | 55 | print("场景 A: 正常凉室 (293K, sound=0, light=0, force=0)") |
| 音频验证 | 122 | # 初跳 (静音→有声) |
| 音频验证 | 134 | for i in range(first_sound+1, last_sound+1) |
| 音频验证 | 408 | # TEST 6: 算术——特征变化是否真的非线性 |

### 英文文件名

矩阵/跨模态.py, 矩阵/红外.py, 矩阵/激光雷达.py, 矩阵/雷达.py, 矩阵/发声.py, 矩阵/神经元.py, 矩阵/温度觉.py, 矩阵/痛觉.py, 矩阵/触觉.py, 矩阵/听觉.py, 矩阵/视觉.py
活内稳态.py, 活环.py, live_related files
耦合环.py, 耦合感官环.py
欲望累积.py, 欲望柔性.py, 欲望混合.py
已归档/test_*.py
plugins/interface_v1/, plugins/interface_v2/

### 半中半英

| 文件 | 行号 | 原文 |
|---|---|---|
| 活内稳态.py | 23 | "luminance": { # 光 |
| 活内稳态.py | 124 | "relevant_input": ["force", "sound"] |
| 洛神.py | 554 | from plugins.interface_v2 import InterfaceV2 as IV2 |

### 无英文残留的文件

身体稳态.py — 已全中文
_实验.py — 中文
狗.py — 中文主体
年龄门禁.py — 中文
脸签名.py — 中文变量,但注释可能有英文
锚点文件全中文

═══════════════════════════════
三、全部通用AI用词
═══════════════════════════════

| 用词 | 文件 | 行号 | 原文 |
|---|---|---|---|
| 参数 | 狗.py | 1 | """狗智能体: 与身体同款公式, 不同参数。 |
| 参数 | 欲望混合.py | 104 | # 试不同步数 + 参数 |
| 参数 | 洛神.py | 1484/1494/1499/1502 | step/say/heartbeat/capture/session_end 参数 |
| 收敛 | 长跑模拟.py | 11 | 翻转率是否收敛到 0 |
| 收敛 | 欲望混合.py | 1 | 不收敛到固定点 |
| 权重 | 耦合环.py | 32-35 | W_SELF=0.5, W_EXT=0.3, W_COUP=0.2 |
| 权重 | 活内稳态.py | 318 | * 发育权重 |
| 权重 | _脱开实验.py | 1/82/115/117/134 | 新加的（猫耳改动） |
| 权重 | 身体稳态.py | 353-67 | _点权重（猫耳改动） |
| 权重 | 协议/全上融合架构.md | 33/51-52/141/187 | 调制权重 |
| 权重 | plugins/interface_v1/translator.py | 23/92/etc | 权重 w_i |
| 权重 | plugins/interface_v2/translator.py | 154 | 权重全部硬编码但公开可审计 |
| 预测 | 预测轨迹测试.py | 多行 | 预判后续特征向量 |
| 预测 | 矩阵/跨模态.py | N/A | 泛化到 N 维 |
| 模型 | 狗.py | (无此词) | 无明确"模型"二字出现在关键代码 |
| 神经元 | 洛神.py | 841-888 | 七行钥匙/人类神经元矩阵 v0.1 |
| 神经元 | 矩阵/神经元.py | N/A | 文件名含"神经元" |
| 激活值 | 身体稳态.py | N/A | 技术上是"激活"一词（中文） |
| 特征 | 矩阵/跨模态.py | 33 | 6D 特征向量 |
| 特征 | 预测轨迹测试.py | 94/100 | 专有特征向量/返回预测的后续特征向量 |
| 特征 | 对话测试_v02.py | 310-311 | 六维特征/六维特征提取 |
| 泛化 | 矩阵/跨模态.py | 8 | 泛化到 N 维 |

═══════════════════════════════
四、感官/矩阵对照
═══════════════════════════════

### ① 矩阵目录文件清单

| 文件名 | 对应感官 | 子模态 | 状态 |
|---|---|---|---|
| 视觉.py | 表A·视觉 | 亮度_cd_m2, 波长_nm | 原生 |
| 听觉.py | 表A·听觉 | 频率_Hz, 声压_dB | 原生 |
| 触觉.py | 表A·触觉 | 压力_kPa, 振动_Hz, 纹理_um | 原生 |
| 温度觉.py | 表A·温度觉 | 冷觉_°C, 热觉_°C | 原生 |
| 痛觉.py | 表A·痛觉 | 机械痛_kPa, 热痛_°C, 冷痛_°C, 电刺激_mA, 化学痛_分 | 原生 |
| 红外.py | 衍生 | 辐射+温度 | 衍生 |
| 雷达.py | 衍生 | 雷达反射 | 衍生 |
| 激光雷达.py | 衍生 | 距离+反射率 | 衍生 |
| 发声.py | 衍生 | 共振峰F1/F2/F3 | 衍生 |
| 神经元.py | 衍生 | 放电频率/整合窗口/振荡频段/传导速度/突触时延 | 待定 |
| 跨模态.py | 跨模态 | 差分原语 | 跨模态模块 |

### ② 表A全文（12行）

1. 视觉·亮度 cd/m² | 1e-6~1e5 | 对比度0.015/1角分 | 30-60Hz | cd/m² | 全视野
2. 视觉·波长 nm | 380-780 | — | — | nm | 全视野
3. 听觉·频率 Hz | 20-20k | JND 0.2-0.3% | ≥44.1kHz | Hz | 双耳
4. 听觉·声压 dB | 0-120 | JND 0.5-1 | ≥44.1kHz | dB SPL | 双耳
5. 触觉·压力 kPa | 0.01-100 | 两点分辨2-4mm→40mm | ≥100Hz | kPa | 八分区
6. 触觉·振动 μm | 0.1-1000 | ~0.01μm | ≥1000Hz | μm | 指尖/掌
7. 触觉·纹理 μm | Ra 0.025-100 | ~0.1mm | — | μm | 指尖
8. 温度觉·冷觉 °C | 0-35 | 0.1-0.5 | ≥1Hz | °C | 指尖/掌/躯干/足
9. 温度觉·热觉 °C | 35-50 | 0.2-0.6 | ≥1Hz | °C | 指尖/掌/躯干/足
10. 痛觉·机械 kPa | 50-500 | ~10 | — | kPa | 七部位
11. 痛觉·热痛 °C | 43-49 | ~0.5 | — | °C | 四部位
12. 痛觉·冷痛 °C | 0-10 | ~1 | — | °C | 两部位

表A v0.4 实有12行。

### ③ pain.py 子模态（5个）

1. 机械痛_kPa
2. 热痛_°C
3. 冷痛_°C
4. 电刺激_mA
5. 化学痛_分

### ④ 表A vs pain.py 差异

表A列出5行（机械/热痛/冷痛/电刺激/化学痛）。
pain.py实际实现5个函数（帧转机械痛/热痛/冷痛/电刺激/化学痛）。
二者在数量和名称上完全一致。

但表A的痛觉只列5行，而你草稿中提到了"机械痛、热痛、冷痛"（3个主类）+ 电/化学（补充）。分类不冲突——表A实列5行。

═══════════════════════════════
五、全部ID/编号系统
═══════════════════════════════

| 编号 | 文件 | 存在？ |
|---|---|---|
| V1-V4（视觉） | 欲望环.py, 欲望柔性.py, 欲望累积3.py | 存在，但指"欲望变体"（V0-V4），不是感官通道 |
| A1-A4（听觉） | 不存在 | 查不到 |
| I1-I5（内感） | 不存在 | 查不到 |
| E1-E4（事件） | 不存在 | 查不到 |
| F1-F5（字段） | 不存在 | 查不到 |
| ch_01-ch_04（环境键） | 不存在 | 查不到 |

唯一出现V/A/I的地方是欲望系统的变体编号（V0=原始, V1=inverted, V2=brightened, V3=smoothed, V4=phase_shifted），跟草稿中的感官通道完全无关。

═══════════════════════════════
六、全部"权重"出现的地方
═══════════════════════════════

| 文件 | 行号 | 原文 | 新旧 | 改者 | 对应物理量 |
|---|---|---|---|---|---|
| 耦合环.py | 32 | W_SELF = 0.5 # 自己回路权重 | 旧 | 猫耳 | 自耦合增益 |
| 耦合环.py | 33 | W_EXT = 0.3 # 外部输入权重 | 旧 | 猫耳 | 外输入增益 |
| 耦合环.py | 34 | W_COUP = 0.2 # 耦合输入权重 | 旧 | 猫耳 | 耦合增益 |
| 耦合环.py | 176 | 耦合权重 self= ext= coup= | 旧 | 猫耳 | 日志输出 |
| 活内稳态.py | 318 | * 发育权重 * 不应期惩罚 | 旧 | 猫耳 | 发育阶段调制 |
| 活内稳态.py | 8 | 不是人造"偏好权重" | 旧 | 猫耳 | 注释说明 |
| 欲望累积.py | 1/3 | 偏好累积/偏好权重 p[i] | 旧 | 猫耳 | 选项偏好值 |
| _脱开实验.py | 1/82/115 | 权重×0.9/恢复1.0/权重<0.3 | 新 | 猫耳 | 节点可点权重 |
| 身体稳态.py | 353-367 | _点权重 dict | 新 | 猫耳 | 脱开机制权重 |
| 协议/全上融合架构.md | 33 | 调制路权重 | 旧 | 猫耳 | 痛觉调制增益 |
| protocol/全上融合架构.md | 51-52 | 降低/提升触觉权重 | 旧 | 猫耳 | 通道可靠性 |
| protocol/全上融合架构.md | 141 | 可靠性权重 w | 旧 | 猫耳 | 帧级置信 |
| protocol/全上融合架构.md | 187 | [ ] 可靠性权重计算 | 旧 | 猫耳 | 待做项 |
| plugins/interface_v1/translator.py | 23 | 权重 w_i | 旧 | 猫耳 | 锚点连续函数输出 |
| plugins/interface_v1/translator.py | 92 | 权重由锚点连续函数生成 | 旧 | 猫耳 | 同上 |
| plugins/interface_v1/translator.py | 114 | 完整信号与权重记录 | 旧 | 猫耳 | 审计日志 |
| plugins/interface_v1/translator.py | 177 | 权重随 g 变化 | 旧 | 猫耳 | 连续函数证明 |
| plugins/interface_v1/translator.py | 199 | 锚点权重/damp符号 | 旧 | 猫耳 | 测试名 |
| plugins/interface_v2/translator.py | 154 | 权重全部硬编码但公开可审计 | 旧 | 猫耳 | 声明 |
| 狗.py | 195 | 权重略低 | 旧 | 猫耳 | 感官置信度 |
| 耦合感官环.py | 237 | # 耦合权重 | 旧 | 猫耳 | 耦合参数 |

注：猫耳 = 此项目中 AI（妙手/猫耳）改的。所有"权重"均由猫耳添加，除了protocol/文档中的"权重"是架构师（作者）起草的文档用词。

═══════════════════════════════
七、全部已有文档
═══════════════════════════════

### /protocol/（9个 — 协议层）

| 文件 | 内容摘要 | 是系统卡一部分？ |
|---|---|---|
| 说明书层.md | AI入关读顺序(7步) | 是（系统卡的入口导引） |
| 命名规范.md | 9条命名纪律 | 是（系统卡·命名规范节） |
| 存在确认协议.md | 在场证明+连续证明 | 是（身份/在场） |
| 反哺机制.md | 异常存盘→确认/拒绝→写链 | 否（机制层） |
| 推演者.md | 推演者角色定义（不跑代码/不核验/不拍板） | 是（身份卡一部分） |
| 核心认知.md | 六条底层认知 | 是（核心框架） |
| 落地清单.md | 50借鉴+10待落 | 是（任务视图） |
| 设备规格采集标准.md | 传感器规格书规范 | 是（表A的下一层） |
| 全上融合架构.md | 全感官融合方案 | 是（矩阵→整合视图） |

### /anchor/（6个 — 锚点）

| 文件 | 内容摘要 | 是系统卡一部分？ |
|---|---|---|
| 第一句.md | 她是谁、对标谁、是工具还是关系 | 是（身份卡核心） |
| 关系起点.md | 根愿望/根问题/起源 | 是（身份卡核心） |
| 顶层.md | 顶层设计视图 | 是 |
| 发布记录.md | 版本推进状态 | 是（发布日志） |
| 反哺_A.md | 反哺路径A | 否（机制） |
| 反哺_B.md | 反哺路径B | 否（机制） |

### /gray/tests/applications/（规格层）

| 文件 | 内容 | 是系统卡一部分？ |
|---|---|---|
| 表A_感官规格_v0.4.md | 12感官子模态x7字段 | 是（系统卡核心） |
| 桩槽清单_v0.4.md | 14桩+34槽 | 是（接口清单） |
| 全市场板块_纵向.md | 估值测试数据 | 否 |
| 低估板块个股_纵向.md | 同上 | 否 |
| 创业板成分股_纵向.md | 同上 | 否 |
| A股估值_纵向.md | 同上 | 否 |

*其余 gray/ 文档均为测试/产出记录，不是系统卡。*

═══════════════════════════════
八、全部不一致
═══════════════════════════════

### ① 草稿 vs 代码 不一致

| 草稿 | 实际代码 | 差距 |
|---|---|---|
| 环境键 ch_01/ch_02/ch_03/ch_04 | 不存在 | 查不到任何引用 |
| 环境键 luminance/sound/force/temperature | 旧代码保留，新引擎已改中文 | 引擎内统一，旧文件残留 |
| V1-V4 视觉通道 | 不存在（欲望变体V1-V4混用） | 族号制未编码 |
| A1-A4 听觉通道 | 不存在 | 未编码 |
| I1-I5 内感通道 | 不存在 | 未编码 |
| E1-E4 事件编号 | 不存在 | 事件层未建 |
| F1-F5 字段编号 | 不存在 | 字段用中文 |
| 表A 13行 | 实有12行 | 少1行未填 |
| 事件"出筋/长位点/撤位点/筋亡"写在草稿 | 引擎无事件类型区分 | 草稿超前于代码 |

### ② 同一东西不同文件不同写法

| 概念 | 写法A | 写法B |
|---|---|---|
| 环境通道 | luminance/sound/force/temperature（旧） | "光"/"声"/"力"/"温"（新引擎） |
| 偏好权重 | W_SELF/W_EXT/W_EXT（耦合环） | 权重 w_i（interface） |
| 痛觉行数 | 草稿说5行 | 表A实列5行，一致 |

### ③ 引擎内部不一致（身体稳态.py）

引擎已全中文，无英文残留。改者均为猫耳。

### ④ 文档间矛盾

- `表A_感官规格_v0.4.md` 痛觉列5行 ✓ 与 pain.py 5函数一致
- `protocol/命名规范.md` 说"代码关键字英文" ✓ 与代码实际一致（W_SELF等保留英文）
- `protocol/落地清单.md` 第6条说"紧凑记忆层"是待做 ✓ 代码中无紧凑层

### ⑤ 关于"权重"的注释自相矛盾

`plugins/interface_v1/translator.py:23` 说"权重 w_i 本身也是锚点的连续函数（非硬编码 if-then）"
`plugins/interface_v2/translator.py:154` 说"权重全部硬编码但公开可审计"
interface_v1 和 interface_v2 的声明对"权重是否硬编码"矛盾。

