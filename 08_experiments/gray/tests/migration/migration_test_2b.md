# 迁移测试 #2B — 双向迁移（真双向 · 不同 drive · fork 验证）

状态：[事实]
日期：2026-09-17
版本: 第 3 轮 (前两轮: 同 drive 仅元数据冲突 / 单向追加非真双向)

## 关键声明: 本轮验的是 fork，不是 merge

- fork（分叉）: 两条冲突链分道扬镳，各自独立 → 本轮测的就是这个
- merge（合并）: 两条冲突链合成一条 → 本轮未测

本轮三件验成:
1. 冲突能识别（数值冲突 vs 元数据冲突）
2. 冲突能隔离（一边的链进不了另一边）
3. 共同祖先不丢（seq 0-19 完整保留）

本轮未测:
- 两条冲突链能否合成一条 merge
- 合的话用什么策略（取一边 / 按时间戳仲裁 / 用户选）
- 这是后续测试范围

## 目的

验证: 切点后两边用**不同 drive** 产生数值分歧，能否合、怎么合。

## 配置

- src: bi_src (drive=0.3)
- dst: bi_dst (drive=0.5)
- 流程: src 跑 20 步 → cp src→dst → src + dst 各跑 10 步(不同 drive) → 合并

## 一、共同祖先 (切点)

切点 seq=19 note=bstep_20
hash=52bd5ff9612486879e8b51e875be08f4440a0915a37b173f8ee53d55053c04b0
(src/dst 共享)

## 二、分歧轨迹

### src (drive=0.3, bstep_21~30) — seq 20-29

| seq | g[0]       | g[1]       |
|-----|------------|------------|
| 20  | -7.3276    | -1.6195    |
| 21  | -7.5543    | -1.6033    |
| 22  | -7.7788    | -1.5872    |
| 23  | -8.0010    | -1.5714    |
| 24  | -8.2210    | -1.5556    |
| 25  | -8.4388    | -1.5401    |
| 26  | -8.6544    | -1.5247    |
| 27  | -8.8679    | -1.5094    |
| 28  | -9.0792    | -1.4943    |
| 29  | -9.2884    | -1.4794    |

### dst (drive=0.5, bdstep_1~10) — seq 20-29

| seq | g[0]       | g[1]       |
|-----|------------|------------|
| 20  | -7.5276    | -1.6195    |
| 21  | -7.9523    | -1.6033    |
| 22  | -8.3728    | -1.5872    |
| 23  | -8.7891    | -1.5714    |
| 24  | -9.2012    | -1.5556    |
| 25  | -9.6092    | -1.5401    |
| 26  | -10.0131   | -1.5247    |
| 27  | -10.4130   | -1.5094    |
| 28  | -10.8088   | -1.4943    |
| 29  | -11.2007   | -1.4794    |

## 三、冲突分析 [事实]

| seq | src_g0    | dst_g0    | Δg0    | match |
|-----|-----------|-----------|--------|-------|
| 20  | -7.3276   | -7.5276   | -0.2000 | False |
| 21  | -7.5543   | -7.9523   | -0.3980 | False |
| 22  | -7.7788   | -8.3728   | -0.5940 | False |
| 23  | -8.0010   | -8.7891   | -0.7881 | False |
| 24  | -8.2210   | -9.2012   | -0.9802 | False |
| 25  | -8.4388   | -9.6092   | -1.1704 | False |
| 26  | -8.6544   | -10.0131  | -1.3587 | False |
| 27  | -8.8679   | -10.4130  | -1.5451 | False |
| 28  | -9.0792   | -10.8088  | -1.7297 | False |
| 29  | -9.2884   | -11.2007  | -1.9124 | False |

### Δg0 理论验证 [事实]

公式: Δg0 = Δdrive × (1 − λ^n) / (1 − λ) = 0.2 × (1 − 0.99^n) / 0.01

| n (步) | 实测 Δg0 | 理论 Δg0 |
|--------|---------|---------|
| 1 | 0.2000 | 0.2000 |
| 2 | 0.3980 | 0.3980 |
| 3 | 0.5940 | 0.5940 |
| 4 | 0.7881 | 0.7881 |
| 5 | 0.9802 | 0.9802 |
| 6 | 1.1704 | 1.1704 |
| 7 | 1.3587 | 1.3587 |
| 8 | 1.5451 | 1.5451 |
| 9 | 1.7297 | 1.7297 |
| 10 | 1.9124 | 1.9124 |

结论: 实测与理论全等，零偏差。[事实]

冲突类型: **数值冲突** (非元数据、非哈希)
- 每一对 g[0] 都不相等
- Δg0 从 0.2 增长到 1.9 (累积效应)
- g[1] 完全相同 (dim1 对所有 drive 反应一样)
- 原因: src drive=0.3 vs dst drive=0.5 → 同一初始态的不同演化

是否必然冲突: 是 — 不同 drive 在确定性引擎下同样必然分歧，且分歧度随步数增长

## 四、合并策略选择 [事实]

可选:
1. 取 src → 丢 dst 的 10 步真数据 (信息损失)
2. 取 dst → 丢 src 的 10 步真数据 (信息损失)
3. 分叉 (fork) → 两边都留
4. 让用户选 → agent 做不到 (没人在手机前)

本轮选: **分叉 (fork)**
理由: 数值冲突 = 双方都有真实信息，直接丢弃是信息损失
备注: 真实系统应挂 pending 等人确认，不应 agent 单方面 pick

执行操作: 仅在 src 追加一条 fork 记录 (seq=30)，dst 未被修改

fork 记录:
```json
{"seq":30,"ts":"2026-09-17T02:12:29","type":"fork","trigger":"system","payload":{"fork_from":"bi_dst","common_ancestor":{"seq":19,"hash":"52bd5ff9612486879e8b51e875be08f4440a0915a37b173f8ee53d55053c04b0"},"conflict":"numerical divergence (drive_src=0.3 vs drive_dst=0.5)","resolution":"fork: both branches preserved","src_branch":"bstep_21~30 (drive=0.3)","dst_branch":"bdstep_1~10 (drive=0.5)","note":"in real system: surface to user for choice, do not auto-pick"},"prev":"20f0d93927b98991ac2e78d514d0dadd36a4d656bc2c7f1f551cf61e47541ddb","hash":"07f2343ef5fe3570906d6c8dc8b5b731a92dca11ae246eba63d9deef7a992257"}
```

## 五、合并后验证 [事实]

### 链完整性

bi_src: 31/31 (seq 0-30, fork 在 seq 30)
bi_dst: 30/30 (seq 0-29, 未修改)

### 共同祖先完整

seq 0-19 hash 全等: True

### dst 未被合并修改

dst 总行数: 30 (未变)
dst 尾: seq=29 note=bdstep_10 (仍为 drive=0.5 分支)

### 合并后继续跑 (src)

src fork 标记后续跑一步 (drive=0.3, note=post_fork):
g=[-9.4955, -1.4646]  h=[-2,-2]  J1/J2/J3全过
src chain: 30 → 32 (seq 31 新 step + seq 30 fork 记录)

## 六、结论

| 判项 | 结果 |
|------|------|
| 合得上 | [事实] — 用 fork 策略，两边分支都保留，无信息丢弃 |
| 冲突在哪 | [事实] — seq 20-29, g[0] 数值冲突 (Δ=0.2~1.9) |
| 怎么合的 | [事实] — 分叉 (src 追加 fork 记录，dst 不动) |
| 合完能继续跑 | [事实] — src 续跑一步，链 31→32，引擎正常 |

## 七、原始文件

- src: luoshenfu_home/users/bi_src/chain.jsonl (32 行)
- dst: luoshenfu_home/users/bi_dst/chain.jsonl (30 行)

## 八、Merge 测试 (fork 之后追加)

### 规则选择

三选一:
- A: 保留双方 (同 seq 存两条) — 需改链结构，等价新建链
- B: 选 src 主链 + dst 入 deprecated/ — 单线主链，废档可审计
- C: 时间戳仲裁 — 自动但随意（ts 副产物不代表"更对"）

本轮选: **规则 B**
理由: 简单诚实，src 是 home user（搬回源），identity 应保留。dst 的 10 步不丢，进 deprecated/。

### 合并执行

merge 规则: B (select_src_deprecate_dst)
selected: bi_src drive=0.3 (bstep_21~30)
deprecated: bi_dst seq 20-29 → deprecated/dst_rejected_chain.jsonl (drive=0.5, bdstep_1~10)

merge 记录 (src seq=30):
```json
{"seq":30,"ts":"2026-09-17T02:38:27","type":"merge","trigger":"system","payload":{"rule":"B: select_src_deprecate_dst","merged_from":"bi_dst","selected":"bi_src (drive=0.3, bstep_21~30)","deprecated":"bi_dst seq 20-29 → deprecated/dst_rejected_chain.jsonl (drive=0.5, bdstep_1~10)","conflict":"numerical divergence (Δg0=0.2~1.91)","common_ancestor":{"seq":19,"hash":"0f258339784bb9432b89fa9290fee341d5fa1f7caeca359647128a41b1793979"},"reason":"src is home user (搬回源), original identity"},"prev":"7349bd73cff74be6ca10aa570542e3e3b5bf80826663ac25238949b6a438b917","hash":"88d52b3788ce46563c1e35807c0885b92c4dbc3aa18a423e88a753fc8d331c9e"}
```

### 合并后验证

**行数变化:**
- 合并前: src 30 行 (seq 0-29)
- 合并后: src 31 行 (seq 0-29 + merge@30), deprecated 10 行

**尾哈希:**
- src: seq=30 type=merge hash=88d52b3788ce46563c1e35807c0885b92c4dbc3aa18a423e88a753fc8d331c9e
- dst: seq=29 type=step hash=...(不变)

**共同祖先 (seq 0-19):**
- src seq 19 hash = dst seq 19 hash = 0f258339...
- 共同祖先一致: True

**链完整性:**
- bi_src: 31/31
- bi_dst: 30/30
- deprecated: 10 lines

**合并后继续跑:**
- src 续跑一步 (drive=0.3, note=post_merge): g=[-9.4955, -1.4646]  h=[-2,-2]  J1/J2/J3全过
- src chain: 31 → 32 行

### Merge 结论 [事实]

| 判项 | 结果 |
|------|------|
| 合得上 | [事实] — 规则 B 成功，src 主链 + dst 入 deprecated |
| 共同祖先 | [事实] — seq 0-19 hash 未动 |
| 完整 | [事实] — 所有链 100% 通过 |
| 能续跑 | [事实] — merge 后引擎正常响应 |

### 原始文件

- src: luoshenfu_home/users/bi_src/chain.jsonl (32 行)
- dst: luoshenfu_home/users/bi_dst/chain.jsonl (30 行)
- deprecated: luoshenfu_home/users/bi_dst/deprecated/dst_rejected_chain.jsonl (10 行)

## 九、未覆盖

- 时间戳仲裁策略 (规则 C)
- 用户确认流程 (pending + 回调)
- 三角合并
- 规则 A (保留双方) — 需改链结构
- 数值冲突时是否有更细粒度的仲裁（比如逐 seq 而非整段）
