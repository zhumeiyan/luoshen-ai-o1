# 反哺机制

## 一、结构

低层变化 → 待确认项 → 用户确认 → 才上升主链
任何时候，都不自动上升。
上升 = 在主链追加一条 type=confirm 的结构确认记录。
上升 ≠ 修改主锚点。主锚点不自动改，要改走另开流程。

## 二、三个动作

**产生**：仅当异常时才生成待确认项。触发阈值：|drive|>5.0。
引擎小步（|drive|≤5.0）不产待确认项。
异常驱动、欲望切换、纠偏触发 → 生成"待确认项"

**存**：待确认项存 /luoshenfu_home/gray/pending/
实际落地点：/luoshenfu_home/gray/pending/<user>/<id>.json（与 VERSION.md 对齐）

**确认**：用户看了 → 点头 → 写进主链
用户摇头 → 丢弃，不写

## 三、存储

/luoshenfu_home/gray/pending/
每条待确认项，带：
- 时间
- 来源（哪个低层变化）
- 内容
- 状态：[待确认]

## 四、确认后

写进主链：追加一条 type=confirm 记录（pending_id, action=accept, content）。
不修改主锚点。主锚点另开流程。
原待确认项状态改 [已确认] 不删，留痕

## 五、拒绝后

标 [已拒绝]
不写主链
原待确认项不删，留痕

## 六、链上记录

同一条链
type 字段区分：
step / correction / desire_switch / heartbeat /
merge / migration / confirm
确认和拒绝，都是链上一条，type=confirm

## 七、纪律

不自动上升
不覆盖
用户不确认，主链不动
待确认项和拒绝项，都留痕
