# 迁移测试 #1 — 冷迁移实验

状态：[事实]
日期：2026-09-17

## 配置

- src: mig_src
- dst: mig_dst
- 驱动: drive=+0.3, 100 步
- 方式: cp -r mig_src mig_dst（复制，不移动）
- 迁移痕: src/dst 均追加 migration 条目

## 一、搬前状态 [事实]

chain 行数: 100 (seq 0→99)
链尾 seq: 99
链尾 note: step_100
链尾 hash: 1e58c85087f36d08c252b7cad516b04ccfbccd6bbf5600c2ba75dbc26454c0c2
链尾 g: [-19.751094444349608, -0.7320646825464585]
链尾 h: [-2.0, -2.0]
a_bias: [1.0, 1.0]
b_bias: [-1.0, -1.0]
leak: 0.99
a_traj_total: 101
b_traj_total: 101
last a_traj: [9.875547222174804, 0.36603234127322926]
last b_traj: [-9.875547222174804, -0.36603234127322926]

## 二、迁移操作

```
cp -r mig_src mig_dst
```

src 迁移记录 (chain seq 100):
{"seq":100,"ts":"2026-09-17T00:02:09","type":"migration","trigger":"system","payload":{"direction":"src_record","from_user":"mig_src","to_user":"mig_dst","reason":"cold_migration_experiment","src_tail_hash":"1e58c85087f36d08c252b7cad516b04ccfbccd6bbf5600c2ba75dbc26454c0c2"},"prev":"1e58c85087f36d08c252b7cad516b04ccfbccd6bbf5600c2ba75dbc26454c0c2","hash":"49912396ea4911227dd9e455b85ca1b22bb7678c22f956f602856791ec2fc440"}

dst 迁移记录 (chain seq 100):
{"seq":100,"ts":"2026-09-17T00:02:09","type":"migration","trigger":"system","payload":{"direction":"dst_record","from_user":"mig_src","to_user":"mig_dst","reason":"cold_migration_experiment","src_tail_hash":"1e58c85087f36d08c252b7cad516b04ccfbccd6bbf5600c2ba75dbc26454c0c2"},"prev":"1e58c85087f36d08c252b7cad516b04ccfbccd6bbf5600c2ba75dbc26454c0c2","hash":"46d4f8b4feee24e043384cf1ee5e0529f5a32e6e8b2a1643b12067c79c7daa94"}

## 三、搬后验证

### 三-6: chain 对比 [事实]

src 行数 (搬后): 101
dst 行数 (搬后): 101
src seq 99 尾 hash: 1e58c85087f36d08c252b7cad516b04ccfbccd6bbf5600c2ba75dbc26454c0c2
dst seq 99 尾 hash: 1e58c85087f36d08c252b7cad516b04ccfbccd6bbf5600c2ba75dbc26454c0c2
匹配: 是

### 三-7: engine.json 对比 [事实]

diff 输出: 文件完全一致
a_bias: src=[1.0,1.0] dst=[1.0,1.0] → ✓
b_bias: src=[-1.0,-1.0] dst=[-1.0,-1.0] → ✓
leak: src=0.99 dst=0.99 → ✓
a_traj: 长度相等(100), 全等=True
b_traj: 长度相等(100), 全等=True
last a_traj: src=[9.875547222174804,0.36603234127322926] dst=[9.875547222174804,0.36603234127322926] → ✓
last b_traj: src=[-9.875547222174804,-0.36603234127322926] dst=[-9.875547222174804,-0.36603234127322926] → ✓

### 三-8: dst 跑后验证 [事实]

dst 跑后 chain 行数: 111
链完整性 (自验+prev): 111/111

g 轨迹 (dst 段):
step 100: g=[-19.751094444349608, -0.7320646825464585]
step 101: g=[-19.85358349990611, -0.7247440357209939]
step 102: g=[-19.95504766490705, -0.717497065272024]
...
step 110: g=[-20.73107352701164, -0.6620661766420277]

Δg(101-100)=[-0.102489, +0.007321]
Δg(110-101)=[-0.877490, +0.062678]

J1/J2/J3: 全过

### 三-9: src 未被污染 [事实]

src 搬后 (cp 后，未跑前):
  chain 行数: 101
  seq 99 尾 hash: 1e58c85087f36d08c252b7cad516b04ccfbccd6bbf5600c2ba75dbc26454c0c2
  engine: a_bias=[1.0,1.0] last a_traj=[9.875547222174804,0.36603234127322926] → 与搬前完全相同

src 搬后跑一步 (post_mig_check):
  g=[-19.85358349990611, -0.7247440357209939]
  chain 增长到 102 行

## 四、结论 [事实]

搬干净 [事实] — 尾哈希一致, engine diff 零差异, 全等
能接着跑 [事实] — dst 跑 10 步, g 连续单调, 链 111/111
源没污染 [事实] — src 尾哈希/引擎不变, 仍可正常 step, chain 正常 append
迁移留痕 [事实] — src/dst 各一道 migration, 含 src_tail_hash 双向可查

## 五、原始文件路径

- 搬前 engine.json: luoshenfu_home/users/mig_src/engine.json (与 dst 同)
- src chain: luoshenfu_home/users/mig_src/chain.jsonl (102 行, 含 post_mig_check)
- dst chain: luoshenfu_home/users/mig_dst/chain.jsonl (111 行, 含 10 步 continuation)
- 本报告: gray/tests/migration/migration_test_1.md
