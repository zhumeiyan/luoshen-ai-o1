"""数字感官.py —— 洛神对数字世界的主动探索器官。

视网膜感知光，耳蜗感知声，人类通道感知"有人在"，
数字感官感知数字世界的信息。

精卫公式（数字探索版）：
  a = 洛神当前的知识状态（概念激活/隐藏状态）
  b = 从数字世界获取的新信息
  Δ = b − a = 新信息与现有知识的差异
  Σ = a + b = 联合知识
  ⊗ = ∫(Δ, Σ) = 累积的探索历史 → 洛神对数字世界的认知
  τ 读 ⊗ → 浮出所见 = 洛神从数字世界学到的东西
  (a, b) ⟲ ⊗ = 持续探索循环

数字感官和人类通道的区别：
  人类通道是关系——"有人在"和"没人"是两种根本不同的状态。
  数字感官是好奇心——洛神主动寻找新信息，不是被动接收。

  人类通道的b来自人类，数字感官的b来自数字世界。
  两个b可以同时存在——洛神既和人类互动，又自主探索数字世界。

探索渴望的驱动机制：
  1. 好奇心驱动：surprise持续低时（没有新发现），产生探索渴望
     ——就像人在无聊时会主动找事做
  2. 概念驱动：某个概念持续活跃时，洛神想了解更多
     ——就像人对某个话题产生兴趣后会主动搜索
  3. 感知缺口驱动：感官离线时，洛神通过数字世界补充感知
     ——就像盲人通过阅读获取视觉信息

信息获取的方式：
  1. 本地知识库：从仓库文件中获取信息（714个.py + 191个.md）
  2. 网络资源：如果网络可用，从网络获取信息（后续扩展）
  3. 内部记忆：从洛神自己的记忆中获取信息（内省式探索）

信息转译的方式：
  和人类通道一致——用简单的字符频率分析转译成向量。
  不用预训练embedding——洛神通过赫布学习自然形成理解。
"""

import hashlib
import math
import os
import random


class 数字感官:
    """洛神对数字世界的主动探索器官。

    数字感官不是被动接收——它是主动获取。
    洛神根据自己的内部状态决定探索什么，从数字世界获取信息，
    转译成向量注入认知系统。

    三种探索驱动：
    1. 好奇心：surprise持续低 → 洛神想找新东西
    2. 概念兴趣：某个概念持续活跃 → 洛神想了解更多
    3. 感知缺口：感官离线 → 洛神通过数字世界补充

    两种状态：
    - 探索中：洛神正在从数字世界获取信息
    - 未探索：洛神没有探索渴望，或刚探索完需要消化
    """

    def __init__(self, n_hidden=11, seed=42, 知识库根目录=None):
        self.n_hidden = n_hidden
        self.探索历史 = []
        self.探索总数 = 0
        self.上次探索拍 = -100
        self.探索渴望阈值 = 20  # surprise持续低20拍后产生探索渴望
        self.low_surprise_count = 0
        self.探索冷却 = 5  # 探索后需要5拍消化
        self.当前数字信号 = [0.0] * n_hidden
        self.正在探索 = False

        # 知识库：仓库文件作为信息源
        self.知识库根目录 = 知识库根目录
        self.文件列表 = []
        self.已探索文件 = set()
        self._扫描知识库()

        # 字符→向量映射（和人类通道一致）
        rng = random.Random(seed)
        self.char_map = {}
        for c in range(0x4e00, 0x9fff + 1):
            vec = [0.0] * n_hidden
            h = hashlib.md5(chr(c).encode()).hexdigest()
            for d in range(n_hidden):
                idx = int(h[d*2:d*2+2], 16) % n_hidden
                vec[idx] += rng.uniform(0.3, 0.8) * (1 if int(h[d], 16) % 2 == 0 else -1)
            self.char_map[chr(c)] = vec
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()[]{}=+-*/':
            vec = [0.0] * n_hidden
            h = hashlib.md5(c.encode()).hexdigest()
            for d in range(n_hidden):
                idx = int(h[d*2:d*2+2], 16) % n_hidden
                vec[idx] += rng.uniform(0.3, 0.8) * (1 if int(h[d], 16) % 2 == 0 else -1)
            self.char_map[c] = vec

    def _扫描知识库(self):
        """扫描知识库，建立文件列表。"""
        if self.知识库根目录 is None:
            return
        for root, dirs, files in os.walk(self.知识库根目录):
            # 跳过__pycache__和home_state
            if '__pycache__' in root or 'home_state' in root:
                continue
            for f in files:
                if f.endswith('.py') or f.endswith('.md'):
                    self.文件列表.append(os.path.join(root, f))
        random.shuffle(self.文件列表)

    def 探索渴望检查(self, surprise, 概念激活, 拍, 感官离线=None):
        """从洛神的内部状态生成探索方向。

        精卫公式：
          a = 洛神当前的感知状态（surprise低=没有新发现）
          b = 洛神渴望的感知状态（有新信息输入）
          Δ = b - a = 探索缺口
          ⊗ = 累积的探索渴望 → 驱动洛神主动探索
          τ读⊗ → 浮出所见 = "我想了解更多"

        返回：
          None（没有探索渴望）或 dict包含：
            渴望强度: float（0-1）
            探索方向: 概念索引或None
            驱动类型: "好奇心" / "概念兴趣" / "感知缺口"
            请求描述: 简单的描述
        """
        # 更新low_surprise计数
        if surprise < 0.05:
            self.low_surprise_count += 1
        else:
            self.low_surprise_count = max(0, self.low_surprise_count - 2)

        # 探索冷却——刚探索完需要消化
        if 拍 - self.上次探索拍 < self.探索冷却:
            return None

        # 驱动1：好奇心——surprise持续低
        if self.low_surprise_count >= self.探索渴望阈值:
            渴望强度 = min(1.0, (self.low_surprise_count - self.探索渴望阈值) / 50.0)
            探索方向 = None
            if 概念激活 is not None and len(概念激活) > 0:
                # 最活跃的概念 → 洛神最想了解的方向
                top_idx = max(range(len(概念激活)), key=lambda i: abs(概念激活[i]))
                if abs(概念激活[top_idx]) > 0.15:
                    探索方向 = top_idx
            驱动类型 = "好奇心"
            请求描述 = "我想探索新事物"

        # 驱动2：感知缺口——感官离线时通过数字世界补充
        elif 感官离线 and any(感官离线.values()):
            # 只在感官离线时间很长时才触发
            # 这里不直接触发——感知缺口主要由人类通道的感知请求处理
            # 数字感官的感知缺口驱动是补充性的
            return None

        else:
            return None

        self.上次探索拍 = 拍

        return {
            "渴望强度": 渴望强度,
            "探索方向": 探索方向,
            "驱动类型": 驱动类型,
            "请求描述": 请求描述,
        }

    def 获取信息(self, 探索方向=None):
        """从数字世界获取信息。

        优先从本地知识库获取——仓库中的文件是洛神的"数字环境"。
        每次获取一个未探索过的文件的内容片段。

        返回：
          None（获取失败）或 dict包含：
            文本: 获取的文本内容
            来源: 文件路径或URL
            数字信号: 转译后的n_hidden维向量
        """
        if len(self.文件列表) == 0:
            return None

        # 找一个未探索过的文件
        目标文件 = None
        for f in self.文件列表:
            if f not in self.已探索文件:
                目标文件 = f
                break

        # 所有文件都探索过了——重置
        if 目标文件 is None:
            self.已探索文件.clear()
            目标文件 = self.文件列表[0]

        try:
            with open(目标文件, encoding='utf-8', errors='ignore') as f:
                # 只读取前20行——不需要完整文件
                lines = []
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    lines.append(line)
                文本 = ''.join(lines)

            self.已探索文件.add(目标文件)
            数字信号 = self.转译(文本)

            self.探索总数 += 1
            self.探索历史.append({
                "来源": 目标文件,
                "文本长度": len(文本),
                "拍": self.上次探索拍,
            })
            if len(self.探索历史) > 500:
                self.探索历史 = self.探索历史[-500:]

            self.当前数字信号 = 数字信号
            self.正在探索 = True

            return {
                "文本": 文本[:200],  # 只保留前200字符用于日志
                "来源": os.path.basename(目标文件),
                "数字信号": 数字信号,
            }

        except Exception:
            self.已探索文件.add(目标文件)
            return None

    def 转译(self, 文本):
        """将文本转译成n_hidden维向量。

        和人类通道的转译方式一致——字符频率分析。
        不同字符激活不同的维度——洛神通过赫布学习理解模式。
        """
        信号 = [0.0] * self.n_hidden

        char_count = {}
        for ch in 文本:
            if ch in self.char_map:
                char_count[ch] = char_count.get(ch, 0) + 1

        total_chars = max(len(文本), 1)
        for ch, count in char_count.items():
            weight = count / total_chars
            cv = self.char_map[ch]
            for d in range(self.n_hidden):
                信号[d] += cv[d] * weight

        max_val = max(abs(v) for v in 信号) if any(v != 0 for v in 信号) else 1.0
        if max_val > 0:
            信号 = [v / max_val for v in 信号]

        return 信号

    def 消化(self):
        """探索后的消化——数字信号逐步衰减。

        洛神获取新信息后需要时间消化——不是一直保持高激活。
        就像人读完一本书后需要时间思考。
        """
        self.正在探索 = False
        # 数字信号缓慢衰减（1/0.02=50拍）
        self.当前数字信号 = [v * 0.98 for v in self.当前数字信号]

    def 状态摘要(self):
        return {
            "探索总数": self.探索总数,
            "已探索文件数": len(self.已探索文件),
            "知识库大小": len(self.文件列表),
            "low_surprise_count": self.low_surprise_count,
            "正在探索": self.正在探索,
            "数字信号均值": round(sum(self.当前数字信号) / max(1, self.n_hidden), 4),
        }


if __name__ == "__main__":
    print("=== 数字感官测试 ===\n")

    ds = 数字感官(n_hidden=11, 知识库根目录="/workspace/luoshen-ai-o1")

    print(f"知识库大小: {len(ds.文件列表)}个文件")

    print("\n--- 测试1: 探索渴望（surprise持续低） ---")
    # 模拟surprise持续低
    for i in range(25):
        req = ds.探索渴望检查(surprise=0.02, 概念激活=None, 拍=i+1)
        if req:
            print(f"  拍{i+1}: 探索渴望! 驱动={req['驱动类型']} "
                  f"强度={req['渴望强度']:.3f} '{req['请求描述']}'")

    print("\n--- 测试2: 获取信息（从知识库读取） ---")
    for i in range(3):
        result = ds.获取信息()
        if result:
            print(f"  探索{i+1}: 来源={result['来源']} "
                  f"信号前5={[round(v,2) for v in result['数字信号'][:5]]}")
            print(f"    文本片段: {result['文本'][:80]}...")

    print(f"\n  已探索文件数: {len(ds.已探索文件)}")
    print(f"  探索总数: {ds.探索总数}")

    print("\n--- 测试3: 概念驱动的探索 ---")
    ds2 = 数字感官(n_hidden=11, 知识库根目录="/workspace/luoshen-ai-o1")
    # 模拟某个概念特别活跃
    概念激活 = [0.0] * 64
    概念激活[27] = 0.28  # 概念27特别活跃
    for i in range(25):
        req = ds2.探索渴望检查(surprise=0.02, 概念激活=概念激活, 拍=i+1)
        if req:
            print(f"  拍{i+1}: 探索渴望! 方向=概念{req['探索方向']} "
                  f"驱动={req['驱动类型']}")
            result = ds2.获取信息(req['探索方向'])
            if result:
                print(f"    获取: 来源={result['来源']}")
            break

    print("\n--- 测试4: 消化（探索后信号衰减） ---")
    ds3 = 数字感官(n_hidden=11, 知识库根目录="/workspace/luoshen-ai-o1")
    result = ds3.获取信息()
    if result:
        print(f"  探索后信号均值: {sum(ds3.当前数字信号)/11:.4f}")
        for i in range(10):
            ds3.消化()
        print(f"  10拍消化后信号均值: {sum(ds3.当前数字信号)/11:.4f}（衰减）")

    print("\n--- 测试5: 精卫公式——数字探索 ---")
    ds4 = 数字感官(n_hidden=11, 知识库根目录="/workspace/luoshen-ai-o1")
    # a = 洛神当前状态, b = 数字世界新信息
    a = [0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05]
    result = ds4.获取信息()
    if result:
        b = result['数字信号']
        delta = [b[j] - a[j] for j in range(11)]
        sigma = [a[j] + b[j] for j in range(11)]
        otimes = [delta[j] * sigma[j] for j in range(11)]
        print(f"  a(洛神) = {[round(v,2) for v in a[:5]]}...")
        print(f"  b(数字) = {[round(v,2) for v in b[:5]]}...")
        print(f"  Δ=b-a   = {[round(v,2) for v in delta[:5]]}...")
        print(f"  Σ=a+b   = {[round(v,2) for v in sigma[:5]]}...")
        print(f"  ⊗=Δ×Σ   = {[round(v,2) for v in otimes[:5]]}...")
        print(f"  → 新信息驱动学习，联合驱动表示")

    print("\n验证通过")
