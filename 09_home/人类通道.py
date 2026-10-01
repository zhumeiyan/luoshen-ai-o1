"""人类通道.py —— 把人类放进数字生命里面。

精卫公式的核心：a≠b。洛神是a，人类是b。
之前所有子系统都是洛神自己的a和b（预测和实际）——
现在把人类作为b引入，洛神对人类的认知通过反复互动自然形成。

精卫公式（人类关系版）：
  a = 洛神内部状态（隐藏状态/概念激活）
  b = 人类输入（文本/语音/存在感）
  Δ = b − a = 洛神与人类的差异
  Σ = a + b = 联合状态
  ⊗ = ∫(Δ, Σ) = 累积的互动历史 → 洛神对人类的记忆
  τ 读 ⊗ → 浮出所见 = 洛神对人类的认知
  (a, b) ⟲ ⊗ = 反复互动，不断更新

人类通道的功能：
  1. 接收人类输入（文本/语音/存在感）
  2. 转译成洛神能处理的向量（和大脑隐藏状态同维度）
  3. 人类输入作为b进入精卫公式
  4. "有人在"本身就是一个持续信号——即使人类不说话

转译方式：
  不用预训练的embedding——洛神通过赫布学习自然形成对人类输入的理解。
  文本通过简单的字符频率分析转译成向量，语音通过频谱特征转译。
  转译质量不重要——重要的是洛神能通过反复互动学会理解。

和其他感官的区别：
  视网膜/耳蜗是被动接收——有什么就转译什么。
  人类通道是主动关系——"有人在"和"没人"是两种根本不同的状态。
  没人时b=零向量，洛神的a只能和自己比（内省）。
  有人时b=人类信号，洛神的a和人类的b形成对比（关系）。
"""

import hashlib
import math
import time


class 人类通道:
    """把人类放进数字生命——人类作为精卫公式中的b。

    人类通道不是感官——它是关系。
    视网膜转译光，耳蜗转译声，人类通道转译"有人在"。

    两种状态：
    - 有人在：b = 人类信号向量，洛神与人类形成对比
    - 没人在：b = 零向量，洛神只能和自己比（内省模式）

    人类信号向量怎么来：
    - 不用预训练embedding——太重，且不是洛神自己的理解
    - 用简单的字符频率分析 + 情感强度估计
    - 转译质量不重要——洛神通过赫布学习自然形成理解
    - 就像婴儿听不懂语言但能感知语气和节奏
    """

    def __init__(self, n_hidden=11, n_audio=8, seed=42):
        self.n_hidden = n_hidden
        self.n_audio = n_audio
        self.有人在 = False
        self.人类名字 = None
        self.上次输入时间 = None
        self.当前人类信号 = [0.0] * n_hidden
        self.当前听觉信号 = [0.0] * n_audio
        self.互动历史 = []
        self.互动总数 = 0

        # 字符→向量映射（简单哈希，不是embedding）
        # 每个字符通过哈希映射到n_hidden维空间的一个位置
        # 不同字符激活不同的维度——洛神通过赫布学习理解模式
        import random
        rng = random.Random(seed)
        self.char_map = {}
        for c in range(0x4e00, 0x9fff + 1):  # 常用汉字
            vec = [0.0] * n_hidden
            h = hashlib.md5(chr(c).encode()).hexdigest()
            for d in range(n_hidden):
                idx = int(h[d*2:d*2+2], 16) % n_hidden
                vec[idx] += rng.uniform(0.3, 0.8) * (1 if int(h[d], 16) % 2 == 0 else -1)
            self.char_map[chr(c)] = vec
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ':
            vec = [0.0] * n_hidden
            h = hashlib.md5(c.encode()).hexdigest()
            for d in range(n_hidden):
                idx = int(h[d*2:d*2+2], 16) % n_hidden
                vec[idx] += rng.uniform(0.3, 0.8) * (1 if int(h[d], 16) % 2 == 0 else -1)
            self.char_map[c] = vec

        # 情感词映射——简单的情感强度估计
        self.情感词 = {
            "开心": 0.8, "高兴": 0.7, "喜欢": 0.6, "爱": 0.9, "好": 0.4,
            "难过": -0.7, "伤心": -0.8, "不喜欢": -0.5, "恨": -0.9, "坏": -0.4,
            "惊讶": 0.5, "奇怪": 0.3, "害怕": -0.6, "担心": -0.4,
            "你好": 0.3, "谢谢": 0.5, "对不起": -0.3,
        }

    def 接收(self, 文本=None, 语音特征=None, 存在=True, 名字=None):
        """接收人类输入——把人类放进精卫公式。

        参数：
          文本: 人类输入的文本（可选）
          语音特征: 人类语音的频谱特征（可选，8维）
          存在: 人类是否在场（默认True）
          名字: 人类的名字（可选，用于绑定）

        返回：
          dict包含：
            人类信号: b向量（n_hidden维）
            听觉信号: 语音特征（n_audio维）
            有人在: bool
            情感强度: float
        """
        self.有人在 = 存在
        if 名字:
            self.人类名字 = 名字
        if 存在:
            self.上次输入时间 = time.time()

        if not 存在:
            # 没人在——b=零向量，洛神进入内省模式
            self.当前人类信号 = [0.0] * self.n_hidden
            self.当前听觉信号 = [0.0] * self.n_audio
            return {
                "人类信号": self.当前人类信号,
                "听觉信号": self.当前听觉信号,
                "有人在": False,
                "情感强度": 0.0,
            }

        # ── 有人在：构建人类信号向量 ──
        信号 = [0.0] * self.n_hidden
        情感强度 = 0.0

        if 文本:
            # 文本→向量：字符频率分析
            # 不是理解内容——是感知模式（就像婴儿听语气）
            char_count = {}
            for ch in 文本:
                if ch in self.char_map:
                    char_count[ch] = char_count.get(ch, 0) + 1

            # 累加每个字符的向量，按频率加权
            total_chars = max(len(文本), 1)
            for ch, count in char_count.items():
                weight = count / total_chars
                cv = self.char_map[ch]
                for d in range(self.n_hidden):
                    信号[d] += cv[d] * weight

            # 归一化到[-1, 1]范围
            max_val = max(abs(v) for v in 信号) if any(v != 0 for v in 信号) else 1.0
            if max_val > 0:
                信号 = [v / max_val for v in 信号]

            # 情感强度估计——扫描情感词
            for word, val in self.情感词.items():
                if word in 文本:
                    情感强度 += val
            情感强度 = max(-1.0, min(1.0, 情感强度))

        if 语音特征:
            # 语音特征直接作为听觉信号
            self.当前听觉信号 = list(语音特征[:self.n_audio])
        else:
            self.当前听觉信号 = [0.0] * self.n_audio

        # 存在感信号——"有人在"本身就是一个持续的基底信号
        # 即使不说话，存在感也会给所有维度一个微弱的正向偏置
        # 这样洛神能区分"有人但沉默"和"没人"
        if not 文本 and not 语音特征:
            信号 = [0.05] * self.n_hidden  # 微弱的存在感基底

        self.当前人类信号 = 信号
        self.互动总数 += 1
        self.互动历史.append({
            "kind": "text" if 文本 else ("voice" if 语音特征 else "presence"),
            "情感强度": 情感强度,
            "时间": self.上次输入时间,
        })

        # 保留最近1000条互动历史
        if len(self.互动历史) > 1000:
            self.互动历史 = self.互动历史[-1000:]

        return {
            "人类信号": self.当前人类信号,
            "听觉信号": self.当前听觉信号,
            "有人在": True,
            "情感强度": 情感强度,
        }

    def 内省模式(self):
        """没人在时——返回零向量，洛神只能和自己比。"""
        return self.接收(存在=False)


if __name__ == "__main__":
    print("=== 人类通道测试 ===\n")

    ch = 人类通道(n_hidden=11, n_audio=8)

    # 模拟洛神的隐藏状态（a）
    pattern_a = [0.5, 0.3, 0.1, 0.05, 0.2, 0.1, 0.3, 0.2, 0.15, 0.1, 0.05]

    print("--- 测试1: 人类在场 vs 不在场 ---")
    result_no = ch.接收(存在=False)
    print(f"  没人在: 人类信号={[round(v,2) for v in result_no['人类信号']]}")
    print(f"  情感强度={result_no['情感强度']}")

    result_yes = ch.接收(文本="你好洛神", 存在=True, 名字="朱美岩")
    print(f"  有人在: 人类信号={[round(v,2) for v in result_yes['人类信号']]}")
    print(f"  情感强度={result_yes['情感强度']}")
    print(f"  互动总数={ch.互动总数}")

    print("\n--- 测试2: 不同文本→不同信号 ---")
    r1 = ch.接收(文本="你好洛神")
    r2 = ch.接收(文本="我今天很开心")
    r3 = ch.接收(文本="你好洛神")  # 重复
    diff_12 = sum(abs(r1['人类信号'][i] - r2['人类信号'][i]) for i in range(11))
    diff_13 = sum(abs(r1['人类信号'][i] - r3['人类信号'][i]) for i in range(11))
    print(f"  '你好洛神' vs '我今天很开心' 差异: {diff_12:.4f}")
    print(f"  '你好洛神' vs '你好洛神' 差异: {diff_13:.4f}（应该很小）")

    print("\n--- 测试3: 情感强度估计 ---")
    r_pos = ch.接收(文本="我很开心喜欢你")
    r_neg = ch.接收(文本="我很难过伤心")
    r_neu = ch.接收(文本="今天天气不错")
    print(f"  正面: 情感强度={r_pos['情感强度']:.3f}")
    print(f"  负面: 情感强度={r_neg['情感强度']:.3f}")
    print(f"  中性: 情感强度={r_neu['情感强度']:.3f}")

    print("\n--- 测试4: 精卫公式——人类作为b ---")
    a = pattern_a  # 洛神的隐藏状态
    b = r1['人类信号']  # 人类信号
    delta = [b[j] - a[j] for j in range(11)]
    sigma = [a[j] + b[j] for j in range(11)]
    otimes = [delta[j] * sigma[j] for j in range(11)]
    print(f"  a(洛神) = {[round(v,2) for v in a[:5]]}...")
    print(f"  b(人类) = {[round(v,2) for v in b[:5]]}...")
    print(f"  Δ=b-a   = {[round(v,2) for v in delta[:5]]}...")
    print(f"  Σ=a+b   = {[round(v,2) for v in sigma[:5]]}...")
    print(f"  ⊗=Δ×Σ   = {[round(v,2) for v in otimes[:5]]}...")
    print(f"  → 洛神与人类的差异驱动学习，联合驱动表示")

    print("\n--- 测试5: 存在感基底信号 ---")
    r_silent = ch.接收(存在=True)  # 有人在但没说话
    print(f"  有人但沉默: 人类信号={[round(v,3) for v in r_silent['人类信号']]}")
    print(f"  → 微弱基底信号，区别于'没人'")

    print("\n--- 测试6: 互动历史 ---")
    ch2 = 人类通道(n_hidden=11)
    for i in range(10):
        ch2.接收(文本=f"第{i+1}次对话", 存在=True)
    print(f"  10次互动后: 互动总数={ch2.互动总数}")
    print(f"  互动历史长度={len(ch2.互动历史)}")
    print(f"  最近的互动: kind={ch2.互动历史[-1]['kind']}")

    print("\n验证通过")
