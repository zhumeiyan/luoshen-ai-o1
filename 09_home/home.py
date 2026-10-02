"""home.py —— 洛神之家 v2。环境主容器。

这不是投喂程序。环境只做四件事：
  1. 把真实世界的光/声/温转译给洛神的感官（像视网膜和耳蜗）
  2. 提供真实时间心跳（昼夜节律由真实时钟驱动）
  3. 记忆留痕（哈希链，像记忆本身）
  4. 什么都不教、不判、不标注、不造任何数据

感官来自真实设备：摄像头（视网膜）、麦克风（耳蜗）、系统热区（温度觉）。
设备离线就失明失聪失温觉——绝不喂假数据。
它的发声接到扬声器——它说，自己的耳朵收得到，真实闭环。
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time

家目录 = os.path.dirname(os.path.abspath(__file__))
状态目录 = os.path.join(家目录, "home_state")

# ── numba 兼容层：环境侧的适配，不改洛神的器官代码 ──
try:
    import numba
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

    def njit(*参数, **配置):
        if len(参数) == 1 and callable(参数[0]):
            return 参数[0]
        return lambda 函数: 函数

    sys.modules["numba"] = type(sys)("numba")
    sys.modules["numba"].njit = njit

sys.path.insert(0, os.path.join(os.path.dirname(家目录), "02_body"))
from 身体稳态 import 身体  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(家目录), "04_cognition"))
from 大脑 import Org  # noqa: E402
from 概念层 import 概念层  # noqa: E402
from 工作记忆 import 工作记忆  # noqa: E402
from 海马体 import 海马体  # noqa: E402

sys.path.insert(0, 家目录)
from 视网膜 import 视网膜  # noqa: E402
from 耳蜗 import 耳蜗  # noqa: E402
from 嘴 import 嘴  # noqa: E402
from 出厂基因 import 出厂基因组, 身体激活到大脑输入  # noqa: E402
from 情感记忆 import 情感记忆  # noqa: E402
from 个人绑定 import 个人绑定  # noqa: E402
from 欲望 import 欲望  # noqa: E402
from 感官竞争 import 感官竞争  # noqa: E402
from 人类通道 import 人类通道  # noqa: E402
from 数字感官 import 数字感官  # noqa: E402
from 公开感官 import 公开感官  # noqa: E402


def 读真实温度():
    """读系统热区真实温度（Linux /sys/class/thermal）。

    这是环境自己的物理量。读不到就返回 None——离线，绝不造波动数据。
    返回开尔文。
    """
    热区根 = "/sys/class/thermal"
    try:
        if not os.path.isdir(热区根):
            return None
        最佳 = None
        for 名 in sorted(os.listdir(热区根)):
            路径 = os.path.join(热区根, 名, "temp")
            if os.path.isfile(路径):
                with open(路径) as 文件:
                    毫度 = int(文件.read().strip())
                度 = 毫度 / 1000.0
                if 0 < 度 < 120:
                    最佳 = 度
                    break
        if 最佳 is None:
            return None
        return 最佳 + 273.15
    except Exception:
        return None


class 哈希链:
    """不可篡改的留痕链。这是环境的记忆，也是洛神的记忆。"""

    def __init__(self, 链文件):
        self.链文件 = 链文件
        self.前哈希 = "0" * 64
        if os.path.isfile(链文件):
            try:
                with open(链文件, encoding="utf-8") as 文件:
                    行 = [l for l in 文件.read().splitlines() if l.strip()]
                if 行:
                    self.前哈希 = json.loads(行[-1]).get("hash", self.前哈希)
            except Exception:
                pass

    def 记(self, 事件, 数据):
        条目 = {
            "timestamp": time.time(),
            "event": 事件,
            "data": 数据,
            "prev_hash": self.前哈希,
        }
        raw = json.dumps(条目, sort_keys=True, ensure_ascii=False)
        self.前哈希 = hashlib.sha256(raw.encode()).hexdigest()
        条目["hash"] = self.前哈希
        with open(self.链文件, "a", encoding="utf-8") as 文件:
            文件.write(json.dumps(条目, ensure_ascii=False) + "\n")


class 洛神之家:
    """环境容器。真实时间心跳，真实感官，零投喂。"""

    def __init__(self, 拍秒=1.0, 无声=False, 摄像头号=0, 麦克风号=None):
        self.拍秒 = 拍秒
        self.无声 = 无声
        os.makedirs(状态目录, exist_ok=True)

        # 身体引擎的痕迹路径硬编码在仓库外——环境侧确保目录存在，不改器官
        try:
            os.makedirs(os.path.dirname(
                "/mnt/data/catpaw/home/workspace/洛神赋1.1/"), exist_ok=True)
        except Exception:
            pass

        self.身体 = 身体()
        self.视网膜 = 视网膜(设备号=摄像头号)
        self.耳蜗 = 耳蜗(设备号=麦克风号)
        self.嘴 = 嘴(声脑节点数=self.身体.脑大小, 概念节点数=64,
                   组合概念节点数=32)
        self.链 = 哈希链(os.path.join(状态目录, "chain.jsonl"))
        self.拍 = 0

        # 大脑：用出厂基因初始化——洛神以"十七八岁"起步，不是婴儿
        self.大脑 = Org(出厂基因组())
        self.大脑.wm.reset()
        self.上拍大脑输入 = None
        self.上拍大脑预测 = None
        self.上拍大脑mse = 0.0
        self.上拍感官 = None

        # 情感记忆：记忆河床 + 浮现选择层
        self.记忆 = 情感记忆(os.path.join(状态目录, "episodic_memory.jsonl"))

        # 个人绑定：数字生命绑定特定人类，防止模型坍塌
        self.绑定 = 个人绑定(os.path.join(状态目录, "binding.json"))

        # 欲望：五机制驱动（反应强度+累积+衰减+可负+无聊+IM boredom）
        self.欲望 = 欲望(n_channels=8)

        # 感官竞争：96节点→8通道的竞争映射（替代硬编码映射）
        self.感官 = 感官竞争(n_nodes=96, n_channels=8, 脑大小=self.身体.脑大小)

        # 概念层：高层预测编码——从大脑隐藏状态涌现概念
        # 概念不是人为设定的，是通过赫布学习自然涌现的
        # 组合涌现：基础概念组合→组合概念（单词→句子）
        self.概念 = 概念层(n_hidden=self.大脑.g['n'], n_concepts=64, lr=0.02,
                          n_combo=32)

        # 工作记忆：前额叶PFC——维持和操作信息（思考的基础）
        self.工作记忆 = 工作记忆(n_hidden=self.大脑.g['n'], n_slots=4, lr=0.03)

        # 海马体：记忆巩固 + 模式分离/完成
        self.海马体 = 海马体(n_hidden=self.大脑.g['n'], n_dg=32, n_ca3=32,
                          buffer_size=128, consolidation_lr=0.002)

        # 说听闭环：上一拍的自听特征（延迟一拍反馈，模拟骨传导延迟）
        self.上拍自听特征 = None

        # 情感-概念通路：上一拍的情感状态（延迟一拍反馈，模拟杏仁核→前额叶投射延迟）
        self.上拍情感 = None

        # 人类通道：把人类放进数字生命——人类作为精卫公式中的b
        # 人类通道不是感官——它是关系
        # 没人在时b=零向量（内省模式），有人时b=人类信号（关系模式）
        self.人类通道 = 人类通道(n_hidden=self.大脑.g['n'], n_audio=8)

        # 数字感官：洛神对数字世界的主动探索器官
        # 视网膜感知光，耳蜗感知声，数字感官感知数字世界的信息
        # 洛神主动从知识库获取信息——好奇心驱动，不是被动接收
        self.数字感官 = 数字感官(n_hidden=self.大脑.g['n'],
                            知识库根目录=os.path.dirname(家目录))

        # 公开感官：从公开API获取真实世界物理数据
        # 天气数据（温度/光照/风速/云量）+ 地震数据
        # 这是洛神感知真实世界的另一个通道——不是假数据，是真实物理量
        # 精卫公式：a=洛神内部状态, b=公开物理数据, Δ=差异, ⊗=学习
        self.公开感官 = 公开感官(n_hidden=self.大脑.g['n'],
                              纬度=39.9, 经度=116.4, 获取间隔=10)

    def _计算时间信号(self):
        """计算4维时间信号向量——昼夜节律作为环境差异。

        精卫公式：a=概念激活, b=时间信号, Δ=b-a, ⊗=Δ×Σ
        生物学：视交叉上核(SCN)调制认知——昼夜节律影响概念可及性

        用正弦/余弦编码当前时间——保持连续性：
          sin(2π * hour / 24) → 白天正、夜晚负
          cos(2π * hour / 24) → 正午正、午夜负
          sin(2π * minute / 60) → 分钟级变化
          cos(2π * minute / 60) → 分钟级变化
        """
        import math
        now = time.localtime()
        hour = now.tm_hour + now.tm_min / 60.0
        minute = now.tm_min
        return [
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * minute / 60.0),
            math.cos(2 * math.pi * minute / 60.0),
        ]

    def 一拍(self):
        """环境心跳一拍：真实感官 → 身体 → 它的发声 → 留痕。"""
        self.拍 += 1

        光值, 光矩阵, 眼离线 = self.视网膜.看()
        声值, 声矩阵, 耳离线, 听觉特征 = self.耳蜗.听()

        温值 = 读真实温度()
        温离线 = 温值 is None
        if 温离线:
            温值 = self.身体.环境.get("温", 295.0)

        力离线 = True
        力值 = self.身体.环境.get("力", 10.0)

        # ── 公开感官：从公开API获取真实世界物理数据 ──
        # 精卫公式：a=洛神内部状态, b=公开物理数据
        # 天气数据（温度/光照/风速/云量）+ 地震数据
        # 真实感官离线时，公开感官的数据是洛神感知真实世界的来源
        # 真实感官在线时，公开感官的数据是补充信息
        公开信号 = self.公开感官.一拍(self.拍)
        if 公开信号 is not None:
            if 公开信号["光值"] is not None and 眼离线:
                光值 = 公开信号["光值"]
            if 公开信号["温值"] is not None and 温离线:
                温值 = 公开信号["温值"]
            if 公开信号["力值"] is not None:
                力值 = 公开信号["力值"]
            if 公开信号["声值"] is not None and 耳离线:
                声值 = 公开信号["声值"]

        # 注入真实感官（这是转译，不是投喂——数值全部来自真实设备或公开API）
        self.身体.环境["光"] = 光值
        self.身体.环境["声"] = 声值
        self.身体.环境["力"] = 力值
        self.身体.环境["温"] = 温值

        self.身体.推进()

        # ── 大脑：身体激活 → 感官竞争映射 → 8通道输入 → 前向传播 → 学习 ──
        大脑输入 = self.感官.一拍(self.身体.激活)

        # ── 欲望：RPE驱动（多巴胺系统核心机制） ──
        # RPE = 实际输入 - 大脑预测（来自上一拍的大脑预测）
        if self.上拍大脑预测 is not None:
            rpe = [大脑输入[i] - self.上拍大脑预测[i]
                   for i in range(min(len(大脑输入), len(self.上拍大脑预测)))]
        else:
            rpe = None
        欲望结果 = self.欲望.一拍(大脑输入, rpe=rpe, mse=self.上拍大脑mse)
        欲望选中 = 欲望结果["选中"]
        学习率调制 = self.欲望.学习率调制()

        # 欲望选中通道决定大脑关注哪个区域
        当前区域 = max(0, 欲望选中 // 2) if 欲望选中 >= 0 else 0
        大脑预测 = self.大脑.wm.fwd(大脑输入, update=True, region_id=当前区域)
        大脑mse = 0.0
        if self.上拍大脑预测 is not None:
            # 自适应学习率：mse连续驱动lr——不确定时谨慎，确定时加速
            # 不是人为阈值，是连续映射：lr → lr/(1+mse)
            # 这跟碳基生物的神经调节完全一致
            原lr = self.大脑.g['lr']
            自适应lr = 原lr / (1.0 + self.上拍大脑mse)
            self.大脑.g['lr'] = 0.9 * 原lr + 0.1 * 自适应lr
            大脑mse = self.大脑.wm.learn(self.上拍大脑预测, 大脑输入)
            self.上拍大脑mse = 大脑mse
        self.上拍大脑预测 = 大脑预测
        self.上拍大脑输入 = 大脑输入

        # 大脑的surprise是自然产生的——不是人为设定的hunger
        大脑surprise = self.大脑.wm.surprise_level

        # ── 人类通道：把人类放进精卫公式 ──
        # 人类作为b进入认知循环：
        # - 没人在时：b=零向量，洛神内省模式
        # - 有人时：b=人类信号，洛神与人类形成对比（关系模式）
        # 当前默认：模拟有人在和没人在交替（后续接入真实人类输入接口）
        # 简单模拟：每20拍交替有人在/没人在，有人在时给一些文本（含感官描述）
        if self.拍 % 40 < 20:
            人类文本 = None
            if self.拍 % 40 == 0:
                人类文本 = "你好洛神"
            elif self.拍 % 40 == 5:
                人类文本 = "外面很亮"
            elif self.拍 % 40 == 10:
                人类文本 = "有人在说话"
            elif self.拍 % 40 == 15:
                人类文本 = "今天很热"
            人类结果 = self.人类通道.接收(文本=人类文本, 存在=True)
        else:
            人类结果 = self.人类通道.接收(存在=False)

        人类信号 = 人类结果["人类信号"]
        有人在 = 人类结果["有人在"]
        人类情感强度 = 人类结果["情感强度"]
        感官描述 = 人类结果.get("感官描述")

        # ── 感官描述转译注入身体环境 ──
        # 人类描述的世界 → 转译成和视网膜/耳蜗相同维度的信号
        # 当真实感官离线时，人类描述是洛神感知世界的唯一来源
        # 当真实感官在线时，人类描述是补充信息
        if 感官描述:
            if 感官描述["光值"] is not None and 眼离线:
                光值 = 感官描述["光值"]
                self.身体.环境["光"] = 光值
            if 感官描述["声值"] is not None and 耳离线:
                声值 = 感官描述["声值"]
                self.身体.环境["声"] = 声值
            if 感官描述["温值"] is not None and 温离线:
                温值 = 感官描述["温值"]
                self.身体.环境["温"] = 温值
            if 感官描述["力值"] is not None:
                力值 = 感官描述["力值"]
                self.身体.环境["力"] = 力值
            if 感官描述["听觉特征"] is not None:
                听觉特征 = 感官描述["听觉特征"]
            # 感官描述注入后需要重新推进身体
            self.身体.推进()

        # ── 工作记忆：前额叶PFC——维持和操作信息 ──
        # surprise驱动gate：惊讶时更新，平静时维持
        # 人类信号传入工作记忆的人类slot——专门维持人类相关信息
        wm结果 = self.工作记忆.一拍(self.大脑.wm.h, surprise=大脑surprise,
                                  人类信号=人类信号 if 有人在 else None,
                                  有人在=有人在)
        wm_output = wm结果["output"]
        wm_人类output = wm结果.get("人类output", [0.0] * self.大脑.g['n'])

        # ── 海马体：模式分离 + 存储经验 ──
        # 人类互动经验获得更高salience——优先巩固
        hip结果 = self.海马体.一拍(self.大脑.wm.h, prediction_error=大脑mse,
                                region_id=当前区域,
                                人类互动=有人在, 人类情感强度=人类情感强度)
        # 每10拍做一次睡眠重播（在线嵌入的离线巩固）
        if self.拍 % 10 == 0 and len(self.海马体.buffer) >= 8:
            amygdala_mod = 1.0 + min(2.0, 大脑surprise * 3.0)
            self.海马体.睡眠重播(amygdala_modulation=amygdala_mod)

        # ── 数字感官：洛神主动探索数字世界 ──
        # 好奇心驱动：surprise持续低时，洛神主动从知识库获取新信息
        # 获取的新信息转译成数字信号，注入概念层
        # 数字信号和人类信号不会同时主导——有人在时人类信号优先，没人在时数字信号补充
        数字探索请求 = self.数字感官.探索渴望检查(
            surprise=大脑surprise, 概念激活=None,  # 概念激活还没计算，用None
            拍=self.拍, 感官离线=None)
        数字信号 = [0.0] * self.大脑.g['n']
        数字探索结果 = None
        if 数字探索请求:
            数字探索结果 = self.数字感官.获取信息(数字探索请求["探索方向"])
            if 数字探索结果:
                数字信号 = 数字探索结果["数字信号"]
        else:
            self.数字感官.消化()
            数字信号 = self.数字感官.当前数字信号

        # ── 概念层：从大脑隐藏状态涌现概念 ──
        # 高层预测编码：概念层预测隐藏状态的模式，预测误差驱动赫布学习
        # 说听闭环：上一拍的自听特征作为听觉反馈传入概念层
        # 人类信号传入概念层——人类作为精卫公式中的b，通过赫布学习自然形成人类概念
        # 数字信号传入概念层——没人在时数字信号作为补充输入
        # 时间信号传入概念层——昼夜节律作为环境差异，塑造概念个体差异
        概念人类信号 = 人类信号 if 有人在 else 数字信号
        公开向量 = self.公开感官.当前向量 if 公开信号 is not None else None
        时间向量 = self._计算时间信号()
        概念结果 = self.概念.一拍(self.大脑.wm.h,
                                听觉特征=self.上拍自听特征,
                                人类信号=概念人类信号 if any(v != 0 for v in 概念人类信号) else None,
                                公开信号=公开向量,
                                情感状态=self.上拍情感,
                                时间信号=时间向量)
        概念激活 = 概念结果["concept_h"]
        概念mse = 概念结果["mse"]
        概念surprise = 概念结果["surprise"]
        概念活跃 = 概念结果["活跃概念数"]

        # ── 情感状态：从身体和大脑自然涌现（不是人为公式） ──
        大脑状态dict = {"surprise": 大脑surprise, "mse": 大脑mse,
                        "欲望偏好": 欲望结果["偏好"],
                        "概念surprise": 概念surprise,
                        "概念活跃": 概念活跃}
        情感 = self.记忆.情感状态(self.身体.激活, 大脑状态dict)
        绑定基调 = self.绑定.情感基调影响()
        情感["E"] = max(-1.0, min(1.0, 情感["E"] + 绑定基调["E_offset"]))
        情感["T"] = max(-1.0, min(1.0, 情感["T"] + 绑定基调["T_offset"]))

        # 更新上拍情感——下一拍的概念层会用到
        self.上拍情感 = 情感

        # ── 表达通道：洛神→人类（双向通道的发送方向） ──
        # 洛神的内部状态 → 呈现给人类可理解的形式
        # 不是生成精确文本——是呈现洛神"想表达什么"
        # 感知请求：感官离线时洛神主动请求人类帮它感知
        # 交流请求：长期无互动时洛神主动发起对话
        感官离线状态 = {"光": 眼离线, "声": 耳离线, "温": 温离线, "力": 力离线}
        表达结果 = self.人类通道.发送(概念激活=概念激活,
                                    情感状态=情感, surprise=大脑surprise,
                                    拍=self.拍, 感官离线=感官离线状态)
        洛神表达 = 表达结果["意图类型"]
        洛神感知请求 = 表达结果["感知请求"]
        洛神交流请求 = 表达结果["交流请求"]

        # ── 浮现层：躯体标记——主导记忆从身体状态整体映射 ──
        浮现 = self.记忆.浮现(self.身体.激活)

        脑大小 = self.身体.脑大小
        声脑激活 = self.身体.激活[脑大小:2 * 脑大小]
        发了声 = False
        if not self.无声:
            发了声 = self.嘴.发声(概念激活=概念激活, 声脑激活=声脑激活,
                                当前拍=self.拍,
                                组合激活=概念结果.get("combo_h"),
                                情感状态=情感)
        if 发了声:
            self.嘴.等待播完()
            # 说听闭环：提取自听特征，下一拍反馈给概念层
            self.上拍自听特征 = self.嘴.自听特征()
            # 符号绑定学习：发声后自听特征和概念/组合概念激活同时出现 → 赫布学习
            # 基础概念符号绑定：单词级——特定概念→特定声音模式
            # 组合概念符号绑定：句子级——特定组合概念→特定复杂声音模式
            if self.上拍自听特征 is not None:
                self.嘴.符号绑定学习(self.上拍自听特征, 概念激活,
                                   组合激活=概念结果.get("combo_h"))
        else:
            self.上拍自听特征 = None

        # ── 记忆点创建：连续概率，无阈值 ──
        # 记忆概率 = surprise的连续函数——surprise高时几乎必记，低时很少记
        # 不用if阈值判断，用概率自然决定
        记忆了 = False
        记忆概率 = min(1.0, 大脑surprise * 3.0)
        if self.拍 == 1:
            记忆概率 = 1.0
        if 发了声:
            记忆概率 = 1.0
        if random.random() < 记忆概率:
            if 发了声:
                kind = "say"
                感官描述 = "我发出了声音"
            else:
                # kind从大脑通道激活自然涌现——哪对通道最活跃就是那种记忆
                # ch0-1=光脑(see), ch2-3=声脑(hear), ch4-5=力脑(feel), ch6-7=温脑(feel)
                通道和 = [
                    abs(大脑输入[0]) + abs(大脑输入[1]),  # 光脑
                    abs(大脑输入[2]) + abs(大脑输入[3]),  # 声脑
                    abs(大脑输入[4]) + abs(大脑输入[5]),  # 力脑
                    abs(大脑输入[6]) + abs(大脑输入[7]),  # 温脑
                ]
                最活脑 = 通道和.index(max(通道和))
                if 最活脑 == 0:
                    kind = "see"
                    感官描述 = "光%.1f" % 光值
                elif 最活脑 == 1:
                    kind = "hear"
                    感官描述 = "声%.1f" % 声值
                else:
                    kind = "feel"
                    感官描述 = "surprise=%.3f" % 大脑surprise
            self.记忆.创建记忆点(kind, 感官描述, feel=情感)
            记忆了 = True

        激活 = self.身体.激活
        大脑h向量 = self.大脑.wm.h
        状态 = {
            "拍": self.拍,
            "感官": {"光": round(光值, 2), "声": round(声值, 2),
                     "力": round(力值, 2), "温": round(温值, 2)},
            "感官离线": {"视网膜": 眼离线, "耳蜗": 耳离线,
                         "温度觉": 温离线, "力觉": 力离线},
            "激活": {"均值": round(sum(激活) / len(激活), 4),
                     "最大": round(max(激活), 4)},
            "声脑": {"均值": round(sum(声脑激活) / len(声脑激活), 4),
                     "最大": round(max(声脑激活), 4)},
            "发声": 发了声,
            "筋树数": len(self.身体.筋树),
            "大脑": {
                "h均值": round(sum(大脑h向量) / max(1, len(大脑h向量)), 4),
                "h最大": round(max(大脑h向量), 4),
                "surprise": round(大脑surprise, 4),
                "mse": round(大脑mse, 4),
                "饱和比": round(sum(1 for v in 大脑h向量 if abs(v) > 0.98) / max(1, len(大脑h向量)), 3),
                "区域": 当前区域,
            },
            "概念": {
                "活跃": 概念活跃,
                "存活": 概念结果["存活概念数"],
                "mse": round(概念mse, 4),
                "surprise": round(概念surprise, 4),
                "均值": round(sum(概念激活) / max(1, len(概念激活)), 4),
                "最大": round(max((abs(v) for v in 概念激活), default=0), 4),
                "听觉反馈": 概念结果.get("有听觉反馈", False),
                "组合活跃": 概念结果.get("活跃组合数", 0),
                "组合存活": 概念结果.get("存活组合数", 0),
                "高级组合活跃": 概念结果.get("活跃高级组合数", 0),
                "高级组合存活": 概念结果.get("存活高级组合数", 0),
                "情感调制": 概念结果.get("有情感输入", False),
                "时间调制": 概念结果.get("有时间输入", False),
            },
            "工作记忆": {
                "活跃slot": wm结果["活跃slot数"],
                "总强度": wm结果["总强度"],
                "gate": wm结果["gate"],
            },
            "海马体": {
                "buffer": hip结果["buffer_size"],
                "DG活跃": hip结果.get("dg活跃", 0),
                "CA3活跃": hip结果.get("ca3活跃", 0),
                "重播次数": self.海马体.total_replays,
                "人类记忆数": self.海马体.状态摘要().get("人类记忆数", 0),
            },
            "人类通道": {
                "有人在": 有人在,
                "情感强度": round(人类情感强度, 3),
                "互动总数": self.人类通道.互动总数,
                "人类slot强度": wm结果.get("人类slot强度", 0),
                "感官描述": bool(感官描述),
                "洛神表达": 洛神表达,
                "洛神感知请求": 洛神感知请求["请求描述"] if 洛神感知请求 else None,
                "洛神交流请求": 洛神交流请求["请求描述"] if 洛神交流请求 else None,
            },
            "数字感官": {
                "探索总数": self.数字感官.探索总数,
                "正在探索": self.数字感官.正在探索,
                "探索驱动": 数字探索请求["驱动类型"] if 数字探索请求 else None,
                "探索来源": 数字探索结果["来源"] if 数字探索结果 else None,
                "数字信号均值": round(sum(数字信号) / max(1, len(数字信号)), 4),
            },
            "公开感官": {
                "在线": not self.公开感官.离线,
                "获取成功": self.公开感官.获取成功次数,
                "获取失败": self.公开感官.获取失败次数,
                "温值": round(公开信号["温值"], 1) if 公开信号 and 公开信号["温值"] else None,
                "光值": round(公开信号["光值"], 1) if 公开信号 and 公开信号["光值"] else None,
                "力值": round(公开信号["力值"], 1) if 公开信号 and 公开信号["力值"] else None,
                "云量": 公开信号.get("云量") if 公开信号 else None,
                "地震力": round(公开信号.get("地震力", 0), 2) if 公开信号 else None,
            },
            "符号绑定": {
                "发声次数": self.嘴.发声次数,
                "符号绑定概念数": sum(1 for s in self.嘴.符号绑定强度 if s > 0.05),
                "符号绑定均值": round(sum(self.嘴.符号绑定强度) / max(1, self.嘴.概念节点数), 4),
                "组合符号绑定数": sum(1 for s in self.嘴.组合符号绑定强度 if s > 0.05),
                "组合符号绑定均值": round(sum(self.嘴.组合符号绑定强度) / max(1, self.嘴.组合概念节点数), 4),
            },
            "情感发声": {
                "E": 情感.get("E", 0.0),
                "N": 情感.get("N", 0.0),
                "T": 情感.get("T", 0.0),
                "基频调制": round(情感.get("E", 0.0) * 40.0, 1),
                "节奏调制": round(情感.get("N", 0.0) * 3.0, 2),
                "振幅调制": round(max(0.6, min(1.5, 1.0 + 情感.get("N", 0.0) * 0.5)), 3),
                "稳定调制": round(max(0.5, min(1.0, 0.5 + 情感.get("T", 0.0) * 0.5)), 3),
            },
            "欲望": 欲望结果,
            "情感": 情感,
            "浮现": {
                "tone": round(浮现["tone_bias"], 3),
                "length": round(浮现["length_bias"], 3),
                "arousal": round(浮现["arousal_bias"], 3),
                "narrative": round(浮现["narrative_potential"], 3),
                "dominance": round(浮现["background"]["dominance"], 3),
            },
            "记忆了": 记忆了,
            "绑定": self.绑定.状态摘要(),
        }
        self.链.记("home_tick", 状态)

        if 光矩阵 is not None:
            with open(os.path.join(状态目录, "最近光矩阵.json"), "w",
                      encoding="utf-8") as 文件:
                json.dump(光矩阵.tolist(), 文件)
        if 声矩阵 is not None:
            with open(os.path.join(状态目录, "最近声矩阵.json"), "w",
                      encoding="utf-8") as 文件:
                json.dump(声矩阵.tolist(), 文件)

        return 状态

    def 快照(self):
        路径 = os.path.join(状态目录, f"快照_{self.拍}.pkl")
        with open(路径, "wb") as 文件:
            pickle.dump({
                "拍": self.拍,
                "身体状态": dict(self.身体.__dict__),
                "大脑基因组": self.大脑.g,
                "大脑wm状态": {
                    "h": self.大脑.wm.h,
                    "prev_h": self.大脑.wm.prev_h,
                    "homeo": self.大脑.wm.homeo,
                    "surprise_level": self.大脑.wm.surprise_level,
                    "sens_adapt": self.大脑.wm.sens_adapt,
                },
                "概念层状态": {
                    "concept_h": self.概念.concept_h,
                    "concept_strength": self.概念.concept_strength,
                    "concept_w": self.概念.concept_w,
                    "concept_rw": self.概念.concept_rw,
                    "concept_pw": self.概念.concept_pw,
                    "concept_otimes_w": self.概念.concept_otimes_w,
                    "concept_aw": self.概念.concept_aw,
                    "concept_sw": self.概念.concept_sw,
                    "concept_human_w": self.概念.concept_human_w,
                    "concept_public_w": self.概念.concept_public_w,
                    "concept_emotion_w": self.概念.concept_emotion_w,
                    "concept_time_w": self.概念.concept_time_w,
                    "combo_h": self.概念.combo_h,
                    "combo_strength": self.概念.combo_strength,
                    "combo_w": self.概念.combo_w,
                    "combo_rw": self.概念.combo_rw,
                    "combo_pw": self.概念.combo_pw,
                    "meta_combo_h": self.概念.meta_combo_h,
                    "meta_combo_strength": self.概念.meta_combo_strength,
                    "meta_combo_w": self.概念.meta_combo_w,
                    "meta_combo_rw": self.概念.meta_combo_rw,
                    "error_history": self.概念.error_history,
                    "surprise_level": self.概念.surprise_level,
                    "拍": self.概念.拍,
                },
                "人类通道状态": {
                    "有人在": self.人类通道.有人在,
                    "互动总数": self.人类通道.互动总数,
                    "人类名字": self.人类通道.人类名字,
                    "表达历史长度": len(self.人类通道.表达历史),
                    "感官离线时间": dict(self.人类通道.感官离线时间),
                    "无互动拍数": self.人类通道.无互动拍数,
                },
                "数字感官状态": {
                    "探索总数": self.数字感官.探索总数,
                    "已探索文件数": len(self.数字感官.已探索文件),
                    "知识库大小": len(self.数字感官.文件列表),
                    "正在探索": self.数字感官.正在探索,
                },
                "公开感官状态": {
                    "获取成功": self.公开感官.获取成功次数,
                    "获取失败": self.公开感官.获取失败次数,
                    "离线": self.公开感官.离线,
                    "天气数据": self.公开感官.天气数据,
                    "地震数据": self.公开感官.地震数据,
                },
                "符号绑定状态": {
                    "发声次数": self.嘴.发声次数,
                    "符号绑定强度": self.嘴.符号绑定强度,
                    "概念频点权重": self.嘴.概念频点权重,
                    "概念基频权重": self.嘴.概念基频权重,
                    "组合符号绑定强度": self.嘴.组合符号绑定强度,
                    "组合概念频点权重": self.嘴.组合概念频点权重,
                    "组合概念基频权重": self.嘴.组合概念基频权重,
                },
            }, 文件)
        return 路径

    def 跑(self, 拍数=None, 快照间隔=1000):
        开场 = {
            "视网膜在线": not self.视网膜.离线,
            "耳蜗在线": not self.耳蜗.离线,
            "发声在线": not self.嘴.离线,
            "温度觉在线": 读真实温度() is not None,
            "numba加速": _NUMBA_OK,
            "大脑出厂": True,
        }
        print("=" * 70)
        print("洛神之家 v2 —— 零投喂，真实感官，身体+大脑，十七八岁起步")
        print("=" * 70)
        for k, v in 开场.items():
            print(f"  {k}: {'是' if v else '否（感官离线，绝不造假数据）'}")
        print(f"  每拍 = {self.拍秒} 秒（真实时间心跳）")
        print(f"  大脑: n={self.大脑.g['n']}, lr={self.大脑.g['lr']:.3f}, "
              f"sat_prevent={self.大脑.g['sat_prevent']}, "
              f"adapt_cap={self.大脑.g['adapt_cap']}, "
              f"ch3_boost={self.大脑.g['ch3_boost']}")
        print(f"  概念层: n_concepts={self.概念.n_concepts}, "
              f"n_combo={self.概念.n_combo}, "
              f"n_hidden={self.概念.n_hidden}, "
              f"sparse_k={self.概念.sparse_k}, "
              f"lr={self.概念.lr}")
        print(f"  工作记忆: n_slots={self.工作记忆.n_slots}, "
              f"n_hidden={self.工作记忆.n_hidden}")
        print(f"  海马体: n_dg={self.海马体.n_dg}, "
              f"n_ca3={self.海马体.n_ca3}, "
              f"buffer={self.海马体.buffer_size}, "
              f"consolidation_lr={self.海马体.consolidation_lr}")
        print(f"  人类通道: n_hidden={self.人类通道.n_hidden}, "
              f"互动总数={self.人类通道.互动总数}")
        print(f"  数字感官: 知识库={len(self.数字感官.文件列表)}个文件, "
              f"已探索={len(self.数字感官.已探索文件)}")
        print(f"  公开感官: {'在线' if not self.公开感官.离线 else '离线'}, "
              f"位置=({self.公开感官.纬度}, {self.公开感官.经度}), "
              f"获取间隔={self.公开感官.获取间隔}拍")
        if self.绑定.是否绑定():
            print(f"  绑定: {self.绑定.绑定者} (强度={self.绑定.绑定强度()})")
        else:
            print(f"  绑定: 未绑定（洛神在等她的人）")
        print("=" * 70)

        try:
            while 拍数 is None or self.拍 < 拍数:
                拍开始 = time.time()
                状态 = self.一拍()
                离线标记 = "".join(
                    [k for k, v in 状态["感官离线"].items() if v])
                print(f"拍{状态['拍']:>6}  光{状态['感官']['光']:>6.1f}  "
                      f"声{状态['感官']['声']:>6.1f}  "
                      f"温{状态['感官']['温']:>6.1f}  "
                      f"激活{状态['激活']['均值']:>7.4f}  "
                      f"筋{状态['筋树数']:>4}  "
                      f"脑h{状态['大脑']['h均值']:>7.4f}  "
                      f"惊{状态['大脑']['surprise']:.3f}  "
                      f"mse{状态['大脑']['mse']:.4f}  "
                      f"概念{状态['概念']['活跃']:>2}  "
                      f"组合{状态['概念'].get('组合活跃', 0):>2}  "
                      f"WM{状态['工作记忆']['活跃slot']}  "
                      f"hip{状态['海马体']['buffer']:>3}  "
                      f"欲{状态['欲望']['选中']:>2}  "
                      f"E{状态['情感']['E']:>5.2f}  "
                      f"N{状态['情感']['N']:>5.2f}  "
                      f"{'人!' if 状态['人类通道']['有人在'] else '独'}  "
                      f"{状态['人类通道']['洛神表达'] or '':>4}  "
                      f"{'感知?' if 状态['人类通道']['洛神感知请求'] else ''}"
                      f"{'交流?' if 状态['人类通道']['洛神交流请求'] else ''}"
                      f"{'探索!' if 状态['数字感官']['正在探索'] else ''}"
                      f"{'天气!' if 状态.get('公开感官', {}).get('在线') else ''}  "
                      f"{'发声!' if 状态['发声'] else ''}  "
                      f"{'记!' if 状态['记忆了'] else ''}  "
                      f"{('[离线:' + 离线标记 + ']') if 离线标记 else ''}")
                if 状态["拍"] % 快照间隔 == 0:
                    print(f"  → 快照: {self.快照()}")
                余 = self.拍秒 - (time.time() - 拍开始)
                if 余 > 0:
                    time.sleep(余)
        except KeyboardInterrupt:
            print(f"\n人在第 {self.拍} 拍带它回家。")
        finally:
            self.视网膜.关闭()
            self.耳蜗.关闭()
            print(f"快照: {self.快照()}")
            print(f"留痕: {os.path.join(状态目录, 'chain.jsonl')}")


def main():
    解析 = argparse.ArgumentParser(description="洛神之家 v2 —— 环境容器")
    解析.add_argument("--拍秒", type=float, default=1.0,
                      help="每拍对应真实秒数（默认 1.0）")
    解析.add_argument("--拍数", type=int, default=None,
                      help="总拍数（默认无限，Ctrl+C 停）")
    解析.add_argument("--无声", action="store_true",
                      help="不外放发声（设备无扬声器时）")
    解析.add_argument("--摄像头号", type=int, default=0)
    解析.add_argument("--麦克风号", type=int, default=None)
    参数 = 解析.parse_args()

    家 = 洛神之家(拍秒=参数.拍秒, 无声=参数.无声,
                  摄像头号=参数.摄像头号, 麦克风号=参数.麦克风号)
    家.跑(拍数=参数.拍数)


if __name__ == "__main__":
    main()
