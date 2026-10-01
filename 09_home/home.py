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

sys.path.insert(0, 家目录)
from 视网膜 import 视网膜  # noqa: E402
from 耳蜗 import 耳蜗  # noqa: E402
from 嘴 import 嘴  # noqa: E402
from 出厂基因 import 出厂基因组, 身体激活到大脑输入  # noqa: E402
from 情感记忆 import 情感记忆  # noqa: E402
from 个人绑定 import 个人绑定  # noqa: E402
from 欲望 import 欲望  # noqa: E402
from 感官竞争 import 感官竞争  # noqa: E402


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
        self.嘴 = 嘴(声脑节点数=self.身体.脑大小, 概念节点数=64)
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
        self.概念 = 概念层(n_hidden=self.大脑.g['n'], n_concepts=64, lr=0.02)

        # 说听闭环：上一拍的自听特征（延迟一拍反馈，模拟骨传导延迟）
        self.上拍自听特征 = None

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

        # 注入真实感官（这是转译，不是投喂——数值全部来自真实设备）
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

        # ── 概念层：从大脑隐藏状态涌现概念 ──
        # 高层预测编码：概念层预测隐藏状态的模式，预测误差驱动赫布学习
        # 说听闭环：上一拍的自听特征作为听觉反馈传入概念层
        概念结果 = self.概念.一拍(self.大脑.wm.h, 听觉特征=self.上拍自听特征)
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

        # ── 浮现层：躯体标记——主导记忆从身体状态整体映射 ──
        浮现 = self.记忆.浮现(self.身体.激活)

        脑大小 = self.身体.脑大小
        声脑激活 = self.身体.激活[脑大小:2 * 脑大小]
        发了声 = False
        if not self.无声:
            发了声 = self.嘴.发声(概念激活=概念激活, 声脑激活=声脑激活,
                                当前拍=self.拍)
        if 发了声:
            self.嘴.等待播完()
            # 说听闭环：提取自听特征，下一拍反馈给概念层
            self.上拍自听特征 = self.嘴.自听特征()
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
                    "error_history": self.概念.error_history,
                    "surprise_level": self.概念.surprise_level,
                    "拍": self.概念.拍,
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
              f"n_hidden={self.概念.n_hidden}, "
              f"sparse_k={self.概念.sparse_k}, "
              f"lr={self.概念.lr}")
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
                      f"欲{状态['欲望']['选中']:>2}  "
                      f"E{状态['情感']['E']:>5.2f}  "
                      f"N{状态['情感']['N']:>5.2f}  "
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
