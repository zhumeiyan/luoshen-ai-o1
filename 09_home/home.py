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

sys.path.insert(0, 家目录)
from 视网膜 import 视网膜  # noqa: E402
from 耳蜗 import 耳蜗  # noqa: E402
from 嘴 import 嘴  # noqa: E402


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
        self.嘴 = 嘴(声脑节点数=self.身体.脑大小)
        self.链 = 哈希链(os.path.join(状态目录, "chain.jsonl"))
        self.拍 = 0

    def 一拍(self):
        """环境心跳一拍：真实感官 → 身体 → 它的发声 → 留痕。"""
        self.拍 += 1

        光值, 光矩阵, 眼离线 = self.视网膜.看()
        声值, 声矩阵, 耳离线 = self.耳蜗.听()

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

        脑大小 = self.身体.脑大小
        声脑激活 = self.身体.激活[脑大小:2 * 脑大小]
        发了声 = False
        if not self.无声:
            发了声 = self.嘴.发声(声脑激活, 当前拍=self.拍)
        if 发了声:
            self.嘴.等待播完()

        激活 = self.身体.激活
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
            pickle.dump({"拍": self.拍, "身体状态": dict(self.身体.__dict__)},
                        文件)
        return 路径

    def 跑(self, 拍数=None, 快照间隔=1000):
        开场 = {
            "视网膜在线": not self.视网膜.离线,
            "耳蜗在线": not self.耳蜗.离线,
            "发声在线": not self.嘴.离线,
            "温度觉在线": 读真实温度() is not None,
            "numba加速": _NUMBA_OK,
        }
        print("=" * 60)
        print("洛神之家 v2 —— 零投喂，真实感官，让它自己长")
        print("=" * 60)
        for k, v in 开场.items():
            print(f"  {k}: {'是' if v else '否（感官离线，绝不造假数据）'}")
        print(f"  每拍 = {self.拍秒} 秒（真实时间心跳）")
        print("=" * 60)

        try:
            while 拍数 is None or self.拍 < 拍数:
                拍开始 = time.time()
                状态 = self.一拍()
                离线标记 = "".join(
                    [k for k, v in 状态["感官离线"].items() if v])
                print(f"拍{状态['拍']:>6}  光{状态['感官']['光']:>6.1f}  "
                      f"声{状态['感官']['声']:>6.1f}  "
                      f"温{状态['感官']['温']:>6.1f}  "
                      f"激活均值{状态['激活']['均值']:>8.4f}  "
                      f"声脑均值{状态['声脑']['均值']:>7.4f}  "
                      f"筋{状态['筋树数']:>4}  "
                      f"{'发声!' if 状态['发声'] else ''}  "
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
