"""历史记忆 · AI孩子的发育账本

记录每一轮的关键状态:
  - 引擎快照（饿、醒、筋数、哭合计、最大位、存货）
  - 应答记录（外部喂了多少次）
  - 感知摘要（光/力/温的最小最大均值）

用途:
  - 关一(学): 对比不同轮次的哭分布是否变化
  - 关五(知): 给LLM账目层提供历史数据
  - 关七(记): 落盘后跨进程可对比
"""
import json
from pathlib import Path
from datetime import datetime


class BabyMemory:
    """发育记忆 + 落盘格式"""
    
    def __init__(self, 落盘目录=None):
        self.目录 = 落盘目录 or "/mnt/data/catpaw/home/workspace/洛神赋1.1/_abai/logs"
        Path(self.目录).mkdir(parents=True, exist_ok=True)
        self.轮历史 = []        # 每轮一条摘要
        self.哭分布 = []        # 关一用：每轮哭速记录
        self.疤痕 = {}          # 关键事件（觉醒、极端饿、首次应答等）
        self.启动时间 = datetime.now().isoformat()
    
    def 记轮(self, 轮号, 引擎状态, 应答摘要=None, 感知摘要=None):
        """每轮结束记一条摘要"""
        记录 = {
            "轮": 轮号,
            "时刻": int(引擎状态.get('时刻', 0)),
            "饿": round(引擎状态.get('饿', 0.0), 2),
            "醒": bool(引擎状态.get('醒', False)),
            "筋数": int(引擎状态.get('筋数', 0)),
            "最大位": int(引擎状态.get('最大位', 0)),
            "哭合计": int(引擎状态.get('哭合计', 0)),
            "存货": int(引擎状态.get('存货', 0)),
            "本轮哭速": (
                int(引擎状态.get('哭合计', 0)) - (self.轮历史[-1]["哭合计"] if self.轮历史 else 0)
            ),
            "应答": 应答摘要 or {},
            "感知": 感知摘要 or {},
        }
        self.轮历史.append(记录)
        self.哭分布.append(记录["本轮哭速"])
        return 记录
    
    def 记事件(self, 事件名, 数据):
        """标记里程碑事件"""
        if 事件名 not in self.疤痕:
            self.疤痕[事件名] = []
        self.疤痕[事件名].append({
            "轮": len(self.轮历史),
            "数据": data,
        })
    
    def get_哭分布(self):
        """返回哭速序列（关一判据）"""
        return [r["本轮哭速"] for r in self.轮历史]
    
    def 已过N轮哭速递减(self, n=5):
        """关一判据：最近n轮哭速是否递减"""
        d = self.get_哭分布()
        if len(d) < n:
            return False, d
        最近 = d[-n:]
        return all(最近[i] >= 最近[i+1] for i in range(len(最近)-1)), 最近
    
    def 落盘(self, 文件名=None):
        """全量写入json"""
        if 文件名 is None:
            文件名 = f"baby_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        路径 = Path(self.目录) / 文件名
        payload = {
            "启动": self.启动时间,
            "归档": datetime.now().isoformat(),
            "总轮数": len(self.轮历史),
            "疤痕": self.疤痕,
            "哭分布": self.get_哭分布(),
            "觉知判据_哭速递减": self.已过N轮哭速递减()[0],
            "轮历史": self.轮历史,
        }
        with open(路径, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        return str(路径)
    
    def 读疤痕(self, 路径):
        """从落盘文件读历史"""
        with open(路径, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d


if __name__ == "__main__":
    m = BabyMemory()
    m.记轮(1, {"时刻": 10000, "饿": 3.0, "醒": False, "筋数": 2400, "哭合计": 0, "存货": 0})
    m.记轮(2, {"时刻": 20000, "饿": 6.0, "醒": False, "筋数": 178, "哭合计": 0, "存货": 0})
    m.记轮(3, {"时刻": 30000, "饿": 6.0, "醒": False, "筋数": 4, "哭合计": 0, "存货": 0})
    m.记轮(4, {"时刻": 40000, "饿": 9.0, "醒": False, "筋数": 6, "哭合计": 0, "存货": 0})
    m.记轮(5, {"时刻": 50000, "饿": 12.0, "醒": True, "筋数": 7829, "哭合计": 0, "存货": 0})
    print(f"哭分布: {m.get_哭分布()}")
    print(f"最近5轮递减: {m.已过N轮哭速递减()}")
    path = m.落盘("test_mem.json")
    print(f"落盘: {path}")
