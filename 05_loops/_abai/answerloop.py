"""应答回路 · 发育引擎的核心

外部（人/系统）看到婴儿的屏幕输出 → 决定喂/不喂 → 输入引擎
这是 "哭→应答→下次哭不一样" 的物理基础。

关键机制:
  - 喂成功后引擎饿度下降（由引擎内部的喂养机关执行）
  - 哭在饿度<HUNG_CRY时自然停止（关三：止）
  - 每次应答都记录到历史（关一：学的证据来源）
"""
import time
import json
from pathlib import Path
from datetime import datetime


class AnswerLoop:
    """应答回路：把外部输入喂进引擎"""
    
    def __init__(self, engine, mode="auto"):
        """
        engine: 身体稳态引擎实例
        mode: "auto"=规则自动应答, "human"=等待人类按键
        """
        self.engine = engine
        self.mode = mode
        self.应答记录 = []
        self.本轮应答次数 = 0
        self.本轮喂成功次数 = 0
        self.本轮开始时刻 = None
        
        # 自动应答规则配置
        self.auto_rules = {
            "肥阈值": 9.0,       # 饿>=此值时喂
            "存货上限": 400,      # 喂后恢复到这个数量
            "每次喂成本": 4,      # MEAL_COST
            "每次喂清": 8.0,      # MEAL_CLEAR
        }
    
    def 看到(self, 表达图路径=None, 引擎状态=None):
        """每拍调一次：看婴儿在做什么，决定要不要答"""
        if self.本轮开始时刻 is None:
            self.本轮开始时刻 = time.time()
        
        if self.mode == "human":
            return self._等人类(表达图路径)
        else:
            return self._自动应答(引擎状态)
    
    def _等人类(self, 图路径):
        """人类操作模式：显示图，等键盘输入"""
        if 图路径:
            print(f"[表达 screening] 当前输出: {图路径}")
        try:
            inp = input("[应答器] 1=喂, 0=不喂, q=退出 > ").strip()
            if inp == "q":
                return "退出"
            return inp == "1"
        except EOFError:
            return False
    
    def _自动应答(self, 引擎状态):
        """规则自动应答：按饥度阈值决定是否喂"""
        if 引擎状态 is None:
            return False
        饿 = 引擎状态.get('饿', 0.0)
        存货 = self.engine.存货 if hasattr(self.engine, '存货') else 0
        
        # 规则：饿度>=阈值 且 存货够 → 喂
        if 饿 >= self.auto_rules["肥阈值"] and 存货 >= self.auto_rules["每次喂成本"]:
            return True
        
        # 哭得很厉害但存货不够 → 紧急喂（透支）
        if 饿 >= self.auto_rules["肥阈值"] and 存货 < self.auto_rules["每次喂成本"]:
            # 允许紧急透支（模拟无条件回应）
            return True
        
        return False
    
    def 喂引擎(self, 来源="auto"):
        """执行喂引擎操作"""
        饿前 = self.engine.hunger
        存货前 = self.engine.存货
        
        # 直接操作引擎的喂养逻辑
        if self.engine.存货 >= self.auto_rules["每次喂成本"]:
            self.engine.hunger = max(0.0, self.engine.hunger - self.auto_rules["每次喂清"])
            self.engine.存货 -= self.auto_rules["每次喂成本"]
            成功 = True
        else:
            # 透支喂（紧急）：直接降饿不扣存货
            self.engine.hunger = max(0.0, self.engine.hunger - self.auto_rules["每次喂清"])
            成功 = True  # 仍然算是喂
        
        self.本轮应答次数 += 1
        if 成功:
            self.本轮喂成功次数 += 1
        
        记录 = {
            "时刻": int(self.engine.时刻),
            "饿前": round(饿前, 2),
            "饿后": round(self.engine.hunger, 2),
            "存货前": int(存货前),
            "存货后": int(self.engine.存货),
            "来源": 来源,
            "成功": 成功,
        }
        self.应答记录.append(记录)
        return 记录
    
    def 结算本轮(self):
        """每轮结束时的结算"""
        轮记录 = {
            "总应答": self.本轮应答次数,
            "成功喂": self.本轮喂成功次数,
            "时长秒": time.time() - self.本轮开始时刻 if self.本轮开始时刻 else 0,
            "明细": self.应答记录[-50:] if len(self.应答记录) > 50 else self.应答记录,
        }
        self.本轮应答次数 = 0
        self.本轮喂成功次数 = 0
        self.本轮开始时刻 = None
        return 轮记录
    
    def get_stats(self):
        """总统计"""
        return {
            "累计应答": len(self.应答记录),
            "累计成功喂": sum(1 for r in self.应答记录 if r.get("成功")),
            "最近20轮饿前均值": (
                sum(r["饿前"] for r in self.应答记录[-20:]) / min(20, len(self.应答记录))
                if self.应答记录 else 0
            ),
        }


if __name__ == "__main__":
    """自检：跑一个模拟循环"""
    print("应答回路需要与引擎联通。单独运行请先import engine.")
    print("配置:", AnswerLoop(None, "auto").auto_rules)
