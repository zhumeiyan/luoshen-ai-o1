"""主循环 · AI孩子的发育引擎

感知 → 引擎推进 → 表达 → 应答 → 记忆 → 下一帧
每帧循环约30fps（视频帧率）+ 引擎推进。

与身体稳态.py的关系:
  不修改引擎本体任何代码。
  1. 从引擎读状态（饿、醒、筋数、激活、哭合计）
  2. 把感知帧的驱动物写入引擎环境
  3. 引擎自己推一轮
  4. 从引擎读输出，调表达选择器画图
  5. 调应答回路决定是否喂引擎
  6. 调记忆模块记账
"""
import time
import numpy as np
from pathlib import Path
from _abai.perception import PerceptionProcessor
from _abai.expression import ExpressionSelector
from _abai.answerloop import AnswerLoop
from _abai.memory import BabyMemory


class BabyLoop:
    """发育主循环"""
    
    def __init__(self, engine, video_path=None, ticks_per_frame=100,
                 应答模式="auto", 落盘目录=None):
        self.engine = engine
        self.ticks_per_frame = ticks_per_frame  # 引擎每帧推多少tick
        
        self.感知 = PerceptionProcessor(source=video_path)
        self.表达 = ExpressionSelector(200, 200)
        self.应答 = AnswerLoop(engine, mode=应答模式)
        self.记忆 = BabyMemory(落盘目录)
        
        self.帧数 = 0
        self.表达目录 = Path(self.记忆.目录) / "expressions"
        self.表达目录.mkdir(parents=True, exist_ok=True)
        self.运行中 = False
    
    def _引擎状态摘要(self):
        """从引擎读当前状态"""
        eng = self.engine
        # 筋树：活筋（位点>0）才计入
        活筋 = [j for j in eng.筋树.values() if j.get('位点', 0) > 0]
        筋数 = len(活筋)
        最大位 = max((j['位点'] for j in 活筋), default=0) if 活筋 else 0
        
        return {
            "时刻": int(eng.时刻),
            "饿": float(eng.hunger),
            "醒": bool(eng.awake),
            "存货": int(getattr(eng, '存货', 0)),
            "哭合计": int(getattr(eng, '哭合计', 0)),
            "筋数": 筋数,
            "最大位": 最大位,
        }
    
    def _写引擎环境(self, 感知结果):
        """把感知驱动物写入引擎环境（引擎已有的接口）"""
        if 感知结果 is None:
            return
        # 引擎四脑:
        #   光 = 环境亮度(0-1) × 放大 → 与环境基线量级匹配
        #   力 = 帧间运动(0-1)
        #   声 = 脉动代理(已缩放)
        #   温 = 肤色占比映射到280-310K
        
        # 引擎使用环境字典 self.环境 = {"光":50.0,"声":45.0,"力":10.0,"温":295.0}
        
        # 亮度映射: 0-1 → 10-100 (引擎基线是50)
        光 = 10.0 + 感知结果["光"] * 90.0
        self.engine.环境["光"] = 光
        
        # 力: 帧间运动 0-0.1 → 放大到引擎量级 0-30
        力 = 10.0 + 感知结果["力"] * 200.0
        self.engine.环境["力"] = min(40.0, 力)
        
        # 声: 色彩脉动 → 引擎基线45
        声 = 45.0 + 感知结果["声"]
        self.engine.环境["声"] = min(100.0, 声)
        
        # 温: 来自肤色 → 295±15K
        self.engine.环境["温"] = 感知结果["温"]
    
    def step(self):
        """执行一帧（感知→引擎→表达→应答→记忆）"""
        # 1. 感知
        感知 = self.感知.处理()
        if 感知 is None:
            return False  # 视频帧耗尽
        
        # 2. 写入引擎环境
        self._写引擎环境(感知)
        
        # 3. 引擎推N个tick
        for _ in range(self.ticks_per_frame):
            self.engine.推进()
        
        self.帧数 += 1
        
        # 4. 读取引擎状态
        状态 = self._引擎状态摘要()
        
        # 5. 表达选择器画图（每10帧画一次，减少IO）
        表达路径 = None
        if self.帧数 % 10 == 0:
            表达路径 = str(self.表达目录 / f"frame_{self.帧数:06d}.png")
            self.表达.render_to_file(状态, 表达路径)
        
        # 6. 应答
        if self.应答.mode == "human":
            结果 = self.应答.看到(表达路径, 状态)
            if 结果 == "退出":
                return False
            if 结果:
                self.应答.喂引擎("human")
        else:
            # 自动应答：按阈值
            if self.应答._自动应答(状态):
                self.应答.喂引擎("auto")
        
        # 7. 每100帧记一次记忆
        if self.帧数 % 100 == 0:
            self.记忆.记轮(
                self.帧数,
                状态,
                答摘要=self.应答.get_stats(),
                感知摘要={"光均值": 感知["光"], "力均值": 感知["力"], "温": 感知["温"], "皮肤": 感知["皮肤"]},
            )
            print(f"[帧{self.帧数:5d}] 饿={状态['饿']:.1f} 醒={状态['醒']} 筋={状态['筋数']} 哭={状态['哭合计']} 存={状态['存货']}")
        
        return True
    
    def run(self, max_frames=1000):
        """跑起来"""
        print(f"=== AI孩子启动 ===")
        print(f"视频源: {self.感知.源}")
        print(f"引擎每帧推: {self.ticks_per_frame} ticks")
        print(f"应答模式: {self.应答.mode}")
        print(f"最大帧数: {max_frames}")
        print(f"表达目录: {self.表达目录}")
        
        self.运行中 = True
        t0 = time.time()
        
        try:
            for _ in range(max_frames):
                if not self.step():
                    break
        except KeyboardInterrupt:
            print("中断")
        
        dt = time.time() - t0
        self.运行中 = False
        
        # 结算
        路径 = self.记忆.落盘()
        print(f"\n=== 完成 ===")
        print(f"总帧数: {self.帧数}")
        print(f"耗时: {dt:.1f}秒")
        print(f"哭分布: {self.记忆.get_哭分布()}")
        print(f"落盘: {路径}")
        return 路径


if __name__ == "__main__":
    print("需要先import身体稳态引擎。示例:")
    print("  from 身体稳态 import FourBody")
    print("  e = FourBody(种子=7)")
    print("  loop = BabyLoop(e, video_path='...', ticks_per_frame=100)")
    print("  loop.run(max_frames=200)")
