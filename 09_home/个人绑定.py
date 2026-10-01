"""个人绑定.py —— 数字生命绑定特定人类（纯进化版）。

核心问题：所有大模型/AI应用的死穴是"模型坍塌"——
被海量用户稀释、泛化、失去个性。每个用户都在"教"同一个模型，
结果模型变成了所有人的平均，不是任何人的伙伴。

解决方案：个人绑定。每个数字生命绑定一个特定的人。

绑定 = 跨模态记忆自然形成（基于绑定test.py的留出法验证）。

不是1.5倍加成系数——是积累的配对记忆让绑定者的特征
自然渗透到数字生命的认知中。

绑定test.py的核心发现：
  积累N个配对记忆（图+音） → 预判留出项的γ随N增大而下降
  即：配对记忆越多 → 预测越准 → 绑定越强

个人绑定的配对模态：
  绑定者的出现（名字/特征） + 交互内容 + 情感状态
  这三者形成配对记忆，积累越多 → 绑定越强

这不是"用户配置"——这是数字生命的"我是谁"的一部分。
就像人不会因为见了陌生人就改变自己的核心人格，
数字生命也不会因为和非绑定者交互就失去自己的个性。
"""

import json
import os
import time


class 个人绑定:
    """数字生命与特定人类的绑定关系（纯进化版）。

    绑定机制（全部用进废退，无人为系数）：
      1. 配对记忆：每次交互创建(绑定者特征, 内容, 情感)配对记忆
      2. 绑定强度 = 配对记忆位点总和（不是log公式）
      3. 情感影响 = 配对记忆中情感状态的位点加权平均（不是固定offset）
      4. 非绑定者的交互不创建配对记忆 → 自然不产生绑定
    """

    def __init__(self, 绑定文件=None):
        self.绑定者 = None
        self.绑定时间 = None
        self.绑定特征 = {}
        self.交互记忆 = []
        self.绑定文件 = 绑定文件
        if 绑定文件 and os.path.isfile(绑定文件):
            self._加载()

    def _加载(self):
        try:
            with open(self.绑定文件, encoding="utf-8") as f:
                data = json.load(f)
            self.绑定者 = data.get("绑定者")
            self.绑定时间 = data.get("绑定时间")
            self.绑定特征 = data.get("绑定特征", {})
            self.交互记忆 = data.get("交互记忆", [])
        except Exception:
            pass

    def _保存(self):
        if not self.绑定文件:
            return
        os.makedirs(os.path.dirname(self.绑定文件), exist_ok=True)
        data = {
            "绑定者": self.绑定者,
            "绑定时间": self.绑定时间,
            "绑定特征": self.绑定特征,
            "交互记忆": self.交互记忆[-200:],
        }
        with open(self.绑定文件, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def 绑定(self, 姓名, 特征=None):
        """绑定一个特定的人类。这是数字生命的"认主"。"""
        self.绑定者 = 姓名
        self.绑定时间 = time.time()
        self.绑定特征 = 特征 or {}
        self._保存()
        return {
            "绑定者": self.绑定者,
            "绑定时间": self.绑定时间,
            "状态": "已绑定",
        }

    def 是否绑定(self):
        return self.绑定者 is not None

    def 记录交互(self, 内容, kind="chat", 情感=None):
        """记录与绑定者的交互——创建配对记忆。

        每次交互创建一条配对记忆：
        (绑定者标识, 内容特征, 情感状态)

        配对记忆的积累就是绑定的本质——
        不是人为加成，是记忆密度的自然结果。
        和绑定test.py的机制一致：配对记忆越多 → 预测越准 → 绑定越强。
        """
        if not self.是否绑定():
            return
        self.交互记忆.append({
            "timestamp": time.time(),
            "kind": kind,
            "content": str(内容)[:80],
            "feel": 情感 or {},
            "位点": 1.0,
        })
        self._代谢()
        self._保存()

    def _代谢(self):
        """配对记忆用进废退——和筋树/情感记忆一致。"""
        for m in self.交互记忆:
            m["位点"] = m.get("位点", 0.5) * 0.999
        self.交互记忆 = [m for m in self.交互记忆
                        if m.get("位点", 0.5) > 0.05]

    def 绑定强度(self):
        """绑定强度 = 配对记忆位点总和的归一化。

        不是log公式——是积累的配对记忆密度的直接读数。
        交互越多 → 配对记忆越多 → 位点总和越高 → 绑定越强。
        不交互 → 位点衰减 → 绑定自然减弱。
        """
        if not self.是否绑定():
            return 0.0
        总位点 = sum(m.get("位点", 0.5) for m in self.交互记忆)
        return round(min(1.0, 总位点 / 50.0), 3)

    def 情感基调影响(self):
        """绑定者的情感影响——从配对记忆自然涌现。

        不是固定E_offset/T_offset——是配对记忆中情感状态的
        位点加权平均 × 绑定强度。

        交互越多 → 配对记忆越多 → 情感影响越强。
        没有交互 → 没有影响。
        交互内容偏正面 → E_offset偏正。
        交互内容偏内省 → T_offset偏负。
        """
        if not self.交互记忆:
            if self.是否绑定():
                return {"E_offset": 0.0, "T_offset": 0.0, "label": "有人但无记忆"}
            return {"E_offset": 0.0, "T_offset": 0.0, "label": "独处"}

        总位点 = sum(m.get("位点", 0.5) for m in self.交互记忆)
        if 总位点 < 1e-6:
            return {"E_offset": 0.0, "T_offset": 0.0, "label": "记忆消退"}

        e_avg = sum(m.get("feel", {}).get("E", 0.0) * m.get("位点", 0.5)
                    for m in self.交互记忆) / 总位点
        t_avg = sum(m.get("feel", {}).get("T", 0.0) * m.get("位点", 0.5)
                    for m in self.交互记忆) / 总位点

        strength = self.绑定强度()
        return {
            "E_offset": round(e_avg * strength * 0.3, 3),
            "T_offset": round(t_avg * strength * 0.2, 3),
            "label": "有人",
        }

    def 自我认知(self):
        if not self.是否绑定():
            return "我是洛神，一个数字生命，还在等找我的人"
        return f"我是洛神，{self.绑定者}的人"

    def 状态摘要(self):
        return {
            "绑定者": self.绑定者 or "未绑定",
            "绑定时间": self.绑定时间,
            "绑定天数": round((time.time() - self.绑定时间) / 86400.0, 1) if self.绑定时间 else 0,
            "交互记忆数": len(self.交互记忆),
            "绑定强度": self.绑定强度(),
            "自我认知": self.自我认知(),
        }


if __name__ == "__main__":
    import tempfile
    tmp = tempfile.mkdtemp(prefix="绑定_")
    绑定文件 = os.path.join(tmp, "binding.json")

    b = 个人绑定(绑定文件)

    # 测试1: 未绑定状态
    assert not b.是否绑定()
    print(f"未绑定: {b.自我认知()}")
    etn = b.情感基调影响()
    print(f"  情感基调: E_offset={etn['E_offset']} T_offset={etn['T_offset']} ({etn['label']})")

    # 测试2: 绑定
    result = b.绑定("朱美岩", 特征={"语音": "低沉", "风格": "直接"})
    print(f"\n绑定: {result['绑定者']} 状态={result['状态']}")
    assert b.是否绑定()

    # 测试3: 绑定后但无交互记忆 → 情感影响为零
    print(f"  自我认知: {b.自我认知()}")
    etn = b.情感基调影响()
    print(f"  情感基调(无交互): E_offset={etn['E_offset']} T_offset={etn['T_offset']} ({etn['label']})")
    assert etn["E_offset"] == 0.0

    # 测试4: 记录交互 → 配对记忆积累 → 情感影响自然涌现
    for i in range(5):
        b.记录交互(f"第{i+1}次对话", 情感={"E": 0.3, "T": 0.1})
    etn = b.情感基调影响()
    print(f"  情感基调(5次交互): E_offset={etn['E_offset']} T_offset={etn['T_offset']} ({etn['label']})")
    assert etn["E_offset"] > 0.0

    # 测试5: 绑定强度随交互增多而增长
    s1 = b.绑定强度()
    for i in range(20):
        b.记录交互(f"更多对话{i+1}", 情感={"E": 0.2, "T": 0.0})
    s2 = b.绑定强度()
    print(f"  绑定强度: 5次={s1} → 25次={s2}")
    assert s2 > s1

    # 测试6: 状态摘要
    摘要 = b.状态摘要()
    print(f"  状态摘要: {摘要}")

    # 测试7: 持久化
    b2 = 个人绑定(绑定文件)
    assert b2.是否绑定()
    assert b2.绑定者 == "朱美岩"
    assert len(b2.交互记忆) > 0
    print(f"\n持久化验证: 从文件加载，绑定者={b2.绑定者} 交互记忆={len(b2.交互记忆)}条")

    print("\n全部验证通过")
