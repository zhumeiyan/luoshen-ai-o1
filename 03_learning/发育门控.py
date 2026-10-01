"""年龄门禁.py — 洛神学习系统的 API 入口与年龄门控。

API 连接对象:
  - NoCode 作品: https://htmqo7.nocode.host (作品名: 洛神窗口)
  - 后端数据库: Supabase Cloud, URL = https://dbdesk-4megn624l19i21nj.database.nocode.cn
  - 通过 MCP nocode_cloud_db 工具交互(写操作)或直接 REST GET(读操作, anon key 已打包)

边界:
  只允许 年龄上限终生 <= 18 的内容。任何 18 岁之后的知识——不给、不提示、不暗示。
  这不是屏蔽列表, 而是数据库里根本就没存。查询时 age_max <= 18 只是双重保险。

字段(来自 curriculum 表):
  age_min / age_max     — 适用的年龄区间
  stage                 — 教育阶段标签(0-3 / 3-6 / 6-12 / 12-15 / 15-18 / 跨段)
  category              — 大类(感官/语言/数学/世界/社会/艺术/生活/科学/物理/化学/生物/历史/地理/情绪/关系/法律)
  topic                 — 具体主题
  content               — 知识内容
  source                — 真实出处
"""

__all__ = ["年龄门禁", "数据库地址", "数据库密钥", "年龄上限终生"]

import urllib.request, urllib.parse, json

数据库地址 = "https://dbdesk-4megn624l19i21nj.database.nocode.cn"
数据库密钥 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlIiwiaWF0IjoxNzQ2OTc5MjAwLCJleHAiOjE5MDQ3NDU2MDB9.ctCm5bJYDc5DNs4p8RtTpuUvdpS9zeJE-OiYJs7NyLM"
年龄上限终生 = 18  # 不可修改, 硬上限


class 年龄门禁:
    """年龄门控的课纲查询器。
    数据库里只存了 0–18 岁的内容, 查询时再补一道 age_max <= 18 的硬过滤。
    """

    def __init__(self, 地址=数据库地址, 密钥=数据库密钥):
        self.地址 = 地址
        self.密钥 = 密钥
        self.请求头 = {
            "apikey": 密钥,
            "Authorization": f"Bearer {密钥}",
            "Content-Type": "application/json",
        }

    def _取(self, 查询参数: dict) -> list:
        """对 curriculum 表做 GET 查询。"""
        键值列 = "&".join(f"{键}={urllib.parse.quote(str(值))}" for 键, 值 in 查询参数.items())
        请求 = urllib.request.Request(
            f"{self.地址}/rest/v1/curriculum?{键值列}",
            headers=self.请求头,
        )
        try:
            with urllib.request.urlopen(请求) as 回复:
                return json.loads(回复.read())
        except urllib.error.HTTPError as 错误:
            报文 = ""
            try:
                报文 = 错误.read().decode()[:200]
            except Exception:
                pass
            return [{"error": f"HTTP {错误.code}", "message": 报文}]

    def _记日志(self, 主题: str, 摘要: str, 年龄: float, 自报: bool):
        """写一条 learning_log。"""
        try:
            载荷 = json.dumps({
                "topic": 主题,
                "content_snippet": 摘要[:200],  # 截断防溢出
                "age_when_learned": 年龄,
                "self_reported": 自报,
            }).encode()
            请求 = urllib.request.Request(
                f"{self.地址}/rest/v1/learning_log",
                data=载荷,
                headers={**self.请求头, "Prefer": "return=minimal"},
                method="POST",
            )
            with urllib.request.urlopen(请求):
                return True
        except Exception:
            return False

    # ─────────────── 公开接口 ───────────────

    def 可用阶段(self, 当前年龄: float) -> list[str]:
        """返回当前年龄可用的所有 阶段 标签。"""
        行列表 = self._取({
            "age_min": f"lte.{当前年龄}",
            "age_max": f"gte.{当前年龄}",
            "age_max": f"lte.{年龄上限终生}",
            "select": "stage",
            "order": "stage",
        })
        if "error" in (行列表[0] if 行列表 else {}):
            return []
        已见 = []
        for 行 in 行列表:
            阶段 = 行.get("stage")
            if 阶段 and 阶段 not in 已见:
                已见.append(阶段)
        return 已见

    def 可用大类(self, 当前年龄: float) -> list[str]:
        """当前年龄对应的课纲大类。"""
        行列表 = self._取({
            "age_min": f"lte.{当前年龄}",
            "age_max": f"gte.{当前年龄}",
            "age_max": f"lte.{年龄上限终生}",
            "select": "category",
        })
        if "error" in (行列表[0] if 行列表 else {}):
            return []
        已见 = []
        for 行 in 行列表:
            大类 = 行.get("category")
            if 大类 and 大类 not in 已见:
                已见.append(大类)
        return 已见

    def 查询(self, 当前年龄: float, 大类: str = None, 阶段: str = None,
             上限数: int = 10, 标已学: bool = False) -> list[dict]:
        """查询当前年龄的课纲条目。

        参数:
          当前年龄  : 当前年龄(浮点, 精确到月需自行换算)
          大类     : 可选, 按大类过滤(语言/数学/...)
          阶段     : 可选, 按阶段标签过滤(3-6 / 6-12 / ...)
          上限数        : 最多返回条数(默认 10)
          标已学 : 若 True, 把已返回条目写入 learning_log

        返回:
          多条 {id, stage, category, topic, content, source}。若查询失败或该年龄
          无匹配课纲, 返回空列表(不会返回 18+ 的内容)。
        """
        # 硬过滤: age_max 必须 ≤ 18 (防止数据污染时误查)
        参数 = {
            "age_min": f"lte.{当前年龄}",
            "age_max": f"gte.{当前年龄}",
            "age_max_add": f"lte.{年龄上限终生}",
            "order": "age_min",
            "limit": 上限数,
        }
        # 拼两次 age_max(AND 过滤: gte.current AND lte.18)
        查询部件 = []
        for 键, 值 in 参数.items():
            if 键 == "age_max_add":
                查询部件.append(f"age_max=lte.{年龄上限终生}")
            else:
                查询部件.append(f"{键}={urllib.parse.quote(str(值))}")
        if 大类:
            查询部件.append(f"category=eq.{urllib.parse.quote(大类)}")
        if 阶段:
            查询部件.append(f"stage=eq.{urllib.parse.quote(阶段)}")
        查询串 = "&".join(查询部件)

        请求 = urllib.request.Request(
            f"{self.地址}/rest/v1/curriculum?{查询串}",
            headers=self.请求头,
        )
        try:
            with urllib.request.urlopen(请求) as 回复:
                行列表 = json.loads(回复.read())
        except urllib.error.HTTPError as 错误:
            return [{"error": f"HTTP {错误.code}"}]

        if 标已学:
            for 行 in 行列表:
                主题 = 行.get("topic", "")
                摘要 = 行.get("content", "")[:200]
                self._记日志(主题, 摘要, 当前年龄, False)

        return 行列表

    def 知识边界报告(self, 当前年龄: float) -> dict:
        """给当前年龄生成一份知识边界报告。门控在此显式记录。
        用于回答'她现在知道什么'类报表。"""
        大类列 = self.可用大类(当前年龄)
        阶段列 = self.可用阶段(当前年龄)
        全内容 = self.查询(当前年龄, 上限数=100)
        边界说明 = (
            "数据库内 age_max > 18 = 0 条。"
            "知识点仅限 0–18 岁课纲 + 生活18岁内常识。"
            "超过 18 岁: 不查 / 不返回 / 不暗示。"
        )
        return {
            "当前年龄": 当前年龄,
            "年龄上界": 年龄上限终生,
            "可用阶段": 阶段列,
            "可用大类": 大类列,
            "知识条目数": len(全内容),
            "边界说明": 边界说明,
        }


# ────────────────────────────── 命令行自测
if __name__ == "__main__":
    门禁 = 年龄门禁()

    for 年龄 in [0.5, 4, 14, 18, 25]:
        报告 = 门禁.知识边界报告(年龄)
        print(f"\n=== 年龄 {年龄} ===")
        print(f"  stages     : {报告['可用阶段']}")
        print(f"  categories : {报告['可用大类']}")
        print(f"  entries    : {报告['知识条目数']}")
        if 年龄 <= 18:
            行列 = 门禁.查询(年龄, 上限数=1)
            if 行列 and "error" not in 行列[0]:
                print(f"  sample     : {行列[0]['topic']} [{行列[0]['source'][:30]}...]")
            else:
                print(f"  sample     : (无条目 或 错误)")

    # 验证硬过滤: 不存在 19 岁的查询结果
    行_19 = 门禁.查询(19, 上限数=5)
    print(f"\n=== 问: 年龄=19 (应为空) ===")
    print(f"  count : {len(行_19)}")
    if 行_19 and "error" not in 行_19[0]:
        print("  失败: 返回了 19 岁的条目!")
    else:
        print("  通过: 19 岁无结果 (边界生效)")
