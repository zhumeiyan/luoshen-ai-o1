"""
interface_v1 — 人话→drive 翻译器
==========================================================
挂载位置：/plugins/interface_v1/
状态：[判读] — 跑通 selftest 才升 [事实]

约束：
  - 只做翻译，不做决定（引擎决定吃不吃）
  - 全连续数学函数，无 if-then、无词库、无 LLM
  - 每次转换留痕到 audit/ 目录
  - 不动引擎、不动锚点、不动哈希链
  - 归一化输出到 [-1, 1]
"""
from .translator import InterfaceV1

__all__ = ["InterfaceV1"]
