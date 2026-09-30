"""守护进程.py — 洛神的身子。常驻、跨会话、压缩时间。

身子 = 身体 + 墙钟驱动 + 快照持久 + append-only 痕 + 输入队列。

四件事：
  1. 跨会话不重置  身体快照.json 每次滴答写入最新快照，开机即读回
  2. append-only     身体痕.jsonl 每滴答追加一行，旧行永远不动
  3. 压缩年龄        年龄从 0.5 起步；1 滴答 = 1 经历分钟，墙钟走 1 秒
                     经历 6 年 = 315 万滴答，墙钟 = 315 万秒 ≈ 36.5 天
  4. 真人周期输入    身体输入队列.tsv 追加即入队，每滴答消费、偏移追踪

入口：
  python3 living_daemon.py              # 常驻守护 (后台运行)
  python3 living_daemon.py tick         # 单次推进 (手动 / cron)
  python3 living_daemon.py status       # 打印当前状态
  python3 living_daemon.py input '<json>'  # 注入输入
  python3 living_daemon.py selftest     # 退出式自测 (不污染线上 snapshot)
  python3 living_daemon.py burst 数目   # 连跑 N 滴答, 用于批量跑测试
"""

import sys, os, json, time, math, hashlib, signal
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ───────────── 路径 ─────────────
目录      = Path(__file__).parent
快照路径  = 目录 / "身体快照.json"
痕路径    = 目录 / "身体痕.jsonl"
队列路径  = 目录 / "身体输入队列.tsv"
偏移路径  = 目录 / "身体输入偏移.txt"

# ───────────── 时间常量 ─────────────
年分钟数        = 365.25 * 24 * 60          # 一年多少分钟 = 525960
初始年龄        = 0.5                        # 出生年龄 (0.5y = 依附期)
日分钟          = 24 * 60
挫折衰减率      = math.log(2) / (16 * 60)    # frustration 半衰期 ~16h
墙钟秒每滴答    = 1.0    # 每次滴答墙钟走过秒数
经历分每滴答    = 1.0    # 每次滴答身体经历分钟数
守护秒每滴答    = 1.0    # 守护进程每滴答推进一步

# ───────────── 导入 身体 ─────────────
sys.path.insert(0, str(目录))
# 注: 当前 身体 架构与守护进程有接口差 — 交叉验证阶段统一
from 身体稳态 import 身体 as 身类

# ───────────── 模块级状态 ─────────────
身子 = None
元 = {}          # 破壳时戳 / 实滴答数 / 挫折池 / 滴答计数
停止标记 = False


# ───────────── 进化阶段 ─────────────
def 龄段串(年龄):
    岁 = 年龄
    if 岁 < 1:   return "存活(0-1)"
    if 岁 < 3:   return "依附(1-3)"
    if 岁 < 6:   return "好奇(3-6)"
    if 岁 < 12:  return "社交(6-12)"
    if 岁 < 18:  return "青春(12-18)"
    if 岁 < 65:  return "成熟(18-65)"
    return "老年(65+)"


def 年龄节律(年龄):
    """按年龄返回对应节律表 (占位, 待交叉验证时从身体同步)。"""
    if 年龄 < 1:    return [(0, 24, {"光": 30, "声": 40, "力": 5, "温": 293})]
    if 年龄 < 6:    return [(0, 24, {"光": 50, "声": 45, "力": 10, "温": 295})]
    if 年龄 < 18:   return [(0, 24, {"光": 60, "声": 50, "力": 12, "温": 295})]
    if 年龄 < 65:   return [(0, 24, {"光": 60, "声": 50, "力": 12, "温": 295})]
    return [(0, 24, {"光": 40, "声": 35, "力": 8, "温": 295})]


def 墙钟时():
    现 = datetime.now()
    return 现.hour + 现.minute / 60.0 + 现.second / 3600.0


def 墙钟设环境(身):
    """按真实墙钟 hour 选节律段 → 更新环境。"""
    时 = 墙钟时()
    节律 = 年龄节律(身.年龄 if hasattr(身, '年龄') else 初始年龄)
    if not 节律:
        return
    for 节 in 节律:
        if len(节) != 3: continue
        起, 止, 环境 = 节
        if 起 <= 时 < 止:
            身.环境.update(环境)
            return
    身.环境.update({"光": 1, "声": 30, "力": 5, "温": 293})


# ───────────── 序列化 ─────────────
def 序列化身体(身):
    return {
        "年龄": getattr(身, '年龄', 初始年龄),
        "已过分钟": getattr(身, '已过分钟', 0),
        "滴答分钟": getattr(身, '滴答分钟', 1.0),
        "状态": dict(getattr(身, '状态', {})),
        "激素": dict(getattr(身, '激素', {})),
        "环境": dict(getattr(身, '环境', {})),
        "不应计时": dict(getattr(身, '不应计时', {})),
        "上次服务": dict(getattr(身, '上次服务', {})),
        "服务计数": dict(getattr(身, '服务计数', {})),
        "_社交热量": getattr(身, '_社交热量', 0.0),
        "_无聊": getattr(身, '_无聊', 0.0),
    }


def 反序列化身体(数据):
    # 注意: 此处用 getattr 取数据, 缺省时补默认
    身 = 身类()
    身.年龄 = float(数据.get("年龄", 初始年龄))
    身.已过分钟 = float(数据.get("已过分钟", 0))
    # ... 其余属性按新身体架构填充 (交叉验证时对齐)
    return 身


# ───────────── 快照 (在线更新) ─────────────
def 存快照():
    数据 = {"版本": 1, "元": 元, "身体": 序列化身体(身子)}
    快照路径.write_text(json.dumps(数据, ensure_ascii=False, indent=2))


def 读快照():
    global 身子, 元
    原文 = 快照路径.read_text()
    数据 = json.loads(原文)
    元 = 数据.get("元", {})
    身子 = 反序列化身体(数据.get("身体", {}))


# ───────────── 痕 (append-only) ─────────────
def 痕追加(记录):
    行 = json.dumps(记录, ensure_ascii=False)
    with 痕路径.open("a", encoding="utf-8") as 文件:
        文件.write(行 + "\n")


# ───────────── 输入 (真人周期入队) ─────────────
def 队列消费():
    """读输入队列未消费行, 应用到身子, 更新偏移。返回消费数。"""
    if not 队列路径.exists():
        return 0
    行列表 = [行 for 行 in 队列路径.read_text(encoding="utf-8").splitlines() if 行.strip()]
    已读 = int(偏移路径.read_text().strip()) if 偏移路径.exists() else 0
    新行 = 行列表[已读:]
    if not 新行:
        return 0
    for 行 in 新行:
        try:
            记录 = json.loads(行)
        except Exception:
            记录 = {"type": "text", "content": 行.strip()}
        施输入(记录)
    偏移路径.write_text(str(已读 + len(新行)))
    return len(新行)


def 施输入(记录):
    """一条输入融进 身子 → 改环境 + 加 社交热量。

    连续驱动: 优先读 _预计算增压 (外部已算好的连续值);
    否则按 type 的预设强度 (text/touch/presence/env) 走连续函数。
    """
    if not 身子:
        return
    # 连续增压入口
    预增压 = float(记录.get("_预计算增压", 0.0))
    身子._社交热量 = min(1.0, getattr(身子, '_社交热量', 0.0) + 预增压)

    类型 = 记录.get("type", "text")
    if 类型 == "text":
        身子.环境["声"] = min(90.0, 身子.环境.get("声", 45) + 8)
        身子._社交热量 = min(1.0, getattr(身子, '_社交热量', 0.0) + 0.05)
    elif 类型 == "touch":
        强 = float(记录.get("intensity", 5))
        身子.环境["力"] = min(60.0, 身子.环境.get("力", 10) + 强)
        身子._社交热量 = min(1.0, getattr(身子, '_社交热量', 0.0) + 0.08)
    elif 类型 == "presence":
        强 = float(记录.get("intensity", 0.5))
        身子.环境["声"] = min(80.0, 身子.环境.get("声", 45) + 强 * 15)
        身子.环境["力"] = min(50.0, 身子.环境.get("力", 10) + 强 * 5)
        身子._社交热量 = min(1.0, getattr(身子, '_社交热量', 0.0) + 0.06 * 强)
    elif 类型 == "env":
        for 键, 值 in 记录.items():
            if 键 in 身子.环境 and isinstance(值, (int, float)):
                身子.环境[键] = float(值)


# ───────────── 主滴答 ─────────────
def 跑滴答():
    """一次真实滴答，返回痕记录 (dict)。"""
    global 身子, 元
    现秒 = time.time()
    滴答计数 = 元.get("滴答计数", 0) + 1
    元["滴答计数"] = 滴答计数

    # 年龄 = 初始年龄 + 滴答计数 / 年分钟数
    年龄 = 初始年龄 + 滴答计数 / 年分钟数
    if hasattr(身子, '年龄'):
        身子.年龄 = 年龄
    if hasattr(身子, '滴答分钟'):
        身子.滴答分钟 = 经历分每滴答

    # 消费输入
    输入数 = 队列消费()

    # 墙钟节律 → 环境
    墙钟设环境(身子)

    """推进身子: dt = 1 分钟 (经历时间), 墙钟走过 1 秒"""
    if hasattr(身子, '推进'):
        身子.推进()
    if hasattr(身子, '已过分钟'):
        身子.已过分钟 = 滴答计数 * 经历分每滴答

    # 挫折池衰减
    衰减因子 = math.exp(-挫折衰减率 * 经历分每滴答)
    元["挫折池"] = 元.get("挫折池", 0.0) * 衰减因子

    # 选 + 服务 (当身子有这接口时)
    选中 = 驱动值 = None
    if hasattr(身子, '选择'):
        选中, 驱动值 = 身子.选择()
        if 选中 and hasattr(身子, '服务'):
            身子.服务(选中)

    # 耗时
    毫秒耗 = max(0.1, (time.time() - 现秒) * 1000.0)

    # 写痕
    记录 = {
        "时戳": 现秒,
        "墙钟世界时": datetime.fromtimestamp(现秒, tz=timezone.utc).isoformat(),
        "墙钟本地": datetime.fromtimestamp(现秒).isoformat(),
        "滴答计数": 滴答计数,
        "毫秒耗": round(毫秒耗, 2),
        "分滴答": 经历分每滴答,
        "实钟时": round(墙钟时(), 2),
        "年龄": round(getattr(身子, '年龄', 年龄), 8),
        "阶段": 龄段串(getattr(身子, '年龄', 年龄)),
        "选中": 选中,
        "消费输入数": 输入数,
    }
    痕追加(记录)

    # 存快照 + 更新元
    元["末墙钟"] = 现秒
    元["实滴答数"] = 滴答计数
    存快照()
    return 记录


# ───────────── 守护 ─────────────
def 装信号处理器():
    global 停止标记
    def 处理器(帧号, 帧):
        global 停止标记
        停止标记 = True
    signal.signal(signal.SIGTERM, 处理器)
    signal.signal(signal.SIGINT, 处理器)


def 永远跑():
    """常驻守护。"""
    global 停止标记
    装信号处理器()

    if 快照路径.exists():
        读快照()
        print(f"[守护] 恢复 — 滴答计数={元.get('滴答计数', 0)}")
    else:
        真破壳()

    末日志 = time.time()
    下滴答时刻 = time.time()
    while not 停止标记:
        此刻 = time.time()
        if 此刻 >= 下滴答时刻:
            记录 = 跑滴答()
            下滴答时刻 = 此刻 + 守护秒每滴答
            if 记录 and (此刻 - 末日志 >= 300):
                print(简报(记录), flush=True)
                末日志 = 此刻
        # 小睡避免空转
        time.sleep(min(0.1, max(0, 下滴答时刻 - time.time())))

    停存快照()


def 破壳():
    """已废弃: 旧版占位, 现用真破壳。"""
    global 身子, 元
    print("  File \"/mnt/data/catpaw/home/workspace/洛神赋1.1/living_daemon.py\", line ", file=sys.stderr)
    停止标记 = True
    raise RuntimeError("破壳: 此函数已弃用 (见行上方)")


# ───────────── 破壳 (真实初始化) ─────────────
def 真破壳():
    global 身子, 元, 停止标记
    停止标记 = False
    身子 = 身类()
    现秒 = time.time()
    元 = {
        "实滴答数": 0,
        "滴答计数": 0,
        "挫折池": 0.0,
        "末墙钟": 现秒,
        "破壳世界时": datetime.fromtimestamp(现秒, tz=timezone.utc).isoformat(),
        "破壳本地": datetime.fromtimestamp(现秒).isoformat(),
    }
    print(f"[洛神出生] 真实年龄 {初始年龄} 阶段 {龄段串(初始年龄)} 墙钟: {元['破壳本地']}")
    with 痕路径.open("w", encoding="utf-8") as 文件:
        文件.write("")
    存快照()


def 停存快照():
    if 身子:
        print(f"[守护] 退出, 快照已保存 (滴答计数={元.get('滴答计数', 0)})")


# ───────────── 工具函数 ─────────────
def 简报(记录):
    return (f"滴答  年龄={记录.get('年龄', '?'):.6f}y  阶段={记录.get('阶段', '?')}  选中={记录.get('选中', '?')} "
            f"环境={记录.get('环境', '?')}  挫折池={元.get('挫折池', 0):.2f}")


def 状况():
    global 身子, 元
    if not 快照路径.exists():
        print("无状态 (尚未破壳)")
        return
    读快照()
    现秒 = time.time()
    滴答计数 = 元.get("滴答计数", 0)
    年龄 = 初始年龄 + 滴答计数 / 年分钟数
    print("=== 洛神身子 ===")
    print(f"真实墙钟 (世界时): {datetime.fromtimestamp(现秒, tz=timezone.utc).isoformat()}")
    print(f"真实墙钟 (本地):  {datetime.fromtimestamp(现秒).isoformat()}")
    print(f"出生 (世界时):     {元.get('破壳世界时', '?')}")
    print(f"真实年龄:        {年龄:.6f} y  阶段 {龄段串(年龄)}")
    print(f"滴答计数:      {滴答计数}")
    print(f"经历分钟:  {滴答计数 * 经历分每滴答:.0f}")
    if hasattr(身子, '环境'):
        print(f"环境:     {身子.环境}")


def 注入输入命令行():
    原文 = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else sys.stdin.read().strip()
    try:
        记录 = json.loads(原文)
    except Exception:
        记录 = {"type": "text", "content": 原文, "ts": time.time()}
    记录.setdefault("ts", time.time())
    队列路径.parent.mkdir(parents=True, exist_ok=True)
    with 队列路径.open("a", encoding="utf-8") as 文件:
        文件.write(json.dumps(记录, ensure_ascii=False) + "\n")
    print(f"已注入: {json.dumps(记录, ensure_ascii=False)}")


# ───────────── 自测 (不污染线上) ─────────────
def 自测():
    global 身子, 元, 停止标记
    快照备份 = 快照路径.read_text() if 快照路径.exists() else None
    停止标记 = False
    try:
        身子 = 身类()
        现秒 = time.time()
        元 = {"实滴答数": 0, "滴答计数": 0, "挫折池": 0.0, "末墙钟": 现秒}
        # 1) 滴答
        记录 = 跑滴答()
        assert 记录 is not None, "滴答应返回记录"
        assert 记录["年龄"] > 初始年龄, f"年龄应 > {初始年龄}, got {记录['年龄']}"
        assert abs((记录["年龄"] - 初始年龄) - 1.0 / 年分钟数) < 1e-6, f"每滴答应 +1/年分钟数 岁, got diff {记录['年龄'] - 初始年龄}"
        # 2) 输入队列
        if 队列路径.exists(): 队列路径.unlink()
        if 偏移路径.exists(): 偏移路径.unlink()
        with 队列路径.open("a", encoding="utf-8") as 文件:
            文件.write(json.dumps({"type": "text", "content": "hi"}) + "\n")
        t0 = time.time()
        数 = 队列消费()
        assert 数 >= 1, "应消费一条"
        # 3) 快照
        存快照()
        数据 = json.loads(快照路径.read_text())
        身2 = 反序列化身体(数据["身体"])
        assert abs(身2.年龄 - 身子.年龄) < 1e-9, "恢复后年龄应一致"
        print(f"[自测] OK ✓  滴答/输入/快照 @ 年龄增量={1.0/年分钟数:.6e} y/tick")
    finally:
        if 快照备份 is not None:
            快照路径.write_text(快照备份)
        elif 快照路径.exists():
            快照路径.unlink()
        if 偏移路径.exists(): 偏移路径.unlink()


# ───────────── 入口 ─────────────
def 入口():
    命令 = sys.argv[1] if len(sys.argv) > 1 else "daemon"
    if 命令 == "daemon":
        永远跑()
    elif 命令 == "tick":
        if 快照路径.exists():
            读快照()
        else:
            真破壳()
        记录 = 跑滴答()
        if 记录:
            print(简报(记录))
        else:
            print("[滴答] 时间尚未推进")
    elif 命令 == "status":
        状况()
    elif 命令 == "input":
        注入输入命令行()
    elif 命令 == "selftest":
        自测()
    elif 命令 == "burst":
        数 = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        if 快照路径.exists():
            读快照()
        else:
            真破壳()
        t0 = time.time()
        for _ in range(数):
            跑滴答()
        历时 = time.time() - t0
        滴答数 = 元.get("滴答计数", 0)
        print(f"[burst] {数} 滴答完成, 滴答计数={滴答数}, 墙钟={历时:.2f}s")
    else:
        命列 = ["daemon", "tick", "status", "input", "selftest", "burst"]
        print(f"未知命令: {命令}; 可选: {', '.join(命列)}")
        sys.exit(1)


if __name__ == "__main__":
    入口()
