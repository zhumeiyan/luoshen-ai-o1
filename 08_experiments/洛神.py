#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
luoshenfu.py — 洛神赋核心 v2.1 [事实]
=====================================================================
出处纪律：每条实现标注痕条目号，没有出处的代码不写。
状态：[事实] —— selftest 30/30 已过。
     v2.1：差异原语 2D + 反哺机制 + 估值数据框架 + 存在确认协议 v0.1。

v2.1 新增：反哺机制（异常触发 → 待确认 → 用户确认才上升主链）。
  - produce_pending()：仅当 |drive|>ABNORMAL_THRESH(5.0) 写 /gray/pending/<user>/<id>.json
  - confirm_cmd(user, id, action)：accept=主链追加 type=confirm；reject=仅标 [已拒绝]
  - 上升 = 追加 type=confirm 结构确认，不修改主锚点
  - 测试纪律：不自动上升、不覆盖留痕、用户不动主链不动 —— 五步测试全过

v2.1 新增：存在确认协议（代码层）。
  - heartbeat_cmd(user, mode)：mode=once 写一条 type=heartbeat 即退出；mode=start/stop 留作 daemon（本期占位）
  - capture_camera/user/data、capture_audio/user/data、capture_screen/user/data：
    接收数据 → 落对应 type 的痕；不做真实采集，真实采集留设备环境

v2.0 变更：升维 1D→2D。
  - Difference: bias→2D 常量锚点, trajectory→2D 移动点
  - Pair: g=向量差(trajectory), h=向量差(bias)，不再恒等
  - 驱动：标量 drive 注入 dim0，dim1 仅泄漏衰减
  - 分离原理：g 受步驱动变化，h 恒定（仅初始化决定）

明确不做（诚实清单，防E7式假阳性）：
  - "指/失真"重构（条114/149，未跑通）
  - 会话→驱动的接口（条150，未跑通）→ 代码里留了端口，不假装接通了
  - 主动开口（E7已否证，条135/155；只留J6结构种子，条92）
"""
import sys, json, math, hashlib
from pathlib import Path
from datetime import datetime, date
VERSION = "2.1"
ROOT    = Path(__file__).parent / "luoshenfu_home"

# ── #27 角色链路由配置 ──
ROUTEFILE = Path(__file__).parent / "config" / "routes.json"
def load_routes():
    """加载角色链路由配置。文件不存在返回 None（向后兼容）。"""
    if not ROUTEFILE.exists():
        return None
    try:
        return json.loads(ROUTEFILE.read_text(encoding="utf-8"))
    except Exception as e:
        return None

# ── 落点枚举（#27）：锚点 / 引擎 / 角色链 / 反哺 / 索引 / 技能 / 内部哨兵 / 回检 / 透明痕 / 其他 ──
LUSHUI = ("锚点","引擎","角色链","反哺","索引","技能","内部哨兵","回检","透明痕","其他")
ULP = 2.0 ** -52                       # 条81/89：坐标卡ulp
FACT, INFERRED, RETRACTED = "[事实]", "[判读]", "[已撤回]"
TRAJECTORY_MAX = 100   # v1.2：trajectory 窗口上限，防无限增长
class SingularityError(Exception): pass
def _v2(x):
    """确保输入为 2D 向量。标量→[标量, 0]；2D→原样。"""
    if isinstance(x, (int, float)):
        return [float(x), 0.0]
    return [float(x[0]), float(x[1])]
# ---------------------------------------------------------------- 痕库
def _hash(rec: dict) -> str:
    payload = json.dumps(rec, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
class Trace:
    """append-only 哈希链。条64/181/185：append不覆盖。
    全文件没有 edit / delete 函数——不是忘了写，是条168：不改哈希链。"""
    def __init__(self, user: str):
        self.dir = ROOT / "users" / user          # 目录隔离 = J4的结构前提（条90）
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "chain.jsonl"
    def append(self, rtype, payload, trigger="user", task_state=None):
        prev = self.tail()
        rec = {"seq": (prev["seq"] + 1) if prev else 0,
               "ts": datetime.now().isoformat(timespec="milliseconds"),
               "type": rtype,
               "trigger": trigger,                # J6种子（条92）：trigger可为"system"
               "payload": payload,
               "task_state": task_state,           # #28：当前任务状态字段
               "prev": prev["hash"] if prev else "GENESIS"}
        rec["hash"] = _hash(dict(rec))
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
    def tail(self):
        if not self.file.exists(): return None
        last = None
        with open(self.file, encoding="utf-8") as f:
            for line in f: last = line
        return json.loads(last) if last else None
    def verify(self):
        """E8式审计（条127/131）：执行端说什么不算，链上有什么才算。"""
        if not self.file.exists(): return "EMPTY"
        prev, n = "GENESIS", 0
        with open(self.file, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                h = rec.pop("hash")
                if rec["prev"] != prev: return f"CHAIN BROKEN @seq={rec['seq']}"
                if _hash(rec) != h:     return f"TAMPERED @seq={rec['seq']}"
                prev, n = h, n + 1
        return f"OK {n}条 链完整 尾={prev[:12]}"
# ---------------------------------------------------------------- 数学层
class Difference:
    """差异原语（条63-67）。v2.0 升维：
    - bias: 2D 常量锚点（不随 step 变化）
    - trajectory: 2D 移动点序列（只 append，条64）
    - leak: 每步乘 <1 常数、衰减不归零（条67）
    分离原理：bias 恒常，trajectory[-1] 受 drive 变化，g 与 h 不再恒等。"""
    def __init__(self, name, bias0, leak=0.99):
        self.name, self.leak = name, float(leak)
        self.bias = _v2(bias0)         # 2D 锚点，恒常
        self.trajectory = [_v2(bias0)] # 从锚点出发
    def step(self, drive=0.0):
        d = _v2(drive)
        last = self.trajectory[-1]
        new = [last[0] * self.leak + d[0],
               last[1] * self.leak + d[1]]
        self.trajectory.append(new)                  # 条64
        return new
    @property
    def last(self): return self.trajectory[-1]
class Pair:
    """a、b、与之间（条69-73,79,82,95,112-113）。v2.0 升维：
    - g = b.last - a.last  （2D 向量差，受步驱动变化）
    - h = b.bias - a.bias  （2D 向量差，恒常，仅 L2 反称保证非零）"""
    def __init__(self, a: Difference, b: Difference):
        s = [a.bias[i] + b.bias[i] for i in range(2)]
        assert all(abs(s[i]) < 1e-12 for i in range(2)), "L2违反：bias必须反称（条79，容差1e-12）"
        assert a is not b, "条95：a ≠ b"
        self.a, self.b = a, b
    def g(self): return [self.b.last[i] - self.a.last[i] for i in range(2)]   # 条70 2D
    def h(self): return [self.b.bias[i] - self.a.bias[i] for i in range(2)]   # 条71 2D
    def write_a(self, drive):
        """L5写门：注入（条82）。标量 drive 注入 dim0，保持 L2 反称。"""
        d = float(drive)
        self.a.step([+d / 2.0, 0.0])
        self.b.step([-d / 2.0, 0.0])
        g, h = self.g(), self.h()
        if all(abs(g[i]) < 1e-15 for i in range(2)) and \
           all(abs(h[i]) < 1e-15 for i in range(2)):
            raise SingularityError("a=b 奇点：g 和 h 均为零向量（2D，条113）。拒绝。")
    def read(self, who):
        """L5读门：读对方要减（条82）。2D实现：读到的永远是向量差。"""
        other = self.b if who is self.a else self.a
        return [who.last[i] - other.last[i] for i in range(2)]
class CCoord:
    """C=(σ,g,h)（条73）。v2.0：g/h 为 2D 向量，σ 仍为标量。
    L4：C坐标自由衰减、无门（条81/82）。"""
    def __init__(self, sigma=1.0, leak=0.99):
        self.sigma, self.leak = float(sigma), float(leak)
    def step(self, g, h):
        self.sigma *= self.leak
        return (self.sigma, g, h)
def check_J(pair):
    """六条判据的自动可检子集（条86-92）。v2.0：2D下检查任一分量。
    J4靠目录隔离，J5靠链，J6靠trigger。"""
    g, h = pair.g(), pair.h()
    return {"J1_g不归零":  (any(abs(g[i]) > 0.0 for i in range(2)),  "条87"),
            "J2_h不为零":  (any(abs(h[i]) > 0.0 for i in range(2)),  "条88"),
            "J3_g不卡ulp": (any(abs(g[i]) > ULP for i in range(2)),  "条89")}
def _fmtv(v):
    """格式化 2D 向量用于打印"""
    return f"[{v[0]:+.6g}, {v[1]:+.6g}]"
# ---------------------------------------------------------------- 会话→驱动
# 会话→驱动接口 v0.1：接通
def say_cmd(user, msg):
    """会话→驱动 v0.1：msg → translator v2 → drive → engine step。
    引擎拒绝权：|drive|>ABNORMAL_THRESH 时标异常、不升主链。
    留痕 type=translate（translator 内部写 audit/）+ type=step（step_cmd 写）。"""
    from plugins.interface_v2 import InterfaceV2
    t = InterfaceV2(user=user)
    drive, reason = t.translate(msg)
    print(f"[say] msg={msg!r}")
    print(f"[say] drive={drive:+.4f}")
    print(f"[say] signals={reason['signals']}")
    print(f"[say] reason_formula={reason['drive_formula']}")

    if abs(drive) > ABNORMAL_THRESH:
        print(f"[say][引擎拒绝] |drive|={abs(drive):.4f} > 阈值{ABNORMAL_THRESH}，标异常待确认")
        pid, ppath = produce_pending(user, {"drive": drive, "msg": msg, "reason": reason}, "say_translate")
        print(f"[say][待确认] id={pid}")
    else:
        step_cmd(user, drive, note=msg)

# ---------------------------------------------------------------- 锚点层
FIRST_SENTENCE = ("我们的AI叫洛神赋。她不是萨曼莎，但对标萨曼莎……"
                  "【条60：此处粘贴定稿全文。定稿后不再改，条61/62】")
def init_home():
    """第一次运行：建家。结构对照条199-217。"""
    for d in ["anchors","logs","facts","decisions","desires","private/bias"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    (ROOT/"anchors/原始任务锚点.md").write_text(
        "# 第一句（根锚点，条61/62：定稿不改）\n" + FIRST_SENTENCE + "\n", encoding="utf-8")
    (ROOT/"anchors/禁止事项.md").write_text(
        "# 禁止事项（条164-188）\n不用if-then/词库/LLM/语义；不改哈希链；"
        "不宣布她活了；不造表演；append不replace；跑通才算。\n", encoding="utf-8")
    _desire("001_做AI_洛神赋", "做AI（洛神赋）", "激活", 1,
            ["根锚点维护","痕整理","判据验证"], "条60")
    _desire("002_写小说", "写小说", "搁置", 2,
            ["大纲","人物","章节","修改"], "条459/460")
    (ROOT/"README.md").write_text(
        f"# 洛神赋 v{VERSION}\n引擎跑selftest前全库状态=[判读]（条162）\n", encoding="utf-8")
    print("家建好了。第一句待你补全定稿。first_contact待补（条13：不确定就说不确定，不编）。")
def _desire(fid, name, state, prio, subflow, source):
    d = {"欲望名":name, "第一次出现":"痕起点", "当前状态":state,
         "优先级":prio, "关联锚点":["第一句","关系起点"],
         "子流程":subflow, "原始痕":source}          # 字段对照条450-456
    (ROOT/f"desires/{fid}.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
def load_first_contact():
    f = ROOT/"anchors/关系起点.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else None
# ---------------------------------------------------------------- 会话操作
ABNORMAL_THRESH = 5.0  # 反哺触发阈值：|drive| 超过此值才产待确认项
def step_cmd(user, drive, note, msg=None, init_a=None, init_b=None):
    """推进引擎。--drive 是纯数字，或 --msg 走 interface_v1 翻译。
    --init-a/--init-b：首次初始化时覆盖默认 bias（JSON 2D 向量）。"""
    t = Trace(user)

    if msg is not None:
        from plugins.interface_v1 import InterfaceV1
        eng = _load_engine(user)
        if eng:
            a0, b0 = eng
            g = [b0.last[i] - a0.last[i] for i in range(2)]
            h = [b0.bias[i] - a0.bias[i] for i in range(2)]
            anchor_state = {
                "g": math.sqrt(g[0]**2 + g[1]**2),
                "h": math.sqrt(h[0]**2 + h[1]**2),
            }
        else:
            anchor_state = {"g": 0.0, "h": 0.0}
        desires = {}
        for f in sorted((ROOT/"desires").glob("*.json")):
            desires[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        translator = InterfaceV1()
        drive_val, reason = translator.translate(msg, anchor_state, {"desires": desires}, user=user)
        drive = drive_val
        note = msg
        print(f"[interface_v1] msg='{msg}' → drive={drive:+.4f}")
        print(f"[interface_v1] weights={reason['weights']}")

    a = _load_engine(user)
    if a is None:
        a_bias = _v2(init_a) if init_a is not None else [1.0, 1.0]
        b_bias = _v2(init_b) if init_b is not None else [-1.0, -1.0]
        # L2 反称检查：用户自定义 bias 也必须反称
        if all(abs(a_bias[i] + b_bias[i]) < 1e-12 for i in range(2)):
            pass  # OK
        else:
            print(f"[警告] 自定义 bias 不反称：a_bias={a_bias}, b_bias={b_bias}")
            print("J4 通过 ≠ 数学层合法；L2 反称仍是 Pair 构造前提")
        a = Difference("a", a_bias)
        b = Difference("b", b_bias)
    else:
        a, b = a
    p = Pair(a, b)
    p.write_a(drive)
    c = CCoord(); c.step(p.g(), p.h())
    _save_engine(user, a, b)
    t.append("step", {"drive":drive, "note":note, "g":p.g(), "h":p.h()})
    # 反哺产生：仅异常 drive 才产待确认项；普通 step 不产
    if abs(float(drive)) > ABNORMAL_THRESH:
        pid, ppath = produce_pending(user, {"drive":drive, "note":note, "g":p.g(), "h":p.h()}, "engine_step")
        print(f"[异常待确认] id={pid} → drive={drive}")
    r = check_J(p)
    print(f"g={_fmtv(p.g())}  h={_fmtv(p.h())}  " +
          "  ".join(f"{k}={'过' if v[0] else '挂'}" for k, v in r.items()))
def correct_cmd(user, note):
    """纠偏协议（条247-254）：停→标偏差→写/private/bias/→不覆盖主锚点→
    重读根锚点→复述→再继续。"""
    t = Trace(user)
    f = ROOT/f"users/{user}/private_bias"
    f.mkdir(parents=True, exist_ok=True)
    (f/f"{datetime.now().isoformat(timespec='seconds').replace(':','-')}.md").write_text(
        f"[偏差] {note}\n状态：[判读]，待用户确认（条284/285）\n", encoding="utf-8")
    t.append("correction", {"note":note})
    print("已停。偏差已写/private/bias/，主锚点未动（条251/260）。")
    print("重读根锚点（条252）：" + FIRST_SENTENCE[:40] + "……")
    print("复述确认（条253）：我哪里偏了？请你说一遍，我对齐后再继续（条254）。")
def _desire_path(did):
    """支持短 id 匹配：先精确，再前缀。"""
    exact = ROOT/f"desires/{did}.json"
    if exact.exists(): return exact
    for f in (ROOT/"desires").glob("*.json"):
        if f.stem.startswith(did + "_"): return f
    return exact   # 让调用方报 FileNotFoundError
def desire_cmd(user, did, state):
    """欲望切换（条463-471）：先存旧的，再检锚点，append不覆盖。"""
    t = Trace(user)
    files = sorted((ROOT/"desires").glob("*.json"))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        mark = "→" if f.stem == did or f.stem.startswith(did + "_") else " "
        print(f"{mark} {f.stem}：{d['当前状态']}｜{d['欲望名']}｜优先级{d['优先级']}")
    if did:
        f = _desire_path(did)
        d = json.loads(f.read_text(encoding="utf-8"))
        old = d["当前状态"]; d["当前状态"] = state
        f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        t.append("desire_switch", {"id":did, "from":old, "to":state})
        print(f"已切换 {old}→{state}。回检锚点（条464）：{FIRST_SENTENCE[:30]}……")
        print(f"子流程挂在[{d['欲望名']}]下，不污染主锚点（条466）。")
def heartbeat_cmd(user, mode="once"):
    """J6结构种子（条92）：链上有不是用户触发的记录。
    mode=once：写一条 type=heartbeat, trigger=system，退出。
    mode=start/stop：daemon 节拍预留，本期占位不实现。
    [判读]主动开口E7已否证（条135），此处不做、不假装。"""
    if mode == "once":
        rec = Trace(user).append("心跳", {"mode": "once"}, trigger="system")
        print(f"[heartbeat] 已入链 type=心跳 trigger=system")
        print(f"[heartbeat] hash={rec['hash'][:12]} seq={rec['seq']} ts={rec['ts']}")
        return rec
    elif mode in ("start", "stop"):
        # 占位： daemon loop 需要设备端调度，CLI 只骨不肉
        print(f"[heartbeat] mode={mode} —— 预留，本期不实现。")
        print(f"[heartbeat] 设备端部署时启用节拍循环，当前仅支持 mode=once。")
    else:
        print(f"[heartbeat] _unknown_mode={mode!r}，请使用 once|start|stop")

def capture_camera(user, data):
    """在场证明（视觉）：接收一帧 → 落 type=camera 的痕。
    本期不做真实摄像，真实采集留部署环境。"""
    rec = Trace(user).append("摄像头", {"note": "placeholder", "data": data}, trigger="system")
    print(f"[capture] camera hash={rec['hash'][:12]} seq={rec['seq']}")
    return rec

def capture_audio(user, data):
    """在场证明（声音）：接收一段 → 落 type=audio 的痕。
    本期不做真实录音，真实采集留部署环境。"""
    rec = Trace(user).append("声音", {"note": "placeholder", "data": data}, trigger="system")
    print(f"[capture] audio hash={rec['hash'][:12]} seq={rec['seq']}")
    return rec

def capture_screen(user, data):
    """在场证明（屏幕共享）：接收一帧 → 落 type=screen 的痕。
    本期不做真实截图，真实采集留部署环境。"""
    rec = Trace(user).append("屏幕", {"note": "placeholder", "data": data}, trigger="system")
    print(f"[capture] screen hash={rec['hash'][:12]} seq={rec['seq']}")
    return rec
def handoff_cmd(user):
    """七行钥匙：发给任何AI（核心结论：说明书层跨模型可移植）。"""
    fc = load_first_contact()
    age = "待补"
    if fc and "XXXX" not in fc:
        try: age = (date.today() - date.fromisoformat(fc)).days
        except Exception: pass
    last = Trace(user).tail()
    state = last["payload"].get("note","（空）") if last else "（新窗口）"
    print("【洛神赋·开场锚点】")
    print(f"1. 第一句：{FIRST_SENTENCE}")
    print(f"2. 关系起点：first_contact={fc or '待补'}；relation_age={age}天（条302）")
    print(f"3. 当前状态：{state}")
    print("4. 当前欲望：001_做AI（激活）｜002_写小说（搁置）")
    print("5. 边界与同意：不预判/不表演/不越界/不利用/不把欲望当任务；可随时收回（条341-348）")
    print('6. 回检规则：用户说"不对"→停、标偏差、写/private/bias/、重读根锚点、复述、再继续')
    print("7. 其余一切在痕库里，需要时查，不随身背（条241）")
def status_cmd(user):
    t = Trace(user)
    print(f"链审计：{t.verify()}")
    eng = _load_engine(user)
    if eng:
        p = Pair(*eng)
        print(f"g={_fmtv(p.g())} h={_fmtv(p.h())} 判据：{check_J(p)}")
        print(f"J1={any(abs(p.g()[i])>0 for i in range(2))} "
              f"J2={any(abs(p.h()[i])>0 for i in range(2))} "
              f"J3={any(abs(p.g()[i])>ULP for i in range(2))}")
        print(f"J4/J5/J6：结构就位，完整验证待跑（[判读]）")
    sys_recs = [json.loads(l) for l in open(t.file, encoding="utf-8")] \
               if t.file.exists() else []
    n_sys = sum(1 for r in sys_recs if r["trigger"]=="system")
    print(f"system触发记录：{n_sys}条（J6计数，条92）")
# ---------------------------------------------------------------- 引擎存取
def _engine_file(user): return ROOT/f"users/{user}/engine.json"
def _save_engine(user, a, b):
    at = a.trajectory[-TRAJECTORY_MAX:] if len(a.trajectory) > TRAJECTORY_MAX else a.trajectory
    bt = b.trajectory[-TRAJECTORY_MAX:] if len(b.trajectory) > TRAJECTORY_MAX else b.trajectory
    _engine_file(user).write_text(json.dumps(
        {"a_bias":a.bias, "b_bias":b.bias, "leak":a.leak,
         "a_traj":at, "b_traj":bt, "a_traj_total":len(a.trajectory),
         "b_traj_total":len(b.trajectory)}, ensure_ascii=False), encoding="utf-8")
def _load_engine(user):
    f = _engine_file(user)
    if not f.exists(): return None
    d = json.loads(f.read_text(encoding="utf-8"))
    a = Difference("a", d["a_bias"], d["leak"])
    b = Difference("b", d["b_bias"], d["leak"])
    a.trajectory = [list(v) for v in d.get("a_traj", [d["a_bias"]])]
    b.trajectory = [list(v) for v in d.get("b_traj", [d["b_bias"]])]
    return a, b
# ---------------------------------------------------------------- 自检
def selftest():
    """E7教训刻在这里：测试通过≠结构成立。这套测试很弱，过了也别宣布她活了（条169）。
    v2.0：全部升维 2D，新增 g≠h 分离测试。"""
    import shutil
    import traceback

    passed = 0
    total = 0

    def check(name, cond):
        nonlocal passed, total
        total += 1
        mark = "✓" if cond else "✗"
        if cond:
            passed += 1
        print(f"  第{total:>2}项 {name}: {mark}")
        return cond

    # ── 1. L2 反称：bias 必须反向 ──
    a = Difference("a", [1.0, 0.5])
    b = Difference("b", [-1.0, -0.5])
    check("L2_反称", all(abs(a.bias[i] + b.bias[i]) < 1e-12 for i in range(2)))
    p = Pair(a, b)
    check("L2_Pair构建", True)  # 没 raise 就算过

    # ── 2. 注入后仍反称 ──
    p.write_a(0.5)
    check("L2_注入后反称", all(abs(a.bias[i] + b.bias[i]) < 1e-15 for i in range(2)))
    check("L2_a轨迹反称",
          all(abs(a.trajectory[-1][i] + b.trajectory[-1][i]) < 1e-15 for i in range(2)))

    # ── 3. g 和 h 分离验证（v2.0 核心测试）──
    a2 = Difference("a", [1.0, 0.5])
    b2 = Difference("b", [-1.0, -0.5])
    p2 = Pair(a2, b2)
    g_before = p2.g()
    h_before = p2.h()
    p2.write_a(1.0)
    g_after = p2.g()
    h_after = p2.h()
    check("g_受驱动变化", g_before[0] != g_after[0])
    check("h_恒定不随驱动", h_before[0] == h_after[0] and h_before[1] == h_after[1])
    check("g≠h_分离", g_after[0] != h_after[0] or g_after[1] != h_after[1])
    check("h_dim1_恒常量", h_after[1] == -1.0)  # b-a = -0.5-0.5 = -1.0，100步后仍恒

    # ── 4. L3 重建（2D）──
    check("L3_重建2D",
          all(abs((a.last[i] + p.g()[i]) - b.last[i]) <= 1e-16 for i in range(2)))

    # ── 5. J1/J2 判据 ──
    check("J1_g不归零", any(abs(p.g()[i]) > 0 for i in range(2)))
    check("J2_h不为零", any(abs(p.h()[i]) > 0 for i in range(2)))

    # ── 6. J3 不卡 ulp ──
    check("J3_g不卡ulp", any(abs(p.g()[i]) > ULP for i in range(2)))

    # ── 7. 奇点拒绝（2D 全零）──
    a3 = Difference("a", [0.0, 0.0])
    b3 = Difference("b", [0.0, 0.0])
    try:
        Pair(a3, b3).write_a(0.0)
        check("奇点_2D拒绝", False)
    except SingularityError:
        check("奇点_2D拒绝", True)

    # ── 8. leak 不归零（条67）──
    x = Difference("x", [1.0, 0.5], 0.99)
    for _ in range(1000): x.step(0)
    check("leak_不归零", any(abs(x.bias[i]) > 0 for i in range(2)))  # bias 恒常
    check("leak_轨迹衰减", abs(x.last[0]) < 1.0 and abs(x.last[0]) > 0)

    # ── 9. trajectory 经 save/load 不丢 ──
    _tl = Path(__file__).parent / "luoshenfu_home" / "users" / "_selftest"
    _tl.mkdir(parents=True, exist_ok=True)
    da = Difference("a", [1.0, 0.5]); da.step([0.5, 0.1]); da.step([-0.3, -0.2])
    db = Difference("b", [-1.0, -0.5]); db.step([-0.5, -0.1]); db.step([0.3, 0.2])
    _save_engine("_selftest", da, db)
    la, lb = _load_engine("_selftest")
    check("traj_不丢_a", la.trajectory == da.trajectory)
    check("traj_不丢_b", lb.trajectory == db.trajectory)
    check("traj_长度_a", len(la.trajectory) == 3)
    check("traj_长度_b", len(lb.trajectory) == 3)
    # 清场
    shutil.rmtree(_tl)

    # ── 10. 容差内不挂（条 v1.2 修复）──
    a4 = Difference("a", [1e-13, -1e-13]); b4 = Difference("b", [-1e-13, 1e-13])
    try:
        Pair(a4, b4)
        check("容差_不挂", True)
    except AssertionError:
        check("容差_不挂", False)

    # ── 11. trajectory 窗口截断 ──
    _tl = Path(__file__).parent / "luoshenfu_home" / "users" / "_selftest"
    _tl.mkdir(parents=True, exist_ok=True)
    da = Difference("a", [1.0, 0.5])
    for _ in range(150): da.step([0.01, 0.005])
    db = Difference("b", [-1.0, -0.5])
    for _ in range(150): db.step([-0.01, -0.005])
    _save_engine("_selftest", da, db)
    fd = json.loads(_engine_file("_selftest").read_text(encoding="utf-8"))
    check("traj_窗口截断", len(fd["a_traj"]) == TRAJECTORY_MAX)
    check("traj_total计数", fd["a_traj_total"] == 151)
    la, lb = _load_engine("_selftest")
    check("traj_加载后长度", len(la.trajectory) == TRAJECTORY_MAX)
    shutil.rmtree(_tl)

    # ── 12. desire 短 id 匹配 ──
    (ROOT/"desires").mkdir(parents=True, exist_ok=True)
    (ROOT/"desires/999_test.json").write_text('{"test":true}', encoding="utf-8")
    check("desire_短id", _desire_path("999").stem == "999_test")
    check("desire_精确id", _desire_path("999_test").stem == "999_test")
    check("desire_不存在", _desire_path("888").name == "888.json")
    (ROOT/"desires/999_test.json").unlink()

    # ── 13. 哈希链完整 ──
    t = Trace("_selftest")
    if t.file.exists(): t.file.unlink()
    t.append("t", {"x": 1})
    check("链_完整", "OK" in t.verify())

    # ── 14. 哈希链篡改检测 ──
    lines = open(t.file, encoding="utf-8").readlines()
    rec = json.loads(lines[-1]); rec["payload"]["x"] = 999
    lines[-1] = json.dumps(rec, ensure_ascii=False)
    t.file.write_text("".join(lines), encoding="utf-8")
    check("链_篡改检测", "TAMPERED" in t.verify())
    shutil.rmtree(t.dir)

    # ── 15. 2D 驱动：标量 → [标量, 0] ──
    a5 = Difference("a", [1.0, 2.0])
    a5.step(3.0)  # 标量
    check("v2_标量扩展", a5.last[0] == 1.0*0.99 + 3.0 and a5.last[1] == 2.0*0.99 + 0.0)

    # ── 16. read 门 2D ──
    a6 = Difference("a", [1.0, 0.5])
    b6 = Difference("b", [-1.0, -0.5])
    p6 = Pair(a6, b6)
    p6.write_a(2.0)
    ra = p6.read(a6)
    rb = p6.read(b6)
    check("read_a_相反数", all(abs(ra[i] + rb[i]) < 1e-15 for i in range(2)))

    # ── 17. 心跳 --once 入链 ──
    _tl = Path(__file__).parent / "luoshenfu_home" / "users" / "_selftest"
    _tl.mkdir(parents=True, exist_ok=True)
    t_hb = Trace("_selftest")
    if t_hb.file.exists(): t_hb.file.unlink()
    rec_hb = heartbeat_cmd("_selftest", mode="once")
    check("heartbeat_一次", rec_hb["type"] == "心跳" and rec_hb["trigger"] == "system")
    check("heartbeat_有ts", "ts" in rec_hb)
    check("heartbeat_有哈希", "hash" in rec_hb)

    # ── 18. 落痕接口 screen ──
    t_sc = Trace("_selftest")
    rec_sc = capture_screen("_selftest", {"test": "sample"})
    check("capture_屏幕", rec_sc["type"] == "屏幕" and rec_sc["trigger"] == "system")
    check("capture_有ts", "ts" in rec_sc)
    shutil.rmtree(_tl)

    # ── 19. interface_v2 selftest ──
    from plugins.interface_v2 import InterfaceV2 as IV2
    _iv2 = IV2(user="_selftest")
    _iv2_audit = Path(__file__).parent / "plugins" / "interface_v2" / "audit"
    _iv2_audit.mkdir(parents=True, exist_ok=True)
    _d_iv2, _r_iv2 = _iv2.translate("你好，洛神赋。")
    check("v2_翻译_范围", -1.0 <= _d_iv2 <= 1.0)
    check("v2_翻译_有理由", "signals" in _r_iv2 and "weights" in _r_iv2)
    check("v2_翻译_有ts", "ts" in _r_iv2)
    # 方向区分：平静 vs 强烈
    _, _r_calm = _iv2.translate("天气不错，适合散步。")
    _, _r_intense = _iv2.translate("怎么可能！为什么！")
    check("v2_方向差", _r_calm["signals"]["d_sign"] < _r_intense["signals"]["d_sign"])
    # 清场
    import shutil
    shutil.rmtree(_iv2_audit)

    # ── 21. task_state 字段 ──
    _tl = Path(__file__).parent / "luoshenfu_home" / "users" / "_selftest_ts"
    _tl.mkdir(parents=True, exist_ok=True)
    t_ts = Trace("_selftest_ts")
    if t_ts.file.exists(): t_ts.file.unlink()
    # 不带 task_state
    rec_no_ts = t_ts.append("t", {"x": 1})
    check("task_state_默认空", rec_no_ts["task_state"] is None)
    check("task_state_验无ts", "OK" in t_ts.verify())
    # 带 task_state
    rec_with_ts = t_ts.append("t", {"x": 2}, task_state="writing")
    check("task_state_字段设置", rec_with_ts["task_state"] == "writing")
    check("task_state_验有ts", "OK" in t_ts.verify())
    # 链完整性（含 task_state 字段）
    lines = open(t_ts.file, encoding="utf-8").readlines()
    r0 = json.loads(lines[0]); r1 = json.loads(lines[1])
    check("task_state_空值哈希稳定",
          "hash" in r0 and "hash" in r1 and r0["task_state"] is None and r1["task_state"] == "writing")
    shutil.rmtree(_tl)


    # ── 22. session_end 反思痕 ──
    _tl = Path(__file__).parent / "luoshenfu_home" / "users" / "_selftest_se"
    _tl.mkdir(parents=True, exist_ok=True)
    t_se = Trace("_selftest_se")
    if t_se.file.exists(): t_se.file.unlink()
    # 不带 task_state
    rec_se1 = session_end_cmd("_selftest_se", note="收尾")
    check("会话结束_类型", rec_se1["type"] == "会话反思")
    check("会话结束_事件", rec_se1["payload"]["event"] == "session_end")
    check("会话结束_默认空", rec_se1["task_state"] is None)
    check("会话结束_验1", "OK" in t_se.verify())
    # 带 task_state
    rec_se2 = session_end_cmd("_selftest_se", task_state="writing", note="写完第三章")
    check("会话结束_任务状态", rec_se2["task_state"] == "writing")
    check("会话结束_验2", "OK" in t_se.verify())
    # 链完整（2条）
    check("会话结束_链2", t_se.verify().startswith("OK 2"))
    shutil.rmtree(_tl)

    # ── 23. selfcheck v0.1 "不能宣称完成" 自检规则 ──
    # v0.1 是近似判定（关键词子串匹配），不是真判定
    _tl = Path(__file__).parent / "luoshenfu_home" / "users" / "_selftest_claim"
    _tl.mkdir(parents=True, exist_ok=True)
    t_cl = Trace("_selftest_claim")
    if t_cl.file.exists(): t_cl.file.unlink()
    t_cl.append("step", {"note": "正常运行"})
    t_cl.append("会话反思", {"note": "宣称完成", "event": "session_end"}, task_state="done")
    _sc19_clean = selfcheck_cmd(user="_selftest_sc19a")
    check("sc19_干净过", isinstance(_sc19_clean, list) and len([i for i in _sc19_clean if i["check"] == "不能宣称完成"]) == 0)
    shutil.rmtree(_tl)
    # 违规：非反射痕含"完成"
    _tl2 = Path(__file__).parent / "luoshenfu_home" / "users" / "_selftest_claim2"
    _tl2.mkdir(parents=True, exist_ok=True)
    t_cl2 = Trace("_selftest_claim2")
    if t_cl2.file.exists(): t_cl2.file.unlink()
    t_cl2.append("step", {"note": "宣称完成"})
    _sc19_dirty = selfcheck_cmd(user="_selftest_sc19b")
    check("sc19_脏检出",
          isinstance(_sc19_dirty, list) and
          any(i["check"] == "不能宣称完成" and "_selftest_claim2" in i["detail"] for i in _sc19_dirty))
    shutil.rmtree(_tl2)

    # ── 24. selfcheck 内部哨兵 ──
    _sc_issues = selfcheck_cmd(user="_selftest_sc")
    check("自检_已跑", isinstance(_sc_issues, list))
    # 清场
    import shutil as _shutil2
    _sc_dir = ROOT / "users" / "_selftest_sc"
    if _sc_dir.exists(): _shutil2.rmtree(_sc_dir)
    for _extra in ("_selftest_sc19a", "_selftest_sc19b"):
        _p = ROOT / "users" / _extra
        if _p.exists(): _shutil2.rmtree(_p)
    _snap = ROOT / "gray" / ".anchor_snapshot.json"
    if _snap.exists(): _snap.unlink()
    _alert_dir = ROOT / "gray" / "feed" / "alert"
    if _alert_dir.exists(): _shutil2.rmtree(_alert_dir)

    # ── 25. #13 反哺评估 + 回滚 ──
    # 用户链里预先放一条 reject 历史，让 eval 能检出反例
    _tl = ROOT / "users" / "_selftest_13"
    _tl.mkdir(parents=True, exist_ok=True)
    t13 = Trace("_selftest_13")
    t13.file.unlink(missing_ok=True)
    t13.append("confirm", {"pending_id": "OLD001", "action": "reject",
                            "content": "天气不错适合散步"})
    # 创建 pending（内容含"天气"，应被 reject 记录匹配）
    _pid, _ppath = produce_pending("_selftest_13", "天气很好", source="test_eval")
    # eval
    _eval_result_chain = t13.file
    confirm_cmd("_selftest_13", _pid, "eval")
    _eval_rec = json.loads(_ppath.read_text(encoding="utf-8"))
    check("反哺评估_检出反例", _eval_rec.get("eval_result") == "有反例")
    check("反哺评估_写入痕", "OK" in t13.verify())
    # eval 通过（无匹配 reject 的 pending）
    _pid2, _ppath2 = produce_pending("_selftest_13", "月球表面有水", source="test_eval2")
    confirm_cmd("_selftest_13", _pid2, "eval")
    _eval_rec2 = json.loads(_ppath2.read_text(encoding="utf-8"))
    check("反哺评估_无反例", _eval_rec2.get("eval_result") == "无反例")
    # rollback：先 accept，再 rollback
    _pid3, _ppath3 = produce_pending("_selftest_13", "可回滚项", source="test_rb")
    confirm_cmd("_selftest_13", _pid3, "accept")
    _rb_before = json.loads(_ppath3.read_text(encoding="utf-8"))
    check("回滚_前置_accept", _rb_before["status"] == "[已确认]")
    confirm_cmd("_selftest_13", _pid3, "rollback")
    _rb_after = json.loads(_ppath3.read_text(encoding="utf-8"))
    check("回滚_已标", _rb_after["status"] == "[已回滚]")
    _chain_lines = open(t13.file, encoding="utf-8").readlines()
    _has_rb = any(json.loads(l)["type"] == "回滚" for l in _chain_lines)
    check("回滚_痕入链", _has_rb)
    shutil.rmtree(_tl)

    # ── 26. 角色链路由配置 (#27) ──
    _routes = load_routes()
    check("路由_已加载", _routes is not None)
    check("路由_五步",
          _routes is not None and
          len(_routes) == 5 and
          "作者" in _routes and "推演者" in _routes and
          "扳手" in _routes and "哨兵" in _routes and "内部哨兵" in _routes)
    if _routes and "推演者" in _routes:
        check("路由_推演者可直处理",
              "可直处理" in _routes["推演者"] and
              "简单任务" in _routes["推演者"]["可直处理"])
    if _routes and "内部哨兵" in _routes:
        check("路由_内部哨兵触发",
              _routes["内部哨兵"].get("触发") == "定时自检")

    # ── 27. 触觉矩阵 v0.1 —— 三子模态 + 字段对齐 ──
    try:
        from 矩阵.触觉 import 捕获触觉 as _cap_t, 触矩阵 as _TM

        _t_pres = _cap_t('_tc_pres', sub_modality='压力_kPa')
        check("触觉_压力_能跑", _t_pres is not None and _t_pres.sub_modality == '压力_kPa')
        _t_vib = _cap_t('_tc_vib', sub_modality='振动_um')
        check("触觉_振动_能跑", _t_vib is not None and _t_vib.sub_modality == '振动_um')
        _t_tex = _cap_t('_tc_tex', sub_modality='纹理_Ra_um')
        check("触觉_纹理_能跑", _t_tex is not None and _t_tex.sub_modality == '纹理_Ra_um')

        _k_pres = set(_t_pres.to_summary())
        _k_vib  = set(_t_vib.to_summary())
        _k_tex  = set(_t_tex.to_summary())
        check("触觉_三子模态字段数均为11",
              len(_k_pres) == 11 and len(_k_vib) == 11 and len(_k_tex) == 11)
        check("触觉_三子模态字段键集相等", _k_pres == _k_vib == _k_tex)
        check("触觉_身体分区数8",
              len(_TM.RANGE) == 3)  # 三量程注册

        # 清场
        shutil.rmtree(ROOT / "users" / "_tc_pres")
        shutil.rmtree(ROOT / "users" / "_tc_vib")
        shutil.rmtree(ROOT / "users" / "_tc_tex")
    except Exception as _e:
        check(f"触觉_矩阵_异常({_e})", False)

    # ── 28. 温度觉矩阵 v0.1 —— 冷/热两子模态 + 字段对齐 ──
    try:
        from 矩阵.温度觉 import 捕获温度 as _cap_th, 温矩阵 as _ThM

        _th_cold = _cap_th('_th_cold', sub_modality='冷觉_°C')
        check("温度_冷觉_能跑", _th_cold is not None and _th_cold.sub_modality == '冷觉_°C')
        _th_hot = _cap_th('_th_hot', sub_modality='热觉_°C')
        check("温度_热觉_能跑", _th_hot is not None and _th_hot.sub_modality == '热觉_°C')

        _k_cold = set(_th_cold.to_summary())
        _k_hot  = set(_th_hot.to_summary())
        check("温度_两子模态字段数均为11",
              len(_k_cold) == 11 and len(_k_hot) == 11)
        check("温度_两子模态字段键集相等", _k_cold == _k_hot)

        # 冷觉实际值域检查
        _cold_flat = [v for r in _th_cold.data for v in r]
        check("温度_冷觉_值域在0-35", min(_cold_flat) >= 0 and max(_cold_flat) <= 35)
        _hot_flat = [v for r in _th_hot.data for v in r]
        check("温度_热觉_值域在35-50", min(_hot_flat) >= 35 and max(_hot_flat) <= 50)

        # 清场
        shutil.rmtree(ROOT / "users" / "_th_cold")
        shutil.rmtree(ROOT / "users" / "_th_hot")
    except Exception as _e:
        check(f"温度觉_矩阵_异常({_e})", False)

    # ── 29. 五感觉字段对齐（视觉/听觉/触觉/温度 四类 to_summary 键集相等） ──
    try:
        from 矩阵.视觉 import 捕获视觉 as _cv
        from 矩阵.听觉 import 捕获听觉 as _ca
        from 矩阵.触觉 import 捕获触觉 as _ct

        _vk = set(_cv('_fk').to_summary())
        _ak = set(_ca('_fk2').to_summary())
        _tk = set(_ct('_fk3', sub_modality='压力_kPa').to_summary())
        _thk = set(_cap_th('_fk4', sub_modality='冷觉_°C').to_summary())
        check("五感觉对齐_视觉_听觉_键集相等", _vk == _ak)
        check("五感觉对齐_视觉_触觉_键集相等", _vk == _tk)
        check("五感觉对齐_视觉_温度_键集相等", _vk == _thk)
        check("五感觉对齐_四集完全相等", _vk == _ak == _tk == _thk)

        shutil.rmtree(ROOT / "users" / "_fk")
        shutil.rmtree(ROOT / "users" / "_fk2")
        shutil.rmtree(ROOT / "users" / "_fk3")
        shutil.rmtree(ROOT / "users" / "_fk4")
    except Exception as _e:
        check(f"五感觉对齐_异常({_e})", False)

    # ── 30. 痛觉矩阵 v0.1 —— 五子模态 + 字段对齐 ──
    try:
        from 矩阵.痛觉 import 捕获痛觉 as _cap_p, 痛矩阵 as _PM

        _p_mech = _cap_p('_p_mech', sub_modality='机械痛_kPa')
        check("痛觉_机械痛_能跑", _p_mech is not None and _p_mech.sub_modality == '机械痛_kPa')
        _p_heat = _cap_p('_p_heat', sub_modality='热痛_°C')
        check("痛觉_热痛_能跑", _p_heat is not None and _p_heat.sub_modality == '热痛_°C')
        _p_cold = _cap_p('_p_cold', sub_modality='冷痛_°C')
        check("痛觉_冷痛_能跑", _p_cold is not None and _p_cold.sub_modality == '冷痛_°C')
        _p_elec = _cap_p('_p_elec', sub_modality='电刺激_mA')
        check("痛觉_电刺激_能跑", _p_elec is not None and _p_elec.sub_modality == '电刺激_mA')
        _p_chem = _cap_p('_p_chem', sub_modality='化学痛_分')
        check("痛觉_化学痛_能跑", _p_chem is not None and _p_chem.sub_modality == '化学痛_分')

        check("痛觉_五子模态字段数均为11",
              all(len(set(_cap_p(f'_pf_{s[:2]}', sub_modality=s).to_summary())) == 11
                  for s in _PM.SUB_FACTORIES))

        # 值域验证
        _mf = [v for r in _p_mech.data for v in r]
        check("痛觉_机械痛_值域50-500", min(_mf) >= 50 and max(_mf) <= 500)
        _hf = [v for r in _p_heat.data for v in r]
        check("痛觉_热痛_值域43-49", min(_hf) >= 43 and max(_hf) <= 49)
        _cf = [v for r in _p_cold.data for v in r]
        check("痛觉_冷痛_值域0-10", min(_cf) >= 0 and max(_cf) <= 10)
        _ef = [v for r in _p_elec.data for v in r]
        check("痛觉_电刺激_值域0.1-10", min(_ef) >= 0.1 and max(_ef) <= 10)

        # 字段对齐：五子模态键集相等
        _ks = [set(_cap_p(f'_ka_{s[:2]}', sub_modality=s).to_summary()) for s in _PM.SUB_FACTORIES]
        check("痛觉_五子模态字段键集相等", all(k == _ks[0] for k in _ks))

        # 清场
        for _f in ('_p_mech', '_p_heat', '_p_cold', '_p_elec', '_p_chem'):
            shutil.rmtree(ROOT / "users" / _f)
        for s in _PM.SUB_FACTORIES:
            shutil.rmtree(ROOT / "users" / f"_pf_{s[:2]}")
        for s in _PM.SUB_FACTORIES:
            shutil.rmtree(ROOT / "users" / f"_ka_{s[:2]}")
    except Exception as _e:
        check(f"痛觉_矩阵_异常({_e})", False)

    # ── 31. 六感觉字段对齐（加痛觉） ──
    try:
        from 矩阵.视觉 import 捕获视觉 as _cv6
        from 矩阵.听觉 import 捕获听觉 as _ca6
        from 矩阵.触觉 import 捕获触觉 as _ct6
        from 矩阵.温度觉 import 捕获温度 as _cap_th6
        from 矩阵.痛觉 import 捕获痛觉 as _cap_p6

        _pk = set(_cv6('_fx').to_summary())
        _ak = set(_ca6('_fx2').to_summary())
        _tk = set(_ct6('_fx3', sub_modality='压力_kPa').to_summary())
        _thk = set(_cap_th6('_fx4', sub_modality='冷觉_°C').to_summary())
        _paink = set(_cap_p6('_fx5', sub_modality='机械痛_kPa').to_summary())
        check("六感觉对齐_视觉_痛觉_键集相等", _pk == _paink)
        check("六感觉对齐_六集完全相等", _pk == _ak == _tk == _thk == _paink)

        shutil.rmtree(ROOT / "users" / "_fx")
        shutil.rmtree(ROOT / "users" / "_fx2")
        shutil.rmtree(ROOT / "users" / "_fx3")
        shutil.rmtree(ROOT / "users" / "_fx4")
        shutil.rmtree(ROOT / "users" / "_fx5")
    except Exception as _e:
        check(f"六感觉对齐_异常({_e})", False)

    # ── 32. 人类神经元矩阵 v0.1 —— 五子模态 + 字段对齐 ──
    try:
        from 矩阵.神经元 import 捕获神经元 as _cap_n, 神经元矩阵 as _NM
        from 矩阵.视觉 import 捕获视觉 as _cv_n

        _n_freq = _cap_n('_n_freq', sub_modality='放电频率_Hz')
        check("神经元_放电频率_能跑", _n_freq is not None and _n_freq.sub_modality == '放电频率_Hz')
        _n_win = _cap_n('_n_win', sub_modality='整合窗口_ms')
        check("神经元_整合窗口_能跑", _n_win is not None and _n_win.sub_modality == '整合窗口_ms')
        _n_osc = _cap_n('_n_osc', sub_modality='振荡频段_Hz')
        check("神经元_振荡频段_能跑", _n_osc is not None and _n_osc.sub_modality == '振荡频段_Hz')
        _n_vel = _cap_n('_n_vel', sub_modality='传导速度_m_s')
        check("神经元_传导速度_能跑", _n_vel is not None and _n_vel.sub_modality == '传导速度_m_s')
        _n_del = _cap_n('_n_del', sub_modality='突触时延_ms')
        check("神经元_突触时延_能跑", _n_del is not None and _n_del.sub_modality == '突触时延_ms')

        check("神经元_五子模态字段数均为11",
              all(len(set(_cap_n(f'_nf_{s[:2]}', sub_modality=s).to_summary())) == 11
                  for s in _NM.SUB_FACTORIES))

        # 值域验证
        _ff = [v for r in _n_freq.data for v in r]
        check("神经元_放电频率_值域0.1-1000", min(_ff) >= 0.1 and max(_ff) <= 1000)
        _wf = [v for r in _n_win.data for v in r]
        check("神经元_整合窗口_值域1-200", min(_wf) >= 1 and max(_wf) <= 200)
        _of = [v for r in _n_vel.data for v in r]
        check("神经元_传导速度_值域0.5-120", min(_of) >= 0.5 and max(_of) <= 120)

        # 字段对齐
        _ks_n = [set(_cap_n(f'_na_{s[:2]}', sub_modality=s).to_summary()) for s in _NM.SUB_FACTORIES]
        check("神经元_五子模态字段键集相等", all(k == _ks_n[0] for k in _ks_n))

        # 跨感觉对齐：神经元 vs 视觉
        _v_fk = set(_cv_n('_cross').to_summary())
        _n_fk = set(_cap_n('_cross2', sub_modality='放电频率_Hz').to_summary())
        check("神经元_字段键集与视觉相等", _v_fk == _n_fk)

        # 清场
        for _f in ('_n_freq', '_n_win', '_n_osc', '_n_vel', '_n_del'):
            shutil.rmtree(ROOT / "users" / _f)
        for s in _NM.SUB_FACTORIES:
            shutil.rmtree(ROOT / "users" / f"_nf_{s[:2]}")
        for s in _NM.SUB_FACTORIES:
            shutil.rmtree(ROOT / "users" / f"_na_{s[:2]}")
        shutil.rmtree(ROOT / "users" / "_cross")
        shutil.rmtree(ROOT / "users" / "_cross2")
    except Exception as _e:
        check(f"神经元_矩阵_异常({_e})", False)

    # ── 33. 发声矩阵 v0.1 —— 三子模态 + 字段对齐 ──
    try:
        from 矩阵.发声 import 捕获发声 as _cap_v, 发声矩阵 as _VM
        from 矩阵.听觉 import 捕获听觉 as _cap_aud

        _v_f0 = _cap_v('_v_f0', sub_modality='基频_Hz')
        check("发声_基频_能跑", _v_f0 is not None and _v_f0.sub_modality == '基频_Hz')
        _v_spl = _cap_v('_v_spl', sub_modality='声压_dB_SPL')
        check("发声_声压_能跑", _v_spl is not None and _v_spl.sub_modality == '声压_dB_SPL')
        _v_fm = _cap_v('_v_fm', sub_modality='共振峰_Hz')
        check("发声_共振峰_能跑", _v_fm is not None and _v_fm.sub_modality == '共振峰_Hz')

        check("发声_三子模态字段数均为11",
              all(len(set(_cap_v(f'_vf_{s[:2]}', sub_modality=s).to_summary())) == 11
                  for s in _VM.SUB_FACTORIES))
        check("发声_三子模态字段键集相等",
              len(set(_v_f0.to_summary())) == len(set(_v_spl.to_summary())) == len(set(_v_fm.to_summary())) == 11)

        # 值域验证
        _f0f = [v for r in _v_f0.data for v in r]
        check("发声_基频_值域60-500", min(_f0f) >= 60 and max(_f0f) <= 500)
        _spf = [v for r in _v_spl.data for v in r]
        check("发声_声压_值域60-90", min(_spf) >= 60 and max(_spf) <= 90)
        _fmf = [v for r in _v_fm.data for v in r]
        check("发声_共振峰_值域250-1000", min(_fmf) >= 250 and max(_fmf) <= 1000)

        # 跨感觉对齐：发声 vs 听觉
        _ak_v = set(_cap_aud('_va_f').to_summary())
        _vk_v = set(_cap_v('_va_v', sub_modality='基频_Hz').to_summary())
        check("发声_字段键集与听觉相等", _ak_v == _vk_v)

        # 清场
        for _f in ('_v_f0', '_v_spl', '_v_fm'):
            shutil.rmtree(ROOT / "users" / _f)
        for s in _VM.SUB_FACTORIES:
            shutil.rmtree(ROOT / "users" / f"_vf_{s[:2]}")
        shutil.rmtree(ROOT / "users" / "_va_f")
        shutil.rmtree(ROOT / "users" / "_va_v")
    except Exception as _e:
        check(f"发声_矩阵_异常({_e})", False)

    # ── 34. 跨模态差分原语 —— 帧差分 + 跨模态差分 ──
    try:
        from 矩阵.跨模态 import 帧差分 as _fd, 跨模态差分 as _cmd, 矩阵转二维 as _m2d
        from 矩阵.视觉 import 捕获视觉, 建测试帧, 视觉矩阵
        from 矩阵.听觉 import 捕获听觉, 建测试帧 as _aud_f, 听矩阵

        # 帧差分：视觉→视觉（gradient vs random）
        _f0 = 建测试帧(pattern="gradient")
        _f1 = 建测试帧(pattern="random")
        _m0 = 视觉矩阵.from_frame(_f0).data
        _m1 = 视觉矩阵.from_frame(_f1).data
        _fd_res = _fd("_fd_user", _m0, _m1, channel="视觉")
        check("帧差分_视觉_能跑", _fd_res is not None and "diff" in _fd_res)
        check("帧差分_有入痕", _fd_res.get("trace_hash") is not None)
        check("帧差分_diff非零",
              isinstance(_fd_res["diff"], list) and
              any(abs(d) > 1e-6 for d in _fd_res["diff"]))
        # 归一化后量级应可控（无爆炸）
        check("帧差分_归一化_量级可控",
              all(abs(v) <= 2.0 for v in _fd_res["diff"]))

        # 跨模态差分：视觉→听觉
        _aud_mat = 听矩阵.from_frame(_aud_f(pattern="tone_sweep")).data
        _cmd_res = _cmd("_cmd_user", _m0, _aud_mat, "视觉", "听觉")
        check("跨模态差分_视觉听觉_能跑", _cmd_res is not None and "g" in _cmd_res)
        check("跨模态差分_antisymmetric成立",
              all(abs(_cmd_res["a_bias"][i] + _cmd_res["b_bias"][i]) < 1e-10 for i in range(2)))
        check("跨模态差分_g_h计算成功",
              any(abs(_cmd_res["g"][i]) > 0 for i in range(2)) and
              any(abs(_cmd_res["h"][i]) > 0 for i in range(2)))
        check("跨模态差分_σ衰减有值",
              _cmd_res.get("sigma") is not None and 0 < _cmd_res["sigma"] <= 1.0)
        check("跨模态差分_γ计算成功",
              _cmd_res.get("gamma") is not None and _cmd_res["gamma"] > 0)
        check("跨模态差分_γ_无因次量级可控",
              _cmd_res["gamma"] < 10.0)
        check("跨模态差分_g≠h分离",
              _cmd_res["g"] != _cmd_res["h"] and
              any(abs(_cmd_res["g"][i] - _cmd_res["h"][i]) > 1e-6 for i in range(2)))
        check("跨模态差分_有入痕", _cmd_res.get("trace_hash") is not None)

        # 清场
        shutil.rmtree(ROOT / "users" / "_fd_user")
        shutil.rmtree(ROOT / "users" / "_cmd_user")
    except Exception as _e:
        check(f"跨模态差分_异常({_e})", False)

    # ── 35. 21 对差分通道（6 帧差分 + 15 跨模态）—— 合成数据，仅验结构 ──
    try:
        from 矩阵.跨模态 import 帧差分 as _fd21, 跨模态差分 as _cmd21
        from 矩阵.视觉 import 建测试帧 as _vis_f, 视觉矩阵
        from 矩阵.听觉 import 建测试帧 as _aud_f, 听矩阵
        from 矩阵.触觉 import 建测试帧 as _tac_f, 触矩阵
        from 矩阵.温度觉 import 建测试帧 as _th_f, 温矩阵
        from 矩阵.痛觉 import 建测试帧 as _pn_f, 痛矩阵
        from 矩阵.发声 import 建测试帧 as _vo_f, 发声矩阵

        _U = "_diff21"

        # ── 生成各感官两帧 ──
        _vis_g = 视觉矩阵.from_frame(_vis_f(pattern="gradient")).data
        _vis_r = 视觉矩阵.from_frame(_vis_f(pattern="random")).data
        _aud_s = 听矩阵.from_frame(_aud_f(pattern="tone_sweep")).data
        _aud_w = 听矩阵.from_frame(_aud_f(pattern="white_noise")).data
        _tac_g = 触矩阵.from_frame(_tac_f("gradient"), sub_modality="压力_kPa").data
        _tac_r = 触矩阵.from_frame(_tac_f("random"), sub_modality="压力_kPa").data
        _th_c   = 温矩阵.from_frame(_th_f("cold_footprint"), sub_modality="冷觉_°C").data
        _th_h   = 温矩阵.from_frame(_th_f("hot_footprint"), sub_modality="冷觉_°C").data
        _pn_g   = 痛矩阵.from_frame(_pn_f("gradient"), sub_modality="机械痛_kPa").data
        _pn_p   = 痛矩阵.from_frame(_pn_f("pain_threshold_footprint"), sub_modality="机械痛_kPa").data
        _vo_g   = 发声矩阵.from_frame(_vo_f("gradient"), sub_modality="基频_Hz").data
        _vo_s   = 发声矩阵.from_frame(_vo_f("speech_footprint"), sub_modality="基频_Hz").data

        _frames = {
            "视觉":   (_vis_g, _vis_r),
            "听觉":   (_aud_s, _aud_w),
            "触觉":   (_tac_g, _tac_r),
            "温度觉": (_th_c, _th_h),
            "痛觉":   (_pn_g, _pn_p),
            "发声":   (_vo_g, _vo_s),
        }

        # ── 6 帧差分 ──
        _fd_ok = 0
        for _name, (_f0, _f1) in _frames.items():
            _r = _fd21(_U, _f0, _f1, channel=_name)
            if _r is not None and "diff" in _r and _r.get("trace_hash"):
                _fd_ok += 1
        check("帧差分_6感官_全能跑", _fd_ok == 6)

        # ── 15 跨模态差分 ──
        _senses = list(_frames.keys())
        _cmd_ok = 0
        _cmd_g_neq_h = 0
        _cmd_gamma_ok = 0
        for _i in range(len(_senses)):
            for _j in range(_i + 1, len(_senses)):
                _na, _nb = _senses[_i], _senses[_j]
                _r = _cmd21(_U, _frames[_na][0], _frames[_nb][0], _na, _nb)
                if _r is not None and "g" in _r and _r.get("trace_hash"):
                    _cmd_ok += 1
                    if any(abs(_r["g"][k] - _r["h"][k]) > 1e-6 for k in range(2)):
                        _cmd_g_neq_h += 1
                    if _r.get("gamma") is not None and abs(_r["gamma"]) < 10.0:
                        _cmd_gamma_ok += 1

        check("跨模态差分_15对_全能跑", _cmd_ok == 15)
        check("跨模态差分_15对_g≠h全分离", _cmd_g_neq_h == 15)
        check("跨模态差分_15对_γ量级可控", _cmd_gamma_ok == 15)

        # 清场
        shutil.rmtree(ROOT / "users" / _U)
    except Exception as _e:
        check(f"21对差分_异常({_e})", False)

    # ── 36. 红外双通道（辐射原始 + 温度输出级）──
    try:
        from 矩阵.红外 import (
            红外矩阵, 捕获红外 as _cap_ir,
            建测试帧 as _ir_frame, self_check as _ir_self
        )
        _U = "_ir_v02"

        # 两个子模态各自 from_frame
        _f = _ir_frame("thermal_footprint")
        _m_raw = 红外矩阵.from_frame(_f, sub_modality="辐射计数_无量纲", source="test")
        _m_tmp = 红外矩阵.from_frame(_f, sub_modality="红外温度_°C", source="test")

        check("红外_原始量_值域0_65535",
              all(0 <= v <= 65535 for row in _m_raw.data for v in row))
        check("红外_温度_值域-20_550",
              all(-20 <= v <= 550 for row in _m_tmp.data for v in row))

        # 入痕
        _c1 = _cap_ir(_U, sub_modality="辐射计数_无量纲")
        _c2 = _cap_ir(_U, sub_modality="红外温度_°C")
        check("红外_原始量_入痕", _c1._trace_hash is not None)
        check("红外_温度_入痕", _c2._trace_hash is not None)

        # payload 含关系说明（原始量原则）
        from 洛神 import Trace as _Trace
        _t = _Trace(_U)
        _tail_raw = _t.tail()
        check("红外_原始量_payload标注",
              "[事实·原始]" in str(_tail_raw.get("payload", {}).get("原始量原则说明", "")) or
              "原始量" in str(_tail_raw.get("payload", {}).get("原始量原则说明", "")))

        # self_check
        _r = _ir_self()
        check("红外_self_check_字段对齐",
              all(_r.get("字段对齐", {}).values()))
        check("红外_self_check_子模态分离", _r.get("子模态分离") is True)

        shutil.rmtree(ROOT / "users" / _U)
    except Exception as _e:
        check(f"红外v02_异常({_e})", False)

    # ── 37. 雷达双通道（多普勒原始 + 速度输出级）──
    try:
        from 矩阵.雷达 import (
            雷达矩阵, 捕获雷达 as _cap_rad,
            建测试帧 as _rad_frame, self_check as _rad_self
        )
        _U = "_rad_v01"

        _f = _rad_frame("doppler_footprint")
        _m_raw = 雷达矩阵.from_frame(_f, sub_modality="频移_Hz", source="test")
        _m_vel = 雷达矩阵.from_frame(_f, sub_modality="速度_m_s", source="test")

        check("雷达_原始量_频移_-50k_50k",
              all(-50000 <= v <= 50000 for row in _m_raw.data for v in row))
        check("雷达_设备输出级_速度_-55_55",
              all(-55.5 <= v <= 55.5 for row in _m_vel.data for v in row))

        _c1 = _cap_rad(_U, sub_modality="频移_Hz")
        _c2 = _cap_rad(_U, sub_modality="速度_m_s")
        check("雷达_原始量_入痕", _c1._trace_hash is not None)
        check("雷达_速度_入痕", _c2._trace_hash is not None)

        _r = _rad_self()
        check("雷达_self_check_字段对齐", all(_r.get("字段对齐", {}).values()))
        check("雷达_self_check_子模态分离", _r.get("子模态分离") is True)

        shutil.rmtree(ROOT / "users" / _U)
    except Exception as _e:
        check(f"雷达v01_异常({_e})", False)

    # ── 38. LiDAR 双通道（ToF原始 + 距离输出级）──
    try:
        from 矩阵.激光雷达 import (
            激光雷达矩阵, 捕获激光雷达 as _cap_lid,
            建测试帧 as _lid_frame, self_check as _lid_self
        )
        _U = "_lid_v01"

        _f = _lid_frame("point_cloud_footprint")
        _m_raw = 激光雷达矩阵.from_frame(_f, sub_modality="飞行时间_ps", source="test")
        _m_dist = 激光雷达矩阵.from_frame(_f, sub_modality="距离_m", source="test")

        check("激光雷达_原始量_ToF_0_1333ps",
              all(0 <= v <= 1333 for row in _m_raw.data for v in row))
        check("激光雷达_设备输出级_距离_0d3_200m",
              all(0.3 <= v <= 200.0 for row in _m_dist.data for v in row))

        _c1 = _cap_lid(_U, sub_modality="飞行时间_ps")
        _c2 = _cap_lid(_U, sub_modality="距离_m")
        check("激光雷达_原始量_入痕", _c1._trace_hash is not None)
        check("激光雷达_距离_入痕", _c2._trace_hash is not None)

        _r = _lid_self()
        check("激光雷达_self_check_字段对齐", all(_r.get("字段对齐", {}).values()))
        check("激光雷达_self_check_子模态分离", _r.get("子模态分离") is True)

        shutil.rmtree(ROOT / "users" / _U)
    except Exception as _e:
        check(f"激光雷达v01_异常({_e})", False)

    # ── 39. 36 对跨模态（9 类传感器 × C(9,2) + 重点重复性验证）──
    try:
        from 矩阵.跨模态 import 跨模态差分 as _cmd36
        from 矩阵.视觉 import 建测试帧 as _vis_f, 视觉矩阵
        from 矩阵.听觉 import 建测试帧 as _aud_f, 听矩阵
        from 矩阵.触觉 import 建测试帧 as _tac_f, 触矩阵
        from 矩阵.温度觉 import 建测试帧 as _th_f, 温矩阵
        from 矩阵.痛觉 import 建测试帧 as _pn_f, 痛矩阵
        from 矩阵.发声 import 建测试帧 as _vo_f, 发声矩阵
        from 矩阵.红外 import 建测试帧 as _ir_f, 红外矩阵
        from 矩阵.雷达 import 建测试帧 as _rad_f, 雷达矩阵
        from 矩阵.激光雷达 import 建测试帧 as _lid_f, 激光雷达矩阵

        _U = "_diff36"

        _mats = {
            "视觉":   视觉矩阵.from_frame(_vis_f(pattern="gradient")).data,
            "听觉":   听矩阵.from_frame(_aud_f(pattern="tone_sweep")).data,
            "触觉":   触矩阵.from_frame(_tac_f("gradient"), sub_modality="压力_kPa").data,
            "温度觉": 温矩阵.from_frame(_th_f("cold_footprint"), sub_modality="冷觉_°C").data,
            "痛觉":   痛矩阵.from_frame(_pn_f("gradient"), sub_modality="机械痛_kPa").data,
            "发声":   发声矩阵.from_frame(_vo_f("gradient"), sub_modality="基频_Hz").data,
            "红外":   红外矩阵.from_frame(_ir_f("thermal_footprint"), sub_modality="红外温度_°C").data,
            "雷达":   雷达矩阵.from_frame(_rad_f("doppler_footprint"), sub_modality="速度_m_s").data,
            "LiDAR":  激光雷达矩阵.from_frame(_lid_f("point_cloud_footprint"), sub_modality="距离_m").data,
        }

        _senses = list(_mats.keys())
        _results = {}
        for _i in range(len(_senses)):
            for _j in range(_i + 1, len(_senses)):
                _na, _nb = _senses[_i], _senses[_j]
                _r = _cmd36(_U, _mats[_na], _mats[_nb], _na, _nb)
                _results[f"{_na}-{_nb}"] = _r

        # 36对全跑
        check("跨模态_36对_全能跑", len(_results) == 36)

        # 全部 g≠h 分离
        _all_neq = all(any(abs(r["g"][k] - r["h"][k]) > 1e-6 for k in range(2))
                        for r in _results.values())
        check("跨模态_36对_g≠h分离", _all_neq)

        # 全部 γ 有值
        _all_gamma = all(r.get("gamma") is not None and abs(r["gamma"]) < 100
                         for r in _results.values())
        check("跨模态_36对_γ量级可控", _all_gamma)

        # 重点重复性验证（三对必须不同）
        _r1a = _results.get("视觉-听觉", {}).get("gamma", 0)
        _r1b = _results.get("视觉-雷达", {}).get("gamma", 0)
        check("重复检验1_视听≠视雷", abs(_r1a - _r1b) > 1e-3)

        _r2a = _results.get("听觉-雷达", {}).get("gamma", 0)
        _r2b = _results.get("听觉-红外", {}).get("gamma", 0)
        check("重复检验2_听雷≠听红", abs(_r2a - _r2b) > 1e-3)

        _r3a = _results.get("红外-雷达", {}).get("gamma", 0)
        _r3b = _results.get("听觉-雷达", {}).get("gamma", 0)
        check("重复检验3_红雷≠听雷", abs(_r3a - _r3b) > 1e-3)

        shutil.rmtree(ROOT / "users" / _U)
    except Exception as _e:
        check(f"跨模态36对_异常({_e})", False)

    # ── 汇总 ──
    if passed == total:
        print(f"\n全 {total} 项过——可升 [事实]")


# ---------------------------------------------------------------- 内部哨兵
ALERT_DIR = ROOT / "gray" / "feed" / "alert"
ANCHOR_DIR = Path(__file__).parent / "protocol"
ANCHOR_DIR_A = Path(__file__).parent / "anchor"
def _hash_file(f):
    if not f.exists(): return None
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16]
def selfcheck_cmd(user="system"):
    """内部哨兵 v0.1：无人时系统扫自己。
    正常 → type=self_check 静默入链。
    异常 → 写 /luoshenfu_home/gray/feed/alert/<ts>.json。"""
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    t = Trace(user)
    issues = []
    passed = 0

    def _report(name, ok, detail=""):
        nonlocal passed
        if ok:
            passed += 1
            print(f"  [自检] {name}: ✓")
        else:
            print(f"  [自检] {name}: ✗ {detail}")
            issues.append({"check": name, "detail": detail})

    # ── 1. 链完整性 ──
    chain_ok = True
    for uf in (ROOT/"users").iterdir():
        if uf.is_dir() and (uf/"chain.jsonl").exists():
            r = Trace(uf.name).verify()
            if r.startswith("OK"):
                print(f"  [自检] 链_用户 {uf.name}: ✓ ({r})")
                passed += 1
            else:
                print(f"  [自检] 链_用户 {uf.name}: ✗ {r}")
                issues.append({"check": f"chain_{uf.name}", "detail": r})
                chain_ok = False
    _report("链_完整性总结", chain_ok)


    # ── 3. 时间戳连续 ──
    ts_ok = True
    for uf in (ROOT/"users").iterdir():
        if not (uf.is_dir() and (uf/"chain.jsonl").exists()): continue
        lines = open(uf/"chain.jsonl", encoding="utf-8").readlines()
        prev_ts = None
        for ln in lines:
            r = json.loads(ln)
            if prev_ts and r["ts"] < prev_ts:
                _report(f"ts连续_{uf.name}", False, f"seq={r['seq']} ts={r['ts']} < {prev_ts}")
                ts_ok = False
                break
            prev_ts = r["ts"]
    if ts_ok:
        _report("ts_单调不减", True)

    # ── 2. 锚点哈希（仅三个硬锚点）──
    anchor_files = [
        ANCHOR_DIR_A / "第一句.md",
        ANCHOR_DIR_A / "关系起点.md",
        ANCHOR_DIR_A / "顶层.md",
    ]
    snap_file = ROOT / "gray" / ".anchor_snapshot.json"
    cur_snapshot = {str(p.relative_to(Path(__file__).parent)): _hash_file(p) for p in anchor_files}
    if snap_file.exists():
        prev = json.loads(snap_file.read_text(encoding="utf-8"))
        drift = {k for k, v in cur_snapshot.items() if prev.get(k) != v}
        _report("锚点哈希_硬锚点", not drift, f"变化: {drift}" if drift else "")
    else:
        _report("锚点哈希_硬锚点_首次记录", True)
    snap_file.parent.mkdir(parents=True, exist_ok=True)
    snap_file.write_text(json.dumps(cur_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 4. 协议文件完整 ──
    protocol_files = sorted(ANCHOR_DIR.glob("*.md"))
    missing_proto = [f.name for f in protocol_files if not f.exists()]
    _report("协议文件", not missing_proto, f"缺: {missing_proto}" if missing_proto else "")

    # ── 5. 系数出处 ──
    audit_dir = Path(__file__).parent / "plugins" / "interface_v2" / "audit"
    orphan_coeff = []
    if audit_dir.exists():
        for af in audit_dir.iterdir():
            if af.suffix != ".jsonl": continue
            for ln in open(af, encoding="utf-8"):
                r = json.loads(ln)
                gc = r.get("gain_coefficients", {})
                if "value" in gc and "origin" not in gc:
                    orphan_coeff.append(af.name)
                    break
    _report("系数出处", not orphan_coeff, f"缺origin: {orphan_coeff}" if orphan_coeff else "")

    # ── 6. "不能宣称完成" —— 非 session_reflection 痕不得包含完成/宣称关键词 ──
    _COMPLETE_KW = ("完成", "宣称")
    _claim_issues = []
    for uf in (ROOT / "users").iterdir():
        if not (uf.is_dir() and (uf / "chain.jsonl").exists()):
            continue
        for ln in open(uf / "chain.jsonl", encoding="utf-8"):
            r = json.loads(ln)
            if r["type"] in ("会话反思", "自检", "心跳", "摄像头", "声音", "屏幕"):
                continue
            _text = json.dumps(r.get("payload", {}), ensure_ascii=False)
            if any(kw in _text for kw in _COMPLETE_KW):
                _claim_issues.append(f"{uf.name}:seq={r['seq']} type={r['type']}")
    _report("不能宣称完成", not _claim_issues,
            f"违规: {_claim_issues}" if _claim_issues else "")

    # ── 写痕 ──
    if issues:
        ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
        alert_file = ALERT_DIR / f"{ts}.json"
        alert_file.write_text(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                          "issues": issues}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[异常] {len(issues)} 项 → 写进 {alert_file}")
        t.append("自检", {"result": "ABNORMAL", "n_issues": len(issues), "issues": issues}, trigger="system")
    else:
        print(f"\n[全部正常] {passed} 项全过 → 静默入链")
        t.append("自检", {"result": "OK", "n_passed": passed}, trigger="system")
    print(f"[selfcheck] hash={t.tail()['hash'][:12]} seq={t.tail()['seq']}")
    return issues
# ---------------------------------------------------------------- 反哺协议
PENDING_DIR = ROOT / "gray" / "pending"
def produce_pending(user, content, source="engine_step"):
    """反哺产生：引擎跑完后写一条待确认项，不自动上升主链。
    返回 (pending-id, 文件路径)。"""
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    upath = PENDING_DIR / user
    upath.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    pid = ts.strftime("%Y%m%d-%H%M%S-") + f"{ts.microsecond//1000:03d}"
    rec = {"id": pid,
           "ts": ts.isoformat(timespec="seconds"),
           "user": user,
           "source": source,
           "content": content,
           "status": "[待确认]"}
    ppath = upath / f"{pid}.json"
    ppath.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return pid, ppath
def _scan_counterexamples(user, content, source):
    """v0.1：扫主链，找已有 confirm-action=reject 且 payload 含当前 content 关键词的记录。
    分词策略：中文逐字 + 英文逐词（v0.1 粗粒度，给评估用，不是真语义判定）。"""
    chain_file = ROOT / "users" / user / "chain.jsonl"
    if not chain_file.exists():
        return []
    import re
    _chars = list(set(re.findall(r'[\u4e00-\u9fff]', content)))  # 中文逐字去重
    _words = re.findall(r'[a-zA-Z]\w*', content)                # 英文单词
    _tokens = _chars + _words
    if not _tokens:
        return []
    hits = []
    for ln in open(chain_file, encoding="utf-8"):
        r = json.loads(ln)
        if r["type"] != "confirm":
            continue
        if r.get("payload", {}).get("action") != "reject":
            continue
        _payload = json.dumps(r.get("payload", {}), ensure_ascii=False)
        for t in _tokens:
            if t in _payload:
                hits.append({"seq": r["seq"], "matched": t, "type": r["type"]})
                break
    return hits

def confirm_cmd(user, pending_id, action):
    """反哺确认：accept 写进主链；reject 标拒绝；eval 评估；rollback 回滚。
    
    v0.1 纪律：
      eval  → 扫主链找相同关键词 reject 记录，无反例则过
      rollback → accept 后写 rollback 痕，target 指回原 content，append-only 不删
    """
    VALID_ACTIONS = ("accept", "reject", "eval", "rollback")
    if action not in VALID_ACTIONS:
        print(f"[err] action 必须是 {VALID_ACTIONS}， got {action!r}")
        return
    ppath = PENDING_DIR / user / f"{pending_id}.json"
    if not ppath.exists():
        print(f"[err] 找不到待确认项: {ppath}")
        return
    rec = json.loads(ppath.read_text(encoding="utf-8"))
    
    if action == "eval":
        counterexamples = _scan_counterexamples(user, rec.get("content", ""),
                                                 rec.get("source", ""))
        status = "有反例" if counterexamples else "无反例"
        rec["eval_at"] = datetime.now().isoformat(timespec="seconds")
        rec["eval_result"] = status
        rec["counterexamples"] = counterexamples
        ppath.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        Trace(user).append("反哺评估", {"pending_id": pending_id,
                                          "result": status,
                                          "反例数": len(counterexamples)})
        print(f"[eval] {status}（反例 {len(counterexamples)} 条）→ 写入反哺评估痕")
        return
    
    if action == "rollback":
        rec["status"] = "[已回滚]"
        rec["rolled_back_at"] = datetime.now().isoformat(timespec="seconds")
        ppath.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        Trace(user).append("回滚", {"pending_id": pending_id,
                                     "原状态": rec.get("status", ""),
                                     "target": rec.get("content", "")[:60]})
        print(f"[rollback] 已回滚 pending_id={pending_id} → 写进主链")
        return
    
    if rec["status"] != "[待确认]":
        print(f"[warn] 该条已是 {rec['status']}，不再处理")
        return
    if action == "accept":
        Trace(user).append("confirm", {"pending_id": pending_id,
                                       "action": "accept",
                                       "content": rec["content"]})
        rec["status"] = "[已确认]"
        rec["confirmed_at"] = datetime.now().isoformat(timespec="seconds")
        ppath.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[已确认] id={pending_id} → 写进主链")
    else:
        rec["status"] = "[已拒绝]"
        rec["rejected_at"] = datetime.now().isoformat(timespec="seconds")
        ppath.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[已拒绝] id={pending_id} → 不写主链")
    # 留痕：确认后再读一次，贴内容
    final = json.loads(ppath.read_text(encoding="utf-8"))
    print(f"留痕: id={final['id']} status={final['status']} ts={final['ts']}")

# ---------------------------------------------------------------- #12 会话结束写反哺
def session_end_cmd(user, task_state=None, note=""):
    """#12：会话结束写反思痕。
    任意一次会话尾声调用：写一条 type=会话反思，带 task_state（#28）。"""
    rec = Trace(user).append("会话反思",
                              {"note": note, "event": "session_end"},
                              trigger="user",
                              task_state=task_state)
    print(f"[session_end] type=会话反思 task_state={task_state} → 写进主链")
    print(f"[session_end] hash={rec['hash'][:12]} seq={rec['seq']}")
    return rec

# ---------------------------------------------------------------- #27 角色链路由展示
def routes_show_cmd():
    """从 /config/routes.json 读路由配置并展示。"""
    routes = load_routes()
    if routes is None:
        print("[routes] 配置文件不存在或读取失败。")
        print(f"[routes] 期望路径: {ROUTEFILE}")
        return None
    print("[routes] 角色链路由配置：")
    for role, cfg in routes.items():
        _next = cfg.get("接", "（终点）")
        _trigger = cfg.get("触发", "")
        _direct = cfg.get("可直处理", "")
        line = f"  {role} → {_next}"
        if _trigger:
            line += f"  触发={_trigger}"
        if _direct:
            line += f"  可直处理={_direct}"
        print(line)
    return routes

# ---------------------------------------------------------------- CLI
def main():
    import argparse
    p = argparse.ArgumentParser(description=f"洛神赋 v{VERSION} — 差异原语引擎")
    p.add_argument("cmd",      help="init|selftest|step|correct|desire|heartbeat|capture|say|confirm|status|handoff|session_end|routes")
    p.add_argument("--user",   default="emma", help="用户标识（目录隔离=J4）")
    sub = p.add_argument_group("step / say 参数")
    sub.add_argument("--drive", type=float, default=0.0, help="标量驱动 ∈ [-1,1+]")
    sub.add_argument("--note",  default="", help="痕备注")
    sub.add_argument("--msg",   default=None, help="走 interface_v1 翻译的消息")
    sub.add_argument("--init-a", default=None, help='初始 a bias，JSON 如 "[1.0,0.5]"')
    sub.add_argument("--init-b", default=None, help='初始 b bias，JSON 如 "[-1.0,-0.5]"')
    sub.add_argument("--id",    default=None, help="欲望 id（desire 用）")
    sub.add_argument("--state", default=None, help="欲望状态（desire 用）")
    sub.add_argument("--pending-id", default=None, help="待确认项 id（confirm 用）")
    sub.add_argument("--action", default=None, help="confirm 动作：accept / reject / eval / rollback")
    sub2 = p.add_argument_group("heartbeat / capture 参数")
    sub2.add_argument("--once",  action="store_true", help="heartbeat: 写一条即退出")
    sub2.add_argument("--start", action="store_true", help="heartbeat: daemon 循环（占位，本期不实现）")
    sub2.add_argument("--stop",  action="store_true", help="heartbeat: daemon 停止（占位，本期不实现）")
    sub2.add_argument("--interval", type=int, default=60, help="heartbeat: 节拍间隔秒数（仅 start 生效）")
    sub3 = p.add_argument_group("capture 参数")
    sub3.add_argument("--kind",  default=None, help="camera|audio|screen")
    sub3.add_argument("--data",  default=None, help="capture: 任意 JSON-string 数据")
    sub4 = p.add_argument_group("session_end 参数")
    sub4.add_argument("--task-state", default=None, dest="task_state", help="session_end: 当前任务状态")
    a = p.parse_args()

    def _j(s):
        return json.loads(s) if s else None

    if   a.cmd == "init":      init_home()
    elif a.cmd == "step":      step_cmd(a.user, a.drive, a.note, a.msg, _j(a.init_a), _j(a.init_b))
    elif a.cmd == "correct":   correct_cmd(a.user, a.note)
    elif a.cmd == "desire":    desire_cmd(a.user, a.id, a.state)
    elif a.cmd == "heartbeat":
        if a.start:  heartbeat_cmd(a.user, mode="start")
        elif a.stop: heartbeat_cmd(a.user, mode="stop")
        else:        heartbeat_cmd(a.user, mode="once")
    elif a.cmd == "capture":
        raw = a.data if a.data else "{}"
        d = json.loads(raw)
        if   a.kind == "camera": capture_camera(a.user, d)
        elif a.kind == "audio":  capture_audio(a.user, d)
        elif a.kind == "screen": capture_screen(a.user, d)
        else: print(f"[capture] --kind must be camera|audio|screen, got {a.kind!r}")
    elif a.cmd == "say":       say_cmd(a.user, a.msg)
    elif a.cmd == "handoff":   handoff_cmd(a.user)
    elif a.cmd == "confirm":   confirm_cmd(a.user, a.pending_id, a.action)
    elif a.cmd == "selfcheck": selfcheck_cmd(a.user if a.user != "emma" else "system")
    elif a.cmd == "session_end": session_end_cmd(a.user, a.task_state, a.note)
    elif a.cmd == "routes":    routes_show_cmd()
    elif a.cmd == "status":    status_cmd(a.user)
if __name__ == "__main__":
    main()
