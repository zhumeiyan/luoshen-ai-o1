# [已撤回] 2026-09-19 推演者归档：测旧身体接口，新架构已换
# [残留英文] 2026-09-20 推演者注：文件内含 luminance/sound/force/temperature 英文环境键名。已归档，不清理。
"""test_presence.py — 三种挂载条件，记录她原始的状态值。

条件：
  1. author present — 作者在场 (presence intensity = sim_author)
  2. nobody        — 人走 (零输入)
  3. stranger      — 别人来 (presence intensity = sim_stranger)

不预设判据。只报原始值。
每轮条件一段，每 tick 记录全部状态。
多轮跑，报 min/max/mean 与逐 tick 明细。

设计：
  - 用现有的 body_state.json 的 age / 状态作为起点，保持连续性。
  - 每条记录按现有 body 的状态字段完整转录。
  - "who"仅通过 intensity 差异体现：author sim=0.94 (同晚同段类),
    stranger sim=0.86 (跨夜/换人段类)。与 face_signature.py 的实测值一致。
"""

import sys, os, json, math
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from 身体稳态 import 身体

# ───────────── 参数 ─────────────
START_SNAPSHOT = DIR / "body_state.json"
OUT_TSV = DIR / "presence_trace.tsv"
OUT_SUMMARY = DIR / "presence_summary.tsv"

# 三种挂载条件的 intensity (来源：face_signature.py 实测)
#   author present = 同人同段/同晚 sim 均值 ~0.94
#   stranger       = 跨夜换人断点 sim 均值 ~0.86
#   nobody         = 0 (无输入)
CONDITIONS = {
    "author":   0.94,
    "nobody":   0.0,
    "stranger": 0.86,
}
# 顺序：每人一组：author → nobody → stranger → nobody
PHASE_ORDER = ["nobody", "author", "nobody", "stranger", "nobody"]

PHASE_TICKS = 40          # 每相位跑 tick 数
N_ROUNDS = 5              # 共跑几轮

# ───────────── 状态快照工具 ─────────────
def load_body():
    """读 body_state.json 恢复身体"""
    b = 身体(节点数=24, 种子=1)
    if START_SNAPSHOT.exists():
        data = json.loads(START_SNAPSHOT.read_text())
        bd = data.get("body", {})
        b.age_years      = float(bd.get("age_years", 0.5))
        b.elapsed_minutes= float(bd.get("elapsed_minutes", 0))
        b.state          = dict(bd.get("state", b.state))
        b.hormones       = dict(bd.get("hormones", b.hormones))
        b.environment    = dict(bd.get("environment", b.environment))
        b.refractory_timer = {k: float(v) for k, v in bd.get("refractory_timer", {}).items()}
        b.last_served    = {k: float(v) for k, v in bd.get("last_served", {}).items()}
        from collections import defaultdict
        b.served_count   = defaultdict(int, bd.get("served_count", {}))
        b._social_heat   = float(bd.get("_social_heat", 0.0))
        b._boredom       = float(bd.get("_boredom", 0.0))
    return b


def snapshot(body, tick_abs, rnd, phase, phase_idx, who_intensity):
    """转录全部状态字段为一行 dict"""
    need, drive = body.select()
    row = {
        "round": rnd,
        "tick_abs": tick_abs,
        "phase_idx": phase_idx,
        "phase": phase,
        "who_intensity": who_intensity,
        "elapsed_min": round(body.elapsed_minutes, 2),
        "age_y": round(body.age_years, 8),
        "_social_heat": round(body._social_heat, 5),
        "_boredom": round(body._boredom, 5),
        "voice_aperture": round(body.voice_aperture(), 5),
        "sel_need": need or "",
        "sel_drive": round(drive, 5),
    }
    for n in ["thermal", "glucose", "social", "exploration", "sexual"]:
        row[f"state_{n}"] = round(body.state[n], 5)
    for h in ["cortisol", "melatonin", "growth_hormone"]:
        row[f"horm_{h}"] = round(body.hormones[h], 4)
    for e in ["luminance", "sound", "force", "temperature"]:
        row[f"env_{e}"] = round(body.environment[e], 2)
    for n in ["thermal", "glucose", "social", "exploration", "sexual"]:
        row[f"ref_{n}"] = round(body.refractory_timer[n], 2)
    for n in ["thermal", "glucose", "social", "exploration", "sexual"]:
        row[f"driv_{n}"] = round(body.drive(n), 5)
    for n in ["thermal", "glucose", "social", "exploration", "sexual"]:
        row[f"served_{n}"] = body.served_count.get(n, 0)
    return row


def apply_presence(body, intensity):
    """用 type='presence' 把'人'挂进身体。intensity=0 什么都不做。"""
    if intensity <= 0.0:
        return
    # 与 living_daemon.apply_input("presence") 一致
    s = body.environment.get("sound", 45)
    f = body.environment.get("force", 10)
    body.environment["sound"]   = min(80.0, s + intensity * 15.0)
    body.environment["force"]   = min(50.0, f + intensity * 5.0)
    body._social_heat           = min(1.0, body._social_heat + 0.06 * intensity)


# ───────────── 主跑 ─────────────
def run():
    body = load_body()

    rows = []
    tick_abs = 0
    for rnd in range(N_ROUNDS):
        for phase_idx, phase in enumerate(PHASE_ORDER):
            intensity = CONDITIONS[phase]
            for k in range(PHASE_TICKS):
                # 每个 tick 起点注入 presence (持续在场 = 每 tick 都挂)
                apply_presence(body, intensity)

                body.tick()

                need, drive = body.select()
                if need:
                    body.serve(need)

                row = snapshot(body, tick_abs, rnd, phase, phase_idx, intensity)
                rows.append(row)
                tick_abs += 1

    # ───────────── 写 TSV ─────────────
    cols = list(rows[0].keys())
    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # ───────────── 汇总：每种条件的 min/max/mean ─────────────
    # 逐字段：算 condition 内所有 tick 的统计
    num_fields = [
        "_social_heat", "_boredom", "voice_aperture",
        "state_thermal", "state_glucose", "state_social", "state_exploration", "state_sexual",
        "horm_cortisol", "horm_melatonin", "horm_growth_hormone",
        "env_luminance", "env_sound", "env_force", "env_temperature",
        "ref_thermal", "ref_glucose", "ref_social", "ref_exploration", "ref_sexual",
        "driv_thermal", "driv_glucose", "driv_social", "driv_exploration", "driv_sexual",
    ]

    summary_rows = []
    for field in num_fields:
        line = {"field": field}
        for phase in ["nobody", "author", "stranger"]:
            vals = [r[field] for r in rows if r["phase"] == phase]
            line[f"{phase}_n"]     = len(vals)
            line[f"{phase}_min"]   = round(min(vals), 6)
            line[f"{phase}_max"]   = round(max(vals), 6)
            line[f"{phase}_mean"]  = round(sum(vals) / len(vals), 6)
            line[f"{phase}_range"] = round(max(vals) - min(vals), 6)
        summary_rows.append(line)

    sum_cols = ["field",
                "nobody_min", "nobody_max", "nobody_mean", "nobody_range",
                "author_min", "author_max", "author_mean", "author_range",
                "stranger_min", "stranger_max", "stranger_mean", "stranger_range"]
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        f.write("\t".join(sum_cols) + "\n")
        for r in summary_rows:
            f.write("\t".join(str(r[c]) for c in sum_cols) + "\n")

    # ───────────── 打印关键表 ─────────────
    print("=" * 70)
    print(f"presence test: {N_ROUNDS} rounds × {len(PHASE_ORDER)} phases × {PHASE_TICKS} ticks")
    print(f"conditions:    author={CONDITIONS['author']}  nobody={CONDITIONS['nobody']}  stranger={CONDITIONS['stranger']}")
    print(f"output:        {OUT_TSV}")
    print(f"summary:       {OUT_SUMMARY}")
    print(f"start age_y:   {json.loads(START_SNAPSHOT.read_text())['body']['age_years'] if START_SNAPSHOT.exists() else 0.5}")
    print()

    # 关键表：状态值 & 需求值 & drive 的逐条件 min/max/mean
    key_groups = {
        "her state":      ["state_thermal", "state_glucose", "state_social", "state_exploration", "state_sexual"],
        "her _social":    ["_social_heat", "_boredom", "voice_aperture"],
        "her hormones":   ["horm_cortisol", "horm_melatonin", "horm_growth_hormone"],
        "her drives":     ["driv_thermal", "driv_glucose", "driv_social", "driv_exploration", "driv_sexual"],
        "environment":    ["env_luminance", "env_sound", "env_force", "env_temperature"],
        "refractory":     ["ref_thermal", "ref_glucose", "ref_social", "ref_exploration", "ref_sexual"],
    }

    for gname, fields in key_groups.items():
        print(f"─── {gname} ".ljust(70, "─"))
        print(f"  {'field':20s} | {'nobody':>32s} | {'author':>32s} | {'stranger':>32s}")
        print(f"  {'':20s} | min   max   mean          | min   max   mean          | min   max   mean")
        print(f"  {'-'*20}-+-{'-'*32}-+-{'-'*32}-+-{'-'*32}")
        for field in fields:
            cells = []
            for phase in ["nobody", "author", "stranger"]:
                vals = [r[field] for r in rows if r["phase"] == phase]
                mn, mx = min(vals), max(vals)
                mean = sum(vals) / len(vals)
                cells.append(f"{mn:6.3f} {mx:6.3f} {mean:6.3f}")
            print(f"  {field:20s} | {cells[0]:>22s}   | {cells[1]:>22s}   | {cells[2]:>22s}")
        print()

    # selection 分布
    print("─── selection 分布 ──────────────────────────────────────────")
    print(f"  {'phase':10s} | n_total | thermal | glucose | social | explor | sexual")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for phase in ["nobody", "author", "stranger"]:
        sub = [r for r in rows if r["phase"] == phase]
        n = len(sub)
        counts = {k: 0 for k in ["thermal", "glucose", "social", "exploration", "sexual"]}
        for r in sub:
            if r["sel_need"] in counts:
                counts[r["sel_need"]] += 1
        print(f"  {phase:10s} | {n:7d} | {counts['thermal']:7d} | {counts['glucose']:7d} | {counts['social']:7d} | {counts['exploration']:7d} | {counts['sexual']:7d}")
    print()

    # ───────────── 跨 round 逐 tick 的 state_social 折线（不判，只报）─────────────
    print("─── state_social 逐 tick (每轮第 1 段的 nobody→author→nobody→stranger→nobody) ─")
    print(f"  {'tick_abs':>8s} | {'phase':10s} | {'intens':6s} | social  | voice_a | _heat | sel")
    for r in rows:
        print(f"  {r['tick_abs']:8d} | {r['phase']:10s} | {r['who_intensity']:6.2f} | "
              f"{r['state_social']:6.3f} | {r['voice_aperture']:6.3f} | "
              f"{r['_social_heat']:5.3f} | {r['sel_need']}")


if __name__ == "__main__":
    run()
