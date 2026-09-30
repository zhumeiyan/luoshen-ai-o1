# [已撤回] 2026-09-19 推演者归档：旧引用，新架构已换
# [残留英文] 2026-09-20 推演者注：文件内含 luminance/sound/force/temperature 英文环境键名。已归档，不清理。
"""test_spike_source.py — 29-tick 锯齿的源头是不是 social 自己。

三档：
  A. baseline — full loop (tick + select + serve)，不做任何手脚
  B. no_social_heat — 跟 A 一样，但每次 serve("social") 后扣回 0.7（等价于关掉 social 充能）
  C. no_social_serve — social 被选到时跳过 serve（连 social state 恢复也禁）

每档都记录：heat 曲线、被选的是哪个 need、各 need 的 ref 与 state。
跑两大轮：author / stranger，ref 最长的轮子够到第二次锯齿（跑 80 tick）。
"""

import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from age_homeo import Body, HOMEOSTATIC_NEEDS

OUT = DIR / "spike_source_trace.tsv"
OUT_SUMMARY = DIR / "spike_source_summary.tsv"

PRESENT_TICKS = 30
DECAY_TICKS = 80
N_ROUNDS = 2

INTENSITY = {"author": 0.94, "stranger": 0.86}


def settle(body, intensity, ticks):
    for _ in range(ticks):
        if intensity > 0:
            s = body.environment.get("sound", 45)
            f = body.environment.get("force", 10)
            body.environment["sound"] = min(80.0, s + intensity * 15.0)
            body.environment["force"] = min(50.0, f + intensity * 5.0)
            body._social_heat = min(1.0, body._social_heat + 0.06 * intensity)
        body.tick()
        n, _ = body.select()
        if n:
            body.serve(n)


def run_one(mode, who, intensity, rnd):
    body = Body(age_years=0.5, tick_minutes=1.0)
    settle(body, intensity, PRESENT_TICKS)
    leave_heat = body._social_heat

    rows = []
    for k in range(DECAY_TICKS):
        body.tick()
        need, drv = body.select()

        served = ""
        if need:
            if mode == "no_social_serve" and need == "social":
                served = "skipped"
                # do nothing
            else:
                body.serve(need)
                served = need
                if mode == "no_social_heat" and need == "social":
                    body._social_heat -= 0.7  # 扣回

        rows.append({
            "mode": mode,
            "who": who,
            "round": rnd,
            "tick": k,
            "heat_at_leave": round(leave_heat, 5),
            "heat": round(body._social_heat, 6),
            "sel_need": need or "",
            "sel_drive": round(drv, 5),
            "served": served,
            "voice": round(body.voice_aperture(), 6),
            **{f"ref_{n}": round(body.refractory_timer[n], 2) for n in HOMEOSTATIC_NEEDS},
            **{f"state_{n}": round(body.state[n], 4) for n in HOMEOSTATIC_NEEDS},
        })
    return rows


def run():
    all_rows = []
    for rnd in range(N_ROUNDS):
        for who, intensity in INTENSITY.items():
            for mode in ["baseline", "no_social_heat", "no_social_serve"]:
                all_rows.extend(run_one(mode, who, intensity, rnd))

    cols = ["mode", "who", "round", "tick", "heat_at_leave", "heat",
            "sel_need", "sel_drive", "served", "voice",
            *(f"ref_{n}" for n in HOMEOSTATIC_NEEDS),
            *(f"state_{n}" for n in HOMEOSTATIC_NEEDS)]

    with OUT.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in all_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # 打印每个 mode × who 的 heat 曲线（round 0）
    print("=" * 80)
    print(f"spike source test: {PRESENT_TICKS} settle + {DECAY_TICKS} decay, {N_ROUNDS} rounds")
    print(f"modes: baseline / no_social_heat / no_social_serve")
    print(f"output: {OUT}\n")

    for who in ["author", "stranger"]:
        for mode in ["baseline", "no_social_heat", "no_social_serve"]:
            sub = [r for r in all_rows if r["who"] == who and r["mode"] == mode and r["round"] == 0]
            peak_tick = max(range(DECAY_TICKS), key=lambda k: sub[k]["heat"])
            peak_heat = sub[peak_tick]["heat"]
            last_heat = sub[-1]["heat"]
            # 找第一次 heat 跳增 (后一 tick 比前一 tick 大 0.1 以上)
            jumps = []
            for k in range(1, len(sub)):
                if sub[k]["heat"] - sub[k-1]["heat"] > 0.1:
                    jumps.append((k, round(sub[k-1]["heat"], 4), round(sub[k]["heat"], 4)))
            jump_str = " | ".join(f"tick {j[0]}: {j[1]}→{j[2]}" for j in jumps[:4])
            print(f"{who:10s} {mode:18s}: peak=({peak_tick:2d},{peak_heat:.3f}) last={last_heat:.4f}  jumps: {jump_str}")

        print()

    # 锯齿周期 & 各 need 恢复期对比
    print("── 锯齿周期 vs 各 need 恢复期 ──")
    base = [r for r in all_rows if r["who"] == "author" and r["mode"] == "baseline" and r["round"] == 0]
    jumps = []
    for k in range(1, len(base)):
        if base[k]["heat"] - base[k-1]["heat"] > 0.1:
            jumps.append((k, round(base[k-1]["heat"], 4), round(base[k]["heat"], 4)))
    if len(jumps) >= 2:
        periods = [jumps[i+1][0] - jumps[i][0] for i in range(len(jumps)-1)]
        print(f"baseline author 锯齿 tick: {[j[0] for j in jumps]}")
        print(f"锯齿周期: {periods}")
    else:
        print(f"baseline author: 未观测到第二次锯齿。首次跳 tick {jumps[0] if jumps else 'N/A'}")
        if jumps:
            print(f"首次跳 tick {jumps[0][0]}: heat {jumps[0][1]} → {jumps[0][2]}")

    base0 = base[0]
    print(f"\n各 need 恢复期 (refravtory_minutes baseline, 不含 heat 调制):")
    for n in HOMEOSTATIC_NEEDS:
        init_ref = getattr(Body(age_years=0.5).__class__, "__init__", None)
        pass

    # 直接从 body 看 serve 时的 ref 落地值
    print(f"\n锯齿出现时（tick 29）各 need 的 state + ref（baseline author round 0）:")
    for n in HOMEOSTATIC_NEEDS:
        print(f"  {n:14s}: state={base[29][f'state_'+n]}  ref={base[29][f'ref_'+n]}")

    # 跳增时刻被选的是谁
    if jumps:
        for j_tick, from_h, to_h in jumps[:4]:
            r = base[j_tick]
            print(f"\ntick {j_tick} 跳增: sel_need={r['sel_need']} served={r['served']} state_social={r['state_social']} ref_social={r['ref_social']} state_glucose={r['state_glucose']} ref_glucose={r['ref_glucose']}")


if __name__ == "__main__":
    run()
