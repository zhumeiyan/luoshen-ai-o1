# [已撤回] 2026-09-19 推演者归档：旧引用，新架构已换
# [残留英文] 2026-09-20 推演者注：文件内含 luminance/sound/force/temperature 英文环境键名。已归档，不清理。
"""test_heat_decay.py — 人走之后，_social_heat 往哪走。

两段：
  A. pure decay — 关掉 presence / 关掉 select-serve，只跑 body.tick() 的 _social_heat *= 0.85。
     看热本身是否 slowly drop，积着不走，或一下归零。
  B. full sim — 关掉 presence 但保持 select-serve。看 ref 到期→serve("social")→+0.7 的锯齿。

每种跑 author (intensity=0.94) 与 stranger (intensity=0.86)。
不预设判据。只报原始值。
"""

import sys, math
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from age_homeo import Body

OUT = DIR / "heat_decay_trace.tsv"
PRESENT_TICKS = 30
DECAY_TICKS = 40
N_ROUNDS = 3

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


def run():
    rows = []
    global_tick = 0

    for rnd in range(N_ROUNDS):
        for who, intensity in INTENSITY.items():
            body = Body(age_years=0.5, tick_minutes=1.0)

            # Phase 1 — settle
            settle(body, intensity, PRESENT_TICKS)
            heat_at_leave = body._social_heat

            # Phase 2A — pure: only body.tick()
            for k in range(DECAY_TICKS):
                rows.append({"mode": "pure", "round": rnd, "who": who, "tick": k,
                             "heat_at_leave": round(heat_at_leave, 5),
                             "heat": round(body._social_heat, 6)})
                body.tick()

            # re-settle for 2B
            body2 = Body(age_years=0.5, tick_minutes=1.0)
            settle(body2, intensity, PRESENT_TICKS)
            heat2 = body2._social_heat

            # Phase 2B — full loop: body.tick() + select/serve (no presence)
            for k in range(DECAY_TICKS):
                body2.tick()
                n, _ = body2.select()
                if n:
                    body2.serve(n)
                rows.append({"mode": "full", "round": rnd, "who": who, "tick": k,
                             "heat_at_leave": round(heat2, 5),
                             "heat": round(body2._social_heat, 6)})

    cols = ["mode", "round", "who", "tick", "heat_at_leave", "heat"]
    with OUT.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # summary
    print("=" * 78)
    print(f"heat decay: phases=settle {PRESENT_TICKS}t + decay {DECAY_TICKS}t, {N_ROUNDS} rounds, bodies fresh each round")
    print(f"decays only via body._social_heat *= 0.85 per tick (half-life={math.log(0.5)/math.log(0.85):.2f}t ≈ 4.3t)")
    print(f"output: {OUT}\n")

    for mode in ["pure", "full"]:
        print(f"── {mode} ──")
        print(f"  {'tick':>4s} | {'author':>20s} | {'stranger':>20s} | note")
        print(f"  {'':>4s} | min   mean  max     | min   mean  max     |")
        for k in range(DECAY_TICKS):
            a = [r["heat"] for r in rows if r["mode"] == mode and r["who"] == "author" and r["tick"] == k]
            s = [r["heat"] for r in rows if r["mode"] == mode and r["who"] == "stranger" and r["tick"] == k]
            amean = sum(a)/len(a); smean = sum(s)/len(s)
            note = ""
            if k == 0: note = "← leave"
            elif k == 4: note = f"half-life×1"  # heat ≈ halved
            elif k == 13: note = f"≈3 half-lives → ~12.5%"
            print(f"  {k:4d} | {min(a):.3f} {amean:.3f} {max(a):.3f} | {min(s):.3f} {smean:.3f} {max(s):.3f} | {note}")
        print()


if __name__ == "__main__":
    run()
