# -*- coding: utf-8 -*-
"""
宇宙十二 - 洛神赋版  (Universe Twelve · Ode to the Luo Goddess)

Body  : 4x4 = 16 cells.  Each cell {P, F}.  a = min(|P-F|/2, 3).
Grid  : up/down/left/right neighbors.
Pool  : shared energy B (正池).
Cols  : right 2 cols → forward (s=+1), left 2 cols → backward (s=-1).

Per cell per beat
  ΔE = σ + 1.2 · neighbor_a_mean
  E  = E + ΔE
  P' = F + 0.015·sin(E)
  F' = P - 0.010·cos(E)
  (P,F) = (P',F')
  a  = min(|P-F|/2, 3)
  dx = clip( Σ_cell (a·s), ±0.5);  x = clip(x+dx, 0,140)
  cost = 1.28 + 0.30·Σa + 1.2·dx²
  B = min(600, B + 5.0·G(x)·min(1,Σa/3) - cost)
  B ≤ 0 → dead

Cold snap each beat p=0.004  B -= 90
Birth: B=420, E=0, x~U(5,65), P/F all-4-rows copied from parent's random row + N(0,0.03).
Parent chosen by 落盘行数 (rows where a is small → settled) weighting.

σ = 0.3 + light + sound           (per cell)
  light  = 0.5·exp(-((x-75)/70)²) · (1+0.55·d·s)
  sound  = 0.6·( Σj Mij·Aj + 0.55·s·Σj Mij·sign(xj-x)·Aj )
  d = +1 if x<75 else -1
  s = +1 right cols (col 2,3), -1 left cols (col 0,1)
  Mij = exp(-|xi-xj|/25),  Mii=0,  Aj = Σa of body j
  G(x) = exp(-((x-75)/30)²)

Run : 8 seeds (2100..2107), 16 gens, 24 bodies/gen, ≤520 beats/gen.
Two modes: 亮 (light on) & 暗 (light off, sound on).

Report
  1. per-seed digest
  2. same-seed-twice → digest identical? (engineering)
  3. 远生儿 net displacement: 前段(1-4) mean±se, 末段(13-16) mean±se
  4. 进化 J1: paired t 末4代 - 前1/4代 (per seed, on displacement)
  5. 换能 J2: paired t 亮 - 暗 (mean displacement per seed)
  6. per-seed digests (亮/暗)
"""
import os, sys, math, random, hashlib, time

t0 = time.time()

SEEDS    = [2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107]
N_GENS   = 16
N_BODIES = 24
MAX_BEATS = 520
GRID     = 4
N_CELLS  = GRID * GRID   # 16
FOOD_X   = 75.0
LIGHT_SD = 70.0
FOOD_SD  = 30.0

# per-cell s: column 0,1 = -1 ; column 2,3 = +1
CELL_S = []
for i in range(N_CELLS):
    c = i % GRID
    CELL_S.append(1.0 if c >= 2 else -1.0)

# child P/F copied from ONE random parent row → all 4 child rows


class Body:
    __slots__ = ('B', 'E', 'P', 'F', 'a', 'x', 'birth_x', 'last_dx')

    def __init__(self, B_init=420.0, x_init=None, P_init=None, F_init=None):
        self.B = B_init
        self.E = [0.0] * N_CELLS
        if P_init is None:
            P_init = [random.uniform(-0.1, 0.1) for _ in range(16)]
        if F_init is None:
            F_init = [random.uniform(-0.1, 0.1) for _ in range(16)]
        self.P = list(P_init)
        self.F = list(F_init)
        self.a = [min(abs(self.P[i] - self.F[i]) / 2.0, 3.0) for i in range(N_CELLS)]
        self.x = x_init if x_init is not None else random.uniform(5.0, 65.0)
        self.birth_x = self.x
        self.last_dx = 0.0

    @property
    def sum_a(self):
        s = 0.0
        for v in self.a:
            s += v
        return s

    @property
    def settled_rows(self):
        """落盘行数 — rows where |P-F| is small (relaxed/settled)."""
        cnt = 0
        thr = 0.5
        for r in range(GRID):
            s = 0.0
            for c in range(GRID):
                s += abs(self.P[r * GRID + c] - self.F[r * GRID + c])
            s /= GRID
            if s < thr:
                cnt += 1
        return cnt

    def make_child(self):
        """Child: 16 cells all from parent's ONE random row + N(0,0.03)."""
        prow = random.randint(0, GRID - 1)
        nP, nF = [], []
        for r in range(GRID):
            for c in range(GRID):
                pi = prow * GRID + c
                nP.append(self.P[pi] + random.gauss(0.0, 0.03))
                nF.append(self.F[pi] + random.gauss(0.0, 0.03))
        return Body(B_init=420.0, x_init=None, P_init=nP, F_init=nF)


def neigh_mean(a, idx):
    """Mean a of up/down/left/right grid neighbors from snapshot."""
    r = idx // GRID
    c = idx % GRID
    t = 0.0
    n = 0
    if r > 0:
        t += a[(r - 1) * GRID + c]; n += 1
    if r < GRID - 1:
        t += a[(r + 1) * GRID + c]; n += 1
    if c > 0:
        t += a[r * GRID + (c - 1)]; n += 1
    if c < GRID - 1:
        t += a[r * GRID + (c + 1)]; n += 1
    return t / n if n else 0.0


def weather(x, d, s_cell, other_a_list, other_x_list):
    """Per-cell weather σ."""
    # light
    g_l = math.exp(-((x - FOOD_X) / LIGHT_SD) ** 2)
    light = 0.5 * g_l * (1.0 + 0.55 * d * s_cell)

    # sound
    if not other_a_list:
        sound = 0.0
    else:
        s1 = 0.0   # Σ M·Aj
        s2 = 0.0   # Σ M·sign(xj-x)·Aj
        for aj, xj in zip(other_a_list, other_x_list):
            m = math.exp(-abs(x - xj) / 25.0)
            s1 += m * aj
            if xj > x:
                s2 += m * aj
            elif xj < x:
                s2 -= m * aj
        sound = 0.6 * (s1 + 0.55 * s_cell * s2)

    return 0.3 + light + sound


def G_food(x):
    return math.exp(-((x - FOOD_X) / FOOD_SD) ** 2)


def run_generation(bodies, light_on=True):
    """Run one generation: each alive body steps up to MAX_BEATS beats."""
    for beat in range(MAX_BEATS):
        alive = False
        for b in bodies:
            if b.B > 0.0:
                alive = True
                break
        if not alive:
            break

        # snapshot a & x before updating any body
        a_snap = [tuple(b.a) for b in bodies]
        x_snap = [b.x for b in bodies]

        for bi, b in enumerate(bodies):
            if b.B <= 0.0:
                continue

            d = 1.0 if b.x < FOOD_X else -1.0

            # gather other alive bodies' aggregations
            oa, ox = [], []
            for oi, ob in enumerate(bodies):
                if oi == bi or ob.B <= 0.0:
                    continue
                sa = 0.0
                for v in a_snap[oi]:
                    sa += v
                oa.append(sa)
                ox.append(x_snap[oi])

            new_P = []
            new_F = []
            new_E = []
            new_a = []
            for i in range(N_CELLS):
                s_c = CELL_S[i]
                if light_on:
                    sigma = weather(b.x, d, s_c, oa, ox)
                else:
                    # dark: no light
                    if not oa:
                        sound = 0.0
                    else:
                        s1 = 0.0
                        s2 = 0.0
                        for aj, xj in zip(oa, ox):
                            m = math.exp(-abs(b.x - xj) / 25.0)
                            s1 += m * aj
                            if xj > b.x:
                                s2 += m * aj
                            elif xj < b.x:
                                s2 -= m * aj
                        sound = 0.6 * (s1 + 0.55 * s_c * s2)
                    sigma = 0.3 + sound

                nm = neigh_mean(a_snap[bi], i)
                e_new = b.E[i] + sigma + 1.2 * nm
                p_new = b.F[i] + 0.015 * math.sin(e_new)
                f_new = b.P[i] - 0.010 * math.cos(e_new)
                a_new = abs(p_new - f_new) / 2.0
                if a_new > 3.0:
                    a_new = 3.0
                new_P.append(p_new)
                new_F.append(f_new)
                new_E.append(e_new)
                new_a.append(a_new)

            b.P = new_P
            b.F = new_F
            b.E = new_E
            b.a = new_a

            # output dx from all cells
            dx_raw = 0.0
            for i in range(N_CELLS):
                dx_raw += b.a[i] * CELL_S[i]
            if dx_raw > 0.5:
                dx_raw = 0.5
            elif dx_raw < -0.5:
                dx_raw = -0.5
            b.last_dx = dx_raw
            b.x += dx_raw
            if b.x < 0.0:
                b.x = 0.0
            elif b.x > 140.0:
                b.x = 140.0

            cost = 1.28 + 0.30 * b.sum_a + 1.2 * dx_raw * dx_raw
            food = 5.0 * G_food(b.x) * min(1.0, b.sum_a / 3.0)
            b.B += food - cost
            if b.B > 600.0:
                b.B = 600.0

            if random.random() < 0.004:
                b.B -= 90.0
            if b.B <= 0.0:
                b.B = 0.0

    return bodies


def make_next_gen(bodies):
    """Select parents by settled_rows; produce N_BODIES children."""
    survivors = [b for b in bodies if b.B > 0.0]
    if not survivors:
        return [Body() for _ in range(N_BODIES)]

    w = [max(0.01, b.settled_rows) for b in survivors]
    tw = sum(w)

    children = []
    for _ in range(N_BODIES):
        r = random.uniform(0.0, tw)
        acc = 0.0
        chosen = survivors[-1]
        for j, ww in enumerate(w):
            acc += ww
            if r <= acc:
                chosen = survivors[j]
                break
        children.append(chosen.make_child())
    return children


def gen_stat(bodies, gen_idx):
    alive = [b for b in bodies if b.B > 0.0]
    n_a = len(alive)
    if n_a == 0:
        return {'gen': gen_idx, 'alive': 0, 'disp_mean': 0.0,
                'x_mean': 0.0, 'sum_a': 0.0, 'landed': 0.0}
    disps = [(b.x - b.birth_x) for b in bodies]
    disp_m = sum(disps) / len(disps)
    xs = [b.x for b in alive]
    sa = sum(b.sum_a for b in alive)
    sd = sum(b.settled_rows for b in alive)
    return {
        'gen': gen_idx,
        'alive': n_a,
        'disp_mean': disp_m,
        'x_mean': sum(xs) / n_a,
        'sum_a': sa / n_a,
        'landed': sd / n_a,
    }


def world_digest(stats):
    """Short + long digest for one world run."""
    parts = []
    for s in stats:
        if s['alive'] == 0:
            parts.append("G%d:×" % s['gen'])
        else:
            parts.append("G%d:A%d/d%.1f/l%.1f" % (
                s['gen'], s['alive'], s['disp_mean'], s['landed']))
    raw = "|".join(parts)
    h = hashlib.md5(raw.encode('utf-8')).hexdigest()[:10]
    return h, raw


def run_world(seed, light_on=True):
    random.seed(seed)
    bodies = [Body() for _ in range(N_BODIES)]
    stats = []
    digests = []
    for g in range(N_GENS):
        bodies = run_generation(bodies, light_on=light_on)
        st = gen_stat(bodies, g + 1)
        stats.append(st)
        if g < N_GENS - 1:
            bodies = make_next_gen(bodies)
    return stats


def mean_se(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in vals) / (n - 1)
    return m, math.sqrt(v / n)


def paired_diff_t(group_a, group_b):
    """Paired t over aligned (a_i - b_i). Returns (t_stat, diff_mean, diff_se)."""
    n = len(group_a)
    if n == 0:
        return 0.0, 0.0, 0.0
    ds = [group_a[i] - group_b[i] for i in range(n)]
    m = sum(ds) / n
    if n < 2:
        return m, m, 0.0
    ss = sum((d - m) ** 2 for d in ds) / (n - 1)
    se = math.sqrt(ss / n)
    if se < 1e-12:
        return 9999.0, m, se
    return m / se, m, se


# =========================================================
out_path = os.path.join(os.getcwd(), 'luoshen_universe12.txt')
OUT = open(out_path, 'w', encoding='utf-8')


def O(line=''):
    OUT.write(line + '\n')


O('宇宙十二 - 洛神赋版  仿真结果')
O('=' * 72)
O('规格 : 16格(4x4), %d代 x %d体, %d拍上限' % (N_GENS, N_BODIES, MAX_BEATS))
O('天气 : σ = 0.3 + light + sound')
O('  light = 0.5·exp(-((x-75)/70)²)·(1+0.55·d·s)')
O('  sound = 0.6·[ΣM·Aj + 0.55·s·ΣM·sign(xj-x)·Aj]')
O('  d = +1 if x<75 else -1;  s = +1(col2,3), -1(col0,1)')
O('  Mij = exp(-|xi-xj|/25)  (Mii=0)')
O('食物 : G(x)=exp(-((x-75)/30)²)  电费: 1.28 + 0.30·Σa + 1.2·dx²')
O('生死 : B=min(600, B+5·G·min(1,Σa/3)-cost); B≤0死')
O('寒潮 : p=0.004/拍  B-=90')
O('出生 : B=420, E=0, x~U(5,65), P/F copy parent random row + N(0,0.03)')
O('选择 : 亲代按落盘行数加权抽')
O('种子 : %s' % str(SEEDS))
O('')

# ---- main computation ----
all_stats = {}    # (seed, light_bool) -> [gen_stat × 16]

for seed in SEEDS:
    for lf in [True, False]:
        all_stats[(seed, lf)] = run_world(seed, light_on=lf)

# ---- per-seed digest table ----
O('=' * 72)
O('【§1+§6】每个种子的世界摘要 (digest)')
O('-' * 50)
O('         亮宇宙                               暗宇宙')
O('seed  disp-md5    alive-last  disp-last | disp-md5    alive-last  disp-last')
digest_map = {}
for seed in SEEDS:
    for lf in [True, False]:
        h, raw = world_digest(all_stats[(seed, lf)])
        digest_map[(seed, lf)] = (h, raw)

for seed in SEEDS:
    stL = all_stats[(seed, True)][-1]
    stD = all_stats[(seed, False)][-1]
    hL, _ = digest_map[(seed, True)]
    hD, _ = digest_map[(seed, False)]
    O('%4d  %s  A=%-3d d=%-6.2f | %s  A=%-3d d=%-6.2f' % (
        seed, hL, stL['alive'], stL['disp_mean'],
        hD, stD['alive'], stD['disp_mean']))

O('')
O('【详细 digest raw lines】')
for seed in SEEDS:
    for lf in [True, False]:
        lab = '亮' if lf else '暗'
        h, raw = digest_map[(seed, lf)]
        O('seed=%d [%s] %s  %s' % (seed, lab, h, raw))

# ---- §2 engineering reproducibility ----
O('')
O('=' * 72)
O('【§2】同种子跑两遍 digest 是否一致 (工程层)')
O('-' * 50)
eng_all_ok = True
for seed in SEEDS:
    for lf in [True, False]:
        lab = '亮' if lf else '暗'
        # run two independent re-seeded worlds
        sA = run_world(seed, light_on=lf)
        hA, _ = world_digest(sA)
        sB = run_world(seed, light_on=lf)
        hB, _ = world_digest(sB)
        ok = (hA == hB)
        if not ok:
            eng_all_ok = False
        O('  seed=%d [%s] run1=%s  run2=%s  %s' % (
            seed, lab, hA, hB, 'OK' if ok else 'DIFF!'))
O('')
O('工程可复现性: %s' % ('全部一致' if eng_all_ok else '⚠ 存在不一致'))

# ---- §3 远生儿净位移 ----
O('')
O('=' * 72)
O('【§3】远生儿净位移')
O('前段=gen 1..4 (末段=gen 13..16)')
O('报告: mean ± se (在同段内各gen取disp_mean, 再对8 seed聚合)')
O('-' * 50)
O('               前段(1-4)             末段(13-16)')
O('seed        亮mean±se  暗mean±se  | 亮mean±se  暗mean±se')

disps_early_L = {}
disps_late_L  = {}
disps_early_D = {}
disps_late_D  = {}
for seed in SEEDS:
    stL = all_stats[(seed, True)]
    stD = all_stats[(seed, False)]
    eL = [stL[g]['disp_mean'] for g in range(4)]
    lL = [stL[g]['disp_mean'] for g in range(12, 16)]
    eD = [stD[g]['disp_mean'] for g in range(4)]
    lD = [stD[g]['disp_mean'] for g in range(12, 16)]
    disps_early_L[seed] = eL
    disps_late_L[seed]  = lL
    disps_early_D[seed] = eD
    disps_late_D[seed]  = lD

    emL, esL = mean_se(eL)
    lmL, lsL = mean_se(lL)
    emD, esD = mean_se(eD)
    lmD, lsD = mean_se(lD)
    O('%4d  %7.2f±%5.2f %7.2f±%5.2f | %7.2f±%5.2f %7.2f±%5.2f' % (
        seed, emL, esL, emD, esD, lmL, lsL, lmD, lsD))

# aggregated over 8 seeds
agg_el = [mean_se(v)[0] for v in disps_early_L.values()]
agg_ll = [mean_se(v)[0] for v in disps_late_L.values()]
agg_ed = [mean_se(v)[0] for v in disps_early_D.values()]
agg_ld = [mean_se(v)[0] for v in disps_late_D.values()]
mel, sel = mean_se(agg_el)
mll, sll = mean_se(agg_ll)
med, sed = mean_se(agg_ed)
mld, sld = mean_se(agg_ld)
O('-----')
O('跨seed均值:')
O('  亮前段: %.2f ± %.2f   亮末段: %.2f ± %.2f' % (mel, sel, mll, sll))
O('  暗前段: %.2f ± %.2f   暗末段: %.2f ± %.2f' % (med, sed, mld, sld))

# ---- §4 进化J1 ----
O('')
O('=' * 72)
O('【§4】进化 J1 = 末4代 vs 前1/4代 配对t (远生儿净位移)')
O('配對: (gen13-gen1, gen14-gen2, gen15-gen3, gen16-gen4) per seed')
O('-' * 50)
O('seed   亮t值  亮diff  暗t值  暗diff')
j1_tL, j1_dL = [], []
j1_tD, j1_dD = [], []
for seed in SEEDS:
    eL = disps_early_L[seed]
    lL = disps_late_L[seed]
    eD = disps_early_D[seed]
    lD = disps_late_D[seed]
    tL, mL, _ = paired_diff_t(lL, eL)
    tD, mD, _ = paired_diff_t(lD, eD)
    j1_tL.append(tL); j1_dL.append(mL)
    j1_tD.append(tD); j1_dD.append(mD)
    O('%4d  %7.3f  %7.3f  %7.3f  %7.3f' % (seed, tL, mL, tD, mD))

mtL, stL = mean_se(j1_tL)
mdL, sdL = mean_se(j1_dL)
mtD, stD = mean_se(j1_tD)
mdD, sdD = mean_se(j1_dD)
O('-----')
O('亮宇宙 J1: t均值 %.3f ± %.3f, diff均值 %.3f ± %.3f' % (mtL, stL, mdL, sdL))
O('暗宇宙 J1: t均值 %.3f ± %.3f, diff均值 %.3f ± %.3f' % (mtD, stD, mdD, sdD))

# ---- §5 换能J2 ----
O('')
O('=' * 72)
O('【§5】换能 J2 = 亮宇宙 - 暗宇宙 配对t (远生儿净位移末段-前段进化差)')
O('变量 per seed: Δ进化 = (亮末4代-亮前1/4代) - (暗末4代-暗前1/4代)')
O('-' * 50)
j2_vals = []
for seed in SEEDS:
    eL = disps_early_L[seed]; lL = disps_late_L[seed]
    eD = disps_early_D[seed]; lD = disps_late_D[seed]
    evoL = [lL[i] - eL[i] for i in range(4)]
    evoD = [lD[i] - eD[i] for i in range(4)]
    j2 = sum(evoL) / 4.0 - sum(evoD) / 4.0
    j2_vals.append(j2)

mj2, sj2 = mean_se(j2_vals)
if len(j2_vals) >= 2:
    ss = sum((v - mj2) ** 2 for v in j2_vals)
    se_j2 = math.sqrt(ss / (len(j2_vals) - 1) / len(j2_vals))
    t_j2 = mj2 / se_j2 if se_j2 > 1e-12 else 9999.0
else:
    t_j2 = 0.0
O('seed  Δ进化(亮-暗)')
for i, seed in enumerate(SEEDS):
    O('%4d  %8.3f' % (seed, j2_vals[i]))
O('-----')
O('Δ进化均值: %.3f ± %.3f' % (mj2, sj2))
O('J2 t = %.3f  (df=%d)' % (t_j2, len(j2_vals) - 1))

O('')
O('=' * 72)
O('跑完. 用时 %.1f 秒' % (time.time() - t0))
OUT.close()
print('WROTE', out_path, 'time=%.1fs' % (time.time() - t0))
