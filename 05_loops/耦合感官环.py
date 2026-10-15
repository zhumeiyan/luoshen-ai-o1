"""视听耦合自转 v1.0 —— 两个连续量 + 外部锚点驱动。

去掉阈值。A组和B组是同一个连续量。
嘴动和听觉：两个通道，互耦，外部数据锚定。

循环跑，不停。每圈产出两个数（嘴态、听态），
自己听回（耦合），再转。

外部锚点：真实录像数据（嘴开合度、听觉Δmax），56帧循环。
耦合：嘴 ↔ 听觉，相互驱动。
不判、不升、不收敛检测——先跑起来看它出什么。

纪律：
  - if/then → 若
  - 矛盾体，两个都摆
  - 先有果再找因
  - 可复现 ≠ 真
"""

import json, sys, os, math, struct, time


# ───────────────────────────────────────────
# 一、读真实数据
# ───────────────────────────────────────────

def read_wav_pcm(path):
    with open(path, 'rb') as f:
        f.read(4); f.read(4); f.read(4)
        pcm = None; sr = None; bits = None
        while True:
            ck = f.read(4)
            if len(ck) < 4: break
            cl = struct.unpack('<I', f.read(4))[0]
            if ck == b'fmt ':
                fmt = f.read(cl)
                sr = struct.unpack('<I', fmt[4:8])[0]
                bits = struct.unpack('<H', fmt[14:16])[0]
            elif ck == b'data':
                pcm = f.read(cl); break
            else: f.read(cl)
    return pcm, sr, bits


def decode_pcm(pcm, bits=16):
    n = len(pcm) // 2
    return [s / 32780.0 for s in struct.unpack('<{}h'.format(n), pcm)]


COLS = 64; ROWS = 48
MOUTH_R0 = 23; MOUTH_R1 = 29; MOUTH_C0 = 24; MOUTH_C1 = 34


def build_aud_sparse(sub):
    n = len(sub); bs = max(1, n // COLS)
    zr = []; er = []
    for c in range(COLS):
        s = c*bs; e = min(s+bs, n); blk = sub[s:e]
        if not blk: zr.append(0.0); er.append(0.0); continue
        zc = sum(1 for i in range(1,len(blk)) if blk[i-1]>=0>blk[i])
        sq = sum(x*x for x in blk)/len(blk)
        zr.append(float(zc)); er.append(sq*10000)
    return [zr, er]


def read_ppm(path):
    with open(path,'rb') as f:
        magic = f.readline().strip()
        ln = f.readline().strip()
        while ln.startswith(b'#'): ln = f.readline().strip()
        w,h = map(int, ln.split())
        f.readline(); data = f.read()
    if magic == b'P6':
        px = [(data[i],data[i+1],data[i+2]) for i in range(0,len(data),3)]
        return [[0.299*r+0.587*g+0.114*b for r,g,b in px[y*w:(y+1)*w]] for y in range(h)], w, h
    elif magic == b'P3':
        vs = [float(x) for x in data.split()]
        return [[0.299*vs[(y*w+x)*3]+0.587*vs[(y*w+x)*3+1]+0.114*vs[(y*w+x)*3+2] for x in range(w)] for y in range(h)], w, h


def resize(src, dr=ROWS, dc=COLS):
    if not src or not src[0]: return [[0.0]*dc for _ in range(dr)]
    sr_, sc_ = len(src), len(src[0]); out = []
    for i in range(dr):
        si = i*sr_/dr; i0=int(si); i1=min(i0+1,sr_-1); fy=si-i0
        row = []
        for j in range(dc):
            sj = j*sc_/dc; j0=int(sj); j1=min(j0+1,sc_-1); fx=sj-j0
            a=src[i0][j0]; b=src[i0][j1]; c2=src[i1][j0]; d=src[i1][j1]
            row.append(a*(1-fx)*(1-fy)+b*fx*(1-fy)+c2*(1-fx)*fy+d*fx*fy)
        out.append(row)
    return out


def mouth_openness(frame):
    total = 0.0
    for i in range(MOUTH_R0+1, MOUTH_R1):
        for j in range(MOUTH_C0, MOUTH_C1):
            total += abs(frame[i][j] - frame[i-1][j])
    return total


# ───────────────────────────────────────────
# 二、提取真实锚点
# ───────────────────────────────────────────

def extract_anchors():
    pcm, sr, bits = read_wav_pcm("video_input/audio/speech.wav")
    samples = decode_pcm(pcm, bits)
    spf = int(sr * 0.10); n = len(samples) // spf

    aud_grids = [build_aud_sparse(samples[i*spf:(i+1)*spf]) for i in range(n)]

    vis = []
    for fi in range(n):
        p = os.path.join("video_input/frames", "frame_{:04d}.ppm".format(fi+1))
        if not os.path.exists(p): break
        raw, _, _ = read_ppm(p)
        vis.append(resize(raw, ROWS, COLS))

    n = min(len(aud_grids), len(vis))

    mouth_vals = [mouth_openness(vis[fi]) for fi in range(n)]
    aud_vals = []
    for fi in range(1, n):
        ac = [v for r in aud_grids[fi] for v in r]
        ap = [v for r in aud_grids[fi-1] for v in r]
        aud_vals.append(max(abs(a-b) for a,b in zip(ac,ap)))

    return mouth_vals, aud_vals, n


# ───────────────────────────────────────────
# 三、归一化锚点
# ───────────────────────────────────────────

def normalize(vals):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.5]*len(vals), lo, hi
    return [(v-lo)/(hi-lo) for v in vals], lo, hi


# ───────────────────────────────────────────
# 四、耦合循环
# ───────────────────────────────────────────

def run_loop(mouth_norm, aud_norm, N, total_steps,
             w_mm=0.50, w_ma=0.20, w_me=0.30,
             w_aa=0.50, w_am=0.20, w_ae=0.30,
             log_interval=500, output_path="感官耦合环_trace.jsonl"):
    """耦合两通道闭环跑 total_steps 步。

    每步：
      m_new = w_mm * m_curr + w_ma * a_curr + w_me * ext_mouth[i]
      a_new = w_aa * a_curr + w_am * m_curr + w_ae * ext_aud[i]

    嘴态和听态互驱动。外部锚点周期性注入。
    """
    m = mouth_norm[0]
    a = aud_norm[0] if aud_norm else 0.0

    trace = []
    t0 = time.time()

    for step in range(total_steps):
        i = step % N  # 锚点索引循环

        ext_m = mouth_norm[i]
        ext_a = aud_norm[min(i, len(aud_norm)-1)]

        # 耦合更新
        m_new = w_mm * m + w_ma * a + w_me * ext_m
        a_new = w_aa * a + w_am * m + w_ae * ext_a

        # 记录
        delta_m = m_new - m
        delta_a = a_new - a

        if step % log_interval == 0:
            record = {
                "step": step,
                "i": i,
                "m": round(m, 4),
                "a": round(a, 4),
                "delta_m": round(delta_m, 4),
                "delta_a": round(delta_a, 4),
                "ext_m": round(ext_m, 4),
                "ext_a": round(ext_a, 4),
            }
            trace.append(record)

        m, a = m_new, a_new

    # 最后一条
    trace.append({
        "step": total_steps,
        "i": total_steps % N,
        "m": round(m, 4),
        "a": round(a, 4),
        "delta_m": 0,
        "delta_a": 0,
        "ext_m": round(mouth_norm[total_steps % N], 4),
        "ext_a": round(aud_norm[min(total_steps % N, len(aud_norm)-1)], 4),
        "_final": True
    })

    with open(output_path, 'w') as f:
        for rec in trace:
            f.write(json.dumps(rec) + '\n')

    elapsed = time.time() - t0
    return trace, elapsed


# ───────────────────────────────────────────
# 五、主
# ───────────────────────────────────────────

if __name__ == "__main__":
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50000

    print(f"【视听耦合自转 v1.0】")
    print(f"总步数: {total_steps}")
    print()

    # 锚点
    mouth_vals, aud_vals, N = extract_anchors()
    mouth_norm, m_lo, m_hi = normalize(mouth_vals)
    aud_norm, a_lo, a_hi = normalize(aud_vals)

    print(f"锚点帧数: {N}")
    print(f"嘴开合度  min={m_lo:.1f}  max={m_hi:.1f}")
    print(f"听觉Δmax  min={a_lo:.2f}  max={a_hi:.2f}")
    print()

    # 耦合权重
    w_mm, w_ma, w_me = 0.50, 0.20, 0.30
    w_aa, w_am, w_ae = 0.50, 0.20, 0.30
    print(f"耦合: w_self={w_mm}  w_cross={w_ma}  w_ext={w_me}")
    print()

    trace, elapsed = run_loop(
        mouth_norm, aud_norm, N, total_steps,
        w_mm=w_mm, w_ma=w_ma, w_me=w_me,
        w_aa=w_aa, w_am=w_am, w_ae=w_ae,
        log_interval=500
    )

    print(f"耗时: {elapsed:.2f}s")
    print(f"记录数: {len(trace)}")
    print()

    # 尾态
    final = trace[-1]
    print(f"最终状态: m={final['m']:.4f}  a={final['a']:.4f}")
    print()

    # δ（变化幅度）
    deltas_m = [r['delta_m'] for r in trace if abs(r.get('delta_m',0)) > 1e-9]
    deltas_a = [r['delta_a'] for r in trace if abs(r.get('delta_a',0)) > 1e-9]

    if deltas_m:
        print(f"嘴态δ  mean={sum(abs(d) for d in deltas_m)/len(deltas_m):.6f}")
    if deltas_a:
        print(f"听态δ  mean={sum(abs(d) for d in deltas_a)/len(deltas_a):.6f}")

    # 最后 N 步是否收敛
    last_n = trace[-10:]
    dm_last = [r['delta_m'] for r in last_n]
    da_last = [r['delta_a'] for r in last_n]
    dm_max = max(abs(d) for d in dm_last)
    da_max = max(abs(d) for d in da_last)
    print(f"最后10条记录: |δ_m|_max={dm_max:.6f}  |δ_a|_max={da_max:.6f}")
    print()

    # 一周期内的波动
    period_m = [r['m'] for r in trace[-N:]]
    period_a = [r['a'] for r in trace[-N:]]
    if period_m:
        m_range = max(period_m) - min(period_m)
        a_range = max(period_a) - min(period_a)
        print(f"末周期波动: m_range={m_range:.4f}  a_range={a_range:.4f}")
    print()

    # 嘴态和听态的耦合相关
    ms = [r['m'] for r in trace]
    acs = [r['a'] for r in trace]
    n_c = len(ms)
    mean_m = sum(ms)/n_c
    mean_a = sum(acs)/n_c
    cov = sum((ms[i]-mean_m)*(acs[i]-mean_a) for i in range(n_c))/n_c
    var_m = sum((x-mean_m)**2 for x in ms)/n_c
    var_a = sum((x-mean_a)**2 for x in acs)/n_c
    if var_m > 0 and var_a > 0:
        corr = cov / (var_m ** 0.5 * var_a ** 0.5)
        print(f"嘴态-听态 相关 r = {corr:+.4f}")
    print()

    print("状态：[判读·视听耦合自转]")
    print(f"输出: 感官耦合环_trace.jsonl")
