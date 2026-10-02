#!/usr/bin/env python3
"""
yinyang_evo_v20 — v17o验证后的正推：容量测试
================================================
关键假设：v17o的0.9361天花板可能是n=11的参数容量极限
测试方法：加大网络(n=50)+加大演化(50org×1500gen)

架构保持v17o的核心（双通路+蛋白折叠+Verifier），不折腾额外模块
只加一项：tanh替换为可学习激活（每个节点独立参数化的leaky系数）
"""
import random, math, hashlib, copy, time
import numpy as np

N_CHANNELS = 8; MAX_NODES = 100; N_REGIONS = 9; DELAY = 3

class Pool:
    def __init__(self, seed=42):
        self.seed = seed; self.r = self._mk(seed); self._r()
    def _mk(self, s):
        rng = random.Random(s)
        return [{'id': i, 'sigma': round(0.05 + 0.64 * i / 8, 2), 'proprio': round(i / 8, 2),
                 'freq': round(0.03 + 0.12 * rng.random(), 3), 'phase': round(rng.uniform(0, 2 * math.pi), 3),
                 'type': ['sin', 'logistic', 'lorenz', 'vdP', 'saw', 'rossler', 'henon', 'tent', 'cubic'][i],
                 '_s': 0.3, '_t': 0, '_p': 0.0} for i in range(N_REGIONS)]
    def _r(self):
        for r in self.r: r['_s'] = 0.3; r['_t'] = 0; r['_p'] = 0.0
    def reset(self): self._r()
    def step(self):
        for r in self.r: r['_t'] += 1; r['_p'] += r['freq']
    def sense(self, rid):
        r = self.r[rid]; t = r['_t']; ph = r['_p']
        nv = random.Random(hash((self.seed, rid, t)) % (2**31)).gauss(0, 1)
        n = r['sigma'] * nv; s = r['_s']; rt = r['type']
        if rt == 'sin': v = math.sin(ph + r['phase'])
        elif rt == 'logistic': v = 4*s*(1-s) if 0<s<1 else 0.5
        elif rt == 'lorenz': v = 0.5 + 0.3*math.tanh(s)
        elif rt == 'vdP': v = s - 0.1*(s**3 - s)
        elif rt == 'saw': v = (ph + r['phase']) % 1.0
        elif rt == 'rossler': v = 0.5*math.sin(0.7*ph + s)
        elif rt == 'henon': v = 1 - 1.4*s**2 + 0.3*s
        elif rt == 'tent': v = 2*s if s < 0.5 else 2*(1-s)
        else: v = s**3 - s
        s2 = max(-2, min(2, v + n)); r['_s'] = s2
        return [s2, math.sin(s2*2.5) if abs(s2)<1.5 else 0, math.tanh(s2),
                s2**2*(1 if s2>0 else -1), abs(s2), math.cos(s2*1.7),
                (1 if s2>0 else -1)*math.sqrt(abs(s2)) if abs(s2)<2 else 0, r['proprio']]

def activation(x, leak=0.01):
    """Leaky-tanh: 在饱和区保留微小梯度"""
    if x > 1.5: return 1.5 + leak * (x - 1.5)
    elif x < -1.5: return -1.5 + leak * (x + 1.5)
    else: return math.tanh(x)

class WM:
    def __init__(self, g):
        self.g = g; n = g['n']
        self.h = [0.0]*n; self.prev_h = [0.0]*n
        self.db_history = []; self.input_vec = []; self.err_history = []
    
    def build_input(self, sens):
        self.db_history.append(sens[:])
        if len(self.db_history) > DELAY: self.db_history.pop(0)
        while len(self.db_history) < DELAY: self.db_history.insert(0, [0.0]*N_CHANNELS)
        result = sens[:]
        for frame in self.db_history: result.extend(frame)
        self.input_vec = result[:]
        return result
    
    def fwd(self, sens, update=True):
        x = self.build_input(sens)
        g = self.g; n = g['n']; dec = g['dt'][0]; bb = 0
        nh = [0.0] * n
        for i in range(n):
            a = 0.0
            nn = min(len(g['sw']), len(x))
            for j in range(nn):
                if i < len(g['sw'][j]): a += g['sw'][j][i] * x[j]
            for k in range(min(n, len(g['rw']))):
                if i < len(g['rw'][k]): a += g['rw'][k][i] * self.h[k]
            leak = g['leak'][i] if 'leak' in g and i < len(g['leak']) else 0.01
            nh[i] = activation(a, leak)
        if update:
            for i in range(n): self.h[i] = (1-dec) * nh[i] + dec * self.h[i]
        
        pred = [0.0] * N_CHANNELS
        for i in range(n):
            for c in range(N_CHANNELS):
                if i < len(g['pw']) and c < len(g['pw'][i]):
                    pred[c] += self.h[i] * (g['pw'][i][c] + g['pb'][c])
        n_skip = min(len(g['skip']), len(x))
        for j in range(n_skip):
            for c in range(N_CHANNELS):
                if c < len(g['skip'][j]): pred[c] += g['skip'][j][c] * x[j]
        if update: self.prev_h = self.h[:]
        return pred
    
    def protein_fold(self):
        g = self.g; n = g['n']; fold_lr = g['fold_lr']
        for k in range(n):
            predicted = 0.0
            for i in range(n):
                if k < len(g['rw']) and i < len(g['rw'][k]):
                    predicted += g['rw'][k][i] * self.prev_h[i]
            if abs(predicted) > 0.01:
                leak = g['leak'][k] if 'leak' in g and k < len(g['leak']) else 0.01
                predicted = activation(predicted, leak)
                err = self.h[k] - predicted
                for i in range(n):
                    if k < len(g['rw']) and i < len(g['rw'][k]):
                        delta = fold_lr * self.prev_h[i] * err
                        g['rw'][k][i] += delta
                        g['rw'][k][i] = max(-1.2, min(1.2, g['rw'][k][i]))
    
    def learn(self, pred, actual):
        g = self.g; n = g['n']; lr = g['lr']
        errs = [actual[j] - pred[j] for j in range(N_CHANNELS)]
        mse = sum(e*e for e in errs)/N_CHANNELS
        self.err_history.append(mse)
        for i in range(min(n, len(g['pw']))):
            for c in range(N_CHANNELS):
                if c < len(g['pw'][i]):
                    g['pw'][i][c] += lr * self.h[i] * errs[c]
                    g['pw'][i][c] = max(-3, min(3, g['pw'][i][c]))
        for c in range(N_CHANNELS):
            g['pb'][c] += lr * 1.5 * errs[c]
            g['pb'][c] = max(-5, min(5, g['pb'][c]))
        x = self.input_vec; nn = min(len(g['skip']), len(x))
        for j in range(nn):
            for c in range(N_CHANNELS):
                if c < len(g['skip'][j]):
                    g['skip'][j][c] += lr * 0.5 * x[j] * errs[c]
                    g['skip'][j][c] = max(-2, min(2, g['skip'][j][c]))
        self.protein_fold()
        return mse
    
    def reset(self):
        self.h = [0.0]*len(self.h); self.prev_h = [0.0]*len(self.prev_h)
        self.db_history = []; self.input_vec = []; self.err_history = []

class Org:
    def __init__(self, gn=None):
        self.g = gn or self.rg(); self.wm = WM(self.g)
        self.alive = True; self.age = 0; self._ib = 1.0
    
    def rg(self):
        n = random.randint(20, 60)  # 更大网络
        input_size = N_CHANNELS * (1 + DELAY)
        g = {
            'n': n,
            'sw': [[random.uniform(-0.3, 0.3) for _ in range(n)] for __ in range(input_size)],
            'pw': [[random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)] for __ in range(n)],
            'rw': [[random.uniform(-0.03, 0.03) for _ in range(n)] for __ in range(n)],
            'skip': [[random.uniform(-0.2, 0.2) for _ in range(N_CHANNELS)] for __ in range(input_size)],
            'pb': [random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)],
            'dt': [random.uniform(0.5, 0.9)],
            'lr': random.uniform(0.04, 0.12),
            'fold_lr': random.uniform(0.01, 0.06),
            'leak': [random.uniform(0.005, 0.05) for _ in range(n)],  # 每个节点独立的leak
        }
        for i in range(n): g['rw'][i][i] = random.uniform(0.4, 1.0)
        return g

class Engine:
    def __init__(self, n_org=40, n_gen=1500, seed=42, eval_ticks=350):
        self.n_org = n_org; self.n_gen = n_gen; self.eval_ticks = eval_ticks
        self.pop = [Org() for _ in range(n_org)]
        self.best_ever = 0.0
        self.best_g = None
        self.log = []
        random.seed(seed)
    
    def eval_org(self, org, ticks=None):
        if ticks is None: ticks = self.eval_ticks
        pk = Pool(seed=42); pk.reset()
        org.wm.reset()
        cr = random.randint(0, 8)
        for _ in range(30):
            s = pk.sense(cr); org.wm.fwd(s, update=True); pk.step()
        total_mse = 0.0
        for t in range(ticks):
            s = pk.sense(cr); pred = org.wm.fwd(s, update=True); pk.step()
            ns = pk.sense(cr); pk.step()
            mse = sum((ns[j]-pred[j])**2 for j in range(N_CHANNELS))/N_CHANNELS
            org.wm.learn(pred, ns)
            total_mse += mse
            if random.random() < 0.08: cr = random.randint(0, 8)
        return 1.0 / (1.0 + total_mse / ticks)
    
    def xover(self, a, b):
        n = max(a['n'], b['n']); ins = N_CHANNELS * (1 + DELAY)
        ch = {'n': n, 'sw': [], 'pw': [], 'rw': [], 'skip': [],
              'pb': a['pb'][:] if random.random()<0.5 else b['pb'][:],
              'dt': a['dt'][:] if random.random()<0.5 else b['dt'][:],
              'lr': a['lr'] if random.random()<0.5 else b['lr'],
              'fold_lr': a['fold_lr'] if random.random()<0.5 else b['fold_lr']}
        
        # sw: ins × n
        for j in range(ins):
            src = a if random.random() < 0.5 else b
            row = []
            for i in range(n):
                si = min(i, src['n'] - 1)
                sj = min(j, len(src['sw']) - 1)
                if sj >= 0 and si < len(src['sw'][sj]):
                    row.append(src['sw'][sj][si])
                else:
                    row.append(0.0)
            ch['sw'].append(row)
        
        # pw: n × N_CHANNELS
        for i in range(n):
            src = a if random.random() < 0.5 else b
            si = min(i, src['n'] - 1)
            if si < len(src['pw']) and src['pw'][si]:
                ch['pw'].append(src['pw'][si][:N_CHANNELS])
            else:
                ch['pw'].append([0.0] * N_CHANNELS)
        
        # rw: n × n
        for i in range(n):
            src = a if random.random() < 0.5 else b
            si = min(i, src['n'] - 1)
            row = []
            for k in range(n):
                sk = min(k, src['n'] - 1)
                if si < len(src['rw']) and sk < len(src['rw'][si]):
                    row.append(src['rw'][si][sk])
                else:
                    row.append(0.0)
            ch['rw'].append(row)
        
        # skip: ins × N_CHANNELS
        for j in range(ins):
            src = a if random.random() < 0.5 else b
            sj = min(j, len(src['skip']) - 1)
            if sj >= 0:
                ch['skip'].append(src['skip'][sj][:N_CHANNELS])
            else:
                ch['skip'].append([0.0] * N_CHANNELS)
        
        # leak
        ch['leak'] = []
        for i in range(n):
            if i < len(a.get('leak', [])) and random.random() < 0.5:
                ch['leak'].append(a['leak'][i])
            elif i < len(b.get('leak', [])):
                ch['leak'].append(b['leak'][i])
            else:
                ch['leak'].append(0.02)
        
        return ch
    
    def mutate(self, g, rate=0.04):
        n = g['n']
        for j in range(len(g['sw'])):
            for i in range(len(g['sw'][j])):
                if random.random() < rate:
                    g['sw'][j][i] += random.gauss(0, 0.05)
                    g['sw'][j][i] = max(-3, min(3, g['sw'][j][i]))
        for i in range(n):
            for c in range(N_CHANNELS):
                if i < len(g['pw']) and c < len(g['pw'][i]) and random.random() < rate:
                    g['pw'][i][c] += random.gauss(0, 0.05)
                    g['pw'][i][c] = max(-3, min(3, g['pw'][i][c]))
        for i in range(n):
            for k in range(n):
                if i < len(g['rw']) and k < len(g['rw'][i]) and random.random() < rate * 0.7:
                    g['rw'][i][k] += random.gauss(0, 0.03)
                    g['rw'][i][k] = max(-1.5, min(1.5, g['rw'][i][k]))
        for j in range(len(g['skip'])):
            for c in range(N_CHANNELS):
                if c < len(g['skip'][j]) and random.random() < rate:
                    g['skip'][j][c] += random.gauss(0, 0.05)
                    g['skip'][j][c] = max(-2, min(2, g['skip'][j][c]))
        if random.random() < 0.3:
            idx = random.randint(0, n-1)
            g['leak'][idx] = max(0.001, min(0.1, g['leak'][idx] + random.gauss(0, 0.005)))
        if random.random() < 0.2:
            g['lr'] = max(0.02, min(0.2, g['lr'] + random.gauss(0, 0.015)))
            g['fold_lr'] = max(0.005, min(0.08, g['fold_lr'] + random.gauss(0, 0.01)))
        return g
    
    def evolve(self):
        st = time.time()
        for gen in range(self.n_gen):
            fits = [self.eval_org(o) for o in self.pop]
            bi = fits.index(max(fits))
            bf = fits[bi]; af = sum(fits)/len(fits)
            if bf > self.best_ever:
                self.best_ever = bf; self.best_g = copy.deepcopy(self.pop[bi].g)
            
            if gen % 50 == 0 or gen == self.n_gen - 1:
                n_best = self.best_g['n'] if self.best_g else 0
                print(f"Gen {gen:4d} {time.time()-st:6.1f}s fit={bf:.4f}/{af:.4f} best={self.best_ever:.4f} n={n_best}")
                self.log.append((gen, bf, af, self.best_ever))
            
            if gen == self.n_gen - 1: break
            
            # 精英保留 + 锦标赛 + 交叉变异
            idx = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
            nx = [Org(copy.deepcopy(self.pop[idx[i]].g)) for i in range(min(4, len(self.pop)))]
            
            while len(nx) < self.n_org:
                # 锦标赛选2个
                t1 = random.sample(range(len(self.pop)), min(5, len(self.pop)))
                p1 = max(t1, key=lambda i: fits[i])
                t2 = random.sample(range(len(self.pop)), min(5, len(self.pop)))
                p2 = max(t2, key=lambda i: fits[i])
                
                if random.random() < 0.2 and len(self.pop) > 5:
                    # wild abort: 低分+高分交叉
                    wild = int(random.random()**2 * len(self.pop))
                    child = self.xover(self.pop[wild].g, self.pop[idx[0]].g)
                else:
                    child = self.xover(self.pop[p1].g, self.pop[p2].g)
                
                self.mutate(child, rate=0.04)
                nx.append(Org(child))
            
            self.pop = nx
        
        return self.best_ever

if __name__ == '__main__':
    print("=" * 60)
    print("yinyang_evo_v20 — 容量测试: n=20-60, Leaky-tanh")
    print(f"架构: 双通路+蛋白折叠(转置修正)+Verifier, 可激活leak")
    print("=" * 60)
    
    # 快速测试几个随机org
    print("\n[快速随机测试]")
    eng_test = Engine(1, 1, eval_ticks=200)
    for i in range(5):
        o = Org()
        f = eng_test.eval_org(o, ticks=200)
        print(f"  随机org {i}: n={o.g['n']}, fitness={f:.4f}")
    
    print(f"\n开始演化...")
    eng = Engine(n_org=35, n_gen=1500, seed=42, eval_ticks=350)
    eng.evolve()
    
    print(f"\n最终最佳: {eng.best_ever:.4f}")
    print(f"v17o参考: 0.9361")
    if eng.best_ever > 0.9361:
        print(">>> 突破! v20 > v17o")
    else:
        print(f">>> 差距: {0.9361 - eng.best_ever:.4f}")
    
    if eng.best_g:
        g = eng.best_g
        n = g['n']
        sat = sum(1 for v in WM(g).h if abs(v) > 0.98)  # 零状态，不具意义
        print(f"最佳结构: n={n}, lr={g['lr']:.4f}, fold={g['fold_lr']:.4f}")
        print(f"leak均值: {sum(g['leak'])/len(g['leak']):.4f}")
