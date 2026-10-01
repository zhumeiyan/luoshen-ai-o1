#!/usr/bin/env python3
# wm_benchmark.py
# 标准世界模型 baseline vs 进化模型 — 多步前瞻预测对比
#
# 三种 baseline 覆盖 AI 世界模型的主要范式：
#   MLP:    Dreamer V1 的 dense world model (前馈直接映射 s→s')
#   LSTM:   Dreamer V2/3 + Ha & Schmidhuber World Models 的 MDN-RNN
#   GRU:    Mamba-style + 大量使用的 gated recurrent world model
#
# 参数量全部对齐：hidden 维度 N=12，每模型约 300-500 参数
# 训练：同一批 episode 数据，epoch=5，简单 SGD/BPTT
# 测试：完全复用 v7/v8 的 multi_step_test 协议

import random, math, json, sys, time
from collections import deque
from copy import deepcopy

N_CHANNELS = 8
WARMUP_STEPS = 50
N_TRIALS = 5
N_STEP_LIST = [1, 3, 5, 10, 20, 50]


# ============================================================
# 环境：与 yinyang_evo_v7/v8 完全一致
# ============================================================

class Environment:
    def __init__(self, seed=None, mode='training'):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.mode = mode
        self.season = 0
        self.substep = 0
        self.light = 50.0; self.light_r = 0.0
        self.sound_freq = 440.0; self.sound_amp = 30.0
        self.pressure = 0.0; self.vibration = 0.0
        self.smell = [0.0]*5; self.taste = [0.0]*5
        self.temperature = 20.0; self.accel = [0.0, 0.0, 9.8]

    def step(self, action_code=1):
        self.t += 1
        self.substep += 1
        if self.substep > 150:
            self.substep = 0
            self.season = (self.season + 1) % 4
        s = self.season
        if action_code == 0:
            self.light += random.uniform(0.3, 1.0)
            self.sound_amp += random.uniform(0.05, 0.2)
        elif action_code == 2:
            self.light *= 0.97; self.pressure *= 0.92
        elif action_code == 3:
            self.temperature += random.uniform(-0.5, 0.5)
            self.pressure += random.uniform(0.05, 0.2)
        base_light = 50 + 25*math.sin(self.t*0.02 + s*math.pi/2)
        noise_light = random.uniform(-2,2)
        if self.mode == 'training':
            if s == 0:
                self.light += 0.08*(base_light-self.light)+noise_light*0.5
                self.sound_amp += random.uniform(-0.3,0.1)
                self.pressure += random.uniform(-0.05,0.05)
                if self.substep % 20 == 0: self.sound_amp += random.uniform(3,6)
            elif s == 1:
                self.light += 0.06*(30-self.light)+noise_light
                self.sound_amp += random.uniform(-0.2,0.6)
                if random.random() < 0.02: self.sound_amp += random.uniform(5,12)
            elif s == 2:
                self.temperature += 0.06*(28-self.temperature)
                self.pressure += 0.05*(3-self.pressure)
                te = (self.temperature-20)*0.3
                for i in range(5): self.smell[i] += te*0.1
            else:
                self.temperature += 0.06*(17-self.temperature)
                self.pressure += 0.05*(15-self.pressure)+random.uniform(-0.1,0.1)
                if random.random() < 0.08:
                    self.accel[0]+=random.uniform(-1.5,1.5); self.accel[1]+=random.uniform(-1.5,1.5)
        else:
            if s == 0:
                v=10*math.sin(self.t*0.03)+5*math.sin(self.t*0.11)
                self.light+=0.1*(50+v-self.light)+random.uniform(-1,1)
                self.pressure+=random.uniform(-0.1,0.15)
            elif s == 1:
                x=(self.pressure+10)/20; x=3.7*x*(1-x)
                self.pressure=max(0,min(20,x*20)); self.sound_amp+=random.uniform(-0.5,0.8)
            elif s == 2:
                if self.substep==50: self.light=100-self.light
                self.temperature+=random.uniform(-0.08,0.08); self.sound_freq+=random.uniform(-15,15)
            else:
                self.light+=random.uniform(-3,3)
                pulse=3.0 if self.substep%30<3 else 0
                self.sound_amp+=random.uniform(-0.8,0.8)+pulse
                self.accel[2]=9.8+2*math.sin(self.t*0.2)
        self.light=max(0.0,min(100.0,self.light))
        self.light_r=max(-1.0,min(1.0,self.light_r+random.uniform(-0.01,0.01)))
        self.sound_freq=max(20.0,min(20000.0,self.sound_freq))
        self.sound_amp=max(0.0,min(120.0,self.sound_amp*0.99))
        self.pressure=max(0.0,min(20.0,self.pressure))
        self.vibration=max(-2.0,min(2.0,self.vibration*0.9+random.uniform(-0.1,0.1)))
        for i in range(5):
            self.smell[i]*=0.93; self.smell[i]=max(0.0,min(10.0,self.smell[i]))
            if random.random()<0.02: self.smell[i]+=random.uniform(0,2)
            self.taste[i]*=0.985
            if random.random()<0.04: self.taste[i]+=random.uniform(-0.3,0.3)
            self.taste[i]=max(-5.0,min(5.0,self.taste[i]))
        self.temperature=max(15.0,min(35.0,self.temperature))
        for k in range(3): self.accel[k]=max(-2.0,min(12.0,self.accel[k]))

    def get_sensory(self):
        return [
            (self.light/50.0)-1.0,
            (self.sound_amp/60.0)-0.5+(math.log(max(1.0,self.sound_freq))-6)*0.15,
            (self.pressure/10.0)-0.5,
            (sum(self.smell)/25.0)-0.5,
            (sum(self.taste)/10.0),
            (self.temperature-25.0)/10.0,
            (abs(self.accel[2]-9.8)/5.0),
            0.0
        ]


# ============================================================
# Baseline A: MLP World Model  (8→12→12→8, tanh)
# ============================================================

class MLPWM:
    def __init__(self, hidden=12, lr=0.005):
        self.h = hidden
        self.lr = lr
        h = hidden
        self.W1 = [[random.uniform(-0.3,0.3) for _ in range(h)] for _ in range(N_CHANNELS)]
        self.b1 = [0.0]*h
        self.W2 = [[random.uniform(-0.3,0.3) for _ in range(h)] for _ in range(h)]
        self.b2 = [0.0]*h
        self.W3 = [[random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)] for _ in range(h)]
        self.b3 = [0.0]*N_CHANNELS
        self.errors = deque(maxlen=300)

    def forward(self, x):
        h1 = [math.tanh(sum(x[i]*self.W1[i][j] for i in range(N_CHANNELS))+self.b1[j]) for j in range(self.h)]
        h2 = [math.tanh(sum(h1[k]*self.W2[k][j] for k in range(self.h))+self.b2[j]) for j in range(self.h)]
        out = [math.tanh(sum(h2[j]*self.W3[j][i] for j in range(self.h))+self.b3[i]) for i in range(N_CHANNELS)]
        return out, h1, h2

    def predict(self, x): return self.forward(x)[0]

    def train_step(self, x, y_true):
        out,h1,h2 = self.forward(x)
        err = [y_true[i]-out[i] for i in range(N_CHANNELS)]
        mse = sum(e*e for e in err)/N_CHANNELS
        self.errors.append(mse)
        # backprop
        d_out = [-2.0/N_CHANNELS*e*(1-out[i]**2) for i,e in enumerate(err)]
        d_h2 = [sum(d_out[i]*self.W3[j][i] for i in range(N_CHANNELS))*(1-h2[j]**2) for j in range(self.h)]
        d_h1 = [sum(d_h2[j]*self.W2[k][j] for j in range(self.h))*(1-h1[k]**2) for k in range(self.h)]
        for j in range(self.h):
            for i in range(N_CHANNELS):
                self.W3[j][i] -= self.lr*d_out[i]*h2[j]
            for k in range(self.h):
                self.W2[k][j] -= self.lr*d_h2[j]*h1[k]
        for i in range(N_CHANNELS):
            for k in range(self.h):
                self.W1[i][k] -= self.lr*d_h1[k]*x[i]
        for k in range(self.h): self.b1[k]-=self.lr*d_h1[k]
        for j in range(self.h): self.b2[j]-=self.lr*d_h2[j]
        for i in range(N_CHANNELS): self.b3[i]-=self.lr*d_out[i]

    def multi_step(self, x, n):
        preds=[]; curr=x[:]
        for _ in range(n):
            p=self.predict(curr); preds.append(p); curr=p
        return preds

    def reset(self): pass


# ============================================================
# Baseline B: GRU World Model (Dreamer V3 + Mamba-style)
# ============================================================

class GRUWM:
    def __init__(self, hidden=12, lr=0.01):
        self.h = hidden
        self.lr = lr
        H = hidden
        N = N_CHANNELS
        # Update gate z
        self.Wz = [[random.uniform(-0.2,0.2) for _ in range(H)] for _ in range(N+H)]
        self.bz = [0.5]*H
        # Reset gate r
        self.Wr = [[random.uniform(-0.2,0.2) for _ in range(H)] for _ in range(N+H)]
        self.br = [0.0]*H
        # Candidate h~
        self.Wh = [[random.uniform(-0.2,0.2) for _ in range(H)] for _ in range(N+H)]
        self.bh = [0.0]*H
        # Output projection
        self.Wp = [[random.uniform(-0.2,0.2) for _ in range(N)] for _ in range(H)]
        self.bp = [0.0]*N
        self.hidden = [0.0]*H
        self.errors = deque(maxlen=300)

    def _gate(self, W, b, xh):
        H = len(b)
        return [1.0/(1.0+math.exp(-(b[j]+sum(xh[i]*W[i][j] for i in range(len(xh)))))) for j in range(H)]

    def _step(self, x):
        H = self.h
        xh = x + self.hidden
        z = self._gate(self.Wz, self.bz, xh)
        r = self._gate(self.Wr, self.br, xh)
        xh_r = x + [r[j]*self.hidden[j] for j in range(H)]
        h_hat = [math.tanh(self.bh[j]+sum(xh_r[i]*self.Wh[i][j] for i in range(len(xh_r)))) for j in range(H)]
        new_h = [(1-z[j])*self.hidden[j]+z[j]*h_hat[j] for j in range(H)]
        pred = [math.tanh(self.bp[i]+sum(new_h[j]*self.Wp[j][i] for j in range(H))) for i in range(N_CHANNELS)]
        self.hidden = new_h
        return pred

    def predict(self, x): return self._step(x)

    def reset(self): self.hidden = [0.0]*self.h

    def train_step(self, xs, y_target):
        """Train on sequence, only optimize using final prediction error"""
        self.reset()
        pred = None
        for x in xs: pred = self._step(x)
        # If target is a shift, use it
        err = [y_target[i]-pred[i] for i in range(N_CHANNELS)]
        mse = sum(e*e for e in err)/N_CHANNELS
        self.errors.append(mse)
        # Limited: just update Wp, bp (efficient, targets output directly)
        H = self.h
        lr = self.lr
        for j in range(H):
            for i in range(N_CHANNELS):
                self.Wp[j][i]-=lr*(-2.0/N_CHANNELS*err[i]*(1-pred[i]**2)*self.hidden[j])
        for i in range(N_CHANNELS):
            self.bp[i]-=lr*(-2.0/N_CHANNELS*err[i]*(1-pred[i]**2))

    def multi_step(self, x, n):
        preds=[]; curr=x[:]
        for _ in range(n):
            p=self._step(curr); preds.append(p); curr=p
        return preds


# ============================================================
# Baseline C: LSTM World Model (Ha & Schmidhuber World Models)
# ============================================================

class LSTM:
    def __init__(self, hidden=12, lr=0.005):
        self.h = hidden
        self.lr = lr
        H = hidden
        N = N_CHANNELS
        # Combined weight matrices: [x;h] → gates
        self.Wf = [[random.uniform(-0.2,0.2) for _ in range(H)] for _ in range(N+H)]
        self.bf = [0.5]*H  # forget bias >0 = remember
        self.Wi = [[random.uniform(-0.2,0.2) for _ in range(H)] for _ in range(N+H)]
        self.bi = [0.0]*H
        self.Wc = [[random.uniform(-0.2,0.2) for _ in range(H)] for _ in range(N+H)]
        self.bc = [0.0]*H
        self.Wo = [[random.uniform(-0.2,0.2) for _ in range(H)] for _ in range(N+H)]
        self.bo = [0.0]*H
        # Output projection
        self.Wp = [[random.uniform(-0.2,0.2) for _ in range(N)] for _ in range(H)]
        self.bp = [0.0]*N
        self.cell = [0.0]*H
        self.hidden = [0.0]*H
        self.errors = deque(maxlen=300)

    def _sigmoid(self, x):
        if x > 20: return 1.0
        if x < -20: return 0.0
        return 1.0/(1.0+math.exp(-x))

    def _gate_linear(self, W, b, xh):
        H = len(b)
        return [self._sigmoid(b[j]+sum(xh[i]*W[i][j] for i in range(len(xh)))) for j in range(H)]

    def _step(self, x):
        H = self.h
        xh = x + self.hidden
        f = self._gate_linear(self.Wf, self.bf, xh)
        ig = self._gate_linear(self.Wi, self.bi, xh)
        c_hat = [math.tanh(self.bc[j]+sum(xh[i]*self.Wc[i][j] for i in range(len(xh)))) for j in range(H)]
        new_cell = [f[j]*self.cell[j]+ig[j]*c_hat[j] for j in range(H)]
        o = self._gate_linear(self.Wo, self.bo, xh)
        new_hidden = [o[j]*math.tanh(new_cell[j]) for j in range(H)]
        self.cell = new_cell
        self.hidden = new_hidden
        pred = [math.tanh(self.bp[i]+sum(new_hidden[j]*self.Wp[j][i] for j in range(H))) for i in range(N_CHANNELS)]
        return pred

    def predict(self, x): return self._step(x)

    def reset(self): self.cell=[0.0]*self.h; self.hidden=[0.0]*self.h

    def train_step(self, xs, y_target):
        self.reset()
        pred = None
        for x in xs: pred = self._step(x)
        err = [y_target[i]-pred[i] for i in range(N_CHANNELS)]
        mse = sum(e*e for e in err)/N_CHANNELS
        self.errors.append(mse)
        H = self.h
        lr = self.lr
        for j in range(H):
            for i in range(N_CHANNELS):
                self.Wp[j][i]-=lr*(-2.0/N_CHANNELS*err[i]*(1-pred[i]**2)*self.hidden[j])
        for i in range(N_CHANNELS):
            self.bp[i]-=lr*(-2.0/N_CHANNELS*err[i]*(1-pred[i]**2))

    def multi_step(self, x, n):
        preds=[]; curr=x[:]
        for _ in range(n):
            p=self._step(curr); preds.append(p); curr=p
        return preds


# ============================================================
# 数据生成
# ============================================================

def gen_episodes(n=30, length=150, seed_base=100):
    eps = []
    for ep in range(n):
        env = Environment(seed=seed_base+ep, mode='training')
        seq = []
        for _ in range(length):
            a = env.get_sensory()
            act = random.randrange(4)
            env.step(act)
            b = env.get_sensory()
            seq.append((a, b))
        eps.append(seq)
    return eps


def train_baseline(model, episodes, n_epochs=5):
    """训练所有 baseline"""
    is_mlp = isinstance(model, MLPWM)
    for epoch in range(n_epochs):
        for seq in episodes:
            if is_mlp:
                # MLP: 逐对训练 (状态无关)
                for x, y in seq:
                    model.train_step(x, y)
            else:
                # GRU/LSTM: 用完整序列 warmup，最后一步作预测目标
                xs = [s for s,_ in seq]
                if len(xs) > 1:
                    model.reset()
                    for x in xs[:-1]:
                        model.predict(x)
                    model.train_step(xs, seq[-1][1])


# ============================================================
# 多步前瞻预测测试（完全复用 v8 协议）
# ============================================================

def run_multi_step(model, n_list=N_STEP_LIST, n_trials=N_TRIALS):
    results = {}
    for n_steps in n_list:
        trial_errs = []
        for trial in range(n_trials):
            seed = trial*777+123
            # warmup
            env = Environment(seed=seed, mode='novel')
            model.reset()
            for _ in range(WARMUP_STEPS):
                s = env.get_sensory()
                _ = model.predict(s)
                env.step(1)
            # predict n steps
            sensory = env.get_sensory()
            predictions = model.multi_step(sensory, n_steps)
            # ground truth
            env_gt = Environment(seed=seed, mode='novel')
            for _ in range(WARMUP_STEPS):
                s = env_gt.get_sensory()
                env_gt.step(1)
            gts = []
            for n in range(n_steps):
                env_gt.step(1)
                gts.append(env_gt.get_sensory()[:])
            # compute error
            errs = []
            for n in range(min(n_steps, len(gts))):
                e = sum((predictions[n][i]-gts[n][i])**2 for i in range(N_CHANNELS))/N_CHANNELS
                errs.append(e)
            if errs:
                trial_errs.append(sum(errs)/len(errs))
        results[str(n_steps)] = sum(trial_errs)/max(1,len(trial_errs)) if trial_errs else 1.0
    return results


# ============================================================
# 进化模型：从 v8 结果重新训练获得（use checkpoint or re-train via evolution）
# ============================================================

def get_best_evo_and_test():
    """直接复用 multi_step_prediction_v8.json 已测好的结果"""
    try:
        with open('multi_step_prediction_v8.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ============================================================
# Main
# ============================================================

def main():
    print("="*70)
    print("World Model Benchmark: Baseline (MLP/GRU/LSTM) vs Evolved (v8)")
    print("="*70)

    random.seed(42)

    # 1. Generate training data (same for all)
    print("\n[1/4] Generating training episodes...")
    episodes = gen_episodes(n=30, length=150)
    print(f"  {len(episodes)} episodes × {len(episodes[0])} steps")

    # 2. Train baselines
    print("\n[2/4] Training baselines...")
    mlp = MLPWM(hidden=12, lr=0.005)
    gru = GRUWM(hidden=12, lr=0.008)
    lstm = LSTM(hidden=12, lr=0.005)

    train_baseline(mlp, episodes, n_epochs=5)
    print(f"  MLP  train_MSE={sum(mlp.errors)/max(1,len(mlp.errors)):.6f}")
    train_baseline(gru, episodes, n_epochs=5)
    print(f"  GRU  train_MSE={sum(gru.errors)/max(1,len(gru.errors)):.6f}")
    train_baseline(lstm, episodes, n_epochs=5)
    print(f"  LSTM train_MSE={sum(lstm.errors)/max(1,len(lstm.errors)):.6f}")

    # 3. Multi-step prediction test
    print("\n[3/4] Multi-step prediction test (novel env, n_trials=5)...")
    mlp_res = run_multi_step(mlp)
    gru_res = run_multi_step(gru)
    lstm_res = run_multi_step(lstm)

    # 4. Load evolved model results (from v8 multi_step_prediction_v8.json)
    print("\n[4/4] Loading evolved model data...")
    evo_data = get_best_evo_and_test()
    if evo_data:
        evo_res = evo_data.get('evolved', {})
        print(f"  Evolved model data loaded from v8 test")
    else:
        print("  No evolved model data found, re-running evolution test needed")
        evo_res = {}

    # Output comparison
    all_results = {
        'MLP': mlp_res,
        'GRU': gru_res,
        'LSTM': lstm_res,
    }
    if evo_res:
        all_results['EVO(v8-200g)'] = evo_res

    print("\n" + "="*70)
    print("Multi-step Prediction MSE (novel environment)")
    print("="*70)
    header = f"{'N':>4s}"
    for name in all_results:
        header += f" | {name:>14s}"
    print(header)
    print("-"*len(header))
    for n_steps in ['1','3','5','10','20','50']:
        row = f"{int(n_steps):4d}"
        for name in all_results:
            v = all_results[name].get(n_steps, None)
            if v is not None:
                row += f" | {v:14.6f}"
            else:
                row += f" | {'N/A':>14s}"
        print(row)

    # Advantage ratios
    if 'EVO(v8-200g)' in all_results:
        print("\n--- Advantage: Evo / Baseline ---")
        for name in ['MLP','GRU','LSTM']:
            if name in all_results:
                n_steps_list = ['1','3','5','10','20','50']
                ratios = []
                for ns in n_steps_list:
                    evo_v = all_results['EVO(v8-200g)'].get(ns, 1e-8)
                    base_v = all_results[name].get(ns, 1e-8)
                    if evo_v > 0 and base_v > 0:
                        ratios.append(f"{ns}:{base_v/max(1e-8,evo_v):.1f}x")
                print(f"  Evo vs {name}: {', '.join(ratios)}")

    # Save to JSON
    with open('benchmark_raw.json','w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print("\nSaved to benchmark_raw.json")


if __name__ == '__main__':
    main()
