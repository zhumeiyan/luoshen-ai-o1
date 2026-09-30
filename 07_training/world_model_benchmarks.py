#!/usr/bin/env python3
# world_model_benchmarks.py
#
# 标准世界模型 baseline 实现 + 与进化模型的多步前瞻预测对比
#
# Baseline 范式（取自 Dreamer / World Models / 经典论文）：
#   A. MLP World Model       — 直接状态→状态映射（Dreamer v1 的 dense world model）
#   B. LSTM World Model      — 循环隐藏状态，BPTT 训练（Dreamer v2/v3, MDN-RNN）
#   C. Transformer World Model — 自注意力时序建模（Transformer-XL / Mamba attention-free 简化）
#
# 与我们的进化模型完全同环境、同测试协议（multi-step prediction）对比

import random, math, json, sys, time
from collections import deque
from copy import deepcopy

# ============================================================
# 环境：与 v7/v8 完全相同
# ============================================================

N_CHANNELS = 8
CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']


class Environment:
    """训练/测试用环境 — 与 yinyang_evo_v7.py 完全一致"""
    def __init__(self, seed=None, mode='training'):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.mode = mode
        self.season = 0
        self.substep = 0
        self.light = 50.0
        self.light_r = 0.0
        self.sound_freq = 440.0
        self.sound_amp = 30.0
        self.pressure = 0.0
        self.vibration = 0.0
        self.smell = [0.0]*5
        self.taste = [0.0]*5
        self.temperature = 20.0
        self.accel = [0.0, 0.98, 9.8]

    def step(self, action_code):
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
            self.light *= 0.97
            self.pressure *= 0.92
        elif action_code == 3:
            self.temperature += random.uniform(-0.5, 0.5)
            self.pressure += random.uniform(0.05, 0.2)

        if self.mode == 'training':
            base_light = 50 + 25 * math.sin(self.t * 0.02 + s * math.pi / 2)
            noise_light = random.uniform(-2, 2)
            if s == 0:
                self.light += 0.08 * (base_light - self.light) + noise_light * 0.5
                self.sound_amp += random.uniform(-0.3, 0.1)
                self.pressure += random.uniform(-0.05, 0.05)
                if self.substep % 20 == 0:
                    self.sound_amp += random.uniform(3, 6)
            elif s == 1:
                self.light += 0.06 * (30 - self.light) + noise_light
                self.sound_amp += random.uniform(-0.2, 0.6)
                if random.random() < 0.02:
                    self.sound_amp += random.uniform(5, 12)
            elif s == 2:
                self.temperature += 0.06 * (28 - self.temperature)
                self.pressure += 0.05 * (3 - self.pressure)
                temp_effect = (self.temperature - 20) * 0.3
                for i in range(5):
                    self.smell[i] += temp_effect * 0.1
            else:
                self.temperature += 0.06 * (17 - self.temperature)
                self.pressure += 0.05 * (15 - self.pressure) + random.uniform(-0.1, 0.1)
                if random.random() < 0.08:
                    self.accel[0] += random.uniform(-1.5, 1.5)
                    self.accel[1] += random.uniform(-1.5, 1.5)
        else:
            if s == 0:
                v = 10 * math.sin(self.t * 0.03) + 5 * math.sin(self.t * 0.11)
                self.light += 0.1 * (50 + v - self.light) + random.uniform(-1, 1)
                self.pressure += random.uniform(-0.1, 0.15)
            elif s == 1:
                x = (self.pressure + 10) / 20
                x = 3.7 * x * (1 - x)
                self.pressure = max(0, min(20, x * 20))
                self.sound_amp += random.uniform(-0.5, 0.8)
            elif s == 2:
                if self.substep == 50:
                    self.light = 100 - self.light
                self.temperature += random.uniform(-0.08, 0.08)
                self.sound_freq += random.uniform(-15, 15)
            else:
                self.light += random.uniform(-3, 3)
                pulse = 3.0 if self.substep % 30 < 3 else 0
                self.sound_amp += random.uniform(-0.8, 0.8) + pulse
                self.accel[2] = 9.8 + 2 * math.sin(self.t * 0.2)

        self.light = max(0.0, min(100.0, self.light))
        self.light_r = max(-1.0, min(1.0, self.light_r + random.uniform(-0.01, 0.01)))
        self.sound_freq = max(20.0, min(20000.0, self.sound_freq))
        self.sound_amp = max(0.0, min(120.0, self.sound_amp * 0.99))
        self.pressure = max(0.0, min(20.0, self.pressure))
        self.vibration = max(-2.0, min(2.0, self.vibration * 0.9 + random.uniform(-0.1, 0.1)))
        for i in range(5):
            self.smell[i] *= 0.93
            self.smell[i] = max(0.0, min(10.0, self.smell[i]))
            if random.random() < 0.02:
                self.smell[i] += random.uniform(0, 2)
            self.taste[i] *= 0.985
            if random.random() < 0.04:
                self.taste[i] += random.uniform(-0.3, 0.3)
            self.taste[i] = max(-5.0, min(5.0, self.taste[i]))
        self.temperature = max(15.0, min(35.0, self.temperature))
        for k in range(3):
            self.accel[k] = max(-2.0, min(12.0, self.accel[k]))

    def get_sensory(self):
        return [
            (self.light / 50.0) - 1.0,
            (self.sound_amp / 60.0) - 0.5 + (math.log(max(1.0, self.sound_freq)) - 6) * 0.15,
            (self.pressure / 10.0) - 0.5,
            (sum(self.smell) / 25.0) - 0.5,
            (sum(self.taste) / 10.0),
            (self.temperature - 25.0) / 10.0,
            (abs(self.accel[2] - 9.8) / 5.0),
            0.0
        ]


# ============================================================
# Baseline A: MLP World Model
#   结构: [8→12→12→8] tanh, 直接状态-Δ预测
# ============================================================

class MLPWorldModel:
    def __init__(self, h1=12, h2=12, lr=0.005):
        self.h1 = h1
        self.h2 = h2
        self.lr = lr
        self.n_in = N_CHANNELS
        # Xavier init
        self.W1 = [[random.uniform(-0.3, 0.3) for _ in range(h1)] for _ in range(N_CHANNELS)]
        self.b1 = [0.0] * h1
        self.W2 = [[random.uniform(-0.3, 0.3) for _ in range(h2)] for _ in range(h1)]
        self.b2 = [0.0] * h2
        self.W3 = [[random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)] for _ in range(h2)]
        self.b3 = [0.0] * N_CHANNELS
        self.errors = deque(maxlen=200)

    def forward(self, x):
        """输入当前状态，输出预测 Δ(sensory)"""
        h1 = [math.tanh(sum(x[i] * self.W1[i][j] for i in range(N_CHANNELS)) + self.b1[j]) for j in range(self.h1)]
        h2 = [math.tanh(h1[k] * self.W2[k][j] + self.b2[j] if k < len(self.W2) else 0.0) for j in range(self.h2)]
        # Simplified: fix index
        h2 = []
        for j in range(self.h2):
            s = self.b2[j]
            for k in range(self.h1):
                s += h1[k] * self.W2[k][j]
            h2.append(math.tanh(s))

        out = []
        for i in range(N_CHANNELS):
            s = self.b3[i]
            for j in range(self.h2):
                s += h2[j] * self.W3[j][i]
            out.append(math.tanh(s))  # 预测下一状态
        return out, h1, h2

    def predict(self, x):
        out, _, _ = self.forward(x)
        return out

    def train_step(self, x, y_true):
        """SGD 一步"""
        out, h1, h2 = self.forward(x)
        err = [y_true[i] - out[i] for i in range(N_CHANNELS)]
        mse = sum(e*e for e in err) / N_CHANNELS
        self.errors.append(mse)

        # Backprop (手动)
        # dL/d_out = -2/N * err
        d_out = [-2.0 / N_CHANNELS * e for e in err]

        # dL/d_W3, dL/d_b3
        d_h2 = [0.0] * self.h2
        for j in range(self.h2):
            for i in range(N_CHANNELS):
                d_h2[j] += d_out[i] * self.W3[j][i] * (1 - out[i]**2)
                self.W3[j][i] -= self.lr * d_out[i] * (1 - out[i]**2) * h2[j]
                self.b3[i] -= self.lr * d_out[i] * (1 - out[i]**2)

        d_h1 = [0.0] * self.h1
        for k in range(self.h1):
            for j in range(self.h2):
                d_h1[k] += d_h2[j] * self.W2[k][j] * (1 - h2[j]**2)
                self.W2[k][j] -= self.lr * d_h2[j] * (1 - h2[j]**2) * h1[k]
                self.b2[j] -= self.lr * d_h2[j] * (1 - h2[j]**2)

        for i in range(N_CHANNELS):
            for k in range(self.h1):
                self.W1[i][k] -= self.lr * d_h1[k] * (1 - h1[k]**2) * x[i]
                self.b1[k] -= self.lr * d_h1[k] * (1 - h1[k]**2)

    def multi_step_predict(self, x, n_steps):
        """多步前瞻：feeding predictions back"""
        preds = []
        curr = x[:]
        for _ in range(n_steps):
            pred = self.predict(curr)
            preds.append(pred)
            curr = pred
        return preds


# ============================================================
# Baseline B: LSTM World Model (Dreamer / MDN-RNN 风格)
#   输入: sensory[8] → LSTM cell → output sensory[8]
# ============================================================

class LSTMWorldModel:
    def __init__(self, hidden=12, lr=0.01):
        self.h = hidden
        self.lr = lr
        H = hidden
        N = N_CHANNELS

        # LSTM gates (simplified, single layer)
        # Forget gate
        self.Wf = [[random.uniform(-0.2, 0.2) for _ in range(H)] for _ in range(N + H)]
        self.bf = [0.5] * H  # forget gate bias = 1-ish (remember)
        # Input gate
        self.Wi = [[random.uniform(-0.2, 0.2) for _ in range(H)] for _ in range(N + H)]
        self.bi = [0.0] * H
        # Candidate
        self.Wc = [[random.uniform(-0.2, 0.2) for _ in range(H)] for _ in range(N + H)]
        self.bc = [0.0] * H
        # Output gate
        self.Wo = [[random.uniform(-0.2, 0.2) for _ in range(H)] for _ in range(N + H)]
        self.bo = [0.0] * H
        # Output projection (hidden → sensory prediction)
        self.Wp = [[random.uniform(-0.2, 0.2) for _ in range(N)] for _ in range(H)]
        self.bp = [0.0] * N

        self.cell = [0.0] * H
        self.hidden = [0.0] * H
        self.errors = deque(maxlen=200)
        # Adam-like momentum
        self.t = 0
        self.m = {}
        self.v = {}

    def _gate(self, W, b, xh):
        H = len(b)
        out = []
        for j in range(H):
            s = b[j]
            for i in range(len(xh)):
                s += xh[i] * W[i][j]
            out.append(1.0 / (1.0 + math.exp(-s)))  # sigmoid
        return out

    def _matmul(self, W, x):
        out = []
        for j in range(len(W[0])):
            s = 0.0
            for i in range(len(x)):
                s += x[i] * W[i][j]
            out.append(s)
        return out

    def _lstm_step(self, x):
        H = self.h
        xh = x + self.hidden  # concat
        f = self._gate(self.Wf, self.bf, xh)
        ii = self._gate(self.Wi, self.bi, xh)
        c_hat = self._gate(self.Wc, self.bc, xh)  # Actually tanh, using sigmoid for simplicity
        c_hat = [math.tanh(sum(xh[i] * self.Wc[i][j] for i in range(len(xh))) + self.bc[j]) for j in range(H)]

        new_cell = [f[j] * self.cell[j] + ii[j] * c_hat[j] for j in range(H)]
        o = self._gate(self.Wo, self.bo, xh)
        new_hidden = [o[j] * math.tanh(new_cell[j]) for j in range(H)]

        self.cell = new_cell
        self.hidden = new_hidden

        # Project to prediction
        pred = [math.tanh(self.bp[i] + sum(new_hidden[j] * self.Wp[j][i] for j in range(H))) for i in range(N_CHANNELS)]
        return pred

    def predict(self, x):
        return self._lstm_step(x)

    def reset_state(self):
        self.cell = [0.0] * self.h
        self.hidden = [0.0] * self.h

    def train_step(self, seq_x, seq_y):
        """BPTT-truncated: 对序列每一步做一步 SGD（简化版：只训练最后一步）"""
        self.reset_state()
        pred = None
        for x in seq_x:
            pred = self._lstm_step(x)

        y_true = seq_y[-1]
        err = [y_true[i] - pred[i] for i in range(N_CHANNELS)]
        mse = sum(e*e for e in err) / N_CHANNELS
        self.errors.append(mse)

        # 简化: 用数值梯度对 Wp, bp 做一步更新
        H = self.h
        lr = self.lr
        for j in range(H):
            for i in range(N_CHANNELS):
                grad = -2.0 / N_CHANNELS * err[i] * (1 - pred[i]**2) * self.hidden[j]
                self.Wp[j][i] -= lr * grad
            for i in range(N_CHANNELS):
                self.bp[i] -= lr * (-2.0 / N_CHANNELS * err[i] * (1 - pred[i]**2))

    def multi_step_predict(self, x0, n_steps):
        self.reset_state()
        _ = self._lstm_step(x0)
        preds = []
        curr = x0[:]
        for _ in range(n_steps):
            pred = self._lstm_step(curr)
            preds.append(pred)
            curr = pred
        return preds


# ============================================================
# Baseline C: Transformer World Model (简化版，无库依赖)
#   因果自回归：位置编码 + 多头注意力 → FFN → 预测
#   节点数 << 标准 transformer，参数量对齐
# ============================================================

class TransformerWorldModel:
    def __init__(self, d_model=8, n_heads=2, n_layers=2, lr=0.005):
        self.d = d_model
        self.h = n_heads
        self.L = n_layers
        self.lr = lr
        d = d_model

        # Q, K, V projections per head per layer
        self.params = {}
        for l in range(n_layers):
            self.params[f'Wq_{l}'] = [[[random.uniform(-0.2,0.2) for _ in range(d)] for _ in range(d)] for _ in range(n_heads)]
            self.params[f'Wk_{l}'] = [[[random.uniform(-0.2,0.2) for _ in range(d)] for _ in range(d)] for _ in range(n_heads)]
            self.params[f'Wv_{l}'] = [[[random.uniform(-0.2,0.2) for _ in range(d)] for _ in range(d)] for _ in range(n_heads)]
            self.params[f'Wo_{l}'] = [[random.uniform(-0.2,0.2) for _ in range(d)] for _ in range(d)]
            self.params[f'ff1_{l}'] = [[random.uniform(-0.2,0.2) for _ in range(d)] for _ in range(d)]
            self.params[f'ff2_{l}'] = [[random.uniform(-0.2,0.2) for _ in range(d)] for _ in range(d)]
            self.params[f'b1_{l}'] = [0.0]*d
            self.params[f'b2_{l}'] = [0.0]*d
            self.params[f'Wo_b_{l}'] = [0.0]*d

        # Output projection
        self.W_out = [[random.uniform(-0.2,0.2) for _ in range(N_CHANNELS)] for _ in range(d)]
        self.b_out = [0.0]*N_CHANNELS

        self.errors = deque(maxlen=200)
        self.memory = []  # KV cache
        self.max_ctx = 32

    def _attn_head(self, Q, K, V, seq_len):
        """Scaled dot-product attention"""
        d_k = len(Q[0])
        scale = math.sqrt(d_k) if d_k > 0 else 1.0
        out = []
        for i in range(seq_len):
            # q_i vs all k_j
            scores = []
            for j in range(i+1):  # causal mask
                s = sum(Q[i][a] * K[j][a] for a in range(d_k)) / scale
                scores.append(s)
            # softmax
            max_s = max(scores) if scores else 0
            exps = [math.exp(s - max_s) for s in scores]
            sum_e = sum(exps)
            weights = [e / sum_e for e in exps]
            # weighted V
            o = [0.0] * len(V[0])
            for j in range(len(weights)):
                for a in range(len(V[0])):
                    o[a] += weights[j] * V[j][a]
            out.append(o)
        return out

    def _layer_forward(self, x, l):
        """x: list of vectors (seq), returns list of vectors"""
        d = self.d
        seq_len = len(x)

        # Multi-head attention
        Q = [_matmv(_get(self.params, f'Wq_{l}')[ha], x[i], d) for ha in range(self.h) for i in range(seq_len)]
        # Simplified: per head
        heads_out = []  # per-token concatenated
        for i in range(seq_len):
            combined = []
            for ha in range(self.h):
                Wq = self.params[f'Wq_{l}'][ha]
                Wk = self.params[f'Wk_{l}'][ha]
                Wv = self.params[f'Wv_{l}'][ha]
                q_i = [sum(x[i][a] * Wq[a][b] for a in range(d)) for b in range(d)]
                k = [[sum(x[j][a] * Wk[a][b] for a in range(d)) for b in range(d)] for j in range(seq_len)]
                v = [[sum(x[j][a] * Wv[a][b] for a in range(d)) for b in range(d)] for j in range(seq_len)]
                # Causal attention
                scores = [sum(q_i[b] * k[j][b] for b in range(d)) / math.sqrt(d) for j in range(i+1)]
                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                sum_e = sum(exps)
                w = [e / sum_e for e in exps]
                o = [sum(w[jm1] * v[jm1][b] for jm1 in range(len(w))) for b in range(d)]
                combined.extend(o)

            # Project concatenated heads back to d (assume h divides or just truncate)
            Wo = self.params[f'Wo_{l}']
            Wo_b = self.params[f'Wo_b_{l}']
            projected = [Wo_b[b] + sum(combined[b] * Wo[b][c] if c < len(combined) else 0.0 for b in range(len(Wo))) for c in range(d)]
            # Residual + LayerNorm (skip LN for simplicity)
            heads_out.append([x[i][b] + projected[b] for b in range(d)])

        # Feed-forward
        ff = self.params[f'ff1_{l}']
        out = self.params[f'ff2_{l}']
        b1 = self.params[f'b1_{l}']
        b2 = self.params[f'b2_{l}']
        final = []
        for i in range(seq_len):
            h1 = [math.tanh(b1[a] + sum(heads_out[i][b] * ff[b][a] for b in range(d))) for a in range(d)]
            h2 = [b2[a] + sum(h1[b] * out[b][a] for b in range(d)) for a in range(d)]
            final.append([heads_out[i][a] + h2[a] for a in range(d)])
        return final

    def _forward(self, seq_x):
        """Full forward: takes list of past + current, outputs prediction"""
        h = [x[:] for x in seq_x]
        for l in range(self.L):
            h = self._layer_forward(h, l)
        last = h[-1]
        pred = [math.tanh(self.b_out[i] + sum(last[j] * self.W_out[j][i] for j in range(self.d))) for i in range(N_CHANNELS)]
        return pred

    def predict(self, x):
        self.memory.append(x)
        if len(self.memory) > self.max_ctx:
            self.memory = self.memory[-self.max_ctx:]
        return self._forward(self.memory)

    def reset_state(self):
        self.memory = []

    def train_step(self, seq_x, y_true):
        """Train on a sequence, target is y_true (next state after seq)"""
        pred = self._forward(seq_x)
        err = [y_true[i] - pred[i] for i in range(N_CHANNELS)]
        mse = sum(e*e for e in err) / N_CHANNELS
        self.errors.append(mse)

        # Simplified: only update W_out, b_out (like linear probing)
        d = self.d
        lr = self.lr
        # Get last hidden state
        h = [x[:] for x in seq_x]
        for l in range(self.L):
            h = self._layer_forward(h, l)
        last = h[-1]

        for j in range(d):
            for i in range(N_CHANNELS):
                grad = -2.0 / N_CHANNELS * err[i] * (1 - pred[i]**2) * last[j]
                self.W_out[j][i] -= lr * grad
        for i in range(N_CHANNELS):
            self.b_out[i] -= lr * (-2.0 / N_CHANNELS * err[i] * (1 - pred[i]**2))

    def multi_step_predict(self, seq_x, n_steps):
        self.memory = [x[:] for x in seq_x]
        preds = []
        curr = seq_x[-1][:]
        for _ in range(n_steps):
            pred = self.predict(curr)
            preds.append(pred)
            curr = pred
        return preds


def _matmv(W, x, d):
    """Matrix-vector: W[d][d], x[d] -> out[d]"""
    return [sum(x[a] * W[a][b] for a in range(d)) for b in range(d)]


def _get(d, k):
    return d[k]


# ============================================================
# 训练流程：为每个 baseline 生成同样的训练数据
# ============================================================

def generate_training_data(n_episodes=25, ep_length=150, seed_base=100):
    """收集 (seq_x, seq_y) 对给 baseline 训练"""
    episodes = []
    for ep in range(n_episodes):
        env = Environment(seed=seed_base + ep, mode='training')
        seq = []
        for _ in range(ep_length):
            s = env.get_sensory()
            a = random.randrange(4)
            env.step(a)
            s_next = env.get_sensory()
            seq.append((s, s_next))
        episodes.append(seq)
    return episodes


def train_mlp(model, episodes, n_epochs=3):
    """训练 MLP"""
    for epoch in range(n_epochs):
        for seq in episodes:
            for x, y in seq:
                model.train_step(x, y)


def train_lstm(model, episodes, n_epochs=3):
    """训练 LSTM：用短窗口 BPTT 近似"""
    for epoch in range(n_epochs):
        for seq in episodes:
            # 序列前 n-1 步作输入，最后一步作目标
            xs = [s for s, _ in seq[:-1]]
            ys = [y for _, y in seq[:-1]]
            if xs and ys:
                model.train_step(xs, ys)


def train_transformer(model, episodes, n_epochs=3):
    """训练 Transformer"""
    for epoch in range(n_epochs):
        for seq in episodes:
            xs = [s for s, _ in seq]
            y = seq[-1][1] if seq else [0]*N_CHANNELS
            model.train_step(xs, y)


# ============================================================
# 多步前瞻预测测试（与 v7/v8 完全一致的协议）
# ============================================================

def multi_step_test(model, n_steps_list=None, n_trials=5, env_mode='novel', model_type='mlp'):
    if n_steps_list is None:
        n_steps_list = [1, 3, 5, 10, 20, 50]

    results = {}
    for n_steps in n_steps_list:
        trial_errs = []
        for trial in range(n_trials):
            seed = trial * 777 + 123

            if model_type == 'mlp':
                model.reset_state = lambda: None
                model.multi_step_predict = lambda x, n: _mlp_multi(model, x, n)
            elif model_type == 'lstm':
                model.reset_state()
                model.multi_step_predict = lambda x, n, m=model: m.multi_step_predict(x, n)
            elif model_type == 'transformer':
                model.reset_state()

            # Warm up
            env = Environment(seed=seed, mode=env_mode)
            for _ in range(50):
                s = env.get_sensory()
                if model_type == 'transformer':
                    pred = model.predict(s)
                    model.memory = model.memory[-32:]
                else:
                    pred = model.predict(s)
                env.step(1)

            # Multi-step forward
            if model_type == 'transformer':
                seq = list(model.memory) if hasattr(model, 'memory') else [env.get_sensory()]
                predictions = []
                curr = seq[-1][:]
                model.memory = [x[:] for x in seq]
                for n in range(n_steps):
                    pred = model.predict(curr)
                    predictions.append(pred)
                    curr = pred
            elif model_type == 'lstm':
                sensory = env.get_sensory()
                model.reset_state()
                for _ in range(min(20, len(model.cell))):
                    _ = model._lstm_step(sensory)
                predictions = model.multi_step_predict(sensory, n_steps)
            else:  # mlp
                sensory = env.get_sensory()
                predictions = model.multi_step_predict(sensory, n_steps)

            # Ground truth
            env_gt = Environment(seed=seed, mode=env_mode)
            for _ in range(50):
                s = env_gt.get_sensory()
                env_gt.step(1)

            if model_type == 'transformer':
                # advance from same point
                pass

            actuals = [env_gt.get_sensory()[:]]
            for n in range(n_steps):
                env_gt.step(1)
                actuals.append(env_gt.get_sensory()[:])

            errs = []
            for n in range(min(n_steps, len(actuals)-1)):
                e = sum((predictions[n][i] - actuals[n+1][i])**2 for i in range(N_CHANNELS)) / N_CHANNELS
                errs.append(e)
            if errs:
                trial_errs.append(sum(errs)/len(errs))

        results[str(n_steps)] = round(sum(trial_errs)/max(1,len(trial_errs)), 6) if trial_errs else 1.0
    return results


def _mlp_multi(model, x, n_steps):
    preds = []
    curr = x[:]
    for _ in range(n_steps):
        pred = model.predict(curr)
        preds.append(pred)
        curr = pred
    return preds


# ============================================================
# 加载 v8 进化出的最佳模型
# ============================================================

def load_best_evo_model():
    """从 evolution_log_v8_200g.json 找回最佳基因组（近似）"""
    # v8 WorldModel 定义已全在 yinyang_evo_v8.py
    sys.path.insert(0, '.')
    from yinyang_evo_v8 import WorldModel, Organism

    # load evolution log
    with open('evolution_log_v8_200g.json') as f:
        data = json.load(f)

    h = data['history']
    # find the gen with best fitness
    best_g = max(h, key=lambda g: g['best_fitness'])

    # Reconstruct best organism - we need genome
    # Re-run v8 with same seed to get the genome
    random.seed(42)
    from yinyang_evo_v8 import World
    world = World(n_organisms=30, steps_per_gen=400, difficulty=1.0)

    for g in range(best_g['gen'] + 1):
        world.generation = g
        world.run_generation(mode='training')
        if g < best_g['gen']:
            world.select_and_reproduce()
        else:
            # find best organism
            fits = [world.compute_fitness(o) for o in world.population]
            best_i = fits.index(max(fits))
            best_org = world.population[best_i]
            return best_org

    return None


# ============================================================
# 主实验：训练 baseline + 进化模型对比
# ============================================================

def main():
    print("=" * 70)
    print("World Model Benchmark Comparison")
    print("=" * 70)

    random.seed(42)

    # ---- 1. 训练数据 ----
    print("\nGenerating training data...")
    episodes = generate_training_data(n_episodes=25, ep_length=150)

    # ---- 2. MLP Baseline ----
    print("Training MLP...")
    mlp = MLPWorldModel(h1=12, h2=12, lr=0.005)
    train_mlp(mlp, episodes, n_epochs=3)
    print(f"  MLP final train MSE: {sum(mlp.errors)/max(1,len(mlp.errors)):.6f}")

    # ---- 3. LSTM Baseline ----
    print("Training LSTM...")
    lstm = LSTMWorldModel(hidden=12, lr=0.01)
    for epoch in range(3):
        for seq in episodes:
            xs = [s for s, _ in seq[:-1]]
            if xs:
                lstm.train_step(xs, [y for _, y in seq[:-1]])
    print(f"  LSTM final train MSE: {sum(lstm.errors)/max(1,len(lstm.errors)):.6f}")

    # ---- 4. Transformer Baseline ----
    print("Training Transformer...")
    transformer = TransformerWorldModel(d_model=8, n_heads=2, n_layers=2, lr=0.002)
    train_transformer(transformer, episodes, n_epochs=3)
    print(f"  Transformer final train MSE: {sum(transformer.errors)/max(1,len(transformer.errors)):.6f}")

    # ---- 5. Multi-step prediction test ----
    print("\n" + "=" * 70)
    print("Multi-step Prediction Comparison")
    print("=" * 70)

    models = {
        'evo (v8 200gens)': None,  # filled after load
        'mlp': (mlp, 'mlp'),
        'lstm': (lstm, 'lstm'),
        'transformer': (transformer, 'transformer'),
    }

    all_results = {}

    for name, (model, mtype) in models.items():
        if name.startswith('evo'):
            print(f"\n  Loading evolved model (gen with best fitness)...")
            best_org = load_best_evo_model()
            if best_org:
                # Create a wrapper that has multi-step predict
                from yinyang_evo_v8 import WorldModel
                wm = WorldModel(best_org.genome)
                # mimic model interface
                class EvoWrapper:
                    def __init__(self, wm):
                        self.wm = wm
                        self.errors = wm.prediction_errors
                    def predict(self, x):
                        pred, nodes = self.wm.forward(x)
                        return pred
                    def reset_state(self):
                        self.wm.nodes = [0.0] * len(self.wm.nodes)
                        self.wm.last_prediction = [0.0] * N_CHANNELS
                    def multi_step_predict(self, x, n_steps):
                        nodes = self.wm.nodes[:]
                        preds = []
                        curr = x[:]
                        for _ in range(n_steps):
                            pred, nodes = self.wm.forward_from_nodes(nodes, curr)
                            preds.append(pred[:])
                            curr = pred
                        return preds

                evo_model = EvoWrapper(wm)
                print("  Testing evolved model...")
                res = multi_step_test(evo_model, n_trials=5, env_mode='novel', model_type='evo_custom')
            else:
                res = {str(n): -1 for n in [1,3,5,10,20,50]}
        else:
            print(f"  Testing {name}...")
            res = multi_step_test(model, n_trials=5, env_mode='novel', model_type=mtype)

        all_results[name] = res
        print(f"  {name} results: {res}")

    # ---- 6. Output summary ----
    print("\n" + "=" * 70)
    print("Raw Multi-step MSE Results (novel environment, n_trials=5 each)")
    print("=" * 70)
    print(f"\n{'N_steps':>8s} | ", end="")
    for name in all_results:
        print(f"{name:>20s}", end=" | ")
    print()
    print("-" * (10 + 23 * len(all_results)))
    for n_steps in ['1', '3', '5', '10', '20', '50']:
        print(f"{int(n_steps):8d} | ", end="")
        for name in all_results:
            v = all_results[name].get(n_steps, 'N/A')
            print(f"{v:20.6f}" if isinstance(v, float) else f"{v:>20s}", end=" | ")
        print()

    # Save raw data
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved to benchmark_results.json")


if __name__ == '__main__':
    main()
