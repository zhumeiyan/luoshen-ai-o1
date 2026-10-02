# yinyang_evo_v10.py
# 阴阳扣数字生命系统 · v10 — Intrinsic Motivation drives LEARNING, not ACTION
#
# v9 的失败根因：IM 强制行动选择（好奇→选 ATTACK）把感官轨迹搅乱，WM 被迫适应混乱输入→精度崩溃。
# v10 解耦：IM 只管学多快，不管做什么。WM 自主决定行动，IM 只驱动学习速率。
#
# 设计：
#   - 行动 = WM 自己选（通过 action_w softmax，BM 通过进化优化）
#   - 学习速率 = base_lr × IM_multiplier
#     - 好奇时（误差快 > 基线 ×1.3）：lr × 1.5（这个区域信息密度高，多学）
#     - 精通时（误差下降中）：lr × 1.2（正在进步，保持学习）
#     - 无聊时（连续可预测）：lr × 0.5（没新信息，省点力气）
#   - 内在适合度仍占 25%，但基于"学到了多少"而非"去了哪里"
#   - 删除 agency reward（需要因果关系，与 v9 一样是噪声源）
#   - mastery reward 保留（自身能力进步）
#   - novelty 保留（去过新地方 = 学到新东西）
#   - boredom 保留（太可预测 → 负奖励）

import random, math, json
from collections import deque, Counter

MAX_NODES = 12
N_CHANNELS = 8
N_ACTIONS = 4


class Environment:
    def __init__(self, seed=None, mode='training'):
        if seed is not None: random.seed(seed)
        self.t = 0; self.mode = mode; self.season = 0; self.substep = 0
        self.light = 50.0; self.light_r = 0.0; self.sound_freq = 440.0; self.sound_amp = 30.0
        self.pressure = 0.0; self.vibration = 0.0; self.smell = [0.0]*5; self.taste = [0.0]*5
        self.temperature = 20.0; self.accel = [0.0, 0.0, 9.8]

    def step(self, action_code=1):
        self.t += 1; self.substep += 1
        if self.substep > 150: self.substep = 0; self.season = (self.season + 1) % 4
        s = self.season
        if action_code == 0: self.light += random.uniform(0.3,1.0); self.sound_amp += random.uniform(0.05,0.2)
        elif action_code == 2: self.light *= 0.97; self.pressure *= 0.92
        elif action_code == 3: self.temperature += random.uniform(-0.5,0.5); self.pressure += random.uniform(0.05,0.2)
        base_light = 50 + 25*math.sin(self.t*0.02 + s*math.pi/2)
        noise_light = random.uniform(-2,2)
        if self.mode == 'training':
            if s == 0:
                self.light += 0.08*(base_light-self.light)+noise_light*0.5
                self.sound_amp += random.uniform(-0.3,0.1); self.pressure += random.uniform(-0.05,0.05)
                if self.substep%20==0: self.sound_amp += random.uniform(3,6)
            elif s == 1:
                self.light += 0.06*(30-self.light)+noise_light
                self.sound_amp += random.uniform(-0.2,0.6)
                if random.random()<0.02: self.sound_amp += random.uniform(5,12)
            elif s == 2:
                self.temperature += 0.06*(28-self.temperature)
                self.pressure += 0.05*(3-self.pressure)
                te = (self.temperature-20)*0.3
                for i in range(5): self.smell[i] += te*0.1
            else:
                self.temperature += 0.06*(17-self.temperature)
                self.pressure += 0.05*(15-self.pressure)+random.uniform(-0.1,0.1)
                if random.random()<0.08: self.accel[0]+=random.uniform(-1.5,1.5); self.accel[1]+=random.uniform(-1.5,1.5)
        else:
            if s == 0:
                v=10*math.sin(self.t*0.03)+5*math.sin(self.t*0.11)
                self.light+=0.1*(50+v-self.light)+random.uniform(-1,1); self.pressure+=random.uniform(-0.1,0.15)
            elif s == 1:
                x=(self.pressure+10)/20; x=3.7*x*(1-x); self.pressure=max(0,min(20,x*20)); self.sound_amp+=random.uniform(-0.5,0.8)
            elif s == 2:
                if self.substep==50: self.light=100-self.light
                self.temperature+=random.uniform(-0.08,0.08); self.sound_freq+=random.uniform(-15,15)
            else:
                self.light+=random.uniform(-3,3); pulse=3.0 if self.substep%30<3 else 0
                self.sound_amp+=random.uniform(-0.8,0.8)+pulse; self.accel[2]=9.8+2*math.sin(self.t*0.2)
        self.light=max(0.0,min(100.0,self.light)); self.light_r=max(-1.0,min(1.0,self.light_r+random.uniform(-0.01,0.01)))
        self.sound_freq=max(20.0,min(20000.0,self.sound_freq)); self.sound_amp=max(0.0,min(120.0,self.sound_amp*0.99))
        self.pressure=max(0.0,min(20.0,self.pressure)); self.vibration=max(-2.0,min(2.0,self.vibration*0.9+random.uniform(-0.1,0.1)))
        for i in range(5):
            self.smell[i]*=0.93; self.smell[i]=max(0.0,min(10.0,self.smell[i]))
            if random.random()<0.02: self.smell[i]+=random.uniform(0,2)
            self.taste[i]*=0.985
            if random.random()<0.04: self.taste[i]+=random.uniform(-0.3,0.3)
            self.taste[i]=max(-5.0,min(5.0,self.taste[i]))
        self.temperature=max(15.0,min(35.0,self.temperature))
        for k in range(3): self.accel[k]=max(-2.0,min(12.0,self.accel[k]))

    def get_sensory(self):
        return [(self.light/50.0)-1.0, (self.sound_amp/60.0)-0.5+(math.log(max(1.0,self.sound_freq))-6)*0.15,
                (self.pressure/10.0)-0.5, (sum(self.smell)/25.0)-0.5, (sum(self.taste)/10.0),
                (self.temperature-25.0)/10.0, (abs(self.accel[2]-9.8)/5.0), 0.0]


def resize_genome(g, new_n):
    old_n = g['params']['n_nodes']; g = {'params': dict(g['params'])}
    g['sensory_w'] = [list(r[:min(old_n,new_n)]) + [random.uniform(-0.3,0.3)]*(new_n-min(old_n,new_n)) for r in g.get('sensory_w',[])] if 'sensory_w' in g else [[random.uniform(-0.3,0.3) for _ in range(new_n)] for _ in range(N_CHANNELS)]
    if len(g['sensory_w']) < N_CHANNELS:
        g['sensory_w'] = [[random.uniform(-0.3,0.3) for _ in range(new_n)] for _ in range(N_CHANNELS)]
    g['predict_w'] = [list(g.get('predict_w',[[]]*new_n)[i]) if i<old_n and i<len(g.get('predict_w',[])) else [random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)] for i in range(new_n)]
    g['recurrent_w'] = [[random.uniform(-0.1,0.1) if i!=j else 0.0 for j in range(new_n)] for i in range(new_n)]
    g['action_w'] = [[random.uniform(-0.2,0.2) for _ in range(new_n)] for _ in range(N_ACTIONS)]
    g['action_bias'] = [random.uniform(-0.1,0.1) for _ in range(N_ACTIONS)]
    g['params']['n_nodes'] = new_n  # CRITICAL: update n_nodes to match new size
    return g


class WorldModel:
    def __init__(self, genome):
        self.params = genome['params']
        n = self.params['n_nodes']
        self.nodes = [0.0]*n; self.last_prediction = [0.0]*N_CHANNELS
        self.sensory_w = genome['sensory_w']; self.predict_w = genome['predict_w']
        self.recurrent_w = genome['recurrent_w']; self.action_w = genome['action_w']
        self.action_bias = genome['action_bias']
        self.prediction_errors = deque(maxlen=80)
        self.cumulative_mse = 0.0; self.n_predictions = 0

    def forward(self, sensory_vec):
        n = len(self.nodes); decay = self.params['wm_decay']
        predicted_next = [math.tanh(sum(self.predict_w[j][i]*self.nodes[j] for j in range(n))*self.params['pred_gain']) for i in range(N_CHANNELS)]
        new_nodes = []
        for j in range(n):
            val = self.nodes[j]*decay
            for i in range(N_CHANNELS): val += self.sensory_w[i][j]*sensory_vec[i]*self.params['sensory_gain']
            for k in range(n): val += self.recurrent_w[j][k]*self.nodes[k]
            if self.last_prediction and len(self.last_prediction)==N_CHANNELS:
                for i in range(N_CHANNELS): val += (sensory_vec[i]-self.last_prediction[i])*self.sensory_w[i][j]*0.05
            new_nodes.append(math.tanh(val))
        self.nodes = new_nodes; self.last_prediction = predicted_next[:]
        return predicted_next[:], new_nodes[:]

    def choose_action(self):
        n = len(self.nodes)
        logits = [self.action_bias[a] + sum(self.action_w[a][j]*self.nodes[j] for j in range(n)) for a in range(N_ACTIONS)]
        max_l = max(logits); exps = [math.exp(l-max_l) for l in logits]; sum_e = sum(exps)
        if sum_e == 0: return random.randrange(N_ACTIONS)
        probs = [e/sum_e for e in exps]; r = random.random(); cum = 0.0
        for i,p in enumerate(probs):
            cum += p
            if r <= cum: return i
        return N_ACTIONS-1

    def forward_from_nodes(self, nodes, sensory_vec):
        n = len(nodes); decay = self.params['wm_decay']
        predicted_next = [math.tanh(sum(self.predict_w[j][i]*nodes[j] for j in range(min(n,len(self.predict_w))))*self.params['pred_gain']) for i in range(N_CHANNELS)]
        new_nodes = []
        for j in range(n):
            val = nodes[j]*decay
            for i in range(N_CHANNELS):
                if j < len(self.sensory_w[i]): val += self.sensory_w[i][j]*sensory_vec[i]*self.params['sensory_gain']
            for k in range(n):
                if k < len(self.recurrent_w[j]) and j < len(self.recurrent_w): val += self.recurrent_w[j][k]*nodes[k]
            if self.last_prediction and len(self.last_prediction)==N_CHANNELS:
                for i in range(N_CHANNELS):
                    if j < len(self.sensory_w[i]): val += (sensory_vec[i]-self.last_prediction[i])*self.sensory_w[i][j]*0.03
            new_nodes.append(math.tanh(val))
        return predicted_next, new_nodes

    def learn(self, actual_next, predicted_next, rate):
        n = len(self.nodes)
        errors = [actual_next[i]-predicted_next[i] for i in range(N_CHANNELS)]
        mse = sum(e*e for e in errors)/N_CHANNELS
        self.prediction_errors.append(mse); self.cumulative_mse += mse; self.n_predictions += 1
        for i in range(N_CHANNELS):
            for j in range(n):
                delta = rate*errors[i]*self.nodes[j]*0.01
                self.predict_w[j][i] = max(-5.0, min(5.0, self.predict_w[j][i]+delta))
        for j in range(n):
            for i in range(N_CHANNELS):
                delta = rate*self.nodes[j]*actual_next[i]*0.001
                self.sensory_w[i][j] = max(-5.0, min(5.0, self.sensory_w[i][j]+delta))


# ============================================================
# v10 内在动机：只管学多快，不管做什么
# ============================================================

class IntrinsicMotivation:
    def __init__(self):
        self.ema_error = 0.05; self.ema_error_fast = 0.05
        self.sensory_history = deque(maxlen=200)
        self.drive_energy = 0.0; self.boredom = 0.0
        self.last_error_for_mastery = None  # 用于 mastery 信号（误差是否下降）

    def get_lr_multiplier(self, current_mse):
        """
        v10 核心：IM 只返回学习速率倍率。
        好奇（误差高于基线）→ 多学；无聊（太可预测）→ 少学。
        不影响行动——WM 自己选行动。
        """
        # EMA 更新
        alpha_slow, alpha_fast = 0.02, 0.1
        self.ema_error = (1-alpha_slow)*self.ema_error + alpha_slow*current_mse
        self.ema_error_fast = (1-alpha_fast)*self.ema_error_fast + alpha_fast*current_mse

        # 好奇：当前误差高于慢基线 → 1.5x
        if self.ema_error_fast > self.ema_error * 1.3:
            return 1.5
        # 精通中（误差下降趋势）→ 1.2x
        if self.last_error_for_mastery is not None:
            if self.ema_error_fast < self.last_error_for_mastery * 0.95:
                self.last_error_for_mastery = self.ema_error_fast
                return 1.2
        self.last_error_for_mastery = self.ema_error_fast

        # 无聊（太可预测）→ 0.5x
        if current_mse < 0.005 and self.ema_error_fast < 0.005:
            self.boredom += 0.0008
            if self.boredom > 0.03:
                return 0.5  # 无聊时少学
            return 0.8
        else:
            self.boredom *= 0.95

        return 1.0  # 默认

    def get_fitness_reward(self, current_mse, sensory_after):
        """
        内在适合度奖励（占 fitness 的 25%）。
        基于"我学到了多少"而非"我去了哪里"。
        - mastery: 误差下降时给正奖励（我变强了）
        - novelty: 见过新状态给正奖励（我拓展了知识边界）
        - boredom_penalty: 太可预测给负奖励（我在原地空转）
        """
        r = 0.0

        # Mastery: 误差下降 → 奖励
        if self.last_error_for_mastery is not None and len(self.sensory_history) > 10:
            if current_mse < 0.01:
                mastery_reward = min(0.04, (0.01 - current_mse) * 0.15)
                r += mastery_reward

        # Novelty: 新状态 → 奖励
        if len(self.sensory_history) > 5:
            dists = [math.sqrt(sum((sensory_after[i]-past[i])**2 for i in range(N_CHANNELS))) for past in list(self.sensory_history)[-20:]]
            if dists:
                min_dist = min(dists); max_dist = max(dists) if max(dists)>0 else 1.0
                novelty = min_dist / max(1e-3, max_dist)
                r += min(0.03, novelty * 0.05)

        self.sensory_history.append(sensory_after[:])

        # Boredom penalty
        if self.boredom > 0.03:
            r -= 0.015

        # 能量衰减
        self.drive_energy = self.drive_energy * 0.99 + r
        self.drive_energy = max(-5.0, min(10.0, self.drive_energy))
        return r


class Organism:
    def __init__(self, genome=None, lineage_id=None):
        self.genome = genome if genome else self.random_genome()
        self.wm = WorldModel(self.genome)
        self.im = IntrinsicMotivation()
        self.senses_prev = None; self.senses_before_action = None
        self.state = [0.0]*N_CHANNELS; self.state_history = []
        self.action_history = []; self.alive = True; self.age = 0
        self.energy = 8.0; self.lineage_id = lineage_id or self._gen_id()
        self.parents = []; self.behavior_trace = []
        self._predicted_next = None; self._action_chosen = None; self.intrinsic_log = deque(maxlen=50)

    def wm_forward_from_nodes(self, nodes, sensory_vec): return self.wm.forward_from_nodes(nodes, sensory_vec)
    def _gen_id(self):
        import hashlib
        return hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]

    @staticmethod
    def random_genome():
        n = random.randint(3, MAX_NODES)
        return {
            'sensory_w': [[random.uniform(-0.3,0.3) for _ in range(n)] for _ in range(N_CHANNELS)],
            'predict_w': [[random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)] for _ in range(n)],
            'recurrent_w': [[random.uniform(-0.1,0.1) if i!=j else 0.0 for j in range(n)] for i in range(n)],
            'action_w': [[random.uniform(-0.2,0.2) for _ in range(n)] for _ in range(N_ACTIONS)],
            'action_bias': [random.uniform(-0.1,0.1) for _ in range(N_ACTIONS)],
            'params': {'n_nodes':n, 'wm_decay':random.uniform(0.7,0.95), 'sensory_gain':random.uniform(0.3,2.0),
                       'pred_gain':random.uniform(0.5,2.0), 'learning_rate':random.uniform(0.01,0.3),
                       'metabolic':random.uniform(0.02,0.12)}
        }

    def plan(self, sensory):
        if self.senses_prev is not None:
            sensory[N_CHANNELS-1] = math.sqrt(sum((sensory[i]-self.senses_prev[i])**2 for i in range(N_CHANNELS)))/N_CHANNELS
        else: sensory[N_CHANNELS-1] = 0.0
        self.senses_before_action = sensory[:]; self.senses_prev = sensory[:]
        self.state = sensory[:]; self.state_history.append(sensory[:])
        if len(self.state_history) > 300: self.state_history = self.state_history[-300:]
        predicted, nodes = self.wm.forward(sensory)
        self._predicted_next = predicted
        action_idx = self.wm.choose_action()  # WM 自主选！不干预！
        self._action_chosen = action_idx
        self.action_history.append(action_idx)
        if len(self.action_history) > 200: self.action_history = self.action_history[-200:]
        return predicted, action_idx

    def learn_and_update(self, actual_next, action_code):
        if not self.alive: return
        predicted = self._predicted_next
        if predicted is None: return
        mse = sum((actual_next[i]-predicted[i])**2 for i in range(N_CHANNELS))/N_CHANNELS

        # v10: IM 驱动学习速率
        lr_mult = self.im.get_lr_multiplier(mse)
        effective_lr = self.wm.params['learning_rate'] * lr_mult
        effective_lr = max(0.001, min(0.5, effective_lr))  # clamp

        self.wm.learn(actual_next, predicted, effective_lr)
        prediction_reward = 1.0 / (1.0 + mse)
        n_nodes = len(self.wm.nodes)
        metabolic_cost = self.wm.params['metabolic']*(1+n_nodes*0.04)
        self.energy += prediction_reward * 0.5
        action_costs = [0.03, 0.0, 0.01, 0.06]
        action_bonus = {0:0.05 if mse<0.3 else 0.0, 1:0.0, 2:0.0, 3:0.08 if mse<0.2 else 0.02}
        if 0 <= self._action_chosen < N_ACTIONS:
            metabolic_cost += action_costs[self._action_chosen]
            self.energy += action_bonus.get(self._action_chosen, 0)
        # v10: 内在动机适合度奖励
        intrinsic_r = self.im.get_fitness_reward(mse, actual_next)
        self.energy -= metabolic_cost
        self.energy = max(-5.0, min(20.0, self.energy))
        self.behavior_trace.append({'action':self._action_chosen,'err':mse,'energy':self.energy,
                                     'intrinsic':intrinsic_r,'lr_mult':lr_mult,'boredom':self.im.boredom,'drive_energy':self.im.drive_energy})
        if len(self.behavior_trace)>200: self.behavior_trace = self.behavior_trace[-200:]
        if self.energy <= 0: self.alive = False
        self.age += 1

    def clone_mutate(self, rate=0.15, magnitude=0.25):
        child_g = self.__class__.random_genome()  # fallback - should use proper copy
        # actually copy genome
        import copy
        child_g = copy.deepcopy(self.genome)
        n = child_g['params']['n_nodes']
        for i in range(N_CHANNELS):
            for j in range(n):
                if random.random()<rate:
                    child_g['sensory_w'][i][j] = max(-5.0,min(5.0,child_g['sensory_w'][i][j]+random.uniform(-magnitude,magnitude)*0.3))
        for i in range(n):
            for j in range(N_CHANNELS):
                if random.random()<rate:
                    child_g['predict_w'][i][j] = max(-5.0,min(5.0,child_g['predict_w'][i][j]+random.uniform(-magnitude,magnitude)*0.3))
        for i in range(n):
            for j in range(n):
                if i!=j and random.random()<rate*0.5:
                    child_g['recurrent_w'][i][j] = max(-5.0,min(5.0,child_g['recurrent_w'][i][j]+random.uniform(-magnitude,magnitude)*0.2))
        for a in range(N_ACTIONS):
            for j in range(n):
                if random.random()<rate:
                    child_g['action_w'][a][j] = max(-5.0,min(5.0,child_g['action_w'][a][j]+random.uniform(-magnitude,magnitude)*0.25))
        if random.random()<rate:
            child_g['params']['wm_decay'] = max(0.5,min(0.98,child_g['params']['wm_decay']+random.uniform(-0.02,0.02)))
        if random.random()<rate:
            child_g['params']['learning_rate'] = max(0.001,min(0.5,child_g['params']['learning_rate']+random.uniform(-0.02,0.02)))
        if random.random()<rate:
            child_g['params']['metabolic'] = max(0.01,min(0.2,child_g['params']['metabolic']+random.uniform(-0.01,0.01)))
        if random.random()<rate*0.8:
            delta = random.choice([-1,1]); new_n = max(2,min(MAX_NODES, child_g['params']['n_nodes']+delta))
            if new_n != child_g['params']['n_nodes']: child_g = resize_genome(child_g, new_n)
        child = Organism(genome=child_g, lineage_id=self._gen_id())
        child.parents = [self.lineage_id]
        return child

    def behavior_fingerprint(self):
        if len(self.action_history)<10: return (0,)*N_ACTIONS+(0,)
        freq = [0]*N_ACTIONS
        for a in self.action_history:
            if 0<=a<N_ACTIONS: freq[a]+=1
        total = max(1,sum(freq))
        if self.wm.prediction_errors:
            avg_err = sum(self.wm.prediction_errors)/len(self.wm.prediction_errors)
        else: avg_err = 1.0
        return tuple(round(f/total,3) for f in freq) + (round(avg_err,4),)


def genome_distance(g1,g2):
    n1=g1['params']['n_nodes']; n2=g2['params']['n_nodes']; min_n=min(n1,n2)
    dist=0.0; count=0
    for i in range(min_n):
        for j in range(N_CHANNELS):
            dist+=(g1['predict_w'][i][j]-g2['predict_w'][i][j])**2; count+=1
    for i in range(N_CHANNELS):
        for j in range(min_n):
            dist+=(g1['sensory_w'][i][j]-g2['sensory_w'][i][j])**2; count+=1
    for key in ['wm_decay','learning_rate','metabolic','sensory_gain','pred_gain']:
        if key in g1['params'] and key in g2['params']:
            dist+=(g1['params'][key]-g2['params'][key])**2; count+=1
    dist+=((n1-n2)*0.5)**2; count+=1
    return math.sqrt(dist/max(1,count))


class World:
    def __init__(self, n_organisms=30, steps_per_gen=400, difficulty=1.0):
        self.n=n_organisms; self.steps_per_gen=steps_per_gen; self.difficulty=difficulty
        self.population=[Organism() for _ in range(n_organisms)]; self.generation=0
        self.history=[]; self.diversity_log=[]

    def aggregate_action(self, actions):
        if not actions: return 1
        counts=[0]*N_ACTIONS
        for a in actions:
            if 0<=a<N_ACTIONS: counts[a]+=1
        return counts.index(max(counts))

    def run_generation(self, mode='training'):
        env = Environment(seed=random.randint(0,999999), mode=mode)
        for step in range(self.steps_per_gen):
            sensory = env.get_sensory(); proposals=[]
            for org in self.population:
                if not org.alive: continue
                _, action = org.plan(sensory); proposals.append(action)
            action_code = self.aggregate_action(proposals)
            env.step(action_code); actual_next = env.get_sensory()
            for org in self.population:
                if not org.alive: continue
                org.learn_and_update(actual_next, action_code)
        return [o for o in self.population if o.alive]

    def novelty_bonus(self, org):
        if len(self.population)<5: return 0.1
        fp = org.behavior_fingerprint(); dists=[]
        for other in self.population:
            if other is org or not other.alive: continue
            ofp = other.behavior_fingerprint()
            if len(ofp)!=len(fp): continue
            dist = math.sqrt(sum((a-b)**2 for a,b in zip(fp,ofp)))
            dists.append(dist)
        if not dists: return 0.1
        dists.sort(); k=min(5,len(dists))
        return min(1.0, sum(dists[:k])/k)

    def compute_fitness(self, org):
        if not org.alive and org.age<20: return 0.0
        survival = min(1.0, org.age/300.0)
        energy_score = min(1.0, max(0, org.energy)/10.0) if org.alive else 0.0
        if org.wm.prediction_errors:
            avg_err = sum(org.wm.prediction_errors)/len(org.wm.prediction_errors)
            prediction_score = 1.0/(1.0+avg_err*5)
        else: prediction_score=0.0
        novelty = self.novelty_bonus(org)
        intrinsic_fitness = min(1.0, max(0, org.im.drive_energy+2.0)/8.0)
        return 0.15*survival + 0.10*energy_score + 0.35*prediction_score + 0.15*novelty + 0.25*intrinsic_fitness

    def select_and_reproduce(self):
        fits = [(o, self.compute_fitness(o)) for o in self.population]
        fits.sort(key=lambda x: x[1], reverse=True)
        n_survive = max(4, self.n//3)
        survivors = [o for o,f in fits[:n_survive]]
        survivor_fits = [f for o,f in fits[:n_survive]]
        new_pop = []
        if survivors and survivor_fits[0]>0.1:
            elite = survivors[0].clone_mutate(rate=0.02, magnitude=0.03)
            new_pop.append(elite)
        attempts = 0
        while len(new_pop)<self.n and attempts<self.n*10:
            if survivors and sum(survivor_fits)>0:
                probs = [max(0.01,f) for f in survivor_fits[:n_survive]]
                sum_p = sum(probs); probs = [p/sum_p for p in probs]
                r = random.random(); cum = 0.0; chosen = survivors[0]
                for i,p in enumerate(probs):
                    cum+=p
                    if r<=cum: chosen = survivors[i]; break
                new_pop.append(chosen.clone_mutate(rate=0.12, magnitude=0.2))
            else: new_pop.append(Organism())
            attempts+=1
        self.population = new_pop[:self.n]
        for o in self.population:
            o.energy=8.0; o.age=0; o.alive=True
            o.wm = WorldModel(o.genome); o.im = IntrinsicMotivation()
            o.intrinsic_log = deque(maxlen=50)

    def genome_diversity(self):
        sample = self.population[:15]; dists=[]
        for i in range(len(sample)):
            for j in range(i+1,len(sample)): dists.append(genome_distance(sample[i].genome,sample[j].genome))
        if not dists: return {'mean':0,'std':0,'min':0,'max':0}
        mean_d = sum(dists)/len(dists); var_d = sum((d-mean_d)**2 for d in dists)/len(dists)
        return {'mean':round(mean_d,4),'std':round(math.sqrt(var_d),4),'min':round(min(dists),4),'max':round(max(dists),4)}

    def checkpoint(self):
        alive = [o for o in self.population if o.alive]
        if not alive: return {'gen':self.generation,'alive':0,'best_fitness':0,'avg_fitness':0,'avg_pred_error':1.0,'intrinsic_drive':0,'avg_boredom':0,'div_mean':0}
        fits = [self.compute_fitness(o) for o in self.population]
        avg_fit = sum(fits)/len(fits)
        intrinsic_drives = [o.im.drive_energy for o in alive]
        avg_intrinsic = sum(intrinsic_drives)/max(1,len(intrinsic_drives))
        boredoms = [o.im.boredom for o in alive]
        avg_boredom = sum(boredoms)/max(1,len(boredoms))
        pred_errs = []
        for o in alive:
            if o.wm.prediction_errors: pred_errs.append(sum(o.wm.prediction_errors)/len(o.wm.prediction_errors))
        pred_errs.sort()
        avg_pred = sum(pred_errs)/max(1,len(pred_errs)) if pred_errs else 1.0
        div = self.genome_diversity()
        return {'gen':self.generation,'alive':len(alive),'best_fitness':max(fits),'avg_fitness':avg_fit,
                'avg_pred_error':avg_pred,'intrinsic_drive':round(avg_intrinsic,4),
                'avg_boredom':round(avg_boredom,4),'div_mean':div['mean']}

    def run(self, num_gens=200):
        print(f"\n{'='*70}")
        print(f"Evolution v10 — IM drives LEARNING, not ACTION")
        print(f"n_orgs={self.n} steps/gen={self.steps_per_gen}")
        print(f"{'='*70}\n")
        for g in range(num_gens):
            self.generation = g
            alive = self.run_generation(mode='training')
            ckpt = self.checkpoint(); self.history.append(ckpt)
            self.select_and_reproduce()
            if g%10==0: self.diversity_log.append({'gen':g,**self.genome_diversity()})
            if g%5==0:
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} | fit={ckpt['best_fitness']:.4f}/{ckpt['avg_fitness']:.4f} | pred={ckpt['avg_pred_error']:.4f} | INTRIN={ckpt['intrinsic_drive']:.3f} boredom={ckpt['avg_boredom']:.3f}")
        with open('evolution_log_v10.json','w') as f:
            json.dump({'history':self.history,'diversity_log':self.diversity_log,'num_gens':num_gens},f,indent=2,default=str)
        print(f"\n保存到 evolution_log_v10.json")
        return self.history[-1]


def run_multi_step_test(world, n_list=None, n_trials=5):
    if n_list is None: n_list = [1,3,5,10,20,50]
    print(f"\n{'='*70}"); print(f"v10 Multi-step Prediction Test"); print(f"{'='*70}")
    fits = [(o, world.compute_fitness(o)) for o in world.population]
    fits.sort(key=lambda x: x[1], reverse=True)
    best_org = fits[0][0]
    results = {}
    for n_steps in n_list:
        trial_errs = []
        for trial in range(n_trials):
            seed = trial*777+123
            env = Environment(seed=seed, mode='novel')
            best_org.wm.nodes = [0.0]*len(best_org.wm.nodes)
            best_org.wm.last_prediction = [0.0]*N_CHANNELS
            for _ in range(50): s=env.get_sensory(); best_org.wm.forward(s); env.step(1)
            sensory = env.get_sensory()
            predictions = []; nodes = list(best_org.wm.nodes); curr = sensory[:]
            for n in range(n_steps):
                pred, nodes = best_org.wm.forward_from_nodes(nodes, curr)
                predictions.append(pred[:]); curr = pred
            env_gt = Environment(seed=seed, mode='novel')
            for _ in range(50): s=env_gt.get_sensory(); env_gt.step(1)
            gts = []
            for n in range(n_steps): env_gt.step(1); gts.append(env_gt.get_sensory()[:])
            errs = [sum((predictions[n][i]-gts[n][i])**2 for i in range(N_CHANNELS))/N_CHANNELS for n in range(min(n_steps,len(gts)))]
            if errs: trial_errs.append(sum(errs)/len(errs))
        results[str(n_steps)] = sum(trial_errs)/max(1,len(trial_errs)) if trial_errs else 1.0
    for n_str,v in results.items():
        print(f"  N={int(n_str):3d} 步 | EVO-IM={v:.5f}")
    with open('multi_step_prediction_v10.json','w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    random.seed(42)
    world = World(n_organisms=30, steps_per_gen=400, difficulty=1.0)
    final = world.run(num_gens=200)
    run_multi_step_test(world)
