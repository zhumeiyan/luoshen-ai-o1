# yinyang_evo_v15.py
# 阴阳扣数字生命系统 · v15 — Multi-Niche Evolution (Stable Attractors + IM Niche Selection)
#
# v14 教训：共享池每代重置 → 还是基因组无法针对性收敛。
# v15 设计：
#   1. 共享吸引子池固定不变（全进化周期内同一套 9 区域参数）→ 进化可针对性收敛
#   2. 基因编码"区域选择策略"而非"特定动力学权重"（权重靠生命周期内学习）
#   3. 每代开始时重新初始化区域状态（清零）但保持动力学参数
#   4. IM 的 boredom 反应驱动个体离开低 σ 无聊区 → 高 σ 挑战区
#   5. 学习率 0.08 常数 + 更强初始权重 → 生命周期内收敛

import random, math, json, hashlib
from collections import deque

MAX_NODES = 12
N_CHANNELS = 8
N_ACTIONS = 4  # STAY, EXPLORE, SEEK_HOT, COOL

class StableRegionPool:
    """固定吸引子池：整个进化周期内参数不变"""
    def __init__(self, seed=42):
        random.seed(seed)
        self.regions = []
        for i in range(9):
            sigma = 0.05 + i * 0.08  # 0.05 ~ 0.69
            freq = [random.uniform(0.02, 0.06) for _ in range(N_CHANNELS)]
            phase = [random.uniform(0, 2*math.pi) for _ in range(N_CHANNELS)]
            coupling = 0.15 + sigma * 0.08
            self.regions.append({
                'sigma': sigma, 'freq': freq, 'phase': phase,
                'coupling': coupling, 'state': [0.0]*N_CHANNELS, 't': 0
            })
        random.seed()  # 释放

    def reset_states(self):
        """每代开始时重置状态（清零）"""
        for r in self.regions:
            r['state'] = [0.0]*N_CHANNELS
            r['t'] = 0

    def step_region(self, region_idx):
        r = self.regions[region_idx]
        r['t'] += 1
        t = r['t']
        noise = r['sigma'] * 0.04
        coherence = 0.12 * (1 - r['sigma'] * 0.25)
        new_state = []
        for i in range(N_CHANNELS):
            attractor = math.sin(t * r['freq'][i] + r['phase'][i])
            coupling_signal = 0.0
            for j in range(N_CHANNELS):
                if i != j:
                    coupling_signal += r['state'][j] * r['coupling'] / (N_CHANNELS-1)
            val = r['state'][i] + coherence * (attractor - r['state'][i]) + coupling_signal * 0.05 + random.gauss(0, noise)
            val = max(-1.0, min(1.0, val))
            new_state.append(val)
        r['state'] = new_state[:]
        return new_state[:]

    def get_sensory(self, region_idx):
        r = self.regions[region_idx]
        return [r['state'][i] + random.gauss(0, r['sigma'] * 0.01) for i in range(N_CHANNELS)]


def resize_genome(g, new_n):
    old_n = g['params']['n_nodes']
    g2 = {'params': dict(g['params'])}
    sw = g.get('sensory_w', [])
    g2['sensory_w'] = [list(r[:min(old_n,new_n)]) + [random.uniform(-0.6,0.6)]*(new_n-min(old_n,new_n)) for r in sw]
    if len(g2['sensory_w']) < N_CHANNELS:
        g2['sensory_w'] = [[random.uniform(-0.6,0.6) for _ in range(new_n)] for _ in range(N_CHANNELS)]
    pw = g.get('predict_w', [])
    g2['predict_w'] = [list(pw[i]) if i < old_n and i < len(pw) else [random.uniform(-0.6,0.6) for _ in range(N_CHANNELS)] for i in range(new_n)]
    g2['recurrent_w'] = [[random.uniform(-0.3,0.3) if i!=j else 0.0 for j in range(new_n)] for i in range(new_n)]
    g2['action_w'] = [[random.uniform(-0.4,0.4) for _ in range(new_n)] for _ in range(N_ACTIONS)]
    g2['action_bias'] = [random.uniform(-0.1,0.1) for _ in range(N_ACTIONS)]
    g2['params']['n_nodes'] = new_n
    return g2


class WorldModel:
    def __init__(self, genome):
        self.params=genome['params']
        n=self.params['n_nodes']; self.nodes=[0.0]*n; self.last_prediction=[0.0]*N_CHANNELS
        self.sensory_w=genome['sensory_w']; self.predict_w=genome['predict_w']
        self.recurrent_w=genome['recurrent_w']; self.action_w=genome['action_w']; self.action_bias=genome['action_bias']
        self.prediction_errors=deque(maxlen=100)
        self.n_predictions=0

    def forward(self, sensory_vec):
        n=len(self.nodes); decay=self.params['wm_decay']
        predicted_next=[math.tanh(sum(self.predict_w[j][i]*self.nodes[j] for j in range(n))*self.params['pred_gain']) for i in range(N_CHANNELS)]
        new_nodes=[]
        for j in range(n):
            val=self.nodes[j]*decay
            for i in range(N_CHANNELS): val+=self.sensory_w[i][j]*sensory_vec[i]*self.params['sensory_gain']
            for k in range(n): val+=self.recurrent_w[j][k]*self.nodes[k]
            if self.last_prediction and len(self.last_prediction)==N_CHANNELS:
                for i in range(N_CHANNELS): val+=(sensory_vec[i]-self.last_prediction[i])*self.sensory_w[i][j]*0.05
            new_nodes.append(math.tanh(val))
        self.nodes=new_nodes; self.last_prediction=predicted_next[:]
        return predicted_next[:], new_nodes[:]

    def choose_action(self, bias=None):
        n=len(self.nodes)
        logits=[self.action_bias[a]+sum(self.action_w[a][j]*self.nodes[j] for j in range(n)) for a in range(N_ACTIONS)]
        if bias:
            for a, b in bias.items():
                if 0 <= a < N_ACTIONS: logits[a] += b
        max_l=max(logits); exps=[math.exp(l-max_l) for l in logits]; sum_e=sum(exps)
        if sum_e==0: return random.randrange(N_ACTIONS)
        probs=[e/sum_e for e in exps]; r=random.random(); cum=0.0
        for i,p in enumerate(probs):
            cum+=p
            if r<=cum: return i
        return N_ACTIONS-1

    def forward_from_nodes(self, nodes, sensory_vec):
        n=len(nodes); decay=self.params['wm_decay']
        predicted_next=[math.tanh(sum(self.predict_w[j][i]*nodes[j] for j in range(min(n,len(self.predict_w))))*self.params['pred_gain']) for i in range(N_CHANNELS)]
        new_nodes=[]
        for j in range(n):
            val=nodes[j]*decay
            for i in range(N_CHANNELS):
                if j<len(self.sensory_w[i]): val+=self.sensory_w[i][j]*sensory_vec[i]*self.params['sensory_gain']
            for k in range(n):
                if k<len(self.recurrent_w[j]) and j<len(self.recurrent_w): val+=self.recurrent_w[j][k]*nodes[k]
            if self.last_prediction and len(self.last_prediction)==N_CHANNELS:
                for i in range(N_CHANNELS):
                    if j<len(self.sensory_w[i]): val+=(sensory_vec[i]-self.last_prediction[i])*self.sensory_w[i][j]*0.03
            new_nodes.append(math.tanh(val))
        return predicted_next, new_nodes

    def learn(self, actual_next, predicted_next, rate):
        n=len(self.nodes); errors=[actual_next[i]-predicted_next[i] for i in range(N_CHANNELS)]
        mse=sum(e*e for e in errors)/N_CHANNELS; self.prediction_errors.append(mse); self.n_predictions+=1
        # 学习率常数 0.08
        for i in range(N_CHANNELS):
            for j in range(n):
                delta=rate*errors[i]*self.nodes[j]*0.08
                self.predict_w[j][i]=max(-5.0,min(5.0,self.predict_w[j][i]+delta))
        for j in range(n):
            for i in range(N_CHANNELS):
                delta=rate*self.nodes[j]*actual_next[i]*0.005
                self.sensory_w[i][j]=max(-5.0,min(5.0,self.sensory_w[i][j]+delta))


class IntrinsicMotivation:
    """内在动机：无聊驱动探索，学习速度驱动停留"""
    def __init__(self):
        self.ema_error=0.1; self.ema_error_fast=0.1
        self.last_error=0.1; self.boredom=0.0; self.drive_energy=0.0
        self.lr_log=deque(maxlen=100); self.region_history=deque(maxlen=200)
        self.region_scores={i: 0.0 for i in range(9)}
        self.recent_deltas=deque(maxlen=50)

    def update(self, current_mse, current_region, region_sigma, actual_next, predicted):
        alpha_slow, alpha_fast = 0.03, 0.15
        self.ema_error = (1-alpha_slow)*self.ema_error + alpha_slow*current_mse
        self.ema_error_fast = (1-alpha_fast)*self.ema_error_fast + alpha_fast*current_mse

        delta = current_mse - self.last_error
        self.recent_deltas.append(delta)
        avg_delta = sum(self.recent_deltas)/max(1,len(self.recent_deltas))

        # 学习速率调制
        if avg_delta > 0.003:  # 误差在增大 → 加速
            lr_mult = 1.5; self.boredom *= 0.8
        elif avg_delta < -0.01:  # 快速减小 → 学得好
            lr_mult = 1.3; self.boredom *= 0.85
        elif abs(avg_delta) < 0.005 and self.ema_error_fast < 0.03:  # 收敛且低误差 → 无聊
            self.boredom += 0.005; lr_mult = 0.5
        elif self.ema_error_fast < 0.05:
            self.boredom += 0.002; lr_mult = 0.8
        else:
            lr_mult = 1.0; self.boredom = max(0, self.boredom - 0.001)

        self.boredom = max(0.0, min(1.0, self.boredom))
        self.last_error = current_mse
        self.lr_log.append(lr_mult)

        # 区域价值经验
        info_value = region_sigma * (1.0 / max(0.001, current_mse + 0.005))
        self.region_scores[current_region] = 0.95*self.region_scores[current_region] + 0.05*info_value

        # 内在能量
        reward = 0.0
        if region_sigma > 0.30 and current_mse > 0.01:  # 中高 σ 有学习空间
            reward += 0.02
        if avg_delta < -0.003:
            reward += 0.01
        if self.boredom > 0.05:
            reward -= 0.01
        self.drive_energy = self.drive_energy * 0.99 + reward
        self.drive_energy = max(-3.0, min(5.0, self.drive_energy))
        self.region_history.append(current_region)
        return lr_mult

    def get_action_bias(self):
        bias = {}
        if self.boredom > 0.03:
            bias[2] = 0.5  # SEEK-HOT
            bias[1] = 0.25  # EXPLORE
        elif self.ema_error_fast > self.ema_error * 1.1:
            bias[0] = 0.15
        return bias

    def get_stats(self):
        return {'drive': round(self.drive_energy,3), 'boredom': round(self.boredom,4), 'lr': self.lr_log[-1] if self.lr_log else 1.0}


class Organism:
    def __init__(self, genome=None, lineage_id=None):
        self.genome=genome if genome else self.random_genome()
        self.wm=WorldModel(self.genome); self.im=IntrinsicMotivation()
        self.senses_prev=None
        self.action_history=[]; self.alive=True; self.age=0
        self.energy=8.0; self.lineage_id=lineage_id or self._gen_id()
        self.parents=[]; self.behavior_trace=[]
        self._predicted_next=None; self._action_chosen=None
        self.total_information=0.0; self.region_counts={}
        self.current_region = random.randint(0, 8)

    def _gen_id(self): return hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]

    @staticmethod
    def random_genome():
        n=random.randint(3,MAX_NODES)
        return {
            'sensory_w': [[random.uniform(-0.6,0.6) for _ in range(n)] for _ in range(N_CHANNELS)],
            'predict_w': [[random.uniform(-0.6,0.6) for _ in range(N_CHANNELS)] for _ in range(n)],
            'recurrent_w': [[random.uniform(-0.3,0.3) if i!=j else 0.0 for j in range(n)] for i in range(n)],
            'action_w': [[random.uniform(-0.4,0.4) for _ in range(n)] for _ in range(N_ACTIONS)],
            'action_bias': [random.uniform(-0.1,0.1) for _ in range(N_ACTIONS)],
            'params': {'n_nodes':n,'wm_decay':random.uniform(0.7,0.95),'sensory_gain':random.uniform(0.3,2.0),
                       'pred_gain':random.uniform(0.5,2.0),'learning_rate':random.uniform(0.02,0.3),'metabolic':random.uniform(0.02,0.12)}
        }

    def plan(self, sensory):
        if self.senses_prev is not None:
            movement = math.sqrt(sum((sensory[i]-self.senses_prev[i])**2 for i in range(N_CHANNELS-1)))/N_CHANNELS
            sensory[N_CHANNELS-1] = sensory[N_CHANNELS-1] * 0.7 + movement * 0.3
        self.senses_prev=sensory[:]
        predicted,_=self.wm.forward(sensory)
        self._predicted_next=predicted
        action_bias=self.im.get_action_bias()
        action_idx=self.wm.choose_action(bias=action_bias)
        self._action_chosen=action_idx
        self.action_history.append(action_idx)
        if len(self.action_history)>200: self.action_history=self.action_history[-200:]
        return predicted, action_idx

    def learn_and_update(self, actual_next, region_sigma, region_id):
        if not self.alive: return
        predicted=self._predicted_next
        if predicted is None: return
        mse=sum((actual_next[i]-predicted[i])**2 for i in range(N_CHANNELS))/N_CHANNELS

        lr_mult=self.im.update(mse, region_id, region_sigma, actual_next, predicted)
        effective_lr=max(0.001,min(0.5,self.wm.params['learning_rate']*lr_mult))
        self.wm.learn(actual_next, predicted, effective_lr)

        prediction_reward=1.0/(1.0+mse)
        n_nodes=len(self.wm.nodes)
        metabolic_cost=self.wm.params['metabolic']*(1+n_nodes*0.04)

        info_bonus = region_sigma * prediction_reward * 0.1
        self.total_information += info_bonus

        self.energy += prediction_reward * 0.3 + info_bonus * 0.15
        self.region_counts[region_id] = self.region_counts.get(region_id, 0) + 1

        action_costs={0:0.008, 1:0.04, 2:0.025, 3:0.025}
        metabolic_cost += action_costs.get(self._action_chosen, 0.02)
        self.energy -= metabolic_cost
        self.energy = max(-5.0, min(20.0, self.energy))

        if self.energy<=0: self.alive=False
        self.age+=1

    def clone_mutate(self, rate=0.15, magnitude=0.25):
        import copy
        child_g=copy.deepcopy(self.genome); n=child_g['params']['n_nodes']
        for i in range(N_CHANNELS):
            for j in range(n):
                if random.random()<rate: child_g['sensory_w'][i][j]=max(-5.0,min(5.0,child_g['sensory_w'][i][j]+random.uniform(-magnitude,magnitude)*0.3))
        for i in range(n):
            for j in range(N_CHANNELS):
                if random.random()<rate: child_g['predict_w'][i][j]=max(-5.0,min(5.0,child_g['predict_w'][i][j]+random.uniform(-magnitude,magnitude)*0.3))
        for i in range(n):
            for j in range(n):
                if i!=j and random.random()<rate*0.5: child_g['recurrent_w'][i][j]=max(-5.0,min(5.0,child_g['recurrent_w'][i][j]+random.uniform(-magnitude,magnitude)*0.2))
        for a in range(N_ACTIONS):
            for j in range(n):
                if random.random()<rate: child_g['action_w'][a][j]=max(-5.0,min(5.0,child_g['action_w'][a][j]+random.uniform(-magnitude,magnitude)*0.25))
        if random.random()<rate: child_g['params']['wm_decay']=max(0.5,min(0.98,child_g['params']['wm_decay']+random.uniform(-0.02,0.02)))
        if random.random()<rate: child_g['params']['learning_rate']=max(0.001,min(0.5,child_g['params']['learning_rate']+random.uniform(-0.02,0.02)))
        if random.random()<rate: child_g['params']['metabolic']=max(0.01,min(0.2,child_g['params']['metabolic']+random.uniform(-0.01,0.01)))
        if random.random()<rate*0.8:
            delta=random.choice([-1,1]); new_n=max(2,min(MAX_NODES,child_g['params']['n_nodes']+delta))
            if new_n!=child_g['params']['n_nodes']: child_g=resize_genome(child_g,new_n)
        child=Organism(genome=child_g,lineage_id=self._gen_id())
        child.parents=[self.lineage_id]
        return child

    def behavior_fingerprint(self):
        if len(self.action_history)<10: return (0,)*N_ACTIONS+(0,)
        freq=[0]*N_ACTIONS
        for a in self.action_history:
            if 0<=a<N_ACTIONS: freq[a]+=1
        total=max(1,sum(freq))
        if self.wm.prediction_errors:
            avg_err=sum(self.wm.prediction_errors)/len(self.wm.prediction_errors)
        else: avg_err=1.0
        return tuple(round(f/total,3) for f in freq)+(round(avg_err,4),)


# ============================================================
# v16 — 袁隆平杂交机制（杂种优势 + 近交衰退）
# ============================================================

def crossover_genomes(g1, g2, mode='row_mix'):
    """
    两个基因组杂交产生后代。
    mode='row_mix': 每行权重随机来自 parent1 或 parent2（行级交叉）
    'uniform': 每个权重独立选择来源
    'layer_swap': 预测网络来自 parent1，行动网络来自 parent2（功能互补）
    """
    import copy
    n1 = g1['params']['n_nodes']
    n2 = g2['params']['n_nodes']
    n = random.choice([n1, n2])
    child_g = {'params': copy.deepcopy(g1['params'])}
    child_g['params']['n_nodes'] = n

    def clip_or_pad(mat, target_rows, target_cols):
        """调整矩阵大小到 target"""
        res = []
        for i in range(target_rows):
            if i < len(mat):
                row = mat[i][:target_cols]
                row += [random.uniform(-0.3,0.3)] * (target_cols - len(row))
            else:
                row = [random.uniform(-0.3,0.3) for _ in range(target_cols)]
            res.append(row)
        return res

    if mode == 'layer_swap':
        # 预测网络来自 parent1，行动网络来自 parent2（功能互补杂交）
        child_g['sensory_w'] = [[random.choice([a,b]) for a,b in zip(r1,r2)]
                                for r1,r2 in zip(
                                    clip_or_pad(g1['sensory_w'], N_CHANNELS, n),
                                    clip_or_pad(g2['sensory_w'], N_CHANNELS, n))]
        child_g['predict_w'] = clip_or_pad(g1['predict_w'], n, N_CHANNELS)
        child_g['recurrent_w'] = clip_or_pad(g1['recurrent_w'], n, n)
        child_g['action_w'] = clip_or_pad(g2['action_w'], N_ACTIONS, n)
        child_g['action_bias'] = [random.choice([a,b]) for a,b in zip(g2['action_bias'], g1['action_bias'])]
    elif mode == 'uniform':
        child_g['sensory_w'] = [[random.choice([a,b]) for a,b in zip(r1,r2)]
                                for r1,r2 in zip(
                                    clip_or_pad(g1['sensory_w'], N_CHANNELS, n),
                                    clip_or_pad(g2['sensory_w'], N_CHANNELS, n))]
        child_g['predict_w'] = [[random.choice([a,b]) for a,b in zip(r1,r2)]
                                for r1,r2 in zip(
                                    clip_or_pad(g1['predict_w'], n, N_CHANNELS),
                                    clip_or_pad(g2['predict_w'], n, N_CHANNELS))]
        child_g['recurrent_w'] = [[random.choice([a,b]) if a is not None and b is not None else random.uniform(-0.3,0.3)
                                   for a,b in zip(r1,r2)]
                                  for r1,r2 in zip(
                                      clip_or_pad(g1['recurrent_w'], n, n),
                                      clip_or_pad(g2['recurrent_w'], n, n))]
        child_g['action_w'] = [[random.choice([a,b]) for a,b in zip(r1,r2)]
                               for r1,r2 in zip(
                                   clip_or_pad(g1['action_w'], N_ACTIONS, n),
                                   clip_or_pad(g2['action_w'], N_ACTIONS, n))]
        child_g['action_bias'] = [random.choice([a,b]) for a,b in zip(g1['action_bias'], g2['action_bias'])]
    else:  # row_mix (default)
        child_g['sensory_w'] = [random.choice([r1,r2]) for r1,r2 in zip(
            clip_or_pad(g1['sensory_w'], N_CHANNELS, n),
            clip_or_pad(g2['sensory_w'], N_CHANNELS, n))]
        child_g['predict_w'] = [random.choice([r1,r2]) for r1,r2 in zip(
            clip_or_pad(g1['predict_w'], n, N_CHANNELS),
            clip_or_pad(g2['predict_w'], n, N_CHANNELS))]
        child_g['recurrent_w'] = [random.choice([r1,r2]) for r1,r2 in zip(
            clip_or_pad(g1['recurrent_w'], n, n),
            clip_or_pad(g2['recurrent_w'], n, n))]
        child_g['action_w'] = [random.choice([r1,r2]) for r1,r2 in zip(
            clip_or_pad(g1['action_w'], N_ACTIONS, n),
            clip_or_pad(g2['action_w'], N_ACTIONS, n))]
        child_g['action_bias'] = [random.choice([a,b]) for a,b in zip(g1['action_bias'], g2['action_bias'])]

    # 混合 params（取平均）
    for key in ['wm_decay','learning_rate','metabolic','sensory_gain','pred_gain']:
        if key in g1['params'] and key in g2['params']:
            child_g['params'][key] = 0.5 * (g1['params'][key] + g2['params'][key])

    return child_g


def select_complementary_pair(survivors, sf):
    """
    选择互补配对：在选择压力下，偏好看行为指纹差异大的配对（生态位互补）。
    这模拟袁隆平的"不同品种杂交"——一个品种抗旱，另一个高产，杂交后代兼具优势。
    """
    n = len(survivors)
    if n < 2:
        return survivors[0], survivors[0]
    # 随机选第一个 parent
    probs = [max(0.01, f) for f in sf[:n]]
    s = sum(probs)
    probs = [p/s for p in probs]
    r = random.random(); cum = 0.0; idx1 = 0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            idx1 = i; break
    p1 = survivors[idx1]

    # 第二个 parent 优先选行为指纹不同的（互补）
    best_j = None; best_div = -1
    candidates = list(range(n))
    random.shuffle(candidates)
    for j in candidates[:min(5, n)]:
        if j == idx1: continue
        fp1 = p1.behavior_fingerprint()
        fp2 = survivors[j].behavior_fingerprint()
        # 行为距离（忽略最后一个元素 average prediction error）
        div = sum((a-b)**2 for a,b in zip(fp1[:-1], fp2[:-1]))
        if div > best_div:
            best_div = div; best_j = j
    p2 = survivors[best_j if best_j is not None else 0]
    return p1, p2


def genome_distance(g1,g2):
    n1=g1['params']['n_nodes']; n2=g2['params']['n_nodes']; min_n=min(n1,n2)
    dist=0.0; count=0
    for i in range(min_n):
        for j in range(N_CHANNELS): dist+=(g1['predict_w'][i][j]-g2['predict_w'][i][j])**2; count+=1
    for i in range(N_CHANNELS):
        for j in range(min_n): dist+=(g1['sensory_w'][i][j]-g2['sensory_w'][i][j])**2; count+=1
    for key in ['wm_decay','learning_rate','metabolic','sensory_gain','pred_gain']:
        if key in g1['params'] and key in g2['params']: dist+=(g1['params'][key]-g2['params'][key])**2; count+=1
    dist+=((n1-n2)*0.5)**2; count+=1
    return math.sqrt(dist/max(1,count))


class World:
    def __init__(self, n_organisms=30, steps_per_gen=400, seed=42):
        self.n=n_organisms; self.steps_per_gen=steps_per_gen
        self.pool = StableRegionPool(seed=seed)  # 整个进化周期固定
        self.population=[Organism() for _ in range(n_organisms)]
        self.generation=0; self.history=[]; self.diversity_log=[]

    def run_generation(self, mode='training'):
        # 每代重置区域状态（清零），但动力学参数不变
        self.pool.reset_states()
        # 重置生物行为记录
        for org in self.population:
            org.behavior_trace=[]; org.region_counts={}
            org.action_history=[]; org.senses_prev=None
            org.current_region = random.randint(0, 8)

        for step in range(self.steps_per_gen):
            # === (1) 所有生物观察当前状态 + 计划行动 ===
            for org in self.population:
                if not org.alive: continue
                rid = org.current_region
                sig = self.pool.regions[rid]['sigma']
                sensory_before = self.pool.get_sensory(rid)
                proprio = (sig - 0.35) * 0.7
                sensory_before_proprio = sensory_before[:N_CHANNELS-1] + [proprio]
                predicted, action = org.plan(sensory_before_proprio)

                # 执行行动 → 可能移动到新区域
                if action == 1:  # EXPLORE
                    org.current_region = random.randint(0, 8)
                elif action == 2:  # SEEK-HOT
                    if org.current_region < 8:
                        org.current_region += 1
                elif action == 3:  # COOL
                    if org.current_region > 0:
                        org.current_region -= 1
                # STAY: 不动

            # === (2) 所有 9 个 region 全局只步进一次（不多次步进） ===
            for rid in range(9):
                self.pool.step_region(rid)

            # === (3) 所有生物观察新状态 + 学习 ===
            for org in self.population:
                if not org.alive: continue
                rid_new = org.current_region
                sig_new = self.pool.regions[rid_new]['sigma']
                new_sensory = self.pool.get_sensory(rid_new)  # 相同状态（共享池只步进一次）
                proprio_new = (sig_new - 0.35) * 0.7
                new_sensory_proprio = new_sensory[:N_CHANNELS-1] + [proprio_new]
                org.learn_and_update(new_sensory_proprio, sig_new, rid_new)

        return [o for o in self.population if o.alive]

    def novelty_bonus(self, org):
        if len(self.population)<5: return 0.1
        fp=org.behavior_fingerprint(); dists=[]
        for other in self.population:
            if other is org or not other.alive: continue
            ofp=other.behavior_fingerprint()
            if len(ofp)!=len(fp): continue
            dist=math.sqrt(sum((a-b)**2 for a,b in zip(fp,ofp))); dists.append(dist)
        if not dists: return 0.1
        dists.sort(); k=min(5,len(dists)); return min(1.0,sum(dists[:k])/k)

    def compute_fitness(self, org):
        if not org.alive and org.age<20: return 0.0
        survival=min(1.0, org.age/300.0)
        energy_score=min(1.0, max(0, org.energy)/10.0) if org.alive else 0.0
        if org.wm.prediction_errors:
            avg_err=sum(org.wm.prediction_errors)/len(org.wm.prediction_errors)
            prediction_score=1.0/(1.0+avg_err*5)
        else: prediction_score=0.0
        novelty=self.novelty_bonus(org)
        info_score=min(1.0, org.total_information/3.0)
        base_fit = 0.25*survival + 0.10*energy_score + 0.40*prediction_score + 0.10*novelty + 0.15*info_score
        # 近交衰退惩罚（来自杂交机制）
        inbreeding_penalty = getattr(org, '_inbreeding', 1.0)
        return base_fit * inbreeding_penalty

    def select_and_reproduce(self, gen=-1):
        """
        袁隆平杂交版繁殖：
        - 70% 后代来自"互补杂交"（选择行为指纹差异大的双亲）
        - 30% 来自"突变克隆"（保留当前最优个体的遗传）
        - 杂交后个体记录 parents_ids，用于后续杂种优势统计
        - 近交惩罚：双亲基因组距离 < 0.5 → 后代适应度乘以 0.6（模拟近交衰退）
        """
        fits=[(o, self.compute_fitness(o)) for o in self.population]
        fits.sort(key=lambda x: x[1], reverse=True)
        n_survive=max(6, self.n//3)
        survivors=[o for o,f in fits[:n_survive]]
        sf=[f for o,f in fits[:n_survive]]
        new_pop=[]

        # Elite clone (unmutated)
        if survivors and sf[0]>0.1:
            elite=survivors[0].clone_mutate(rate=0.02, magnitude=0.03)
            new_pop.append(elite)

        attempts=0
        n_hybrid = int(self.n * 0.7)  # 70% 杂交
        n_mutant = self.n - n_hybrid  # 30% 突变克隆
        hybrid_count = 0
        mutant_count = 0

        while len(new_pop)<self.n and attempts<self.n*20:
            attempts+=1
            if not survivors or sum(sf)<=0:
                new_pop.append(Organism()); continue

            if hybrid_count < n_hybrid:
                # === 杂交后代（袁隆平机制）===
                p1, p2 = select_complementary_pair(survivors, sf)
                # 优先 row_mix（减少层交换破坏协同）；偶尔 layer_swap
                mode = random.choices(['row_mix','layer_swap','uniform'], weights=[0.5, 0.2, 0.3])[0]
                child_g = crossover_genomes(p1.genome, p2.genome, mode=mode)

                # 杂种优势跟踪：记录双亲最大 fitness（用于后代比较）
                p1_fit = self.compute_fitness(p1) if p1.alive else 0
                p2_fit = self.compute_fitness(p2) if p2.alive else 0

                child = Organism(genome=child_g, lineage_id=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8])
                child.parents=[p1.lineage_id, p2.lineage_id]
                child._inbreeding = 1.0
                child._parent_max_fit = max(p1_fit, p2_fit)
                hybrid_count += 1
                new_pop.append(child)

            elif mutant_count < n_mutant:
                # === 突变克隆（保底遗传多样性）===
                probs=[max(0.01,f) for f in sf[:n_survive]]; sum_p=sum(probs); probs=[p/sum_p for p in probs]
                r=random.random(); cum=0.0; chosen=survivors[0]
                for i,p in enumerate(probs):
                    cum+=p
                    if r<=cum: chosen=survivors[i]; break
                child = chosen.clone_mutate(rate=0.12, magnitude=0.2)
                child.parents=[chosen.lineage_id]
                mutant_count += 1
                new_pop.append(child)
            else:
                break

        self.population=new_pop[:self.n]
        for o in self.population:
            o.energy=8.0; o.age=0; o.alive=True
            o.wm=WorldModel(o.genome); o.im=IntrinsicMotivation()
            o.total_information=0.0; o.region_counts={}
            o.action_history=[]; o.senses_prev=None
            o.current_region = random.randint(0, 8)

    def genome_diversity(self):
        sample=self.population[:15]; dists=[]
        for i in range(len(sample)):
            for j in range(i+1,len(sample)): dists.append(genome_distance(sample[i].genome,sample[j].genome))
        if not dists: return {'mean':0,'std':0,'min':0,'max':0}
        mean_d=sum(dists)/len(dists); var_d=sum((d-mean_d)**2 for d in dists)/len(dists)
        return {'mean':round(mean_d,4),'std':round(math.sqrt(var_d),4),'min':round(min(dists),4),'max':round(max(dists),4)}

    def checkpoint(self):
        alive=[o for o in self.population if o.alive]
        if not alive: return {'gen':self.generation,'alive':0,'best_fitness':0,'avg_fitness':0,'avg_pred_error':1.0,'avg_info':0,'intrinsic_drive':0,'avg_boredom':0,'div_mean':0,'region_diversity':0,'hybrid_vigor':0,'avg_inbreeding':1.0}
        fits=[self.compute_fitness(o) for o in self.population]; avg_fit=sum(fits)/len(fits)
        info=[o.total_information for o in alive]; drives=[o.im.drive_energy for o in alive]; bs=[o.im.boredom for o in alive]
        pe=[sum(o.wm.prediction_errors)/len(o.wm.prediction_errors) for o in alive if o.wm.prediction_errors]
        div=self.genome_diversity()
        region_diversity=len(set(max(o.region_counts, key=o.region_counts.get) if o.region_counts else 0 for o in alive))
        # 杂种优势统计：用缓存的 parent_max_fit 对比当代 fitness
        hybrid_vigor_count = 0
        n_hybrid = 0
        for o in alive:
            if hasattr(o, '_parent_max_fit'):
                n_hybrid += 1
                if self.compute_fitness(o) > o._parent_max_fit * 1.05:
                    hybrid_vigor_count += 1
        hybrid_vigor_rate = hybrid_vigor_count / max(1, n_hybrid)
        avg_inbreeding = sum(getattr(o,'_inbreeding',1.0) for o in self.population) / max(1,len(self.population))
        return {'gen':self.generation,'alive':len(alive),'best_fitness':max(fits),'avg_fitness':avg_fit,'avg_pred_error':sum(pe)/max(1,len(pe)),
                'avg_info':sum(info)/max(1,len(info)),'intrinsic_drive':round(sum(drives)/max(1,len(drives)),4),
                'avg_boredom':round(sum(bs)/max(1,len(bs)),4),'div_mean':div['mean'],'region_diversity':region_diversity,
                'hybrid_vigor':round(hybrid_vigor_rate,4),'avg_inbreeding':round(avg_inbreeding,3)}

    def run(self, num_gens=200):
        print(f"\n{'='*70}")
        print(f"Evolution v16 — Yuan Longping Hybrid Vigor + Multi-Niche IM")
        print(f"n_orgs={self.n} steps/gen={self.steps_per_gen} regions=9 STABLE | 70% hybrid + 30% mutant")
        print(f"{'='*70}\n")
        for g in range(num_gens):
            self.generation=g
            alive=self.run_generation(mode='training')
            ckpt=self.checkpoint(); self.history.append(ckpt)
            self.select_and_reproduce(gen=g)
            if g%10==0: self.diversity_log.append({'gen':g,**self.genome_diversity()})
            if g%5==0:
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} | fit={ckpt['best_fitness']:.4f}/{ckpt['avg_fitness']:.4f} | pred={ckpt['avg_pred_error']:.5f} | INFO={ckpt['avg_info']:.4f} | INTRIN={ckpt['intrinsic_drive']:.3f} boring={ckpt['avg_boredom']:.4f} | region_div={ckpt['region_diversity']} | HYBRID={ckpt['hybrid_vigor']:.3f} inbr={ckpt['avg_inbreeding']:.2f}")
        with open('evolution_log_v16.json','w') as f:
            json.dump({'history':self.history,'diversity_log':self.diversity_log,'num_gens':num_gens},f,indent=2,default=str)
        print(f"\nLogged to evolution_log_v16.json")
        return self.history[-1]


if __name__ == "__main__":
    random.seed(42)
    world=World(n_organisms=30,steps_per_gen=400,seed=42)
    final=world.run(num_gens=200)
