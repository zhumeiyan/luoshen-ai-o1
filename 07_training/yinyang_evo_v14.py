# yinyang_evo_v14.py
# 阴阳扣数字生命系统 · v14 — Information Foraging (Fast Learning + Shared Attractors)
#
# v13 教训：独立环境每代重置 → 基因组无法进化特定吸引子权重；
#           学习率常数 0.02 太低 → 400 步无法从随机权重收敛。
#
# v14 修复：
#   1. 学习率常数 0.02 → 0.08（4 倍加速），让 WM 在 400 步内收敛到低 MSE
#   2. 基因组初始权重范围 [-0.3, 0.3] → [-0.6, 0.6]（更强初始信号使节点激活有意义）
#   3. 每代共享同一套 9 区域吸引子参数（同代生物面对相同动力学，进化可以针对性收敛）
#      但每只生物的区域选择仍然是独立的（可学习导航策略）
#   4. 移除第 7 通道感官跳跃（避免 proprio 跳变引入额外误差）

import random, math, json, hashlib
from collections import deque

MAX_NODES = 12
N_CHANNELS = 8
N_ACTIONS = 4  # STAY, EXPLORE, SEEK_HOT, COOL

class SharedRegionPool:
    """共享区域池：每代创建一套 9 区域，所有生物共用同一套吸引子"""
    def __init__(self, seed=None):
        if seed is not None: random.seed(seed)
        self.regions = []
        for i in range(9):
            sigma = 0.05 + i * 0.08  # 0.05 ~ 0.69
            freq = [random.uniform(0.02, 0.06) for _ in range(N_CHANNELS)]
            phase = [random.uniform(0, 2*math.pi) for _ in range(N_CHANNELS)]
            coupling = 0.15 + sigma * 0.08
            state = [0.0]*N_CHANNELS
            self.regions.append({
                'sigma': sigma, 'freq': freq, 'phase': phase,
                'coupling': coupling, 'state': state, 't': 0
            })

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


class IndividualEnv:
    """个体导航器：决定生物当前在哪个区域，所有个体共享同一套区域动力学"""
    def __init__(self, shared_pool):
        self.pool = shared_pool
        self.current_region = random.randint(0, 8)

    def choose_and_step(self, action):
        if action == 1:  # EXPLORE: 随机跳
            self.current_region = random.randint(0, 8)
        elif action == 2:  # SEEK-HOT: 向更高 σ 走
            if self.current_region < 8:
                self.current_region += 1
        elif action == 3:  # COOL: 向更低 σ 走
            if self.current_region > 0:
                self.current_region -= 1
        # STAY: 不动

        region_idx = self.current_region
        sig = self.pool.regions[region_idx]['sigma']
        sensory = self.pool.step_region(region_idx)
        # proprio 编码 sigma（高 σ → 高体感）
        proprio = (sig - 0.35) * 0.7
        return sensory[:N_CHANNELS-1] + [proprio], sig, region_idx


# ============================================================
# WorldModel：更大学习率常数 + 更强初始权重
# ============================================================

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
        # 学习率常数 0.08（v13 是 0.02, 4x 加速）
        for i in range(N_CHANNELS):
            for j in range(n):
                delta=rate*errors[i]*self.nodes[j]*0.08
                self.predict_w[j][i]=max(-5.0,min(5.0,self.predict_w[j][i]+delta))
        for j in range(n):
            for i in range(N_CHANNELS):
                delta=rate*self.nodes[j]*actual_next[i]*0.005
                self.sensory_w[i][j]=max(-5.0,min(5.0,self.sensory_w[i][j]+delta))


class IntrinsicMotivation:
    """内在动机：跟踪学习进度，无聊驱动探索，挑战驱动停留"""
    def __init__(self):
        self.ema_error=0.1; self.ema_error_fast=0.1
        self.last_error_for_trend=0.1; self.boredom=0.0; self.drive_energy=0.0
        self.lr_log=deque(maxlen=100); self.region_history=deque(maxlen=200)
        self.region_scores={i: 0.0 for i in range(9)}
        self.recent_deltas=deque(maxlen=50)  # 更长窗口看趋势

    def update(self, current_mse, current_region, region_sigma, actual_next, predicted):
        alpha_slow, alpha_fast = 0.03, 0.15
        self.ema_error = (1-alpha_slow)*self.ema_error + alpha_slow*current_mse
        self.ema_error_fast = (1-alpha_fast)*self.ema_error_fast + alpha_fast*current_mse

        # 学习进度信号：从上次到现在 MSE 变好还是变差
        delta = current_mse - self.last_error_for_trend
        self.recent_deltas.append(delta)
        avg_delta = sum(self.recent_deltas)/max(1,len(self.recent_deltas))

        # 学习速率调制（现在 env 结构更简单，MSE 能降得更低）
        if avg_delta > 0.002:  # 误差在增大 → 加速学习
            lr_mult = 1.5; self.boredom *= 0.8
        elif avg_delta < -0.01:  # 误差快速减小 → 学得好
            lr_mult = 1.3; self.boredom *= 0.85
        elif abs(avg_delta) < 0.003 and self.ema_error_fast < 0.02:  # 收敛且低误差 → 无聊
            self.boredom += 0.005; lr_mult = 0.5
        elif self.ema_error_fast < 0.05:  # 中误差、稳定 → 略无聊
            self.boredom += 0.002; lr_mult = 0.8
        else:
            lr_mult = 1.0; self.boredom = max(0, self.boredom - 0.001)

        self.boredom = max(0.0, min(1.0, self.boredom))
        self.last_error_for_trend = self.ema_error_fast
        self.lr_log.append(lr_mult)

        # 区域价值
        info_value = region_sigma * (1.0 / max(0.001, current_mse + 0.005))
        self.region_scores[current_region] = 0.95*self.region_scores[current_region] + 0.05*info_value

        # 内在能量累加
        reward = 0.0
        if region_sigma > 0.30 and current_mse > 0.01:  # 中高 σ 区域有学习空间
            reward += 0.02
        if avg_delta < -0.003:  # 正在学习
            reward += 0.01
        if self.boredom > 0.05:  # 无聊惩罚
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
            bias[0] = 0.15  # STAY,继续学
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
        self.env = None

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
            # 用感官移动幅度编码到 proprio 通道（但用平滑版本）
            movement = math.sqrt(sum((sensory[i]-self.senses_prev[i])**2 for i in range(N_CHANNELS-1)))/N_CHANNELS
            sensory[N_CHANNELS-1] = sensory[N_CHANNELS-1] * 0.8 + movement * 0.2
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
    def __init__(self, n_organisms=30, steps_per_gen=400):
        self.n=n_organisms; self.steps_per_gen=steps_per_gen
        self.population=[Organism() for _ in range(n_organisms)]
        self.generation=0; self.history=[]; self.diversity_log=[]
        self.shared_pool = SharedRegionPool(seed=random.randint(0,999999))
        for o in self.population:
            o.env = IndividualEnv(self.shared_pool)

    def run_generation(self, mode='training'):
        # 每代开始时 REINIT 共享池（新一套吸引子参数）
        self.shared_pool = SharedRegionPool(seed=random.randint(0,999999))
        for org in self.population:
            if not org.alive: continue
            org.env = IndividualEnv(self.shared_pool)

        # 重置行为记录
        for org in self.population:
            org.behavior_trace=[]; org.region_counts={}
            org.action_history=[]; org.senses_prev=None

        for step in range(self.steps_per_gen):
            for org in self.population:
                if not org.alive or org.env is None: continue
                # 获取当前区域的感官（注意：每 step 后区域演化了，但还没 action 之前）
                rid = org.env.current_region
                sig = self.shared_pool.regions[rid]['sigma']
                sensory_before = self.shared_pool.get_sensory(rid)
                proprio = (sig - 0.35) * 0.7
                sensory_before_proprio = sensory_before[:N_CHANNELS-1] + [proprio]
                predicted, action = org.plan(sensory_before_proprio)
                new_sensory, sigma_new, rid_new = org.env.choose_and_step(action)
                org.learn_and_update(new_sensory, sigma_new, rid_new)
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
        return 0.25*survival + 0.10*energy_score + 0.40*prediction_score + 0.10*novelty + 0.15*info_score

    def select_and_reproduce(self):
        fits=[(o, self.compute_fitness(o)) for o in self.population]
        fits.sort(key=lambda x: x[1], reverse=True)
        n_survive=max(4, self.n//3)
        survivors=[o for o,f in fits[:n_survive]]
        sf=[f for o,f in fits[:n_survive]]
        new_pop=[]
        if survivors and sf[0]>0.1:
            elite=survivors[0].clone_mutate(rate=0.02, magnitude=0.03); new_pop.append(elite)
        attempts=0
        while len(new_pop)<self.n and attempts<self.n*10:
            if survivors and sum(sf)>0:
                probs=[max(0.01,f) for f in sf[:n_survive]]; sum_p=sum(probs); probs=[p/sum_p for p in probs]
                r=random.random(); cum=0.0; chosen=survivors[0]
                for i,p in enumerate(probs):
                    cum+=p
                    if r<=cum: chosen=survivors[i]; break
                new_pop.append(chosen.clone_mutate(rate=0.12, magnitude=0.2))
            else: new_pop.append(Organism())
            attempts+=1
        self.population=new_pop[:self.n]
        for o in self.population:
            o.energy=8.0; o.age=0; o.alive=True
            o.wm=WorldModel(o.genome); o.im=IntrinsicMotivation()
            o.total_information=0.0; o.region_counts={}
            o.action_history=[]; o.senses_prev=None
            o.env=None

    def genome_diversity(self):
        sample=self.population[:15]; dists=[]
        for i in range(len(sample)):
            for j in range(i+1,len(sample)): dists.append(genome_distance(sample[i].genome,sample[j].genome))
        if not dists: return {'mean':0,'std':0,'min':0,'max':0}
        mean_d=sum(dists)/len(dists); var_d=sum((d-mean_d)**2 for d in dists)/len(dists)
        return {'mean':round(mean_d,4),'std':round(math.sqrt(var_d),4),'min':round(min(dists),4),'max':round(max(dists),4)}

    def checkpoint(self):
        alive=[o for o in self.population if o.alive]
        if not alive: return {'gen':self.generation,'alive':0,'best_fitness':0,'avg_fitness':0,'avg_pred_error':1.0,'avg_info':0,'intrinsic_drive':0,'avg_boredom':0,'div_mean':0,'region_diversity':0}
        fits=[self.compute_fitness(o) for o in self.population]; avg_fit=sum(fits)/len(fits)
        info=[o.total_information for o in alive]; drives=[o.im.drive_energy for o in alive]; bs=[o.im.boredom for o in alive]
        pe=[sum(o.wm.prediction_errors)/len(o.wm.prediction_errors) for o in alive if o.wm.prediction_errors]
        div=self.genome_diversity()
        region_diversity=len(set(max(o.region_counts, key=o.region_counts.get) if o.region_counts else 0 for o in alive))
        return {'gen':self.generation,'alive':len(alive),'best_fitness':max(fits),'avg_fitness':avg_fit,'avg_pred_error':sum(pe)/max(1,len(pe)),
                'avg_info':sum(info)/max(1,len(info)),'intrinsic_drive':round(sum(drives)/max(1,len(drives)),4),
                'avg_boredom':round(sum(bs)/max(1,len(bs)),4),'div_mean':div['mean'],'region_diversity':region_diversity}

    def run(self, num_gens=200):
        print(f"\n{'='*70}")
        print(f"Evolution v14 — Fast Learning (4x lr) + Shared Attractors")
        print(f"n_orgs={self.n} steps/gen={self.steps_per_gen} regions=9 shared per gen")
        print(f"{'='*70}\n")
        for g in range(num_gens):
            self.generation=g
            alive=self.run_generation(mode='training')
            ckpt=self.checkpoint(); self.history.append(ckpt)
            self.select_and_reproduce()
            if g%10==0: self.diversity_log.append({'gen':g,**self.genome_diversity()})
            if g%5==0:
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} | fit={ckpt['best_fitness']:.4f}/{ckpt['avg_fitness']:.4f} | pred={ckpt['avg_pred_error']:.5f} | INFO={ckpt['avg_info']:.4f} | INTRIN={ckpt['intrinsic_drive']:.3f} boredom={ckpt['avg_boredom']:.4f} | region_div={ckpt['region_diversity']}")
        with open('evolution_log_v14.json','w') as f:
            json.dump({'history':self.history,'diversity_log':self.diversity_log,'num_gens':num_gens},f,indent=2,default=str)
        print(f"\n保存到 evolution_log_v14.json")
        return self.history[-1]


if __name__ == "__main__":
    random.seed(42)
    world=World(n_organisms=30,steps_per_gen=400)
    final=world.run(num_gens=200)
