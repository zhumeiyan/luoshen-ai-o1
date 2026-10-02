# yinyang_evo_v12.py
# 阴阳扣数字生命系统 · v12 — Information Foraging Environment
#
# v11 证明：单一预测环境中 IM 任何形式都降低精度。
# v12 切换环境架构：从"均匀预测游戏"到"信息觅食"。
#
# 环境设计：
#   · 9 个空间区域，每个区域有不同的混沌等级 σ（0.05 ~ 0.95）
#   · 生物位于某个区域，感知该区域的"局部感官流"
#   · 区域 σ 越高 → 流越发不可预测 → 可学东西越多 → 信息营养越高
#   · 区域 σ 越低 → 流越发可预测 → 信息营养越低
#   · 生物可以主动移动（"Go to region k" 行动）来寻找信息热点
#   · 适合度 = 预测精度（热点中精度更重要）+ 生存 + 信息积累（总学习进度）
#
# 内在动机的价值：
#   · 在低 σ 区域：预测完美但无聊（boredom+，无新信息）→ 好奇驱动应促使其离开
#   · 在高 σ 区域：可学但预测难 → 精通感促使其留在这里学
#   · 16岁的人类：既需要舒适区安全感，也需要挑战区成长。这正是 v12 的生态位。

import random, math, json, hashlib
from collections import deque

MAX_NODES = 12
N_CHANNELS = 8
N_ACTIONS = 4  # FORAGE, STAY, EXPLORE, SEEK-HOT

class InfoRegion:
    """信息区域：具有特定混沌等级 σ 的子环境"""
    def __init__(self, sigma, seed=None):
        if seed is not None: random.seed(seed)
        self.sigma = sigma  # 混沌等级 [0,1]：0=完全可预测，1=纯噪声
        self.t = 0
        self.state = [0.0]*N_CHANNELS
        self.trend = [random.uniform(-0.5,0.5) for _ in range(N_CHANNELS)]

    def step(self, action=0):
        self.t += 1
        N = N_CHANNELS
        # 基础动力学：有趋势 + 正弦 + 混沌噪声
        noise_scale = self.sigma * 0.15
        trend_strength = (1 - self.sigma) * 0.05
        for i in range(N):
            self.state[i] += self.trend[i] * trend_strength
            self.state[i] += math.sin(self.t * 0.05 + i) * 0.02 * (1 - self.sigma)
            self.state[i] += random.gauss(0, noise_scale)
            # 软反弹
            self.state[i] = max(-1.5, min(1.5, self.state[i]))
        # 偶尔变化趋势
        if random.random() < 0.01 * self.sigma:
            self.trend = [random.uniform(-0.5,0.5) for _ in range(N)]

    def get_sensory(self):
        return [self.state[i] + random.gauss(0, self.sigma * 0.02) for i in range(N_CHANNELS)]


class InformationForagingEnv:
    """信息觅食环境：9 个不同混沌等级的区域 + 位置信息"""
    def __init__(self, seed=None):
        if seed is not None: random.seed(seed)
        # 9 个区域，混沌等级从 0.05 到 0.95 均匀分布
        self.regions = [InfoRegion(sigma=0.05 + i*0.1125, seed=random.randint(0,999999)) for i in range(9)]
        self.current_region = 4  # 从中间开始
        self.time_in_region = 0
        self.t = 0

    def step(self, action, organism_region_choice=None):
        """
        action: 0=STAY, 1=EXPLORE(随机找新区域), 2=SEEK-HOT(向高σ走), 3=COOL(向低σ走)
        返回当前区域的 sensory
        """
        self.t += 1
        self.time_in_region += 1

        # 移动决策
        if action == 1:  # EXPLORE: 随机跳
            self.current_region = random.randint(0, 8)
            self.time_in_region = 0
        elif action == 2:  # SEEK-HOT: 向更高 σ 移动
            if self.current_region < 8:
                self.current_region += 1
                self.time_in_region = 0
        elif action == 3:  # COOL: 向更低 σ 移动
            if self.current_region > 0:
                self.current_region -= 1
                self.time_in_region = 0

        # 当前区域步进
        self.regions[self.current_region].step(action)
        return self.regions[self.current_region].get_sensory()

    def get_region_sigma(self):
        return self.regions[self.current_region].sigma

    def get_sensory_with_proprio(self):
        """包含区域位置信息的 proprio（让 WM 可以通过 region 位置推断信息密度）"""
        base = self.regions[self.current_region].get_sensory()
        # proprio 通道编码当前 region σ 的一种"体感"信号
        sigma = self.get_region_sigma()
        proprio = (sigma - 0.5) * 0.8  # 高 σ = 高 proprio（体感上"兴奋"）
        return base[:N_CHANNELS-1] + [proprio]


# ============================================================
# WorldModel + Organism with Information Bonus
# ============================================================

def resize_genome(g, new_n):
    old_n = g['params']['n_nodes']
    g2 = {'params': dict(g['params'])}
    sw = g.get('sensory_w', [])
    g2['sensory_w'] = [list(r[:min(old_n,new_n)]) + [random.uniform(-0.3,0.3)]*(new_n-min(old_n,new_n)) for r in sw]
    if len(g2['sensory_w']) < N_CHANNELS:
        g2['sensory_w'] = [[random.uniform(-0.3,0.3) for _ in range(new_n)] for _ in range(N_CHANNELS)]
    pw = g.get('predict_w', [])
    g2['predict_w'] = [list(pw[i]) if i < old_n and i < len(pw) else [random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)] for i in range(new_n)]
    g2['recurrent_w'] = [[random.uniform(-0.1,0.1) if i!=j else 0.0 for j in range(new_n)] for i in range(new_n)]
    g2['action_w'] = [[random.uniform(-0.2,0.2) for _ in range(new_n)] for _ in range(N_ACTIONS)]
    g2['action_bias'] = [random.uniform(-0.1,0.1) for _ in range(N_ACTIONS)]
    g2['params']['n_nodes'] = new_n
    return g2


class WorldModel:
    def __init__(self, genome):
        self.params=genome['params']
        n=self.params['n_nodes']; self.nodes=[0.0]*n; self.last_prediction=[0.0]*N_CHANNELS
        self.sensory_w=genome['sensory_w']; self.predict_w=genome['predict_w']
        self.recurrent_w=genome['recurrent_w']; self.action_w=genome['action_w']; self.action_bias=genome['action_bias']
        self.prediction_errors=deque(maxlen=80)
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
        for i in range(N_CHANNELS):
            for j in range(n):
                delta=rate*errors[i]*self.nodes[j]*0.01
                self.predict_w[j][i]=max(-5.0,min(5.0,self.predict_w[j][i]+delta))
        for j in range(n):
            for i in range(N_CHANNELS):
                delta=rate*self.nodes[j]*actual_next[i]*0.001
                self.sensory_w[i][j]=max(-5.0,min(5.0,self.sensory_w[i][j]+delta))


class IntrinsicMotivation:
    def __init__(self):
        self.ema_error=0.05; self.ema_error_fast=0.05
        self.last_error_for_trend=0.05; self.boredom=0.0; self.drive_energy=0.0
        self.lr_log=deque(maxlen=100); self.region_history=deque(maxlen=200)
        self.region_scores={i: 0.0 for i in range(9)}  # 各区域价值经验

    def update(self, current_mse, current_region, region_sigma, actual_next, predicted):
        # EMA 更新
        alpha_slow, alpha_fast = 0.02, 0.1
        self.ema_error = (1-alpha_slow)*self.ema_error + alpha_slow*current_mse
        self.ema_error_fast = (1-alpha_fast)*self.ema_error_fast + alpha_fast*current_mse

        # 学习速率调制
        if self.ema_error_fast > self.ema_error * 1.3:
            lr_mult = 1.5; self.boredom *= 0.9
        elif self.ema_error_fast < self.last_error_for_trend * 0.95:
            lr_mult = 1.2; self.boredom *= 0.95
        elif current_mse < 0.003 and self.ema_error_fast < 0.003:
            self.boredom += 0.002
            lr_mult = 0.5
        else:
            lr_mult = 1.0; self.boredom *= 0.95
        self.last_error_for_trend = self.ema_error_fast
        self.lr_log.append(lr_mult)

        # 区域经验更新（环境复杂度 × 学习效果 = 信息价值）
        info_value = region_sigma * (1.0 / max(0.001, current_mse + 0.01))
        self.region_scores[current_region] = 0.95*self.region_scores[current_region] + 0.05*info_value

        # 内在能量累加：在预测有挑战的区域给予更多内部奖励
        reward = 0.0
        if region_sigma > 0.5 and current_mse > 0.01:  # 正在高信息区域学习
            reward += 0.02
        if self.boredom > 0.05:  # 无聊惩罚
            reward -= 0.01
        self.drive_energy = self.drive_energy * 0.99 + reward
        self.drive_energy = max(-3.0, min(5.0, self.drive_energy))
        self.region_history.append(current_region)

        return lr_mult

    def get_action_bias(self):
        """基于内在动机历史的行动偏置"""
        bias = {}
        # 太无聊 → 偏向 SEEK-HOT (action 2) 或 EXPLORE (action 1)
        if self.boredom > 0.03:
            bias[2] = 0.4  # SEEK-HOT
            bias[1] = 0.2  # EXPLORE
        # 足够好奇 → 留在有挑战的区域
        elif self.ema_error_fast > self.ema_error * 1.2:
            bias[0] = 0.2  # STAY（继续学）
        return bias

    def get_stats(self):
        return {'drive': round(self.drive_energy,3), 'boredom': round(self.boredom,4), 'lr': self.lr_log[-1] if self.lr_log else 1.0}


class Organism:
    def __init__(self, genome=None, lineage_id=None):
        self.genome=genome if genome else self.random_genome()
        self.wm=WorldModel(self.genome); self.im=IntrinsicMotivation()
        self.senses_prev=None; self.senses_before_action=None
        self.state=[0.0]*N_CHANNELS; self.state_history=[]
        self.action_history=[]; self.alive=True; self.age=0
        self.energy=8.0; self.lineage_id=lineage_id or self._gen_id()
        self.parents=[]; self.behavior_trace=[]
        self._predicted_next=None; self._action_chosen=None
        self.total_information=0.0; self.region_counts={}

    def wm_forward_from_nodes(self, nodes, sensory_vec): return self.wm.forward_from_nodes(nodes, sensory_vec)
    def _gen_id(self): return hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]

    @staticmethod
    def random_genome():
        n=random.randint(3,MAX_NODES)
        return {
            'sensory_w': [[random.uniform(-0.3,0.3) for _ in range(n)] for _ in range(N_CHANNELS)],
            'predict_w': [[random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)] for _ in range(n)],
            'recurrent_w': [[random.uniform(-0.1,0.1) if i!=j else 0.0 for j in range(n)] for i in range(n)],
            'action_w': [[random.uniform(-0.2,0.2) for _ in range(n)] for _ in range(N_ACTIONS)],
            'action_bias': [random.uniform(-0.1,0.1) for _ in range(N_ACTIONS)],
            'params': {'n_nodes':n,'wm_decay':random.uniform(0.7,0.95),'sensory_gain':random.uniform(0.3,2.0),
                       'pred_gain':random.uniform(0.5,2.0),'learning_rate':random.uniform(0.01,0.3),'metabolic':random.uniform(0.02,0.12)}
        }

    def plan(self, sensory):
        if self.senses_prev is not None:
            sensory[N_CHANNELS-1] = math.sqrt(sum((sensory[i]-self.senses_prev[i])**2 for i in range(N_CHANNELS)))/N_CHANNELS
        else: sensory[N_CHANNELS-1] = 0.0
        self.senses_before_action=sensory[:]; self.senses_prev=sensory[:]
        self.state=sensory[:]; self.state_history.append(sensory[:])
        if len(self.state_history)>300: self.state_history=self.state_history[-300:]
        predicted,_=self.wm.forward(sensory)
        self._predicted_next=predicted
        # 行动偏置：IM驱动
        action_bias=self.im.get_action_bias()
        action_idx=self.wm.choose_action(bias=action_bias)
        self._action_chosen=action_idx; self.action_history.append(action_idx)
        if len(self.action_history)>200: self.action_history=self.action_history[-200:]
        return predicted, action_idx

    def learn_and_update(self, actual_next, region_sigma, region_id):
        if not self.alive: return
        predicted=self._predicted_next
        if predicted is None: return
        mse=sum((actual_next[i]-predicted[i])**2 for i in range(N_CHANNELS))/N_CHANNELS

        # IM 调制学习速率
        lr_mult=self.im.update(mse, region_id, region_sigma, actual_next, predicted)
        effective_lr=max(0.001,min(0.5,self.wm.params['learning_rate']*lr_mult))

        self.wm.learn(actual_next, predicted, effective_lr)
        prediction_reward=1.0/(1.0+mse)
        n_nodes=len(self.wm.nodes)
        metabolic_cost=self.wm.params['metabolic']*(1+n_nodes*0.04)

        # 信息积累奖励：在高 σ 区域预测好了额外奖励
        info_bonus = region_sigma * prediction_reward * 0.08
        self.total_information += info_bonus

        self.energy += prediction_reward * 0.4 + info_bonus * 0.2

        self.region_counts[region_id] = self.region_counts.get(region_id, 0) + 1

        action_costs={0:0.01, 1:0.05, 2:0.03, 3:0.03}  # STAY cheap, MOVE costly
        metabolic_cost += action_costs.get(self._action_chosen, 0.02)
        self.energy -= metabolic_cost
        self.energy = max(-5.0, min(20.0, self.energy))

        self.behavior_trace.append({'action':self._action_chosen,'err':mse,'energy':self.energy,'region':region_id,'sigma':region_sigma,'lr_mult':lr_mult,'drive':self.im.drive_energy,'boredom':self.im.boredom})
        if len(self.behavior_trace)>200: self.behavior_trace=self.behavior_trace[-200:]
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
        self.population=[Organism() for _ in range(n_organisms)]; self.generation=0
        self.history=[]; self.diversity_log=[]

    def aggregate_action(self, actions):
        if not actions: return 0
        counts=[0]*N_ACTIONS
        for a in actions:
            if 0<=a<N_ACTIONS: counts[a]+=1
        return counts.index(max(counts))

    def run_generation(self, mode='training'):
        env=InformationForagingEnv(seed=random.randint(0,999999))
        for step in range(self.steps_per_gen):
            # 所有生物在同一个环境中共享区域流
            # 但不同的生物可能选择不同的行动（不同移动策略）
            # 为了简化：全群共享同一区域感知，行动决策通过 aggregate 决定移动
            sensory=env.get_sensory_with_proprio(); proposals=[]
            for org in self.population:
                if not org.alive: continue
                _,action=org.plan(sensory); proposals.append(action)
            action_code=self.aggregate_action(proposals)
            # 选择行动的个体感受到的环境 = 行动后的区域
            new_sensory=env.step(action_code)
            region_sigma=env.get_region_sigma(); region_id=env.current_region
            for org in self.population:
                if not org.alive: continue
                org.region_counts[region_id] = org.region_counts.get(region_id, 0) + 1
                org.learn_and_update(new_sensory, region_sigma, region_id)
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
            prediction_score=1.0/(1.0+avg_err*3)  # v12: 更强调预测（系数从5→3）
        else: prediction_score=0.0
        novelty=self.novelty_bonus(org)
        # v12 新增：信息积累奖励（鼓励去高σ区域）
        info_score=min(1.0, org.total_information/5.0)  # 归一化
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

    def genome_diversity(self):
        sample=self.population[:15]; dists=[]
        for i in range(len(sample)):
            for j in range(i+1,len(sample)): dists.append(genome_distance(sample[i].genome,sample[j].genome))
        if not dists: return {'mean':0,'std':0,'min':0,'max':0}
        mean_d=sum(dists)/len(dists); var_d=sum((d-mean_d)**2 for d in dists)/len(dists)
        return {'mean':round(mean_d,4),'std':round(math.sqrt(var_d),4),'min':round(min(dists),4),'max':round(max(dists),4)}

    def checkpoint(self):
        alive=[o for o in self.population if o.alive]
        if not alive: return {'gen':self.generation,'alive':0,'best_fitness':0,'avg_fitness':0,'avg_pred_error':1.0,'avg_info':0,'intrinsic_drive':0,'avg_boredom':0,'div_mean':0}
        fits=[self.compute_fitness(o) for o in self.population]; avg_fit=sum(fits)/len(fits)
        info=[o.total_information for o in alive]; drives=[o.im.drive_energy for o in alive]; bs=[o.im.boredom for o in alive]
        pe=[sum(o.wm.prediction_errors)/len(o.wm.prediction_errors) for o in alive if o.wm.prediction_errors]
        div=self.genome_diversity()
        return {'gen':self.generation,'alive':len(alive),'best_fitness':max(fits),'avg_fitness':avg_fit,'avg_pred_error':sum(pe)/max(1,len(pe)),
                'avg_info':sum(info)/max(1,len(info)),'intrinsic_drive':round(sum(drives)/max(1,len(drives)),4),
                'avg_boredom':round(sum(bs)/max(1,len(bs)),4),'div_mean':div['mean']}

    def run(self, num_gens=200):
        print(f"\n{'='*70}")
        print(f"Evolution v12 — Information Foraging Environment")
        print(f"n_orgs={self.n} steps/gen={self.steps_per_gen} regions=9")
        print(f"{'='*70}\n")
        for g in range(num_gens):
            self.generation=g
            alive=self.run_generation(mode='training')
            ckpt=self.checkpoint(); self.history.append(ckpt)
            self.select_and_reproduce()
            if g%10==0: self.diversity_log.append({'gen':g,**self.genome_diversity()})
            if g%5==0:
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} | fit={ckpt['best_fitness']:.4f}/{ckpt['avg_fitness']:.4f} | pred={ckpt['avg_pred_error']:.4f} | INFO={ckpt['avg_info']:.4f} | INTRIN={ckpt['intrinsic_drive']:.3f} boredom={ckpt['avg_boredom']:.3f}")
        with open('evolution_log_v12.json','w') as f:
            json.dump({'history':self.history,'diversity_log':self.diversity_log,'num_gens':num_gens},f,indent=2,default=str)
        print(f"\n保存到 evolution_log_v12.json")
        return self.history[-1]


if __name__ == "__main__":
    random.seed(42)
    world=World(n_organisms=30,steps_per_gen=400)
    final=world.run(num_gens=200)
