# evo_simple.py
# 极简版 — 撤掉杂交、多区域导航、IM 驱动行动
# 只保留：单环境 + 基础 WM + Hebbian 学习 + 预测选择压
# 目标是搞清楚：最简化条件下生物到底能不能真正"学会"预测

import random, math, json

N_CHANNELS = 8
MAX_NODES = 8  # 更少节点

class SimpleWM:
    def __init__(self, n):
        self.n = n
        self.nodes = [0.0] * n
        self.predict_w = [[random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)] for _ in range(n)]
        self.sensory_w = [[random.uniform(-0.3,0.3) for _ in range(n)] for _ in range(N_CHANNELS)]
        self.recurrent_w = [[0.0]*n for _ in range(n)]
        self.last_pred = [0.0]*N_CHANNELS
        self.errors = []

    def forward(self, sensory):
        pred = [math.tanh(sum(self.predict_w[j][i]*self.nodes[j] for j in range(self.n))) for i in range(N_CHANNELS)]
        new_nodes = []
        for j in range(self.n):
            val = self.nodes[j]*0.8
            for i in range(N_CHANNELS):
                val += self.sensory_w[i][j]*sensory[i]*0.5
            for k in range(self.n):
                val += self.recurrent_w[j][k]*self.nodes[k]*0.1
            # 误差驱动
            if self.last_pred and sum(abs(x) for x in self.last_pred) > 0.01:
                for i in range(N_CHANNELS):
                    err = sensory[i] - self.last_pred[i]
                    val += err * self.sensory_w[i][j] * 0.1
            new_nodes.append(math.tanh(val))
        self.nodes = new_nodes
        self.last_pred = pred[:]
        return pred[:]

    def learn(self, actual, predicted, rate=0.1):
        errors = [actual[i]-predicted[i] for i in range(N_CHANNELS)]
        mse = sum(e*e for e in errors)/N_CHANNELS
        self.errors.append(mse)
        for i in range(N_CHANNELS):
            for j in range(self.n):
                self.predict_w[j][i] += rate*errors[i]*self.nodes[j]*0.05
                self.predict_w[j][i] = max(-3,min(3,self.predict_w[j][i]))

    def forward_from_nodes(self, nodes, sensory):
        """不修改 self.nodes 的只读 forward（用于多步前瞻）"""
        pred = [math.tanh(sum(self.predict_w[j][i]*nodes[j] for j in range(self.n))) for i in range(N_CHANNELS)]
        new_nodes = []
        for j in range(self.n):
            val = nodes[j]*0.8
            for i in range(N_CHANNELS):
                val += self.sensory_w[i][j]*sensory[i]*0.5
            for k in range(self.n):
                val += self.recurrent_w[j][k]*nodes[k]*0.1
            if self.last_pred and sum(abs(x) for x in self.last_pred) > 0.01:
                for i in range(N_CHANNELS):
                    err = sensory[i] - self.last_pred[i]
                    val += err * self.sensory_w[i][j] * 0.1
            new_nodes.append(math.tanh(val))
        return pred[:], new_nodes


class SimpleEnv:
    def __init__(self, seed=None):
        if seed: random.seed(seed)
        self.t = 0
        # 简单正弦 + 噪声
        self.freq = [random.uniform(0.02,0.08) for _ in range(N_CHANNELS)]
        self.phase = [random.uniform(0,2*math.pi) for _ in range(N_CHANNELS)]
    def step(self):
        self.t += 1
    def sensory(self):
        return [math.sin(self.t*self.freq[i]+self.phase[i]) + random.gauss(0,0.05) for i in range(N_CHANNELS)]


def run_simple_evo(n_orgs=20, n_gens=100, steps_per_gen=200):
    # 随机初始化种群
    population = []
    for _ in range(n_orgs):
        n = random.randint(3, MAX_NODES)
        population.append({'n': n, 'wm': SimpleWM(n), 'mse_history': [], 'age':0})
    env = SimpleEnv(seed=42)

    history = []
    for gen in range(n_gens):
        gen_mses = []
        for org in population:
            org['wm'].errors = []
            for step in range(steps_per_gen):
                s = env.sensory()
                pred = org['wm'].forward(s)
                env.step()
                actual = env.sensory()
                org['wm'].learn(actual, pred, rate=0.1)
            avg_mse = sum(org['wm'].errors[50:])/150 if len(org['wm'].errors)>50 else 1.0
            org['mse_history'].append(avg_mse)
            gen_mses.append((org, avg_mse))
        gen_mses.sort(key=lambda x: x[1])
        best_mse = gen_mses[0][1]
        avg_mse = sum(x[1] for x in gen_mses)/len(gen_mses)
        history.append({'gen': gen, 'best_mse': round(best_mse,5), 'avg_mse': round(avg_mse,5)})
        if gen % 10 == 0:
            print(f"GEN {gen:4d} | best MSE={best_mse:.5f} | avg MSE={avg_mse:.5f}")
        # 选择 + 克隆最优的
        survivors = [x[0] for x in gen_mses[:6]]  # 留前 6
        new_pop = []
        for org in survivors:
            # 克隆
            n = org['n']
            child_wm = SimpleWM(n)
            child_wm.predict_w = [row[:] for row in org['wm'].predict_w]
            child_wm.sensory_w = [row[:] for row in org['wm'].sensory_w]
            child_wm.recurrent_w = [row[:] for row in org['wm'].recurrent_w]
            # 小突变
            for i in range(n):
                for j in range(N_CHANNELS):
                    if random.random() < 0.1:
                        child_wm.predict_w[i][j] += random.uniform(-0.2,0.2)
                    if random.random() < 0.1:
                        child_wm.sensory_w[j][i] += random.uniform(-0.1,0.1)
            new_pop.append({'n': n, 'wm': child_wm, 'mse_history': [], 'age':0})
        # 填充剩余（最优克隆+突变）
        while len(new_pop) < n_orgs:
            parent = random.choice(survivors)
            n = parent['n']
            child_wm = SimpleWM(n)
            child_wm.predict_w = [row[:] for row in parent['wm'].predict_w]
            child_wm.sensory_w = [row[:] for row in parent['wm'].sensory_w]
            for i in range(n):
                for j in range(N_CHANNELS):
                    if random.random() < 0.2:
                        child_wm.predict_w[i][j] += random.uniform(-0.3,0.3)
            new_pop.append({'n': n, 'wm': child_wm, 'mse_history': [], 'age':0})
        population = new_pop[:n_orgs]

    # 最终测试
    print("\n=== FINAL TEST ===")
    best_org = history[-1] if history else None
    # 直接测多步预测
    print("\nMulti-step prediction test:")
    env2 = SimpleEnv(seed=999)
    for _ in range(50):
        env2.step()
    final_org = sorted(population, key=lambda o: sum(o['wm'].errors[-50:])/50 if len(o['wm'].errors)>50 else 1)[0]

    for n_step in [1,3,5,10,30]:
        total_mse = 0
        for trial in range(20):
            for _ in range(30):  # warmup
                s = env2.sensory()
                env2.step()
            s = env2.sensory()
            nodes_backup = final_org['wm'].nodes[:]
            pred = final_org['wm'].forward(s)
            for _ in range(n_step-1):
                pred, _ = final_org['wm'].forward_from_nodes(final_org['wm'].nodes[:], pred)
            for _ in range(n_step):
                env2.step()
            actual = env2.sensory()
            mse = sum((actual[i]-pred[i])**2 for i in range(N_CHANNELS))/N_CHANNELS
            total_mse += mse
            final_org['wm'].nodes = nodes_backup
        print(f"  N={n_step:3d}: MSE = {total_mse/20:.5f}")

    with open('evo_simple_log.json','w') as f:
        json.dump({'history': history, 'final_best_mse': history[-1]['best_mse'] if history else None}, f, indent=2)
    print("\nLogged to evo_simple_log.json")
    return history

if __name__ == "__main__":
    random.seed(42)
    run_simple_evo(n_orgs=20, n_gens=100, steps_per_gen=200)
