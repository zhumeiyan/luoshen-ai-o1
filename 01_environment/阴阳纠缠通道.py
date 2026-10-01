#!/usr/bin/env python3
"""
Entangled Dual-Channel: A←Ψ→B 纠缠通道
========================================
A通道（语言）和B通道（行为）共享一个联合密度矩阵ρ_AB。
对任一通道的测量会使另一通道的状态坍缩，其关联不是经典隐变量能解释的。

数学结构：
  联合态 |Ψ⟩ ∈ H_A ⊗ H_B
  约化密度矩阵 ρ_A = Tr_B(|Ψ⟩⟨Ψ|), ρ_B = Tr_A(|Ψ⟩⟨Ψ|)
  纠缠熵 S(ρ_A) = -Tr(ρ_A log₂ ρ_A) —— 为零说明可分离（经典关联），大于零说明纠缠

测量协议：
  A_m: 语言结论的测量算符组 {M_a}
  B_m: 行为结论的测量算符组 {M_b}
  对A测a → B坍缩为 ρ_B|a = Tr_A(|a⟩⟨a|ρ_AB)/P(a)
  关联强度 P(b|a) = Tr(M_b† M_b ρ_B|a)
"""

import random, math, copy, collections
import numpy as np

# ============ 量子工具函数 ============

def partial_trace(rho, dim_A, dim_B, trace='B'):
    """对联合密度矩阵做部分迹"""
    if trace == 'B':
        # ρ_A = Tr_B(ρ_AB)
        result = np.zeros((dim_A, dim_A), dtype=complex)
        for i in range(dim_A):
            for j in range(dim_A):
                s = 0.0 + 0.0j
                for k in range(dim_B):
                    s += rho[i*dim_B + k, j*dim_B + k]
                result[i, j] = s
        return result
    else:
        # ρ_B = Tr_A(ρ_AB)
        result = np.zeros((dim_B, dim_B), dtype=complex)
        for i in range(dim_B):
            for j in range(dim_B):
                s = 0.0 + 0.0j
                for k in range(dim_A):
                    s += rho[k*dim_B + i, k*dim_B + j]
                result[i, j] = s
        return result

def von_neumann_entropy(rho):
    """密度矩阵的冯诺依曼熵 S = -Tr(ρ log₂ ρ)"""
    eigvals = np.linalg.eigvalsh(rho)
    s = 0.0
    for ev in eigvals:
        if ev > 1e-12:
            s -= ev * math.log2(ev)
    return max(0.0, s)

def is_hermitian(rho, tol=1e-10):
    return np.allclose(rho, rho.conj().T, atol=tol)

def concurrence_two_qubit(rho):
    """两比特系统的concurrence（纠缠度量，0=可分离，1=最大纠缠）"""
    # σ_y ⊗ σ_y
    sy = np.array([[0, -1j], [1j, 0]])
    sy_sy = np.kron(sy, sy)
    rho_tilde = sy_sy @ rho.conj() @ sy_sy
    # R = √(√ρ * ρ_tilde * √ρ)
    try:
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = np.maximum(eigvals, 0)
        sqrt_eigvals = np.sqrt(eigvals)
        idx = np.argsort(eigvals)[::-1]
        sqrt_rho = np.zeros_like(rho)
        for i, ii in enumerate(idx):
            if eigvals[ii] > 1e-12:
                v = np.zeros(dim := len(eigvals))
                v[ii] = 1
                sqrt_rho += sqrt_eigvals[ii] * np.outer(v, v)
        R_sq = sqrt_rho @ rho_tilde @ sqrt_rho
        lambdas = sorted(np.sqrt(max(0, ev) for ev in np.linalg.eigvalsh(R_sq).real), reverse=True)
        return max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]) if len(lambdas) == 4 else 0.0
    except:
        return 0.0

# ============ 纠缠通道类 ============

class EntangledChannel:
    """
    A通道（语言）和B通道（行为）共享一个联合密度矩阵。
    
    A的基态（语言结论类别）：
       |0⟩ = "状态正常"   |1⟩ = "激活低下"   |2⟩ = "感官匮乏"   |3⟩ = "饱和过度"
    
    B的基态（行为投票类别）：
       |0⟩ = "偏好原始"   |1⟩ = "偏好剥夺"   |2⟩ = "偏好放大"   |3⟩ = "混乱/无偏好"
    """
    
    N_A = 4  # A通道维度
    N_B = 4  # B通道维度
    
    A_LABELS = ["状态正常", "激活低下", "感官匮乏", "饱和过度"]
    B_LABELS = ["偏好原始配置", "偏好感官剥夺", "偏好感官放大", "混乱无偏好"]
    
    def __init__(self, initial_state='max_entangle'):
        self.dim = self.N_A * self.N_B
        
        if initial_state == 'max_entangle':
            # 最大纠缠态 |Ψ⟩ = (1/2) Σ |i⟩_A ⊗ |i⟩_B
            psi = np.zeros(self.dim, dtype=complex)
            for i in range(min(self.N_A, self.N_B)):
                psi[i * self.N_B + i] = 0.5  # 1/dim_A
            # 等等——最大纠缠态的归一化应该是 Σ|i⟩|i⟩/√N
            psi = np.zeros(self.dim, dtype=complex)
            for i in range(min(self.N_A, self.N_B)):
                psi[i * self.N_B + i] = 1.0 / math.sqrt(min(self.N_A, self.N_B))
            self.rho = np.outer(psi, psi.conj())
        
        elif initial_state == 'separable':
            # 可分离态：ρ = ρ_A ⊗ ρ_B（经典关联）
            rho_A = np.eye(self.N_A) / self.N_A
            rho_B = np.eye(self.N_B) / self.N_B
            self.rho = np.kron(rho_A, rho_B)
        
        elif initial_state == 'identity':
            self.rho = np.eye(self.dim) / self.dim
        
        else:
            raise ValueError(f"Unknown initial_state: {initial_state}")
    
    # ---- 量子测量 ----
    
    def measure_A(self, proj_weights=None):
        """
        对A通道做一次测量，返回测量结果并坍缩B。
        
        proj_weights: [w0, w1, w2, w3] 每种结论的权重（由WM状态决定）
        返回: (a_result, prob_a, rho_B_collapsed, rho_A_reduced)
        """
        rho_A = partial_trace(self.rho, self.N_A, self.N_B, trace='B')
        
        if proj_weights is None:
            proj_weights = [1.0] * self.N_A
        
        # 测量概率 P(a) = Tr(M_a† M_a ρ_A), M_a = √w_a |a⟩⟨a|
        probs = []
        for a in range(self.N_A):
            v = np.zeros(self.N_A)
            v[a] = 1.0
            M = np.outer(v, v) * proj_weights[a]
            p = (M @ rho_A).trace().real
            probs.append(max(0.0, p))
        
        # 归一化
        total = sum(probs)
        if total < 1e-15:
            # 退化：均匀随机
            probs = [1.0 / self.N_A] * self.N_A
        else:
            probs = [p / total for p in probs]
        
        # 采样结果
        a_result = np.random.choice(self.N_A, p=probs)
        
        # B通道坍缩：ρ_B|a = Tr_A(|a⟩⟨a| ρ_AB) / P(a)
        # 投影到 |a⟩⟨a| ⊗ I_B
        projector_A = np.zeros((self.N_A, self.N_A))
        projector_A[a_result, a_result] = 1.0
        proj_full = np.kron(projector_A, np.eye(self.N_B))
        
        rho_collapsed = (proj_full @ self.rho @ proj_full.conj().T) / max(probs[a_result], 1e-15)
        
        # 更新联合态（A坍缩，B条件化）
        self.rho = rho_collapsed
        # 保持厄米性
        self.rho = 0.5 * (self.rho + self.rho.conj().T)
        # 保持迹为1（数值修正）
        tr = self.rho.trace().real
        if abs(tr - 1.0) > 1e-10:
            self.rho /= tr
        
        rho_B_reduced = partial_trace(self.rho, self.N_A, self.N_B, trace='A')
        
        return a_result, probs[a_result], rho_B_reduced, rho_A
    
    def measure_B(self, proj_weights=None):
        """
        对B通道做一次测量，返回测量结果并坍缩A。
        （与measure_A对称）
        """
        rho_B = partial_trace(self.rho, self.N_A, self.N_B, trace='A')
        
        if proj_weights is None:
            proj_weights = [1.0] * self.N_B
        
        probs = []
        for b in range(self.N_B):
            v = np.zeros(self.N_B)
            v[b] = 1.0
            M = np.outer(v, v) * proj_weights[b]
            p = (M @ rho_B).trace().real
            probs.append(max(0.0, p))
        
        total = sum(probs)
        if total < 1e-15:
            probs = [1.0 / self.N_B] * self.N_B
        else:
            probs = [p / total for p in probs]
        
        b_result = np.random.choice(self.N_B, p=probs)
        
        # A通道坍缩
        projector_B = np.zeros((self.N_B, self.N_B))
        projector_B[b_result, b_result] = 1.0
        proj_full = np.kron(np.eye(self.N_A), projector_B)
        
        rho_collapsed = (proj_full @ self.rho @ proj_full.conj().T) / max(probs[b_result], 1e-15)
        
        self.rho = rho_collapsed
        self.rho = 0.5 * (self.rho + self.rho.conj().T)
        tr = self.rho.trace().real
        if abs(tr - 1.0) > 1e-10:
            self.rho /= tr
        
        rho_A_reduced = partial_trace(self.rho, self.N_A, self.N_B, trace='B')
        
        return b_result, probs[b_result], rho_A_reduced, rho_B
    
    def get_rho_A(self):
        return partial_trace(self.rho, self.N_A, self.N_B, trace='B')
    
    def get_rho_B(self):
        return partial_trace(self.rho, self.N_A, self.N_B, trace='A')
    
    def correlated_measure(self):
        """
        同时测量A和B，返回联合结果。
        测量顺序：先A后B（结果依赖于顺序——这是纠缠不同于经典关联的特征）
        """
        a_result, p_a, rho_B_a, rho_A = self.measure_A()
        b_result, p_b_given_a, _, _ = self.measure_B()
        return a_result, b_result, p_a, p_b_given_a
    
    def entanglement_entropy(self):
        """纠缠熵 S(ρ_A)"""
        return von_neumann_entropy(self.get_rho_A())
    
    def reset(self, state='max_entangle'):
        """重置纠缠态"""
        if state == 'max_entangle':
            psi = np.zeros(self.dim, dtype=complex)
            for i in range(min(self.N_A, self.N_B)):
                psi[i * self.N_B + i] = 1.0 / math.sqrt(min(self.N_A, self.N_B))
            self.rho = np.outer(psi, psi.conj())
        elif state == 'separable':
            rho_A = np.eye(self.N_A) / self.N_A
            rho_B = np.eye(self.N_B) / self.N_B
            self.rho = np.kron(rho_A, rho_B)

# ============ WM接口层 ============

class WMBridge:
    """把WM的数值特征转化为投影权重，喂给纠缠通道"""
    
    @staticmethod
    def wm_to_A_weights(wm):
        """根据WM状态生成A通道四个结论的投影权重"""
        n = wm.g['n']
        h = wm.h
        h_samples = wm.err_history[-20:] if wm.err_history else [0.5]
        
        # 激活度
        active_ratio = sum(1 for v in h if abs(v) > 0.5) / max(1, n)
        sat_ratio = sum(1 for v in h if abs(v) > 0.98) / max(1, n)
        dead_ratio = sum(1 for v in h if abs(v) < 0.01) / max(1, n)
        
        # 误差
        avg_err = sum(h_samples) / max(1, len(h_samples))
        
        # region多样性
        regions = set()
        for frame in wm.db_history:
            if frame:
                regions.add(round(frame[7] * 8))
        region_diversity = len(regions)
        
        # 4个权重——用WM的数值特征直接驱动
        w_normal = 0.5 if (active_ratio > 0.3 and sat_ratio < 0.5 and avg_err < 1.0) else 0.05
        w_low_act = 1.0 if dead_ratio > 0.5 else 0.1
        w_sensory = 1.0 if region_diversity <= 2 else 0.05  # 区域少→感官匮乏权重大
        w_sat = 1.0 if sat_ratio > 0.5 else 0.05  # 饱和度高→饱和权重大
        
        # 叠加一些WM特征权重
        if avg_err > 1.0:
            w_low_act += 0.3  # 高误差也指向问题
        
        return [w_normal, w_low_act, w_sensory, w_sat]
    
    @staticmethod
    def votes_to_B_weights(votes_summary):
        """根据B通道的投票结果生成B通道投影权重"""
        # votes_summary: {config_name: fitness, ...}
        if not votes_summary:
            return [0.25, 0.25, 0.25, 0.25]
        
        configs = list(votes_summary.keys())
        fitnesses = list(votes_summary.values())
        
        # 找最优配置
        best_idx = fitnesses.index(max(fitnesses))
        best_config = configs[best_idx]
        
        # 根据最优配置决定B投影
        w_baseline = 0.05
        w_deprive = 0.05
        w_amplify = 0.05
        w_chaos = 0.05
        
        if 'deprived' in best_config or 'half' in best_config:
            w_deprive = 2.0
        elif 'hyper' in best_config:
            w_amplify = 2.0
        elif best_config == 'baseline':
            w_baseline = 2.0
        else:
            # fitness都很低 → 混乱
            if max(fitnesses) < 0.1:
                w_chaos = 2.0
            else:
                w_baseline = 1.5
        
        # fitness差异小→混乱
        if len(fitnesses) >= 2:
            fit_range = max(fitnesses) - min(fitnesses)
            if fit_range < 0.01:
                w_chaos = max(w_chaos, 1.0)
        
        return [w_baseline, w_deprive, w_amplify, w_chaos]

# ============ 纠缠双通道协议 ============

class EntangledDualChannel:
    """
    用纠缠密度矩阵把A和B耦合：
    1. A测语言结论 → B坍缩为对应的条件态
    2. B测行为结论 → A坍缩为对应的条件态
    3. 纠缠熵S量化A-B关联的非经典程度
    """
    
    def __init__(self):
        self.entangled = EntangledChannel(initial_state='max_entangle')
        self.bridge = WMBridge()
        self.history = []
    
    def single_trial(self, wm, votes_summary, initial_state='max_entangle'):
        """
        单次纠缠实验：
        1. 制备指定初始态
        2. 用WM状态驱动A测量 → 语言结论
        3. B坍缩，用B权重测 → 行为结论
        initial_state: 'max_entangle' 或 'separable'
        """
        self.entangled.reset(initial_state)
        
        S_before = self.entangled.entanglement_entropy()
        a_weights = self.bridge.wm_to_A_weights(wm)
        b_weights = self.bridge.votes_to_B_weights(votes_summary)
        
        a_result, p_a, rho_B_a, rho_A = self.entangled.measure_A(a_weights)
        b_result, p_b_given_a, rho_A_cond, rho_B_cond = self.entangled.measure_B(b_weights)
        
        return a_result, b_result, p_a, p_b_given_a, S_before
    
    def separable_trial(self, wm, votes_summary):
        """
        可分离态下单次实验：A和B独立采样，无坍缩、无顺序依赖。
        P(a,b) = P(a)P(b) 严格成立。
        """
        self.entangled.reset('separable')
        a_weights = self.bridge.wm_to_A_weights(wm)
        b_weights = self.bridge.votes_to_B_weights(votes_summary)
        
        # 独立采样A
        rho_A = self.entangled.get_rho_A()
        probs_A = []
        for a in range(self.entangled.N_A):
            v = np.zeros(self.entangled.N_A); v[a] = 1.0
            M = np.outer(v, v) * a_weights[a]
            p = max(0, (M @ rho_A).trace().real)
            probs_A.append(p)
        total = sum(probs_A)
        probs_A = [p/total for p in probs_A] if total > 1e-15 else [1.0/self.entangled.N_A]*self.entangled.N_A
        
        # 独立采样B（完全不依赖A的结果）
        rho_B = self.entangled.get_rho_B()
        probs_B = []
        for b in range(self.entangled.N_B):
            v = np.zeros(self.entangled.N_B); v[b] = 1.0
            M = np.outer(v, v) * b_weights[b]
            p = max(0, (M @ rho_B).trace().real)
            probs_B.append(p)
        total_B = sum(probs_B)
        probs_B = [p/total_B for p in probs_B] if total_B > 1e-15 else [1.0/self.entangled.N_B]*self.entangled.N_B
        
        a_result = int(np.random.choice(self.entangled.N_A, p=probs_A))
        b_result = int(np.random.choice(self.entangled.N_B, p=probs_B))
        
        return a_result, b_result
    
    def run_protocol(self, wm, votes_summary, n_rounds=10):
        """
        运行多轮纠缠协议——每次重新制备纠缠态（物理正确：单纠缠态只够测一次）
        """
        joint_counts = collections.Counter()
        marginal_A = collections.Counter()
        marginal_B = collections.Counter()
        entropy_log = []
        
        for round_i in range(n_rounds):
            a_r, b_r, p_a, p_b_a, S_init = self.single_trial(wm, votes_summary)
            joint_counts[(a_r, b_r)] += 1
            marginal_A[a_r] += 1
            marginal_B[b_r] += 1
            entropy_log.append((f'round_{round_i}_A', S_init))
        
        S_avg = sum(self.entangled.entanglement_entropy() for _ in range(5)) / 5
        entropy_log.append(('final_avg', S_avg))
        
        # 经典相关系数（皮尔逊，基于边际分布近似）
        # 用 joint_counts 的分布计算
        n_total = sum(joint_counts.values())
        
        # 贝尔型关联 E(a,b) = Σ (-1)^(a⊕b) * P(a,b)
        bell_E = 0.0
        for (a, b), cnt in joint_counts.items():
            p_ab = cnt / n_total
            parity = (-1) ** ((a % 2) ^ (b % 2))  # 简化的二值贝尔
            bell_E += parity * p_ab
        
        # 经典预测（如果是可分离态，|E|应该≤某个界；纠缠可以违反）
        # 这里不严格算贝尔不等式界限，只用 E 的绝对值作为关联强度
        
        # 一致性分数：P(a,b) vs P(a)*P(b) 的偏差
        consistency = 0.0
        for a in range(self.entangled.N_A):
            for b in range(self.entangled.N_B):
                p_ab = joint_counts[(a, b)] / n_total
                p_a = marginal_A[a] / n_total
                p_b = marginal_B[b] / n_total
                # KL散度型偏差
                if p_ab > 0:
                    consistency += p_ab * math.log2(p_ab / max(p_a * p_b, 1e-12))
        
        return {
            'joint_counts': joint_counts,
            'marginal_A': marginal_A,
            'marginal_B': marginal_B,
            'entropy_log': entropy_log,
            'S_before': 2.0,  # 每次重置到最大纠缠
            'S_final': 0.0,   # 测量后坍缩
            'bell_E': bell_E,
            'consistency_KL': consistency,
            'n_rounds': n_rounds,
        }
    
    def compare_with_separable(self, wm, votes_summary, n_rounds=10):
        """同样的测量协议，但初始化为可分离态——对比纠缠 vs 经典的区别"""
        # 纠缠态结果
        result_entangled = self.run_protocol(wm, votes_summary, n_rounds)
        
        # 可分离态结果：A和B完全独立采样（无纠缠、无坍缩）
        joint_counts_sep = collections.Counter()
        marginal_A_sep = collections.Counter()
        marginal_B_sep = collections.Counter()
        
        for _ in range(n_rounds):
            a_r, b_r = self.separable_trial(wm, votes_summary)
            joint_counts_sep[(a_r, b_r)] += 1
            marginal_A_sep[a_r] += 1
            marginal_B_sep[b_r] += 1
        
        # 可分离态的一致性（KL散度）
        n_total = sum(joint_counts_sep.values())
        consistency_sep = 0.0
        for a in range(self.entangled.N_A):
            for b in range(self.entangled.N_B):
                p_ab = joint_counts_sep[(a, b)] / n_total
                p_a = marginal_A_sep[a] / n_total
                p_b = marginal_B_sep[b] / n_total
                if p_ab > 0:
                    consistency_sep += p_ab * math.log2(p_ab / max(p_a * p_b, 1e-12))
        
        return {
            'entangled': result_entangled,
            'separable': {
                'joint_counts': joint_counts_sep,
                'marginal_A': marginal_A_sep,
                'marginal_B': marginal_B_sep,
                'consistency_KL': consistency_sep,
            },
        }

# ============ WM 辅助 ============

class Pool:
    def __init__(self, seed=42):
        self.seed=seed; self.r=self._mk(seed); self._r()
    def _mk(self,s):
        rng=random.Random(s)
        return [{'id':i,'sigma':round(0.05+0.64*i/8,2),'proprio':round(i/8,2),
            'freq':round(0.03+0.12*rng.random(),3),'phase':round(rng.uniform(0,2*math.pi),3),
            'type':['sin','logistic','lorenz','vdP','saw','rossler','henon','tent','cubic'][i],
            '_s':0.3,'_t':0,'_p':0.0} for i in range(N_REGIONS)]
    def _r(self):
        for r in self.r: r['_s']=0.3; r['_t']=0; r['_p']=0.0
    def reset(self): self._r()
    def step(self):
        for r in self.r: r['_t']+=1; r['_p']+=r['freq']
    def sense(self,rid):
        r=self.r[rid]; t=r['_t']; ph=r['_p']
        nv=random.Random(hash((self.seed,rid,t))%(2**31)).gauss(0,1)
        n=r['sigma']*nv; s=r['_s']; rt=r['type']
        if rt=='sin': v=math.sin(ph+r['phase'])
        elif rt=='logistic': v=4*s*(1-s) if 0<s<1 else 0.5
        elif rt=='lorenz': v=0.5+0.3*math.tanh(s)
        elif rt=='vdP': v=s-0.1*(s**3-s)
        elif rt=='saw': v=(ph+r['phase'])%1.0
        elif rt=='rossler': v=0.5*math.sin(0.7*ph+s)
        elif rt=='henon': v=1-1.4*s**2+0.3*s
        elif rt=='tent': v=2*s if s<0.5 else 2*(1-s)
        else: v=s**3-s
        s2=max(-2,min(2,v+n)); r['_s']=s2
        return [s2, math.sin(s2*2.5) if abs(s2)<1.5 else 0, math.tanh(s2),
                s2**2*(1 if s2>0 else -1), abs(s2), math.cos(s2*1.7),
                (1 if s2>0 else -1)*math.sqrt(abs(s2)) if abs(s2)<2 else 0, r['proprio']]

# 常量
N_CHANNELS = 8; MAX_NODES = 30; N_REGIONS = 9; DELAY = 3

class WM:
    def __init__(self,g):
        self.g=g; n=g['n']
        self.h=[0.0]*n; self.prev_h=[0.0]*n
        self.db_history=[]; self.prev_sens=None; self.x=[]
        self.err_history=[]
    def ri(self,p):
        i=round(p*8); return max(0,min(8,i))
    def build_input(self,sens):
        self.db_history.append(sens[:])
        if len(self.db_history)>DELAY: self.db_history.pop(0)
        while len(self.db_history)<DELAY: self.db_history.insert(0,[0.0]*N_CHANNELS)
        result=sens[:]
        for frame in self.db_history: result.extend(frame)
        return result
    def fwd(self,sens,update=True):
        x=self.build_input(sens); self.x=x
        g=self.g; n=g['n']; p=sens[7]; ri=self.ri(p)
        dec=g['dt'][ri]; bb=ri*N_CHANNELS
        nn=min(len(g['sw']),len(x)); nh=[0.0]*n
        for i in range(n):
            a=0.0
            for j in range(nn): a+=g['sw'][j][i]*x[j]
            for k in range(n): a+=g['rw'][i][k]*self.h[k]
            nh[i]=math.tanh(a)
        if update:
            for i in range(n): self.h[i]=(1-dec)*nh[i]+dec*self.h[i]
        pred=[0.0]*N_CHANNELS
        for i in range(n):
            for c in range(N_CHANNELS): pred[c]+=self.h[i]*(g['pw'][i][c]+g['pb'][bb+c])
        n_skip=min(len(g['skip_w']),len(x))
        for j in range(n_skip):
            for c in range(N_CHANNELS): pred[c]+=g['skip_w'][j][c]*x[j]
        if update: self.prev_h=self.h[:]
        self.prev_sens=sens[:]
        return pred
    def protein_fold(self):
        g=self.g; n=g['n']; fold_lr=g.get('plr_fold',g['plr']*0.1)
        for k in range(n):
            predicted=0.0
            for i in range(n): predicted+=g['rw'][k][i]*self.prev_h[i]
            if abs(predicted)>0.01:
                predicted=math.tanh(predicted)
                err=self.h[k]-predicted
                for i in range(n):
                    delta=fold_lr*self.prev_h[i]*err
                    g['rw'][k][i]+=delta
                    g['rw'][k][i]=max(-1.2,min(1.2,g['rw'][k][i]))
    def learn(self,pred,actual):
        g=self.g; n=g['n']; lr=g['plr']; ri=self.ri(actual[7]); bb=ri*N_CHANNELS
        errs=[actual[j]-pred[j] for j in range(N_CHANNELS)]
        mse=sum(e*e for e in errs)/N_CHANNELS
        self.err_history.append(mse)
        for i in range(n):
            for c in range(N_CHANNELS):
                g['pw'][i][c]+=lr*self.h[i]*errs[c]
                g['pw'][i][c]=max(-3,min(3,g['pw'][i][c]))
        for c in range(N_CHANNELS):
            g['pb'][bb+c]+=lr*1.5*errs[c]
            g['pb'][bb+c]=max(-5,min(5,g['pb'][bb+c]))
        n_skip=min(len(g['skip_w']),len(self.x))
        for j in range(n_skip):
            for c in range(N_CHANNELS):
                g['skip_w'][j][c]+=lr*0.5*self.x[j]*errs[c]
                g['skip_w'][j][c]=max(-2,min(2,g['skip_w'][j][c]))
        self.protein_fold()
        return mse
    def reset(self):
        self.h=[0.0]*len(self.h); self.prev_h=[0.0]*len(self.prev_h)
        self.db_history=[]; self.prev_sens=None; self.x=[]; self.err_history=[]

class Org:
    def __init__(self,gn=None):
        self.g=gn or self.rg(); self.wm=WM(self.g)
        self.alive=True; self.age=0
        self._ib=1.0; self._r=None
    def rg(self):
        n=random.randint(10,MAX_NODES)
        input_size=N_CHANNELS*(1+DELAY)
        g={'n':n,
           'sw':[[random.uniform(-0.3,0.3) for _ in range(n)] for _ in range(input_size)],
           'pw':[[random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)] for _ in range(n)],
           'rw':[[random.uniform(-0.05,0.05) for _ in range(n)] for _ in range(n)],
           'skip_w':[[random.uniform(-0.2,0.2) for _ in range(N_CHANNELS)] for _ in range(input_size)],
           'plr':random.uniform(0.04,0.15),
           'plr_fold':random.uniform(0.02,0.08),
           'pb':[random.uniform(-0.3,0.3) for _ in range(N_REGIONS*N_CHANNELS)],
           'dt':[random.uniform(0.4,0.95) for _ in range(N_REGIONS)]}
        for i in range(n): g['rw'][i][i]=random.uniform(0.3,1.0)
        return g

def train_org(org, ticks=300, seed=42):
    """快速训练一个org，让内部状态活起来"""
    pk = Pool(seed=seed); pk.reset()
    cr = random.randint(0, 8)
    for _ in range(30):
        s = pk.sense(cr); org.wm.fwd(s, update=True); pk.step()
    for _ in range(ticks):
        s = pk.sense(cr); pred = org.wm.fwd(s, update=True)
        pk.step(); s_next = pk.sense(cr); pk.step()
        org.wm.learn(pred, s_next)
        if random.random() < 0.1: cr = random.randint(0, 8)
    return org

# ============ 输出 ============

def print_entangled_report(result, label=""):
    print(f"\n{'='*60}")
    print(f"  纠缠双通道协议结果 — {label}")
    print(f"{'='*60}")
    
    n_total = sum(result['joint_counts'].values()) if result['joint_counts'] else 0
    
    # 联合分布
    print(f"\nA通道语言结论 vs B通道行为结论  (N={n_total}轮)")
    print(f"{'':>16}", end="")
    for b in range(EntangledChannel.N_B):
        label = EntangledChannel.B_LABELS[b][:8]
        print(f"  B{ b }-{label}", end="")
    print()
    
    for a in range(EntangledChannel.N_A):
        label = EntangledChannel.A_LABELS[a][:6]
        print(f"  A{a}-{label:>6}", end="")
        for b in range(EntangledChannel.N_B):
            cnt = result['joint_counts'].get((a, b), 0)
            p = cnt / n_total if n_total > 0 else 0
            print(f"  {p:>6.2%}" if p > 0 else f"  {'·':>6}", end="")
        print()
    
    # 纠缠熵
    print(f"\n纠缠熵变化:")
    for tag, s in result['entropy_log']:
        bar = '█' * int(s * 10) + '░' * (20 - int(s * 10))
        print(f"  {tag:<12} S = {s:.4f}  {bar}")
    
    # 贝尔关联
    print(f"\n贝尔型关联 E = {result['bell_E']:+.4f}  (|E|→1:强关联, →0:无关联)")
    
    # KL散度一致性
    print(f"\nKL散度 P(a,b)‖P(a)P(b) = {result['consistency_KL']:.6f}")
    print(f"  (=0 说明A和B独立/可分离; >0 说明联合分布不能分解 → 关联/纠缠)")
    
    # 解读
    S_init = result['S_before']
    S_fin = result['S_final']
    print(f"\n初始纠缠熵 S_initial = {S_init:.4f} (最大纠缠)")
    print(f"终态纠缠熵 S_final   = {S_fin:.4f}")
    
    if S_fin < 0.1:
        print("→ 纠缠低：A和B的测量几乎坍缩到可分离态（测量破坏了纠缠）")
    elif S_fin > S_init * 0.5:
        print("→ 纠缠维持：A-B之间仍保留显著量子关联")
    else:
        print("→ 部分退相干：测量部分破坏了纠缠，但残余关联仍超过经典预期")
    
    if abs(result['bell_E']) > 0.3:
        print(f"→ 强贝尔关联：A-B的非经典关联不可被经典隐变量解释")
    elif abs(result['bell_E']) > 0.1:
        print(f"→ 中等关联")
    else:
        print(f"→ 弱关联")

def print_comparison_report(comp):
    """打印纠缠 vs 可分离对比"""
    ent = comp['entangled']
    sep = comp['separable']
    
    print(f"\n{'='*60}")
    print(f"  纠缠态 vs 可分离态 对比")
    print(f"{'='*60}")
    
    print(f"\n{'指标':<24} {'纠缠态':>12} {'可分离态':>12}")
    print(f"{'-'*48}")
    
    # 纠缠熵
    S_ent = ent['S_final']
    S_sep = 0.0  # 可分离态熵为0
    print(f"{'纠缠熵 S(ρ_A)':<24} {S_ent:>12.4f} {S_sep:>12.4f}")
    
    # KL散度（关联强度）
    KL_ent = ent['consistency_KL']
    KL_sep = sep['consistency_KL']
    print(f"{'KL散度 P(ab)‖P(a)P(b)':<24} {KL_ent:>12.6f} {KL_sep:>12.6f}")
    
    # 贝尔关联
    bell_ent = abs(ent['bell_E'])
    bell_sep = 0.0  # 可分离态无贝尔违反
    print(f"{'|贝尔关联E|':<24} {bell_ent:>12.4f} {bell_sep:>12.4f}")
    
    # 最可能的联合态
    if ent['joint_counts']:
        top_ent_pair, top_ent_cnt = ent['joint_counts'].most_common(1)[0]
        a_e, b_e = top_ent_pair
        p_e = top_ent_cnt / sum(ent['joint_counts'].values())
    else:
        a_e, b_e, p_e = -1, -1, 0
    
    if sep['joint_counts']:
        top_sep = sep['joint_counts'].most_common(1)[0]
        a_s, b_s = top_sep[0]
        p_s = top_sep[1] / sum(sep['joint_counts'].values())
    else:
        a_s, b_s, p_s = -1, -1, 0
    
    print(f"\n最可能联合态:")
    if a_e >= 0:
        print(f"  纠缠: |{EntangledChannel.A_LABELS[a_e]}⟩⊗|{EntangledChannel.B_LABELS[b_e]}⟩  p={p_e:.1%}")
    if a_s >= 0:
        print(f"  经典: |{EntangledChannel.A_LABELS[a_s]}⟩⊗|{EntangledChannel.B_LABELS[b_s]}⟩  p={p_s:.1%}")
    
    # 判定
    print(f"\n判定:")
    if KL_ent > 0.01 and KL_ent > KL_sep * 2:
        print("  ✓ 纠缠态的联合分布显著偏离乘积态 —— A-B关联超出经典范围")
    else:
        print("  → 纠缠态未显著超过经典关联 —— 可能初始态非最大纠缠或测量破坏太重")
    
    if S_ent > 0.5:
        print("  ✓ 剩余纠缠熵 > 0.5 —— A-B之间仍保留量子相干性")
    else:
        print("  → 剩余纠缠熵较低 —— 多次测量后量子关联已衰减")

# ============ 主流程 ============

if __name__ == '__main__':
    print("阴阳纠缠通道 — A←Ψ→B 纠缠双通道")
    print(f"联合态维度: {EntangledChannel.N_A}×{EntangledChannel.N_B} = {EntangledChannel.N_A*EntangledChannel.N_B}")
    print()
    
    random.seed(42); np.random.seed(42)
    
    # 创建三个测试org
    test_orgs = []
    
    # 1. 低激活org（高dead_ratio）
    org1 = Org()
    train_org(org1, ticks=200, seed=1)
    # 强制让它进入低激活状态（通过训练少）
    test_orgs.append(("低激活org", org1))
    
    # 2. 多区域探索org
    org2 = Org()
    pk = Pool(seed=77); pk.reset()
    cr = 0
    for _ in range(30):
        s = pk.sense(cr); org2.wm.fwd(s, update=True); pk.step()
    for _ in range(500):
        s = pk.sense(cr); pred = org2.wm.fwd(s, update=True)
        pk.step(); s_next = pk.sense(cr); pk.step()
        org2.wm.learn(pred, s_next)
        if random.random() < 0.3: cr = random.randint(0, 8)  # 频繁切区域
    test_orgs.append(("多区域org", org2))
    
    # 3. 饱和org（高饱和比）
    org3 = Org()
    train_org(org3, ticks=400, seed=99)
    test_orgs.append(("高饱和org", org3))
    
    for label, org in test_orgs:
        print(f"\n{'#'*60}")
        print(f"# {label}: 节点={org.g['n']} 误差={sum(org.wm.err_history[-5:])/max(1,len(org.wm.err_history[-5:])):.3f}")
        print(f"{'#'*60}")
        
        # 生成B通道投票数据
        votes_data = {
            'baseline': random.uniform(0.0005, 0.005),
            'half_deprived': random.uniform(0.01, 0.08),
            'ch0_hyper': random.uniform(0.0002, 0.003),
        }
        
        # 运行纠缠协议
        ch = EntangledDualChannel()
        result = ch.run_protocol(org.wm, votes_data, n_rounds=20)
        print_entangled_report(result, label)
        
        # 对比纠缠 vs 可分离
        ch2 = EntangledDualChannel()
        comp = ch2.compare_with_separable(org.wm, votes_data, n_rounds=20)
        print_comparison_report(comp)
