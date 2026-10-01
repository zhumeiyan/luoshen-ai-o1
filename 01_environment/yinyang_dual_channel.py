#!/usr/bin/env python3
"""
Dual-Channel Sensory Discovery Prototype
=========================================
A通道（语言）：把WM内部状态翻译成结构化文字，可选接LLM生成自然语言
B通道（行为）：给多组感官配置跑WM，测量五项指标投票选最优
交叉验证：A说的话 vs B做的事，一致性报告

不修改v17o的演化逻辑，作为独立诊断工具运行。
"""
import random, math, hashlib, copy, collections

# ====== Constants from v17o =======
N_CHANNELS = 8; MAX_NODES = 30; N_REGIONS = 9; DELAY = 3

# ====== Stripped-down WM/Pool from v17o =======
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

# ====== Org wrapper =======
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

# ====== CHANNEL A: Language Translation =======
class ChannelA:
    """把WM内部状态翻译成文字。可接LLM，也可纯规则生成。"""
    
    def __init__(self, use_llm=False):
        self.use_llm = use_llm
    
    def snapshot_to_text(self, wm, ticks_run):
        """将当前WM状态转为结构化文本"""
        g = wm.g; n = g['n']; h = wm.h
        
        # 激活统计
        active_hi = [v for v in h if abs(v) > 0.5]
        sat_hi = [v for v in h if abs(v) > 0.98]
        dead_hi = [v for v in h if abs(v) < 0.01]
        
        avg_h = sum(h)/n if n > 0 else 0
        max_h = max(h) if h else 0
        min_h = min(h) if h else 0
        
        # 误差历史
        err_hist = wm.err_history[-10:] if wm.err_history else []
        avg_err = sum(err_hist)/len(err_hist) if err_hist else 0
        err_trend = "下降" if len(err_hist) >= 5 and err_hist[-1] < err_hist[-5] else "上升" if len(err_hist) >= 5 else "无数据"
        
        # 权重能量
        rw_energy = sum(g['rw'][i][i]**2 for i in range(n))
        try:
            rw_energy = rw_energy ** 0.5
        except:
            rw_energy = 0
        
        skip_energy = sum(g['skip_w'][j][c]**2 for j in range(len(g['skip_w'])) for c in range(N_CHANNELS))
        try:
            skip_energy = skip_energy ** 0.5
        except:
            skip_energy = 0
        
        # 筋树数（活跃的连接）
        active_connections = sum(1 for i in range(n) for k in range(n) if i != k and abs(g['rw'][i][k]) > 0.1)
        
        # 感知多样性
        region_usage = set()
        for frame in wm.db_history:
            if frame:
                region_usage.add(round(frame[7] * 8))
        
        text = (
            f"【洛神内部状态快照 (已运行{ticks_run}步)】\n"
            f"- 隐状态范围: [{min_h:.3f}, {max_h:.3f}]，均值{avg_h:.3f}\n"
            f"- 激活的隐节点: {len(active_hi)}/{n} ({100*len(active_hi)/n:.0f}%)\n"
            f"- 饱和节点(>0.98): {len(sat_hi)}/{n} ({100*len(sat_hi)/n:.0f}%)\n"
            f"- 静默节点(<0.01): {len(dead_hi)}/{n} ({100*len(dead_hi)/n:.0f}%)\n"
            f"- 近期预测误差: 均值{avg_err:.4f}，趋势{err_trend}\n"
            f"- 回路能量(rw): {rw_energy:.2f}\n"
            f"- 直连能量(skip): {skip_energy:.2f}\n"
            f"- 筋树数(活跃非对角连接): {active_connections}\n"
            f"- 当前使用的环境区域: {sorted(region_usage) if region_usage else '无'}\n"
            f"- 延迟线历史帧数: {len(wm.db_history)}\n"
        )
        
        # 推测缺失的感官
        missing = []
        if len(active_hi) < n * 0.3:
            missing.append("内部激活极低——可能缺少足够的输入变化")
        if len(sat_hi) > n * 0.7:
            missing.append("大部分隐节点饱和——需要更多梯度输入打破饱和")
        if active_connections < 5:
            missing.append("回路过简——时序依赖缺失，长期记忆空间拥挤")
        if skip_energy < 1.0:
            missing.append("直连信号微弱——即时 sensory-motor 联系不成立")
        if len(region_usage) <= 1:
            missing.append("只探索了一个环境区域——感官多样性匮乏，对世界无空间感")
        
        if missing:
            text += "\n【推测缺失维度】\n"
            for m in missing:
                text += f"  * {m}\n"
        else:
            text += "\n【推测缺失维度】暂无明确缺失"
        
        return text
    
    def translate(self, wm, ticks_run):
        """输出文字（规则版或LLM版）"""
        structured = self.snapshot_to_text(wm, ticks_run)
        
        if self.use_llm:
            # LLM接口占位 —— 实际使用时接阿里云/千问
            return structured + f"\n[LLM接口未配置，使用规则输出]"
        
        # 规则版：直接输出结构化文本
        return structured

# ====== CHANNEL B: Behavioral Voting =======
class ChannelB:
    """多组感官配置跑WM，测量五项指标投票"""
    
    def __init__(self):
        # 五组感官配置方案
        self.configs = {
            "baseline":       {"noise": 0.0,  "mask_ch": [],       "amplify_ch": -1, "amp_factor": 1.0},
            "noisy_vision":   {"noise": 0.15, "mask_ch": [],       "amplify_ch": -1, "amp_factor": 1.0},
            "half_deprived":  {"noise": 0.0,  "mask_ch": [2,3,4,5], "amplify_ch": -1, "amp_factor": 1.0},
            "ch0_hyper":      {"noise": 0.0,  "mask_ch": [],       "amplify_ch": 0,   "amp_factor": 3.0},
            "ch7_hyper":      {"noise": 0.0,  "mask_ch": [],       "amplify_ch": 7,   "amp_factor": 3.0},
            "ch3_deprived":   {"noise": 0.0,  "mask_ch": [3],       "amplify_ch": -1, "amp_factor": 1.0},
        }
    
    def perturb_sense(self, sens, config):
        """对感官信号施加配置的扰动"""
        s = sens[:]
        # 加噪声
        if config["noise"] > 0:
            for i in range(min(N_CHANNELS, len(s))):
                if i not in config["mask_ch"]:
                    s[i] += random.gauss(0, config["noise"])
        # 遮罩通道
        for ch in config["mask_ch"]:
            if 0 <= ch < len(s):
                s[ch] = 0.0
        # 放大某通道
        if config["amplify_ch"] >= 0 and config["amplify_ch"] < len(s):
            s[config["amplify_ch"]] *= config["amp_factor"]
        return s
    
    def run_config(self, org, config_name, ticks=200):
        """跑一组配置，返回五项指标"""
        config = self.configs[config_name]
        pk = Pool(seed=42); pk.reset()
        wm = WM(copy.deepcopy(org.g))  # 不污染原org
        wm.reset()
        
        h_all = []
        total_mse = 0.0
        nan_detected = False
        cr = random.randint(0, 8)
        
        # 预热
        for _ in range(30):
            s = pk.sense(cr)
            s = self.perturb_sense(s, config)
            wm.fwd(s, update=True); pk.step()
        
        # 正式测量
        for t in range(ticks):
            s = pk.sense(cr)
            s = self.perturb_sense(s, config)
            pred = wm.fwd(s, update=True)
            pk.step()
            s_next = pk.sense(cr)
            s_next = self.perturb_sense(s_next, config)
            pk.step()
            
            mse = sum((s_next[j]-pred[j])**2 for j in range(N_CHANNELS))/N_CHANNELS
            total_mse += mse
            
            # 检测NaN
            for v in pred:
                if math.isnan(v) or math.isinf(v): nan_detected = True
            for v in wm.h:
                if math.isnan(v) or math.isinf(v): nan_detected = True
            
            h_all.append(wm.h[:])
            
            # 学习
            wm.learn(pred, s_next)
            
            # 偶尔切换区域
            if random.random() < 0.05: cr = random.randint(0, 8)
        
        # 五项指标
        n = org.g['n']
        
        # 1. 总激活
        total_activation = sum(sum(abs(v) for v in h_sample) for h_sample in h_all)
        try:
            total_activation = total_activation / (len(h_all) * n)
        except:
            total_activation = 0
        
        # 2. 最大激活
        max_activation = max(max(abs(v) for v in h_sample) for h_sample in h_all)
        
        # 3. 筋树数
        g = wm.g
        active_r = sum(1 for i in range(n) for k in range(n) if i != k and abs(g['rw'][i][k]) > 0.1)
        
        # 4. 节点亮（任意时刻|h|>0.5为亮）
        lit_nodes = set()
        for h_sample in h_all:
            for idx, v in enumerate(h_sample):
                if abs(v) > 0.5: lit_nodes.add(idx)
        node_lit_ratio = len(lit_nodes) / n if n > 0 else 0
        
        # 5. NaN标志
        has_nan = nan_detected
        
        # fitness (1/(1+avg_mse))
        avg_mse = total_mse / ticks if ticks > 0 else 1
        fitness = 1.0 / (1.0 + avg_mse)
        
        return {
            "config_name": config_name,
            "total_activation": total_activation,
            "max_activation": max_activation,
            "active_recurrent": active_r,
            "node_lit_ratio": node_lit_ratio,
            "has_nan": has_nan,
            "avg_mse": avg_mse,
            "fitness": fitness,
        }
    
    def vote(self, org, ticks_per_config=200):
        """跑全部配置，返回投票结果"""
        results = {}
        for name in self.configs:
            r = self.run_config(org, name, ticks_per_config)
            results[name] = r
        return results
    
    def find_preference(self, results):
        """综合打分：fitness高+激活适中+无NaN = 最优"""
        scored = []
        for name, r in results.items():
            if r["has_nan"]:
                score = -1000  # NaN一票否决
            else:
                score = r["fitness"] * 10  # 拟合度权重最高
                score += min(r["total_activation"], 2.0) * 0.5  # 适度激活加分
                score += r["node_lit_ratio"] * 2.0  # 节点参与度
                score -= r["max_activation"] * 0.1  # 轻微惩罚极端饱和
            scored.append((name, score, r))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

# ====== Cross-validation =======
class CrossValidator:
    """A通道文字 vs B通道行为，一致性检测"""
    
    def __init__(self, channel_a, channel_b):
        self.a = channel_a
        self.b = channel_b
    
    def analyze(self, org):
        """完整分析：跑两通道，输出对比"""
        
        # --- A通道：先跑一段WM，让内部状态活起来 ---
        pk = Pool(seed=42); pk.reset()
        wm_run = WM(copy.deepcopy(org.g)); wm_run.reset()
        cr = random.randint(0, 8)
        for _ in range(30):
            s = pk.sense(cr); wm_run.fwd(s, update=True); pk.step()
        for _ in range(100):
            s = pk.sense(cr); pred = wm_run.fwd(s, update=True)
            pk.step(); s_next = pk.sense(cr); pk.step()
            wm_run.learn(pred, s_next)
            if random.random() < 0.05: cr = random.randint(0, 8)
        
        text_output = self.a.translate(wm_run, 130)
        
        # --- B通道：感官投票 ---
        votes = self.b.vote(org, ticks_per_config=200)
        preference = self.b.find_preference(votes)
        
        # --- 一致性分析 ---
        consistency = self._check_consistency(text_output, preference, votes)
        
        return {
            "text_output": text_output,
            "votes": votes,
            "preference": preference,
            "consistency": consistency,
        }
    
    def _check_consistency(self, text, preference, votes):
        """对比A的文字和B的投票，找一致性"""
        issues = []
        agreements = []
        
        # A是否提到"激活低"
        a_mentions_low_activation = "激活的隐节点: 0/" in text or "极低" in text
        # B是否显示fitness很低
        baseline_fitness = votes.get("baseline", {}).get("fitness", 0)
        b_shows_low_fitness = baseline_fitness < 0.5
        
        if a_mentions_low_activation and b_shows_low_fitness:
            agreements.append("A报告低激活 ↔ B显示低fitness（一致）")
        elif a_mentions_low_activation and not b_shows_low_fitness:
            issues.append("A报告低激活 ↔ B显示正常fitness（矛盾）")
        
        # A是否提到"感官多样性匮乏"
        a_mentions_poor_sensory = "感官多样性匮乏" in text
        # B中多种扰动配置的fitness是否有明显差异
        fitness_range = max(r["fitness"] for r in votes.values() if not r["has_nan"]) - \
                        min(r["fitness"] for r in votes.values() if not r["has_nan"])
        b_shows_sensory_sensitive = fitness_range > 0.2
        
        if a_mentions_poor_sensory and b_shows_sensory_sensitive:
            agreements.append("A报告感官匮乏 ↔ B显示感官变化时fitness波动大（一致：世界模型已经深度依赖当前感官）")
        elif not a_mentions_poor_sensory and b_shows_sensory_sensitive:
            agreements.append("A未明确报告感官匮乏 ↔ B显示感官波动大（信息：当前感官虽被承认但依赖度极高）")
        
        # A是否提到"饱和"
        a_mentions_sat = "节点饱和" in text or "饱和节点(>0.98)" in text
        # baseline配置下max_activation是否>0.9
        b_high_activation = votes.get("baseline", {}).get("max_activation", 0) > 0.9
        
        if a_mentions_sat and b_high_activation:
            agreements.append("A报告饱和 ↔ B显示max_activation极高（一致）")
        
        # 最优配置与A的推测是否对得上
        best_config = preference[0][0]
        best_fitness = preference[0][2]["fitness"]
        worst_config = preference[-1][0]
        worst_fitness = preference[-1][2]["fitness"]
        
        # 总结点
        if best_config == "baseline" and best_fitness > 0.9:
            summary = f"洛神偏爱原始配置（baseline），说明当前感官够用"
        elif best_fitness - worst_fitness > 0.3:
            summary = f"洛神在{best_config}下fitness={best_fitness:.3f}，到{worst_config}只剩{worst_fitness:.3f}——它对当前感官类型敏感"
        else:
            summary = f"各配置fitness差异小（{best_fitness:.3f} vs {worst_fitness:.3f}），感官鲁棒"
        
        return {
            "agreements": agreements,
            "issues": issues,
            "best_config": best_config,
            "best_fitness": best_fitness,
            "worst_config": worst_config,
            "worst_fitness": worst_fitness,
            "summary": summary,
        }

# ====== Display =======
def print_report(report):
    """打印完整报告"""
    print("=" * 60)
    print("【A通道：洛神的语言自述】")
    print("=" * 60)
    print(report["text_output"])
    
    print()
    print("=" * 60)
    print("【B通道：感官配置投票结果】")
    print("=" * 60)
    votes = report["votes"]
    
    print(f"{'配置名':<16} {'fitness':>8} {'总激活':>8} {'最大激活':>8} {'筋树':>6} {'亮节点':>8} {'NaN':>5}")
    print("-" * 62)
    for name in ["baseline", "noisy_vision", "half_deprived", "ch0_hyper", "ch7_hyper", "ch3_deprived"]:
        if name in votes:
            r = votes[name]
            nan_str = "⚠" if r["has_nan"] else "·"
            print(f"{name:<16} {r['fitness']:>8.4f} {r['total_activation']:>8.3f} {r['max_activation']:>8.3f} {r['active_recurrent']:>6} {r['node_lit_ratio']:>7.0%} {nan_str:>5}")
    
    print()
    print("【B通道投票排序（综合分）】")
    for rank, (name, score, r) in enumerate(report["preference"], 1):
        nan_warn = " ⚠含NaN" if r["has_nan"] else ""
        print(f"  #{rank} {name:<16} 综合分={score:>8.2f}  fitness={r['fitness']:.4f}{nan_warn}")
    
    print()
    print("=" * 60)
    print("【交叉验证：说 vs 做】")
    print("=" * 60)
    c = report["consistency"]
    
    if c["agreements"]:
        print("一致性发现：")
        for a in c["agreements"]:
            print(f"  ✓ {a}")
    
    if c["issues"]:
        print("\n矛盾/冲突：")
        for i in c["issues"]:
            print(f"  ✗ {i}")
    
    if not c["agreements"] and not c["issues"]:
        print("  无明确一致或矛盾——需更多数据")
    
    print(f"\n最优配置: {c['best_config']} (fitness={c['best_fitness']:.4f})")
    print(f"最差配置: {c['worst_config']} (fitness={c['worst_fitness']:.4f})")
    print(f"\n结论: {c['summary']}")
    
    print("\n" + "=" * 60)

# ====== Main =======
if __name__ == '__main__':
    print("yinyang_dual_channel — A通道(语言)+B通道(行为)双通道感官发现")
    print(f"N_CHANNELS={N_CHANNELS}, DELAY={DELAY}")
    print()
    
    # 创建一些待测试的orgs
    test_orgs = []
    
    # 1. 完全随机org（刚初始化）
    test_orgs.append(("随机新生org", Org()))
    
    # 2. 预先训练过的org（模拟v17o演化后的个体）
    pre_trained = Org()
    pk = Pool(seed=42); pk.reset()
    cr = random.randint(0, 8)
    for _ in range(50):
        s = pk.sense(cr); pre_trained.wm.fwd(s, update=True); pk.step()
    for _ in range(500):
        s = pk.sense(cr); pred = pre_trained.wm.fwd(s, update=True)
        pk.step(); s_next = pk.sense(cr); pk.step()
        pre_trained.wm.learn(pred, s_next)
        if random.random() < 0.08: cr = random.randint(0, 8)
    test_orgs.append(("500步预训练org", pre_trained))
    
    # 3. 第二个预训练org（不同随机路径）
    pre_trained2 = Org()
    pk2 = Pool(seed=99); pk2.reset()
    cr2 = random.randint(0, 8)
    for _ in range(50):
        s = pk2.sense(cr2); pre_trained2.wm.fwd(s, update=True); pk2.step()
    for _ in range(500):
        s = pk2.sense(cr2); pred = pre_trained2.wm.fwd(s, update=True)
        pk2.step(); s_next = pk2.sense(cr2); pk2.step()
        pre_trained2.wm.learn(pred, s_next)
        if random.random() < 0.08: cr2 = random.randint(0, 8)
    test_orgs.append(("500步预训练org#2 (不同种子)", pre_trained2))
    
    # 初始化通道
    ch_a = ChannelA(use_llm=False)
    ch_b = ChannelB()
    validator = CrossValidator(ch_a, ch_b)
    
    # 逐个分析
    for label, org in test_orgs:
        print(f"\n{'#' * 60}")
        print(f"# 测试对象: {label}")
        print(f"# 节点数: {org.g['n']}  plr={org.g['plr']:.4f}  fold_lr={org.g['plr_fold']:.4f}")
        print(f"{'#' * 60}")
        
        report = validator.analyze(org)
        print_report(report)
