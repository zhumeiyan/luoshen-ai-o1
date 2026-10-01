#!/usr/bin/env python3
"""
v17o — Dual Pathway: Protein-Fold Temporal + Skip-Connection Raw Features
============================================================================
核心改动：给 readout 加 skip connection，让延迟线原始特征直接参与预测

v17n 的问题：
- readout 只看 hidden state (h[i]) → 预测
- hidden 一旦饱和 (tanh→±1) → readout 丢失信息
- 进化只能靠 hidden 当 binary memory → 天花板 0.92

v17o 的方案：
- Pathway A（蛋白折叠时序路径）：sens→rw→h→pw→pred
  - 慢变量，靠蛋白折叠学习时序模式
- Pathway B（跳过连接）：sens→skip_w→pred  
  - 快变量，直接用延迟线当前值预测下一个值
  - 不需要经过 tanh 饱和的非线性
- 两条路径的贡献在 readout 端相加

好处：
1. hidden 饱和不再致命——skip pathway 仍然能线性传递信号
2. 即时预测由 skip 完成，蛋白折叠负责长期记忆
3. 进化分工更自然
"""
import random, math, hashlib, copy, collections

N_CHANNELS = 8; MAX_NODES = 30; N_REGIONS = 9; DELAY = 3

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
        self.h=[0.0]*n
        self.prev_h=[0.0]*n
        self.db_history=[]
        self.prev_sens=None
        self.x=[]  # current delay-line input (for skip pathway)
    def ri(self,p):
        i=round(p*8); return max(0,min(8,i))

    def build_input(self,sens):
        """Build input: current + delayed history"""
        self.db_history.append(sens[:])
        if len(self.db_history)>DELAY:
            self.db_history.pop(0)
        while len(self.db_history)<DELAY:
            self.db_history.insert(0,[0.0]*N_CHANNELS)
        result=sens[:]
        for frame in self.db_history:
            result.extend(frame)
        return result

    def fwd(self,sens,update=True):
        x=self.build_input(sens)
        self.x=x  # save for skip pathway
        g=self.g; n=g['n']; p=sens[7]; ri=self.ri(p)
        dec=g['dt'][ri]; bb=ri*N_CHANNELS
        nn=min(len(g['sw']),len(x))
        nh=[0.0]*n
        for i in range(n):
            a=0.0
            for j in range(nn):
                a+=g['sw'][j][i]*x[j]
            for k in range(n):
                a+=g['rw'][i][k]*self.h[k]
            nh[i]=math.tanh(a)
        if update:
            for i in range(n):
                self.h[i]=(1-dec)*nh[i]+dec*self.h[i]
        # Dual pathway prediction:
        # Pathway A: hidden state -> readout (temporal, protein-folded)
        pred=[0.0]*N_CHANNELS
        for i in range(n):
            for c in range(N_CHANNELS):
                pred[c]+=self.h[i]*(g['pw'][i][c]+g['pb'][bb+c])
        # Pathway B: skip connection (raw delay-line -> readout)
        n_skip=min(len(g['skip_w']),len(x))
        for j in range(n_skip):
            for c in range(N_CHANNELS):
                pred[c]+=g['skip_w'][j][c]*x[j]
        if update:
            self.prev_h=self.h[:]
        self.prev_sens=sens[:]
        return pred

    def protein_fold(self):
        """
        蛋白质折叠（Pathway A 的时序学习）
        fwd中: rw[i][k] = weight FROM k TO i
        预测 h[k] = tanh(sum_i rw[k][i]*prev_h[i])
        更新: rw[k][i] += fold_lr * prev_h[i] * (h[k] - predicted)
        """
        g=self.g; n=g['n']
        fold_lr=g.get('plr_fold',g['plr']*0.1)
        for k in range(n):
            predicted=0.0
            for i in range(n):
                predicted+=g['rw'][k][i]*self.prev_h[i]
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

        # Pathway A learning: hidden state readout (pw/pb)
        for i in range(n):
            for c in range(N_CHANNELS):
                g['pw'][i][c]+=lr*self.h[i]*errs[c]
                g['pw'][i][c]=max(-3,min(3,g['pw'][i][c]))
        for c in range(N_CHANNELS):
            g['pb'][bb+c]+=lr*1.5*errs[c]
            g['pb'][bb+c]=max(-5,min(5,g['pb'][bb+c]))

        # Pathway B learning: skip connection (direct delay-line -> readout)
        n_skip=min(len(g['skip_w']),len(self.x))
        for j in range(n_skip):
            for c in range(N_CHANNELS):
                g['skip_w'][j][c]+=lr*0.5*self.x[j]*errs[c]
                g['skip_w'][j][c]=max(-2,min(2,g['skip_w'][j][c]))

        # Protein folding for temporal memory on Pathway A
        self.protein_fold()

        return mse

    def reset(self):
        self.h=[0.0]*len(self.h)
        self.prev_h=[0.0]*len(self.h)
        self.db_history=[]
        self.prev_sens=None
        self.x=[]

class IM:
    def __init__(self):
        self.ef=self.es=self.efs=0.3; self.bor=0.0; self.drive=0.0; self.rh=[]
    def upd(self,mse,sigma,rid,cr):
        self.ef=0.7*self.ef+0.3*mse
        self.efs=0.3*self.efs+0.7*mse
        self.es=0.99*self.es+0.01*mse
        if self.efs<self.es*0.7:
            self.bor+=0.003; lrm=0.5
        else:
            self.bor*=0.97; lrm=1.0
        self.bor=max(0,min(1,self.bor))
        rw=0.02 if sigma>0.5 and mse>0.01 else 0
        rw-=0.01 if self.bor>0.05 else 0
        self.drive=self.drive*0.99+rw
        self.drive=max(-3,min(5,self.drive))
        self.rh.append(rid)
        return lrm
    def bias(self):
        b={}
        if self.bor>0.05: b[2]=0.4; b[1]=0.2
        elif self.efs>self.ef*1.2: b[0]=0.2
        return b

class Verifier:
    """独立审查程序"""
    def __init__(self):
        self.findings=[]
    def _add(self,check,severity,detail,location=""):
        self.findings.append({'check':check,'sev':severity,'det':detail,'loc':location})
    def run_all(self,org):
        self.findings=[]
        self._dim_check(org)
        self._value_range_check(org)
        self._fold_efficiency_check(org)
        self._activity_variance_check(org)
        self._nan_inf_check(org)
        self._readout_capacity_check(org)
        return self.findings
    def severity_score(self):
        if not self.findings: return 0
        return max(f['sev'] for f in self.findings)

    def _dim_check(self,org):
        g=org.g; n=g['n']; expected_sw=N_CHANNELS*(1+DELAY)
        if len(g['sw'])!=expected_sw:
            self._add('DIM_SW',2,f'sw rows={len(g["sw"])}, expected={expected_sw}')
        for j,row in enumerate(g['sw']):
            if len(row)!=n:
                self._add('DIM_SW_COL',2,f'sw[{j}].len={len(row)}, expected n={n}'); break
        if len(g['pw'])!=n:
            self._add('DIM_PW',2,f'pw rows={len(g["pw"])}, expected n={n}')
        for i,row in enumerate(g['pw']):
            if len(row)!=N_CHANNELS:
                self._add('DIM_PW_COL',2,f'pw[{i}].len={len(row)}, expected N_CHANNELS={N_CHANNELS}'); break
        if len(g['rw'])!=n:
            self._add('DIM_RW',2,f'rw rows={len(g["rw"])}, expected n={n}')
        for i,row in enumerate(g['rw']):
            if len(row)!=n:
                self._add('DIM_RW_COL',2,f'rw[{i}].len={len(row)}, expected n={n}'); break
        if len(g['pb'])!=N_REGIONS*N_CHANNELS:
            self._add('DIM_PB',2,f'pb.len={len(g["pb"])}, expected={N_REGIONS*N_CHANNELS}')
        if len(g['dt'])!=N_REGIONS:
            self._add('DIM_DT',2,f'dt.len={len(g["dt"])}, expected={N_REGIONS}')
        # Skip pathway dimensions
        exp_skip=expected_sw
        if len(g['skip_w'])!=exp_skip:
            self._add('DIM_SKIP',2,f'skip_w rows={len(g["skip_w"])}, expected={exp_skip}')
        for j,row in enumerate(g['skip_w']):
            if len(row)!=N_CHANNELS:
                self._add('DIM_SKIP_COL',2,f'skip_w[{j}].len={len(row)}, expected N_CHANNELS={N_CHANNELS}'); break

    def _value_range_check(self,org):
        g=org.g
        if not (0.001<=g['plr']<=0.5):
            self._add('LR_RANGE',2,f'plr={g["plr"]:.4f} out of [0.001, 0.5]')
        for i,d in enumerate(g['dt']):
            if not (0<d<1):
                self._add('DT_RANGE',2,f'dt[{i}]={d:.3f}, must be in (0,1)'); break

    def _fold_efficiency_check(self,org):
        """蛋白折叠效率"""
        g=org.g; n=g['n']
        pk=Pool(seed=55); org.wm.reset()
        cr=random.randint(0,8)
        rw_save=[row[:] for row in g['rw']]
        for _ in range(15):
            s=pk.sense(cr); org.wm.fwd(s,update=True); pk.step()
        w=org.wm; errs=[]
        for step in range(5):
            e=0.0; cnt=0
            for k in range(n):
                p=sum(g['rw'][k][i]*w.prev_h[i] for i in range(n))
                if abs(p)>0.001:
                    p=math.tanh(p)
                    e+=(w.h[k]-p)**2; cnt+=1
            errs.append(e/max(1,cnt))
            w.protein_fold()
        for i in range(n):
            for k in range(n):
                g['rw'][i][k]=rw_save[i][k]
        final_err=errs[-1]
        start_err=errs[0]
        ratio=final_err/max(0.0001,start_err)
        if final_err>0.025:
            self._add('FOLD_INEFFICIENT',2,
                f'Fold weak: start={start_err:.4f} final={final_err:.4f} ratio={ratio:.3f}')
        elif final_err>0.01:
            self._add('FOLD_SLOW',1,
                f'Fold slow: start={start_err:.4f} final={final_err:.4f} ratio={ratio:.3f}')

    def _activity_variance_check(self,org):
        pk=Pool(seed=99); org.wm.reset()
        h_samples=[]
        cr=random.randint(0,8)
        for _ in range(25):
            s=pk.sense(cr); org.wm.fwd(s,update=True); pk.step()
            h_samples.append(org.wm.h[:])
        n=org.g['n']
        dead_count=0; sat_count=0; total=0
        for h in h_samples[10:]:
            for v in h:
                total+=1
                if abs(v)<0.01: dead_count+=1
                if abs(v)>0.98: sat_count+=1
        dead_ratio=dead_count/max(1,total)
        sat_ratio=sat_count/max(1,total)
        if dead_ratio>0.8:
            self._add('H_DEAD',2,f'{dead_ratio*100:.0f}% hidden dead (<0.01)')
        if sat_ratio>0.85:
            self._add('H_SATURATED',1,f'{sat_ratio*100:.0f}% hidden saturated (>0.98)')

    def _nan_inf_check(self,org):
        pk=Pool(seed=77); org.wm.reset()
        cr=random.randint(0,8)
        bad=False
        for _ in range(35):
            s=pk.sense(cr); pred=org.wm.fwd(s,update=True); pk.step()
            for v in pred:
                if math.isnan(v) or math.isinf(v):
                    self._add('NAN_PRED',2,'pred contains NaN/Inf'); bad=True; break
            for v in org.wm.h:
                if math.isnan(v) or math.isinf(v):
                    self._add('NAN_H',2,'h contains NaN/Inf'); bad=True; break
            if bad: break

    def _readout_capacity_check(self,org):
        g=org.g; n=g['n']
        if n<2: return
        active_nodes=0
        for i in range(n):
            w_norm=sum(g['pw'][i][c]**2 for c in range(N_CHANNELS))**0.5
            if w_norm>0.05: active_nodes+=1
        active_skip=0
        for j in range(len(g['skip_w'])):
            s_norm=sum(g['skip_w'][j][c]**2 for c in range(N_CHANNELS))**0.5
            if s_norm>0.02: active_skip+=1
        if active_nodes<1 and active_skip<1:
            self._add('READOUT_DEAD',2,'Both pathways have no readout capacity')

class Org:
    def __init__(self,gn=None,verify=True):
        self.g=gn or self.rg(); self.wm=WM(self.g); self.im=IM()
        self.alive=True; self.age=0
        self.lid=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]
        self._ib=1.0; self._r=None
        self._vf_findings=[]
        if verify:
            v=Verifier(); v.run_all(self); self._vf_findings=v.findings
            for fd in self._vf_findings:
                if fd['check'] in ('DIM_SW','DIM_SW_COL','DIM_PW','DIM_PW_COL',
                                   'DIM_RW','DIM_RW_COL','DIM_PB','DIM_DT',
                                   'DIM_SKIP','DIM_SKIP_COL',
                                   'NAN_PRED','NAN_H','NODES_TOO_FEW','READOUT_DEAD'):
                    self._ib=min(self._ib,0.2)
                    break

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
        for i in range(n):
            g['rw'][i][i]=random.uniform(0.3,1.0)
        return g

def gdist(a,b):
    n=min(a['n'],b['n']); d=abs(a['n']-b['n'])*0.3
    sn=min(len(a['sw']),len(b['sw']))
    for j in range(sn):
        aj=a['sw'][j]; bj=b['sw'][j]
        for i in range(min(n,len(aj),len(bj))): d+=abs(aj[i]-bj[i])
    for i in range(n):
        if i<len(a['pw']) and i<len(b['pw']):
            for c in range(N_CHANNELS): d+=abs(a['pw'][i][c]-b['pw'][i][c])
    for i in range(n):
        if i<len(a.get('skip_w',[])) and i<len(b.get('skip_w',[])):
            for c in range(N_CHANNELS): d+=abs(a['skip_w'][i][c]-b['skip_w'][i][c])*0.5
    nb=min(len(a['pb']),len(b['pb']))
    for k in range(nb): d+=abs(a['pb'][k]-b['pb'][k])*0.5
    return d/max(1,sn*n+N_CHANNELS*n+nb*0.5)

def xover(a,b):
    input_size=N_CHANNELS*(1+DELAY)
    cn=random.choice([a['n'],b['n']])
    ch={'n':cn,'sw':[[0.0]*cn for _ in range(input_size)],
        'pw':[[0.0]*N_CHANNELS for _ in range(cn)],
        'rw':[[0.0]*cn for _ in range(cn)],
        'skip_w':[[0.0]*N_CHANNELS for _ in range(input_size)],
        'plr':a['plr'],'plr_fold':a['plr_fold'],
        'pb':[0.0]*(N_REGIONS*N_CHANNELS),'dt':[0.0]*N_REGIONS}
    for j in range(input_size):
        for i in range(cn):
            src=a if random.random()<0.5 else b; si=min(i,src['n']-1)
            if j<len(src['sw']) and si<len(src['sw'][j]): ch['sw'][j][i]=src['sw'][j][si]
    for i in range(cn):
        src=a if random.random()<0.5 else b; si=min(i,src['n']-1)
        if si<len(src['pw']):
            for c in range(N_CHANNELS): ch['pw'][i][c]=src['pw'][si][c]
    for i in range(cn):
        src=a if random.random()<0.5 else b; si=min(i,src['n']-1)
        if si<len(src['rw']):
            for k in range(cn):
                sk=min(k,src['n']-1)
                if sk<len(src['rw'][si]): ch['rw'][i][k]=src['rw'][si][sk]
    # Skip pathway crossover
    for j in range(input_size):
        src=a if random.random()<0.5 else b
        if j<len(src['skip_w']):
            for c in range(N_CHANNELS):
                ch['skip_w'][j][c]=src['skip_w'][j][c]
    ip=a if random.random()<0.5 else b
    ch['pb']=ip['pb'][:]; ch['dt']=ip['dt'][:]
    if random.random()<0.5:
        ch['plr']=b['plr']; ch['plr_fold']=b['plr_fold']
    return ch

def comp_pair(sv,sf):
    i1=random.choices(range(len(sv)),weights=sf,k=1)[0]; p1=sv[i1]
    ds=[]
    for j,p2 in enumerate(sv):
        if j==i1: ds.append(0)
        else: ds.append(gdist(p1.g,p2.g))
    i2=max(range(len(sv)),key=lambda j:ds[j]*sf[j])
    return sv[i1],sv[i2]

def mut(g,rate=0.05):
    for j in range(len(g['sw'])):
        for i in range(len(g['sw'][j])):
            if random.random()<rate:
                g['sw'][j][i]+=random.gauss(0,0.08)
                g['sw'][j][i]=max(-3,min(3,g['sw'][j][i]))
    for i in range(len(g['pw'])):
        for c in range(N_CHANNELS):
            if random.random()<rate:
                g['pw'][i][c]+=random.gauss(0,0.08)
                g['pw'][i][c]=max(-3,min(3,g['pw'][i][c]))
    for i in range(len(g['rw'])):
        for k in range(len(g['rw'][i])):
            if random.random()<rate*0.7:
                g['rw'][i][k]+=random.gauss(0,0.04)
                g['rw'][i][k]=max(-1.5,min(1.5,g['rw'][i][k]))
    # Skip pathway mutation
    for j in range(len(g['skip_w'])):
        for c in range(N_CHANNELS):
            if random.random()<rate:
                g['skip_w'][j][c]+=random.gauss(0,0.1)
                g['skip_w'][j][c]=max(-2,min(2,g['skip_w'][j][c]))
    if random.random()<0.3:
        g['plr']=max(0.02,min(0.18,g['plr']+random.gauss(0,0.03)))
        g['plr_fold']=max(0.005,min(0.15,g['plr_fold']+random.gauss(0,0.02)))
    for i in range(len(g['pb'])):
        if random.random()<rate*0.5:
            g['pb'][i]+=random.gauss(0,0.1)
            g['pb'][i]=max(-5,min(5,g['pb'][i]))
    for i in range(len(g['dt'])):
        if random.random()<rate*0.3:
            g['dt'][i]=max(0.1,min(0.99,g['dt'][i]+random.gauss(0,0.03)))
    return g

class Engine:
    def __init__(self,n_org=25,n_gen=1000,seed=71):
        self.n=n_org; self.ng=n_gen
        self.ppl=[Org() for _ in range(n_org)]
        self.hof=[]; self.hof_size=20; self.best_ever=0.0

    def eval_single(self,org,seed,ticks):
        pk=Pool(seed=seed); pk.reset(); org.wm.reset(); org.im=IM()
        total_mse=0.0; rc=collections.Counter()
        cr=random.randint(0,8)
        for _ in range(50):
            s=pk.sense(cr); org.wm.fwd(s,update=True); pk.step()
        for t in range(ticks):
            s_cur=pk.sense(cr); pred=org.wm.fwd(s_cur,update=True)
            pk.step(); s_next=pk.sense(cr); pk.step()
            mse=sum((s_next[j]-pred[j])**2 for j in range(N_CHANNELS))/N_CHANNELS
            org.im.upd(mse,pk.r[cr]['sigma'],cr,cr)
            org.wm.learn(pred,s_next)
            total_mse+=mse; rc[cr]+=1; org.age+=1
            if random.random()<0.05: cr=random.randint(0,8)
        n=sum(rc.values()) if sum(rc.values())>0 else 1
        return total_mse/n

    def fit(self,org,ticks=1000):
        mse_sum=0.0
        for trial in range(2):
            mse_sum+=self.eval_single(org,42+trial*17,ticks)
        raw_f=1/(1+mse_sum/2)
        return raw_f*org._ib

    def update_hof(self,sp):
        for org,r in sp[:3]:
            self.hof.append((r['f'],copy.deepcopy(org.g)))
        self.hof.sort(key=lambda x:x[0],reverse=True)
        seen=set()
        uh=[]
        for f,g in self.hof:
            fp=(g['n'],round(g.get('pw',[[0]])[0][0],3),round(g.get('plr_fold',0),3))
            if fp not in seen:
                seen.add(fp); uh.append((f,g))
        self.hof=uh[:self.hof_size]

    def detect_wa(self,sp):
        bh=sp[len(sp)//2:]
        if len(bh)<3: return []
        wc=[]
        for p in bh:
            if not p[0].im.rh: continue
            bs=p[0].g['pb']
            bss=(sum((b-sum(bs)/len(bs))**2 for b in bs)/len(bs))**0.5
            wc.append((p[0],bss))
        wc.sort(key=lambda x:x[1],reverse=True)
        return [w[0] for w in wc[:max(1,len(bh)//6)]]

    def evolve(self):
        vf_report={'total_orgs':0,'critical':0,'warn':0,'clean':0,'findings_map':{}}
        for gen in range(self.ng):
            res=[]; gen_findings=[]
            for org in self.ppl:
                f=self.fit(org); org._r={'f':f}; res.append(f)
                if org._vf_findings:
                    gen_findings.extend(org._vf_findings)
            vf_report['total_orgs']+=len(self.ppl)
            for fd in gen_findings:
                vf_report['findings_map'][fd['check']]=vf_report['findings_map'].get(fd['check'],0)+1
                if fd['sev']>=2: vf_report['critical']+=1
                else: vf_report['warn']+=1
            if not gen_findings: vf_report['clean']+=len(self.ppl)
            f_list=res
            bf=max(f_list); af=sum(f_list)/len(f_list)
            if bf>self.best_ever: self.best_ever=bf
            idx_sorted=sorted(range(len(f_list)),key=lambda i:f_list[i],reverse=True)
            sp=[(self.ppl[i],{'f':f_list[i]}) for i in idx_sorted]

            if gen%50==0 or gen==self.ng-1:
                vf_top=''
                if gen_findings:
                    vc=collections.Counter(f"{fd['check']}({fd['det'][:40]})" for fd in gen_findings)
                    top3=vc.most_common(3)
                    vf_top=' | VF:'+';'.join(f'{c[0]}' for c in top3)
                print(f"Gen {gen:4d} fit={bf:.4f}/{af:.4f} best={self.best_ever:.4f} vf=[c:{vf_report['critical']} w:{vf_report['warn']}]{vf_top}")

            self.update_hof(sp)
            if gen==self.ng-1:

                # Final verifier summary
                print(f"\n=== VERIFIER REPORT ===")
                print(f"Total orgs: {vf_report['total_orgs']}")
                print(f"Critical: {vf_report['critical']}  Warnings: {vf_report['warn']}  Clean: {vf_report['clean']}")
                if vf_report['findings_map']:
                    for k,v in sorted(vf_report['findings_map'].items(),key=lambda x:-x[1]):
                        print(f"  {k}: {v}")

            if gen==self.ng-1: break

            ELITE=3
            elites=[copy.deepcopy(self.ppl[idx_sorted[i]].g) for i in range(ELITE)]
            hof_inject=[]
            if gen%50==0 and gen>0:
                hof_inject=[copy.deepcopy(g) for f,g in self.hof[:3]]

            n_fixed=len(elites)+len(hof_inject)
            surv=[self.ppl[idx_sorted[i]] for i in range(min(12,self.n))]
            sf=[f_list[idx_sorted[i]] for i in range(min(12,self.n))]
            fs=[f-min(sf)+0.01 for f in sf]; tot=sum(fs); sf=[f/tot for f in fs]

            n_eh=max(1,int((self.n-n_fixed)*0.45))
            n_wh=max(1,int((self.n-n_fixed)*0.25))
            n_rd=max(1,int((self.n-n_fixed)*0.20))
            n_mt=self.n-n_fixed-n_eh-n_wh-n_rd

            np2=[]
            for eg in elites: np2.append(Org(gn=copy.deepcopy(eg),verify=False))
            for eg in hof_inject: np2.append(Org(gn=copy.deepcopy(eg),verify=False))

            ehc=whc=rdc=mtc=0; att=0
            wa=self.detect_wa(sp)
            while len(np2)<self.n and att<self.n*40:
                att+=1
                if not surv or sum(sf)<=0:
                    np2.append(Org()); continue
                if ehc<n_eh:
                    t=max(4,len(surv)//3)
                    p1,p2=comp_pair(surv[:t],sf[:t])
                    cg=xover(p1.g,p2.g); mut(cg,0.05)
                    np2.append(Org(gn=cg)); ehc+=1
                elif whc<n_wh and wa:
                    p1=random.choice(wa); p2=random.choices(surv[:min(8,len(surv))],weights=sf[:min(8,len(sf))],k=1)[0]
                    cg=xover(p1.g,p2.g); mut(cg,0.08)
                    np2.append(Org(gn=cg)); whc+=1
                elif n_rd>0 and rdc<n_rd: np2.append(Org()); rdc+=1
                elif mtc<n_mt:
                    p=random.choices(surv,weights=sf,k=1)[0]
                    np2.append(Org(gn=mut(copy.deepcopy(p.g),0.05))); mtc+=1
            self.ppl=np2
        return self.ppl

if __name__=='__main__':
    print(f"yinyang_evo_v17o — Dual Pathway: Protein-Fold + Skip Connection")
    print(f"DELAY={DELAY}, input_dim={N_CHANNELS*(1+DELAY)}, skip_dim={N_CHANNELS*(1+DELAY)}")
    eng=Engine(n_org=25,n_gen=1000,seed=71)
    pop=eng.evolve()
    best_fit=0; best_org=None
    for org in pop:
        if org._r and org._r['f']>best_fit:
            best_fit=org._r['f']; best_org=org
    print(f"=== FINAL ===")
    print(f"Best fit: {best_fit:.4f}")
    print(f"Best ever: {eng.best_ever:.4f}")
    if best_org:
        print(f"Nodes: {best_org.g['n']} LR: {best_org.g['plr']:.4f} Fold-LR: {best_org.g['plr_fold']:.4f}")
