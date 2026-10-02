#!/usr/bin/env python3
"""
v17j — Online Recurrent Learning
===============================
核心改变：在线学习时，不仅调readout权重(predict_w, bias)，也调recurrent权重
recurrent_w[i][k] += lr_r * h[i] * h[k] * error_signal

这样：
- readout快速学习线性映射（已有）
- recurrent缓慢学习时间结构（新）
- hidden state逐渐获得"预测未来"的能力
"""
import random, math, hashlib, copy, collections

N_CHANNELS = 8; MAX_NODES = 20; N_REGIONS = 9

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
        self.g=g; n=g['n']; self.h=[0.0]*n; self.prev=[0.0]*N_CHANNELS
    def ri(self,p):
        i=round(p*8); return max(0,min(8,i))
    def fwd(self,sens,update=True):
        g=self.g; n=g['n']; p=sens[7]; ri=self.ri(p)
        dec=g['dt'][ri]; bb=ri*N_CHANNELS
        vel=[sens[j]-self.prev[j] for j in range(N_CHANNELS)]
        x=sens+vel; nn=min(len(g['sw']),len(x))
        nh=[0.0]*n
        for i in range(n):
            a=0.0
            for j in range(nn): a+=g['sw'][j][i]*x[j]
            for k in range(n): a+=g['rw'][i][k]*self.h[k]
            nh[i]=math.tanh(a)
        if update:
            for i in range(n): self.h[i]=(1-dec)*nh[i]+dec*self.h[i]
        pred=[0.0]*N_CHANNELS
        for i in range(n):
            for c in range(N_CHANNELS):
                pred[c]+=self.h[i]*(g['pw'][i][c]+g['pb'][bb+c])
        self.prev=sens[:]
        return pred

    def learn(self,pred,actual):
        g=self.g; n=g['n']; lr=g['plr']
        ri=self.ri(actual[7]); bb=ri*N_CHANNELS
        errs=[actual[j]-pred[j] for j in range(N_CHANNELS)]
        mse=sum(e*e for e in errs)/N_CHANNELS
        total_err=sum(abs(e) for e in errs)/N_CHANNELS

        # Readout learning (Hebbian)
        for i in range(n):
            for c in range(N_CHANNELS):
                g['pw'][i][c]+=lr*self.h[i]*errs[c]
                g['pw'][i][c]=max(-3,min(3,g['pw'][i][c]))
        for c in range(N_CHANNELS):
            g['pb'][bb+c]+=lr*1.5*errs[c]
            g['pb'][bb+c]=max(-5,min(5,g['pb'][bb+c]))

        # Recurrent learning (slow, error-gated Hebbian)
        # Goal: strengthen connections that correlate with prediction improvement
        # Simple version: rw[i][k] += lr_r * h[i] * h[k] * mse_signal
        # mse_signal is high when prediction is wrong (system should adjust dynamics)
        lr_r = lr * 0.15  # much slower than readout
        for i in range(n):
            for k in range(n):
                if i != k:  # don't touch self-recall (keep identity memory)
                    delta = lr_r * self.h[i] * self.h[k] * total_err
                    g['rw'][i][k] += delta
                    g['rw'][i][k] = max(-1.0, min(1.0, g['rw'][i][k]))

        return mse

    def reset(self):
        self.h=[0.0]*len(self.h); self.prev=[0.0]*N_CHANNELS

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

class Org:
    def __init__(self,gn=None):
        self.g=gn or self.rg(); self.wm=WM(self.g); self.im=IM()
        self.alive=True; self.age=0
        self.lid=hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]
        self._ib=1.0; self._r=None
    def rg(self):
        n=random.randint(7,MAX_NODES)
        return {'n':n,
            'sw':[[random.uniform(-0.3,0.3) for _ in range(n)] for _ in range(N_CHANNELS*2)],
            'pw':[[random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)] for _ in range(n)],
            'rw':[[random.uniform(-0.1,0.1) for _ in range(n)] for _ in range(n)],
            'plr':random.uniform(0.03,0.12),
            'pb':[random.uniform(-0.3,0.3) for _ in range(N_REGIONS*N_CHANNELS)],
            'dt':[random.uniform(0.4,0.95) for _ in range(N_REGIONS)]}

def gdist(a,b):
    n=min(a['n'],b['n']); d=abs(a['n']-b['n'])*0.3
    for j in range(N_CHANNELS*2):
        aj=a['sw'][j] if j<len(a['sw']) else []; bj=b['sw'][j] if j<len(b['sw']) else []
        for i in range(min(n,len(aj),len(bj))): d+=abs(aj[i]-bj[i])
    for i in range(n):
        if i<len(a['pw']) and i<len(b['pw']):
            for c in range(N_CHANNELS): d+=abs(a['pw'][i][c]-b['pw'][i][c])
    nb=min(len(a['pb']),len(b['pb']))
    for k in range(nb): d+=abs(a['pb'][k]-b['pb'][k])*0.5
    return d/max(1,N_CHANNELS*n*2+N_CHANNELS*n+nb*0.5)

def xover(a,b):
    cn=random.choice([a['n'],b['n']])
    ch={'n':cn,'sw':[[0.0]*cn for _ in range(N_CHANNELS*2)],
        'pw':[[0.0]*N_CHANNELS for _ in range(cn)],
        'rw':[[0.0]*cn for _ in range(cn)],
        'plr':a['plr'],'pb':[0.0]*(N_REGIONS*N_CHANNELS),'dt':[0.0]*N_REGIONS}
    for j in range(N_CHANNELS*2):
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
    ip=a if random.random()<0.5 else b
    ch['pb']=ip['pb'][:]; ch['dt']=ip['dt'][:]
    if random.random()<0.5: ch['plr']=b['plr']
    return ch

def comp_pair(sv,sf):
    i1=random.choices(range(len(sv)),weights=sf,k=1)[0]; p1=sv[i1]
    ds=[]
    for j,p2 in enumerate(sv):
        if j==i1: ds.append(0)
        else: ds.append(gdist(p1.g,p2.g))
    i2=max(range(len(sv)),key=lambda j:ds[j]*sf[j])
    return sv[i1],sv[i2]

def mut(g,rate=0.06):
    for j in range(len(g['sw'])):
        for i in range(len(g['sw'][j])):
            if random.random()<rate:
                g['sw'][j][i]+=random.gauss(0,0.08); g['sw'][j][i]=max(-3,min(3,g['sw'][j][i]))
    for i in range(len(g['pw'])):
        for c in range(N_CHANNELS):
            if random.random()<rate:
                g['pw'][i][c]+=random.gauss(0,0.08); g['pw'][i][c]=max(-3,min(3,g['pw'][i][c]))
    for i in range(len(g['rw'])):
        for k in range(len(g['rw'][i])):
            if random.random()<rate:
                g['rw'][i][k]+=random.gauss(0,0.04); g['rw'][i][k]=max(-1.5,min(1.5,g['rw'][i][k]))
    if random.random()<0.3: g['plr']=max(0.02,min(0.15,g['plr']+random.gauss(0,0.02)))
    for i in range(len(g['pb'])):
        if random.random()<rate*0.5:
            g['pb'][i]+=random.gauss(0,0.1); g['pb'][i]=max(-5,min(5,g['pb'][i]))
    for i in range(len(g['dt'])):
        if random.random()<rate*0.3:
            g['dt'][i]=max(0.1,min(0.99,g['dt'][i]+random.gauss(0,0.03)))
    return g

class Engine:
    def __init__(self,n_org=20,n_gen=500,seed=41):
        self.n=n_org; self.ng=n_gen; self.pk=Pool(seed=seed)
        self.ppl=[Org() for _ in range(n_org)]

    def fit(self,org,ticks=800):
        """True WM with recurrent online learning"""
        self.pk.reset(); org.wm.reset(); org.im=IM()
        total_mse=0.0; info_ticks=0; rc=collections.Counter()

        # Phase 1: warmup (random region)
        cr=random.randint(0,8)
        for _ in range(60):
            s=self.pk.sense(cr); org.wm.fwd(s,update=True); self.pk.step()

        # Phase 2: systematic region coverage + online learning
        ticks_per_region=ticks//N_REGIONS
        for rid in range(N_REGIONS):
            for t in range(ticks_per_region):
                s_curr=self.pk.sense(rid)
                pred=org.wm.fwd(s_curr,update=True)
                self.pk.step()
                s_next=self.pk.sense(rid)
                self.pk.step()

                mse=sum((s_next[j]-pred[j])**2 for j in range(N_CHANNELS))/N_CHANNELS
                org.im.upd(mse,self.pk.r[rid]['sigma'],rid,rid)
                org.wm.learn(pred,s_next)
                total_mse+=mse; rc[rid]+=1; org.age+=1
                if 0.001<mse<0.5: info_ticks+=1

        n_evals=sum(rc.values()) if sum(rc.values())>0 else 1
        return {'f':1/(1+total_mse/n_evals),'m':total_mse/n_evals,
                'info':info_ticks,'div':len([c for c in rc.values() if c>0])}

    def detect_wa(self,sp):
        bh=sp[len(sp)//2:]
        if len(bh)<3: return []
        at=[]
        for p in sp:
            if p[0].im.rh:
                f=[0]*N_REGIONS
                for r in p[0].im.rh: f[r]+=1
                t=sum(f)
                if t>0: f=[x/t for x in f]
                at.append(f)
        if not at: return []
        pa=[0]*N_REGIONS
        for t in at:
            for i in range(N_REGIONS): pa[i]+=t[i]
        pa=[x/len(at) for x in pa]
        wc=[]
        for p in bh:
            if not p[0].im.rh: continue
            f=[0]*N_REGIONS
            for r in p[0].im.rh: f[r]+=1
            t=sum(f)
            if t>0: f=[x/t for x in f]
            d=sum((f[i]-pa[i])**2 for i in range(N_REGIONS))**0.5
            bs=p[0].g['pb']
            bss=(sum((b-sum(bs)/len(bs))**2 for b in bs)/len(bs))**0.5
            wc.append((p[0],d+bss*0.1))
        wc.sort(key=lambda x:x[1],reverse=True)
        return [w[0] for w in wc[:max(2,len(bh)//3)]]

    def evolve(self):
        for gen in range(self.ng):
            res=[]
            for org in self.ppl:
                r=self.fit(org); org._r=r; res.append(r)
            sp=sorted(zip(self.ppl,res),key=lambda x:x[1]['f'],reverse=True)
            af=sum(r['f'] for r in res)/len(res); am=sum(r['m'] for r in res)/len(res)
            bf=sp[0][1]['f']; bm=sp[0][1]['m']
            ab=sum(o.im.bor for o in self.ppl)/len(self.ppl)
            div=sum(r['div'] for r in res)/len(res)
            wa=self.detect_wa(sp)
            if gen%10==0 or gen==self.ng-1:
                print(f"Gen {gen:4d} fit={bf:.4f}/{af:.4f} mse={bm:.4f}/{am:.4f} bor={ab:.4f} div={div:.1f} WA={len(wa)}")
            if gen==self.ng-1: break
            elites=[p[0].g for p in sp[:2]]
            sn=max(8,self.n//2)
            surv=[p[0] for p in sp[:sn]]; sf=[p[1]['f'] for p in sp[:sn]]
            fs=[f-min(sf)+0.01 for f in sf]; tot=sum(fs); sf=[f/tot for f in fs]
            n_eh=max(1,int(self.n*0.4)); n_wh=max(1,int(self.n*0.25))
            n_rd=max(1,int(self.n*0.15)); n_mt=self.n-2-n_eh-n_wh-n_rd
            np2=[]
            for eg in elites:
                o=Org(gn=copy.deepcopy(eg)); o._ib=1.0; np2.append(o)
            ehc=whc=rdc=mtc=0; att=0
            while len(np2)<self.n and att<self.n*50:
                att+=1
                if not surv or sum(sf)<=0: np2.append(Org()); continue
                if ehc<n_eh:
                    t=max(4,len(surv)//3)
                    p1,p2=comp_pair(surv[:t],sf[:t])
                    cg=xover(p1.g,p2.g); mut(cg,0.06)
                    o=Org(gn=cg); o._ib=1.0; ehc+=1; np2.append(o)
                elif whc<n_wh and wa:
                    wa_o=random.choice(wa)
                    el=random.choices(surv[:5],weights=sf[:5],k=1)[0]
                    cg=xover(wa_o.g,el.g); mut(cg,0.10)
                    o=Org(gn=cg); o._ib=1.0; whc+=1; np2.append(o)
                elif n_rd>0 and rdc<n_rd: np2.append(Org()); rdc+=1
                elif mtc<n_mt:
                    p=random.choices(surv,weights=sf,k=1)[0]; cg=mut(copy.deepcopy(p.g),0.06)
                    o=Org(gn=cg); o._ib=1.0; mtc+=1; np2.append(o)
            self.ppl=np2
        for org in self.ppl:
            r=self.fit(org,1000); org._r=r
        return self.ppl

if __name__=='__main__':
    print("yinyang_evo_v17j — Online Recurrent Learning for True WM")
    eng=Engine(n_org=20,n_gen=500,seed=41)
    pop=eng.evolve()
    best=max(pop,key=lambda o:o._r['f'])
    print(f"\n=== FINAL ===")
    print(f"Best fit: {best._r['f']:.4f}")
    print(f"Best mse: {best._r['m']:.4f}")
    print(f"Nodes: {best.g['n']}")
    print(f"LR: {best.g['plr']:.4f}")
    print(f"Decay: {[f'{x:.3f}' for x in best.g['dt']]}")
    print("\nRegional 1-step precision:")
    for rid in range(N_REGIONS):
        pk=Pool(seed=42); pk.reset(); best.wm.reset()
        for _ in range(60): pk.sense(rid); pk.step()
        ms=[]
        for _ in range(50):
            s=pk.sense(rid); pred=best.wm.fwd(s,update=True)
            pk.step(); sn=pk.sense(rid); pk.step()
            mse=sum((sn[j]-pred[j])**2 for j in range(N_CHANNELS))/N_CHANNELS
            ms.append(mse)
        am=sum(ms)/len(ms)
        print(f"  R{rid} σ={pk.r[rid]['sigma']:.2f}: {am:.4f}")
