"""Trace-driven closed-loop emulator and baseline policies (extracted from notebooks/uav_detector_in_the_loop_v3.ipynb).
Runs offline on the detector-score artefacts saved by the Kaggle notebook (results/kaggle_v3/scores_*.npz, meta_*.csv)."""
import os
import numpy as np, pandas as pd, time

CH={'2.4':[2412,2417,2422,2427,2432,2437,2442,2447,2452,2457,2462],
    '5':[5180,5200,5220,5240,5260,5280,5300,5320,5745,5765,5785,5805,5825]}

def a2g_pl(d,f_mhz,eta=1.0):
    return 20*np.log10(np.maximum(d,1.0))+20*np.log10(f_mhz)-27.55+eta
def overlap(band,K):
    if band=='2.4':
        g=np.zeros((K,K))
        for i in range(K):
            for j in range(K): g[i,j]=max(0.0,(20-5*abs(i-j))/20)
        return g
    return np.eye(K)
class Env:
    def __init__(s,band,meta,SCORE,TIDX,domain,jtype,seed,T):
        s.b=band; s.chs=CH[band]; s.K=len(s.chs); s.T=T; s.jtype=jtype
        s.rj=np.random.default_rng(seed); s.ro=np.random.default_rng(seed+1); s.rf=np.random.default_rng(seed+2); s.r=np.random.default_rng(seed+3)
        s.G=overlap(band,s.K); s.fc=float(np.mean(s.chs))
        test=(meta.part=='test').values
        ben_dom='loc_shift' if domain in ('loc_shift','loc_pow') else 'train_dom'
        s.ben=np.where(test&(meta.lab==0).values&(meta.dom==ben_dom).values)[0]
        mal=test&(meta.lab==1).values
        pw_tr=sorted(meta.loc[(meta.lab==1)&(meta.dom=='train_dom'),'pw'].unique())
        pw_all=sorted(meta.loc[meta.lab==1,'pw'].unique())
        s.pclasses=pw_all if domain in ('pow_shift','loc_pow') else pw_tr
        dom_ok=(meta.dom.isin(['train_dom','pow_shift'])).values if domain in ('pow_shift','loc_pow') else (meta.dom=='train_dom').values
        s.mal={p:np.where(mal&dom_ok&(meta.pw==p).values)[0] for p in s.pclasses}
        s.score=SCORE; s.tidx=TIDX
        s.jpos=np.array([1200.0,0,0]); s.Ptx=23; s.Pj=30; s.N=-95; s.gth=5.0; s.K_ric=10**(7.6/10)
        tt=np.arange(600); pr=np.array([s.Pj-a2g_pl(np.linalg.norm(s.uav(t)-s.jpos),s.fc) for t in tt])
        s.pbins=np.quantile(pr,np.linspace(0,1,len(s.pclasses)+1)[1:-1])   # weakest received power -> lowest power class
    def uav(s,t):
        a=2*np.pi*t/600; return np.array([900*np.cos(a),900*np.sin(a),120.0])
    def prx(s,t): return s.Pj-a2g_pl(np.linalg.norm(s.uav(t)-s.jpos),s.fc)
    def power_class(s):
        pr=s.prx(s.t); return s.pclasses[int(np.searchsorted(s.pbins,pr))], pr
    def reset(s):
        s.t=0; s.c=int(s.rj.integers(s.K)); s.on=True; return s.observe()
    def jam_step(s,a_prev):
        s.on = s.rj.random()<(0.95 if s.on else 0.2)
        if s.jtype=='sweep': s.c=(s.c+1)%s.K if s.t%3==0 else s.c
        elif s.jtype=='markov': s.c=s.c if s.rj.random()<0.9 else int(s.rj.integers(s.K))
        elif s.jtype=='reactive': s.c=a_prev if s.rj.random()<0.7 else s.c
    def observe(s):
        if s.on:
            p,_=s.power_class(); i=int(s.ro.choice(s.mal[p])); obs=np.roll(s.score[i],s.c-s.tidx[i])
        else:
            i=int(s.ro.choice(s.ben)); obs=s.score[i]
        s.obs=obs; return obs
    def step(s,a):
        d=np.linalg.norm(s.uav(s.t)); S=s.Ptx-a2g_pl(d,s.fc)
        h=s.rf.normal(np.sqrt(s.K_ric/(s.K_ric+1)),np.sqrt(1/(2*(s.K_ric+1))),2); fade=10*np.log10(h[0]**2+h[1]**2+1e-12)
        J=10**(s.prx(s.t)/10)*s.G[a,s.c] if s.on else 0.0
        sinr=S+fade-10*np.log10(10**(s.N/10)+J)
        ack=int(sinr>=s.gth); jammed=int(s.on and s.G[a,s.c]>0)
        s.t+=1; s.jam_step(a); obs=s.observe()
        return obs,ack,jammed


class Oracle:
    name='Oracle-state'
    def reset(s,env): s.a=0
    def act(s,env,obs):
        if env.on and env.G[s.a,env.c]>0:
            free=[k for k in range(env.K) if env.G[k,env.c]==0]; s.a=int(env.r.choice(free))
        return s.a
    def feedback(s,a,ack,obs): pass
class RandomFH:
    name='Random-FH'
    def reset(s,env): pass
    def act(s,env,obs): return int(env.r.integers(env.K))
    def feedback(s,a,ack,obs): pass
class AckBandit:
    name='ACK-only SW-UCB'
    def reset(s,env,W=50):
        s.K=env.K; s.W=W; s.buf=np.zeros((env.K,W)); s.n=np.zeros(env.K,int); s.ptr=np.zeros(env.K,int); s.t=0; s.a=0
    def act(s,env,obs):
        s.t+=1; n=np.minimum(s.n,s.W)
        mean=np.where(n>0,s.buf.sum(1)/np.maximum(n,1),0)
        v=np.where(n>0,mean+np.sqrt(0.5*np.log(min(s.t,s.W*s.K))/np.maximum(n,1)),1e9)
        s.a=int(np.argmax(v)); return s.a
    def feedback(s,a,ack,obs):
        s.buf[a,s.ptr[a]]=ack; s.ptr[a]=(s.ptr[a]+1)%s.W; s.n[a]+=1
def affected(G,jset):
    """channels whose band overlaps any channel in jset (G = spectral overlap matrix)"""
    if len(jset)==0: return np.zeros(G.shape[0],bool)
    return (G[:,list(jset)]>0).any(1)
def pick_clean(env,aff,pref):
    cand=np.where(~aff)[0]
    if len(cand)==0: return int(np.argmax(pref))
    best=cand[pref[cand]>=pref[cand].max()-1e-12]
    return int(env.r.choice(best))
class DetHard:
    name='Detector hard (0.5)'
    def reset(s,env): s.a=0
    def act(s,env,obs):
        aff=affected(env.G,np.where(obs>=0.5)[0])
        if aff[s.a]: s.a=pick_clean(env,aff,-env.G.dot(obs))
        return s.a
    def feedback(s,a,ack,obs): pass
class DetSoft:
    name='Detector soft (overlap-weighted risk)'
    def reset(s,env): s.a=0
    def act(s,env,obs):
        risk=(env.G*obs[None,:]).max(1)
        k=int(env.r.choice(np.where(risk<=risk.min()+1e-12)[0]))
        if risk[s.a]-risk[k]>0.2: s.a=k
        return s.a
    def feedback(s,a,ack,obs): pass
class FCCS:
    """Feedback-calibrated conformal sensing (proposed).
    Per-channel reference sets hold detector scores seen on slots where the UAV used that channel and got an ACK
    (free labels from the link). Conformal p-value p_k=(1+#{r>=s_k})/(n_k+1); channel k is declared jammed if p_k<alpha_t.
    alpha_t follows adaptive conformal inference: it rises after a NACK on a channel that was declared clean (miss)
    and slowly decays otherwise, tracking a target miss rate. Warm start: training-domain calibration benign scores."""
    name='FCCS (conformal)'
    def __init__(s,warm,target=0.05,gamma=0.01,cap=400): s.warm=np.asarray(warm); s.target=target; s.gamma=gamma; s.cap=cap
    def reset(s,env):
        s.K=env.K; s.Q=[list(s.warm[-s.cap:]) for _ in range(env.K)]; s.R=[np.sort(q) for q in s.Q]; s.alpha=0.05; s.a=0; s.aff=None
    def pvals(s,obs):
        return np.array([(1+len(s.R[k])-np.searchsorted(s.R[k],obs[k],side='left'))/(len(s.R[k])+1) for k in range(s.K)])
    def act(s,env,obs):
        p=s.pvals(obs); s.aff=affected(env.G,np.where(p<s.alpha)[0])
        if s.aff[s.a]: s.a=pick_clean(env,s.aff,(env.G*p[None,:]+(1-env.G)).min(1))
        return s.a
    def feedback(s,a,ack,obs):
        declared_clean=not s.aff[a]
        if ack:
            s.Q[a].append(float(obs[a])); s.Q[a]=s.Q[a][-s.cap:]; s.R[a]=np.sort(s.Q[a])   # FIFO window, sorted copy
        err=float(declared_clean and not ack)
        s.alpha=float(np.clip(s.alpha+s.gamma*(err-s.target),0.001,0.5))

class FBR:
    """Feedback-based recalibration (proposed v3). The detector ranks channels well under shift but its 0.5 threshold
    does not transfer (AUROC stays high while TPR collapses for weak jammers). FBR keeps the detector frozen and learns an
    online Platt map q = sigmoid(w*logit(s)+b) from the link's own ACK/NACK on the channel in use (label y = NACK).
    Starts at the identity map (w=1, b=0), so with no shift it behaves like the hard-threshold detector."""
    name='FBR (online Platt)'
    def __init__(s,lr=0.05): s.lr=lr
    def reset(s,env): s.w=1.0; s.b=0.0; s.a=0
    def q(s,obs):
        z=np.log(np.clip(obs,1e-6,1-1e-6)/(1-np.clip(obs,1e-6,1-1e-6))); return 1/(1+np.exp(-(s.w*z+s.b))),z
    def act(s,env,obs):
        q,_=s.q(obs); aff=affected(env.G,np.where(q>=0.5)[0])
        if aff[s.a]: s.a=pick_clean(env,aff,-env.G.dot(q))
        return s.a
    def feedback(s,a,ack,obs):
        q,z=s.q(obs); y=1.0-ack; g=y-q[a]
        s.w=float(np.clip(s.w+s.lr*g*z[a]*0.1,0.2,5.0)); s.b=float(np.clip(s.b+s.lr*g,-5,15))


def run(env,pol):
    obs=env.reset(); pol.reset(env); acks=0; jam=0; hops=0; a_prev=None; miss=0; falsehop=0
    for t in range(env.T):
        a=pol.act(env,obs)
        if a_prev is not None and a!=a_prev:
            hops+=1
            if not (env.on and env.G[a_prev,env.c]>0): falsehop+=1
        obs2,ack,jammed=env.step(a); pol.feedback(a,ack,obs)
        acks+=ack; jam+=jammed; a_prev=a; obs=obs2
    T=env.T
    return dict(PSR=acks/T,jammed_frac=jam/T,hops_per100=100*hops/T,false_hops_per100=100*falsehop/T)

ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','kaggle_v3')
def load(band,root=ROOT):
    z=np.load(f'{root}/scores_{band}GHz.npz'); meta=pd.read_csv(f'{root}/meta_{band}GHz.csv')
    return meta,z['SC'],z['TI'],z['warm']
