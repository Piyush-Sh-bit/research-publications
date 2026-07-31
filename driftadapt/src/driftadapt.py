"""DriftAdapt corrected experiments on cached frozen-backbone features.

Front-end: feature standardization z=(f-mu)/sd. Source uses fixed source stats;
all adapting methods use batch statistics (test-time normalization, drift-robust),
then train a low-rank logit adapter on z. Fair shared surface; CPU-fast.

Methods: source, ttn, tent, eata, driftadapt (+ ablations).
Streams: sequential, recurrent, gradual.
"""
import os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

CACHE = os.path.dirname(os.path.abspath(__file__))
C = torch.load(os.path.join(CACHE, "cache.pt"))
H = torch.load(os.path.join(CACHE, "head_std.pt"))
W0=H["weight"].clone(); B0=H["bias"].clone(); MU0=H["mu0"].clone(); SD0=H["sd0"].clone()
DIM=W0.shape[1]; NCLS=W0.shape[0]
YTE=C["test_labels"]; STATE_FEATS=C["state_feats"]; NPOOL=YTE.shape[0]
EPS=1e-5

def znorm(f, mu, sd): return (f-mu)/(sd+EPS)
def head(z): return z@W0.t()+B0
def entropy(lg):
    p=F.softmax(lg,-1); lp=F.log_softmax(lg,-1); return -(p*lp).sum(-1)

class Adapter:
    def __init__(self, rank, seed=0):
        g=torch.Generator().manual_seed(seed)
        self.A=(torch.randn(rank,DIM,generator=g)*0.01).requires_grad_(True)
        self.B=torch.zeros(NCLS,rank,requires_grad=True)
        self.centroid=None; self.anchor=self._flat().detach().clone()
        self.mu=None; self.sd=None  # per-state normalization memory
    def params(self): return [self.A,self.B]
    def delta(self,z): return (z@self.A.t())@self.B.t()
    def _flat(self): return torch.cat([self.A.detach().flatten(),self.B.detach().flatten()])
    def set_anchor(self): self.anchor=self._flat().clone()

def make_stream(mode, length, batch, seed, num_states=3, cycles=3):
    rng=np.random.RandomState(seed)
    cs=["shot_noise","defocus_blur","fog","jpeg","brightness"]   # official CIFAR-10-C states
    steps=[]
    if mode=="sequential":
        order=cs[:num_states]; blk=length//len(order)
        for k,s in enumerate(order):
            for _ in range(blk): steps.append(("S",s,k))
    elif mode=="recurrent":
        cyc=cs[:num_states]; blk=max(1,length//(len(cyc)*cycles))
        for _ in range(cycles):
            for k,s in enumerate(cyc):
                for _ in range(blk): steps.append(("S",s,k))
    elif mode=="gradual":
        half=length//2
        ws=np.concatenate([np.linspace(0,1,half),np.linspace(1,0,length-half)])
        for w in ws: steps.append(("G",float(w),int(round(w*4))))
    else: raise ValueError(mode)
    out=[]
    for it in steps:
        idx=rng.randint(0,NPOOL,size=batch)
        if it[0]=="S": f=STATE_FEATS[it[1]][idx]; y=YTE[idx]; sid=it[2]
        else:
            w=it[1]; f=(1-w)*STATE_FEATS["clean"][idx]+w*STATE_FEATS["shot_noise"][idx]; y=YTE[idx]; sid=it[2]
        out.append((f,y,sid))
    return out

class Metrics:
    def __init__(self): self.correct=0;self.total=0;self.per={};self.first={};self.last={};self.traj=[]
    def add(self,pred,y,sid):
        c=(pred==y).sum().item();n=y.numel();self.correct+=c;self.total+=n;a=c/max(n,1)
        self.traj.append(a)
        self.per.setdefault(sid,[]).append(a)
        if sid not in self.first:self.first[sid]=a
        self.last[sid]=a
    def online(self): return self.correct/max(self.total,1)
    def bwt(self):
        d=[self.last[s]-self.first[s] for s,v in self.per.items() if len(v)>1]
        return float(np.mean(d)) if d else 0.0
    def recovery(self):
        r=[float(np.mean(v[len(v)//2:])) for s,v in self.per.items() if len(v)>1]
        return float(np.mean(r)) if r else 0.0

class Router:
    """Windowed change-point detection scoring (tolerance +/- tol steps)."""
    def __init__(self,tol=2):self.det=[];self.ch=[];self.t=-1;self.prev=None;self.tol=tol
    def add(self,det,ts):
        self.t+=1
        if self.prev is not None and ts!=self.prev:self.ch.append(self.t)
        if det:self.det.append(self.t)
        self.prev=ts
    def prec(self):
        if not self.det:return 0.0
        tp=sum(1 for d in self.det if any(abs(d-c)<=self.tol for c in self.ch))
        return tp/len(self.det)
    def rec(self):
        if not self.ch:return 0.0
        m=sum(1 for c in self.ch if any(abs(d-c)<=self.tol for d in self.det))
        return m/len(self.ch)

def batch_stats(f): return f.mean(0), f.std(0)

def run(method, mode, seed=0, length=600, batch=32, num_states=3,
        rank=8, lr=0.02, percentile=0.5, lam_kl=0.5, lam_efdt=2.0, lam_perp=0.01,
        ema=0.9, gclip=1.0, Dball=8.0, reuse_thr=7.0, cusum_h=3.0, cusum_slack=6.0,
        cycles=3, abl=()):
    torch.manual_seed(seed); np.random.seed(seed)
    stream=make_stream(mode,length,batch,seed,num_states,cycles)
    M=Metrics();R=Router();t0=time.time();reg=[]

    if method=="source":
        for f,y,sid in stream:
            with torch.no_grad(): pred=head(znorm(f,MU0,SD0)).argmax(1)
            M.add(pred,y,sid);R.add(False,sid)
        return _pack(M,R,t0,reg)
    if method=="ttn":
        for f,y,sid in stream:
            mu,sd=batch_stats(f)
            with torch.no_grad(): pred=head(znorm(f,mu,sd)).argmax(1)
            M.add(pred,y,sid);R.add(False,sid)
        return _pack(M,R,t0,reg)

    if method in ("tent","eata"):
        ad=Adapter(rank,seed); opt=torch.optim.SGD(ad.params(),lr=lr)
        iA=ad.A.detach().clone(); iB=ad.B.detach().clone()
        mu_g=sd_g=None; mom=0.9
        for f,y,sid in stream:
            bmu,bsd=batch_stats(f)              # global EMA stats (contaminated by drift)
            if mu_g is None: mu_g,sd_g=bmu.clone(),bsd.clone()
            else: mu_g=mom*mu_g+(1-mom)*bmu; sd_g=mom*sd_g+(1-mom)*bsd
            z=znorm(f,mu_g,sd_g)
            lg=head(z)+ad.delta(z)
            M.add(lg.argmax(1).detach(),y,sid);R.add(False,sid)
            if method=="tent":
                loss=entropy(lg).mean()
            else:
                e=entropy(lg);thr=torch.quantile(e.detach(),percentile);m=e<=thr
                loss=(e[m].mean() if m.any() else e.mean()*0.0)+0.05*((ad.A-iA).pow(2).sum()+(ad.B-iB).pow(2).sum())
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(ad.params(),gclip);opt.step()
            reg.append(float(loss.item()))
        return _pack(M,R,t0,reg)

    if method=="driftadapt":
        use_router="no_router" not in abl; use_efdt="no_efdt" not in abl
        use_kl="no_kl" not in abl; use_filter="no_filter" not in abl
        use_ortho="no_ortho" not in abl
        bank=[Adapter(rank,seed)];active=0;frozen=[]
        opt=torch.optim.SGD(bank[active].params(),lr=lr)
        efdt=torch.zeros(rank*DIM+NCLS*rank)
        run_mean=None;g=0.0;fa=0.1
        for f,y,sid in stream:
            bmean=f.mean(0); detected=False
            if run_mean is None: run_mean=bmean.clone()
            else:
                s=float((bmean-run_mean).norm()); g=max(0.0,g+s-cusum_slack)
                if g>cusum_h: detected=True;g=0.0
                run_mean=(1-fa)*run_mean+fa*bmean
            if use_router and detected:
                best=-1;bd=1e9
                for i,a in enumerate(bank):
                    if a.centroid is not None:
                        d=float((bmean-a.centroid).norm())
                        if d<bd:bd=d;best=i
                if best>=0 and bd<reuse_thr:
                    if active!=best:
                        if active not in frozen:frozen.append(active)
                        active=best;opt=torch.optim.SGD(bank[active].params(),lr=lr);bank[active].set_anchor()
                else:
                    if active not in frozen:frozen.append(active)
                    na=Adapter(rank,seed+len(bank))
                    if use_ortho and frozen:_ortho(na,[bank[j] for j in frozen])
                    bank.append(na);active=len(bank)-1;opt=torch.optim.SGD(bank[active].params(),lr=lr);bank[active].set_anchor()
            a=bank[active]
            a.centroid=bmean.clone() if a.centroid is None else 0.9*a.centroid+0.1*bmean
            bmu,bsd=batch_stats(f)             # per-state normalization memory (restored on recurrence)
            if a.mu is None: a.mu,a.sd=bmu.clone(),bsd.clone()
            else: a.mu=0.9*a.mu+0.1*bmu; a.sd=0.9*a.sd+0.1*bsd
            z=znorm(f,a.mu,a.sd)
            use_adapter="with_adapter" in abl   # DEFAULT DriftAdapt is gradient-free (routed norm memory)
            lg=head(z)+a.delta(z) if use_adapter else head(z)
            M.add(lg.argmax(1).detach(),y,sid);R.add(detected,sid)
            if not use_adapter:
                reg.append(0.0); continue
            slog=head(z).detach()
            if use_filter:
                e=entropy(lg);thr=torch.quantile(e.detach(),percentile);msk=e<=thr
                if not msk.any():msk=torch.ones_like(e,dtype=torch.bool)
            else: msk=torch.ones(f.shape[0],dtype=torch.bool)
            yhat=lg.argmax(1).detach()
            if "entropy_loss" in abl:
                loss=entropy(lg).mean()         # ablation: entropy-min instead of filtered pseudo-CE
            else:
                loss=F.cross_entropy(lg[msk],yhat[msk])
            if use_kl:
                pa=F.softmax(lg,-1)
                loss=loss+lam_kl*(pa*(F.log_softmax(lg,-1)-F.log_softmax(slog,-1))).sum(1).mean()
            if use_efdt:
                phi=torch.cat([p.flatten() for p in a.params()])
                loss=loss+lam_efdt*0.5*(efdt*(phi-a.anchor).pow(2)).sum()
            if use_ortho and frozen:
                op=sum((a.A@bank[j].A.t().detach()).pow(2).sum() for j in frozen)
                loss=loss+lam_perp*op
            opt.zero_grad();loss.backward()
            if use_efdt:
                gll=_llgrad(a,z,yhat,msk); efdt=ema*efdt+(1-ema)*gll.pow(2)
            if "no_clip" not in abl: torch.nn.utils.clip_grad_norm_(a.params(),gclip)
            opt.step()
            with torch.no_grad():
                phi=torch.cat([p.flatten() for p in a.params()]);nrm=phi.norm()
                if "no_proj" not in abl and nrm>Dball: a.A.mul_(Dball/nrm);a.B.mul_(Dball/nrm)
                if use_ortho and frozen:_ortho(a,[bank[j] for j in frozen])
                a.set_anchor()
            reg.append(float(loss.item()))
        res=_pack(M,R,t0,reg);res["num_adapters"]=len(bank);return res

    if method=="reservoir":
        # In-regime adaptation of the ReservoirTTA principle (Vray et al., 2025):
        # a bank of per-state ADAPTED MODELS (gradient-trained low-rank adapters), detected by
        # the SAME CUSUM+centroid router and REACTIVATED on recurrence. Each reservoir entry is
        # adapted online by entropy minimization (Tent-style), the canonical TTA loss, plus the
        # same per-state normalization memory. Difference vs DriftAdapt: it stores a per-state
        # *model* (adapter params) rather than only O(d) statistics, and performs a backward pass.
        bank=[Adapter(rank,seed)];active=0;frozen=[]
        opt=torch.optim.SGD(bank[active].params(),lr=lr)
        run_mean=None;g=0.0;fa=0.1
        for f,y,sid in stream:
            bmean=f.mean(0); detected=False
            if run_mean is None: run_mean=bmean.clone()
            else:
                s=float((bmean-run_mean).norm()); g=max(0.0,g+s-cusum_slack)
                if g>cusum_h: detected=True;g=0.0
                run_mean=(1-fa)*run_mean+fa*bmean
            if detected:
                best=-1;bd=1e9
                for i,a in enumerate(bank):
                    if a.centroid is not None:
                        d=float((bmean-a.centroid).norm())
                        if d<bd:bd=d;best=i
                if best>=0 and bd<reuse_thr:
                    if active!=best:
                        if active not in frozen:frozen.append(active)
                        active=best;opt=torch.optim.SGD(bank[active].params(),lr=lr)
                else:
                    if active not in frozen:frozen.append(active)
                    bank.append(Adapter(rank,seed+len(bank)));active=len(bank)-1
                    opt=torch.optim.SGD(bank[active].params(),lr=lr)
            a=bank[active]
            a.centroid=bmean.clone() if a.centroid is None else 0.9*a.centroid+0.1*bmean
            bmu,bsd=batch_stats(f)
            if a.mu is None: a.mu,a.sd=bmu.clone(),bsd.clone()
            else: a.mu=0.9*a.mu+0.1*bmu; a.sd=0.9*a.sd+0.1*bsd
            z=znorm(f,a.mu,a.sd)
            lg=head(z)+a.delta(z)                       # per-state adapted model
            M.add(lg.argmax(1).detach(),y,sid);R.add(detected,sid)
            loss=entropy(lg).mean()                     # Tent-style per-reservoir-model adaptation
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(a.params(),gclip);opt.step()
            reg.append(float(loss.item()))
        res=_pack(M,R,t0,reg);res["num_adapters"]=len(bank);return res
    raise ValueError(method)

def _llgrad(a,z,yhat,msk):
    A=a.A.detach().clone().requires_grad_(True);B=a.B.detach().clone().requires_grad_(True)
    lg=head(z)+(z@A.t())@B.t(); ll=F.cross_entropy(lg[msk],yhat[msk])
    gA,gB=torch.autograd.grad(ll,[A,B]); return torch.cat([gA.flatten(),gB.flatten()])

def _ortho(a,frozen):
    with torch.no_grad():
        As=torch.cat([fa.A.detach() for fa in frozen],0)
        Q,_=torch.linalg.qr(As.t()); P=Q@Q.t(); a.A.copy_(a.A-a.A@P)

def _pack(M,R,t0,reg):
    return {"online":M.online(),"bwt":M.bwt(),"recovery":M.recovery(),
            "router_p":R.prec(),"router_r":R.rec(),
            "ms_per_step":1000*(time.time()-t0)/max(len(M.traj),1),
            "reg_log":reg,"traj":M.traj}

if __name__=="__main__":
    for mode in ["sequential","recurrent","gradual"]:
        print("==",mode,"==")
        for m in ["source","ttn","tent","eata","driftadapt"]:
            r=run(m,mode,seed=0)
            print(f"  {m:11s} acc={r['online']:.3f} bwt={r['bwt']:+.3f} rec={r['recovery']:.3f} "
                  f"P/R={r['router_p']:.2f}/{r['router_r']:.2f} {r.get('num_adapters','')}")
