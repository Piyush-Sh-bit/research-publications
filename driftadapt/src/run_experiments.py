import json, os, numpy as np
from driftadapt import run

SEEDS=[0,1,2,3,4]
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"results.json")
R={}

def agg(method,mode,**kw):
    rs=[run(method,mode,seed=s,**kw) for s in SEEDS]
    keys=["online","bwt","recovery","router_p","router_r","ms_per_step"]
    out={k:[float(np.mean([r[k] for r in rs])),float(np.std([r[k] for r in rs]))] for k in keys}
    out["num_adapters"]=float(np.mean([r.get("num_adapters",0) for r in rs]))
    return out

# 1) MAIN GRID (gradient baselines at lr=0.1 shared operating point)
print("main grid ...")
R["main"]={}
for mode in ["sequential","recurrent","gradual"]:
    R["main"][mode]={}
    for m in ["source","ttn","tent","eata","driftadapt"]:
        a=agg(m,mode,lr=0.1); R["main"][mode][m]=a
        print(f"  {mode:11s} {m:11s} acc={a['online'][0]:.3f}+-{a['online'][1]:.3f} "
              f"bwt={a['bwt'][0]:+.3f} rec={a['recovery'][0]:.3f} P/R={a['router_p'][0]:.2f}/{a['router_r'][0]:.2f} na={a['num_adapters']:.1f}")

# 2) ABLATIONS (recurrent)
print("ablations ...")
R["ablation"]={}
for name,kw in [("DriftAdapt (full)",dict(abl=())),
                ("  - routing (global stats)",dict(abl=("no_router",))),
                ("  + gradient adapter",dict(abl=("with_adapter",),lr=0.1)),
                ("  + entropy adapter",dict(abl=("with_adapter","entropy_loss"),lr=0.5))]:
    a=agg("driftadapt","recurrent",**kw); R["ablation"][name.strip()]=a
    print(f"  {name:28s} acc={a['online'][0]:.3f} bwt={a['bwt'][0]:+.3f} rec={a['recovery'][0]:.3f}")

# 3) LR ROBUSTNESS (recurrent): gradient baselines vs lr; DriftAdapt/TTN are lr-free
print("lr robustness ...")
R["lr"]={"grid":[0.005,0.01,0.02,0.05,0.1,0.2,0.5],"data":{}}
for m in ["tent","eata"]:
    R["lr"]["data"][m]=[agg(m,"recurrent",lr=lr)["online"][0] for lr in R["lr"]["grid"]]
    print(f"  {m:11s} {[round(x,3) for x in R['lr']['data'][m]]}")
R["lr"]["driftadapt"]=agg("driftadapt","recurrent")["online"][0]   # lr-free constant
R["lr"]["ttn"]=agg("ttn","recurrent")["online"][0]
print(f"  driftadapt(const)={R['lr']['driftadapt']:.3f}  ttn(const)={R['lr']['ttn']:.3f}")

# 4) DRIFT FREQUENCY (recurrent, vary cycles => path length); time-avg error
print("drift frequency ...")
R["freq"]={"cycles":[1,2,3,4,6],"data":{}}
for m in ["tent","eata","driftadapt"]:
    kw=dict(lr=0.1) if m in ("tent","eata") else dict()
    R["freq"]["data"][m]=[1.0-agg(m,"recurrent",cycles=cy,**kw)["online"][0] for cy in R["freq"]["cycles"]]
    print(f"  {m:11s} err={[round(x,3) for x in R['freq']['data'][m]]}")

# 5) TRAJECTORIES (recurrent, seed 0)
print("trajectories ...")
R["traj"]={}
for m in ["ttn","tent","eata","driftadapt"]:
    kw=dict(lr=0.1) if m in ("tent","eata") else dict()
    R["traj"][m]=run(m,"recurrent",seed=0,**kw)["traj"]

# 6) Tent best-lr reference (fair text)
R["tent_best"]={}
for mode in ["sequential","recurrent","gradual"]:
    vals={lr:agg("tent",mode,lr=lr)["online"][0] for lr in [0.005,0.01,0.02,0.05,0.1]}
    blr=max(vals,key=vals.get); R["tent_best"][mode]=[blr,vals[blr]]
    print(f"  tent best {mode}: lr={blr} acc={vals[blr]:.3f}")

json.dump(R,open(OUT,"w"),indent=1)
print("saved",OUT)
