"""Nearest-prior-art comparison (Table: DriftAdapt vs ReservoirTTA-style model reservoir).

Reproduces Section "Comparison to the nearest prior art". The baseline `reservoir` is a routed
bank of per-state gradient-adapted models (entropy minimization), reactivated on recurrence -- a
faithful in-regime version of ReservoirTTA (Vray et al., 2025). It is given its BEST per-stream
learning rate so the comparison does not handicap it. DriftAdapt is the gradient-free statistic
reservoir. Writes reservoir_results.json. Run after extract_features.py + train_head.py.
"""
import json, os, numpy as np
from driftadapt import run, DIM, NCLS

SEEDS=[0,1,2,3,4]
RANK=8
LRS=[0.005,0.01,0.02,0.05,0.1]
HERE=os.path.dirname(os.path.abspath(__file__))

def agg(method,mode,**kw):
    rs=[run(method,mode,seed=s,**kw) for s in SEEDS]
    return {k:[float(np.mean([r[k] for r in rs])),float(np.std([r[k] for r in rs]))]
            for k in ["online","bwt","recovery","ms_per_step"]} | {
            "num_adapters":float(np.mean([r.get("num_adapters",0) for r in rs]))}

OUT={}
for mode in ["sequential","recurrent","gradual"]:
    da=agg("driftadapt",mode)
    best=None
    for lr in LRS:
        r=agg("reservoir",mode,lr=lr)
        if best is None or r["online"][0]>best["online"][0]:
            best=r; best["lr"]=lr
    OUT[mode]={"driftadapt":da,"reservoir_bestlr":best}
    print(f"== {mode} ==")
    print(f"  DriftAdapt (stat reservoir)        acc={da['online'][0]:.3f}+-{da['online'][1]:.3f} "
          f"rec={da['recovery'][0]:.3f} K={da['num_adapters']:.1f}")
    print(f"  Reservoir (model bank, lr={best['lr']})  acc={best['online'][0]:.3f}+-{best['online'][1]:.3f} "
          f"rec={best['recovery'][0]:.3f} K={best['num_adapters']:.1f}")

da_mem=2*DIM                      # mu, sd
res_mem=RANK*DIM+NCLS*RANK+2*DIM  # adapter A,B + mu,sd
OUT["per_state_mem_floats"]={"driftadapt":da_mem,"reservoir":res_mem,"ratio":res_mem/da_mem}
print(f"\nper-state memory: DriftAdapt={da_mem} floats  Reservoir={res_mem} floats  "
      f"ratio={res_mem/da_mem:.1f}x  (DriftAdapt is forward-only; reservoir needs a backward pass)")

json.dump(OUT,open(os.path.join(HERE,"reservoir_results.json"),"w"),indent=1)
print("saved reservoir_results.json")
