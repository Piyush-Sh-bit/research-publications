import json, os, numpy as np
from driftadapt import run
SEEDS=[0,1,2,3,4]
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"extended_results.json")
R={}

def acc(method,mode,**kw):
    xs=[run(method,mode,seed=s,**kw)["online"] for s in SEEDS]
    return [float(np.mean(xs)),float(np.std(xs))]

# 1) BATCH-SIZE SENSITIVITY (recurrent): small batches stress normalization estimates
print("batch sensitivity ...")
R["batch"]={"sizes":[8,16,32,64,128],"data":{}}
for m in ["source","ttn","tent","eata","driftadapt"]:
    R["batch"]["data"][m]=[]
    for bs in R["batch"]["sizes"]:
        kw=dict(batch=bs)
        if m in ("tent","eata"): kw["lr"]=0.1
        R["batch"]["data"][m].append(acc(m,"recurrent",**kw)[0])
    print(f"  {m:11s} {[round(100*x,1) for x in R['batch']['data'][m]]}")

# 2) ROUTER THRESHOLD ROBUSTNESS (DriftAdapt, recurrent): acc + #states vs CUSUM h and reuse tau
print("threshold robustness ...")
R["thresh"]={"cusum_h":[1.0,2.0,3.0,4.0,5.0],"reuse_thr":[5.0,6.0,7.0,8.0,9.0],"acc":{},"na":{}}
for h in R["thresh"]["cusum_h"]:
    rs=[run("driftadapt","recurrent",seed=s,cusum_h=h) for s in SEEDS]
    R["thresh"]["acc"][str(h)]=[float(np.mean([r["online"] for r in rs])),float(np.std([r["online"] for r in rs]))]
    R["thresh"]["na"][str(h)]=float(np.mean([r["num_adapters"] for r in rs]))
print("  vs cusum_h: acc=",[round(100*R['thresh']['acc'][str(h)][0],1) for h in R['thresh']['cusum_h']],
      " na=",[round(R['thresh']['na'][str(h)],1) for h in R['thresh']['cusum_h']])
R["thresh"]["acc_tau"]={}; R["thresh"]["na_tau"]={}
for t in R["thresh"]["reuse_thr"]:
    rs=[run("driftadapt","recurrent",seed=s,reuse_thr=t) for s in SEEDS]
    R["thresh"]["acc_tau"][str(t)]=float(np.mean([r["online"] for r in rs]))
    R["thresh"]["na_tau"][str(t)]=float(np.mean([r["num_adapters"] for r in rs]))
print("  vs reuse_tau: acc=",[round(100*R['thresh']['acc_tau'][str(t)],1) for t in R['thresh']['reuse_thr']],
      " na=",[round(R['thresh']['na_tau'][str(t)],1) for t in R['thresh']['reuse_thr']])

# 3) LONG-HORIZON STABILITY (recurrent, many cycles): time-avg error vs cycles (extends to 10)
print("long horizon ...")
R["horizon"]={"cycles":[1,2,3,5,8,10],"data":{}}
for m in ["ttn","tent","eata","driftadapt"]:
    R["horizon"]["data"][m]=[]
    for cy in R["horizon"]["cycles"]:
        kw=dict(cycles=cy)
        if m in ("tent","eata"): kw["lr"]=0.1
        R["horizon"]["data"][m].append(1.0-acc(m,"recurrent",**kw)[0])
    print(f"  {m:11s} err={[round(100*x,1) for x in R['horizon']['data'][m]]}")

json.dump(R,open(OUT,"w"),indent=1); print("saved",OUT)
