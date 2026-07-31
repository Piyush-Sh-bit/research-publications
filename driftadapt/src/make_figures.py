import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(os.path.join(HERE,"results.json")))
OUT=os.path.join(HERE,"figs"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3})
COL={"source":"#888","ttn":"#1f77b4","tent":"#d62728","eata":"#ff7f0e","driftadapt":"#2ca02c"}

def smooth(x,k=15):
    x=np.array(x); c=np.ones(k)/k
    return np.convolve(x,c,mode="valid")

# ---- Fig: accuracy over time (recurrent) ----
plt.figure(figsize=(7,3.6))
for m in ["ttn","tent","eata","driftadapt"]:
    y=smooth(R["traj"][m]); plt.plot(np.arange(len(y)),100*y,label=m.upper() if m!="driftadapt" else "DriftAdapt",
                                     color=COL[m],lw=1.8)
# mark cycle boundaries (3 states x 3 cycles)
L=len(R["traj"]["driftadapt"]); blk=L//9
for c in range(1,9):
    plt.axvline(c*blk,color="k",ls=":",lw=0.5,alpha=0.4)
plt.xlabel("stream step $t$"); plt.ylabel("online accuracy (%)")
plt.title("Recurrent drift: 3 states $\\times$ 3 cycles (dotted = state change)")
plt.legend(ncol=4,fontsize=9,loc="lower center"); plt.tight_layout()
plt.savefig(os.path.join(OUT,"trajectory.png"),dpi=160); plt.close()

# ---- Fig: lr robustness ----
plt.figure(figsize=(6,3.6))
g=R["lr"]["grid"]
for m in ["tent","eata"]:
    plt.plot(g,[100*x for x in R["lr"]["data"][m]],"-o",label=m.upper(),color=COL[m],lw=1.8,ms=4)
plt.axhline(100*R["lr"]["driftadapt"],color=COL["driftadapt"],lw=2.2,label="DriftAdapt (lr-free)")
plt.axhline(100*R["lr"]["ttn"],color=COL["ttn"],ls="--",lw=1.5,label="TTN (lr-free)")
plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel("online accuracy (%)")
plt.title("Robustness to learning rate (recurrent)")
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(OUT,"lr_robust.png"),dpi=160); plt.close()

# ---- Fig: drift frequency ----
plt.figure(figsize=(6,3.6))
cy=R["freq"]["cycles"]
for m in ["tent","eata","driftadapt"]:
    plt.plot(cy,[100*x for x in R["freq"]["data"][m]],"-o",
             label="DriftAdapt" if m=="driftadapt" else m.upper(),color=COL[m],lw=1.8,ms=5)
plt.xlabel("number of recurrence cycles (drift frequency)"); plt.ylabel("time-averaged online error (%)")
plt.title("Robustness to drift frequency (recurrent)")
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(OUT,"drift_freq.png"),dpi=160); plt.close()

print("figures written to",OUT)
print(os.listdir(OUT))
