import os, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
CACHE=os.path.dirname(os.path.abspath(__file__))
c=torch.load(os.path.join(CACHE,"cache.pt")); torch.manual_seed(0)
Xtr,Ytr=c["train_feats"],c["train_labels"]; Yte=c["test_labels"]
mu0=Xtr.mean(0); sd0=Xtr.std(0)+1e-5
Ztr=(Xtr-mu0)/sd0
head=nn.Linear(Xtr.shape[1],10)
opt=torch.optim.Adam(head.parameters(),lr=1e-3,weight_decay=1e-4)
for ep in range(40):
    perm=torch.randperm(len(Ztr))
    for i in range(0,len(Ztr),128):
        b=perm[i:i+128]; loss=F.cross_entropy(head(Ztr[b]),Ytr[b])
        opt.zero_grad(); loss.backward(); opt.step()
torch.save({"weight":head.weight.detach(),"bias":head.bias.detach(),
            "mu0":mu0,"sd0":sd0}, os.path.join(CACHE,"head_std.pt"))
print("trained on standardized feats; loss %.3f"%loss.item())
print(f"{'state':10s} {'src(mu0)':>9s} {'TTN(batch)':>11s}")
for name in c["states"]:
    f=c["state_feats"][name]
    zf=(f-mu0)/sd0
    zb=(f-f.mean(0))/(f.std(0)+1e-5)
    with torch.no_grad():
        a_src=(head(zf).argmax(1)==Yte).float().mean().item()
        a_bn =(head(zb).argmax(1)==Yte).float().mean().item()
    print(f"{name:10s} {a_src:9.3f} {a_bn:11.3f}")
