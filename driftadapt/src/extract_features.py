"""One-time feature extraction + caching for the DriftAdapt pipeline.

Frozen ImageNet ResNet-18 backbone. CONSISTENT transform (interpolate 32->224 +
ImageNet norm) for clean AND corrupted images, so the only distribution shift is the
corruption itself. Features (512-d) are cached to disk.

Corruptions: the OFFICIAL CIFAR-10-C benchmark (Hendrycks & Dietterich, 2019) at a
fixed severity, not synthetic approximations. Download once:

    curl -L -o CIFAR-10-C.tar https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
    tar -xf CIFAR-10-C.tar          # -> ./CIFAR-10-C/{gaussian_noise,defocus_blur,...}.npy

and point C10C_DIR at the extracted folder (or set the CIFAR10C_DIR env var). Each
corruption .npy is [50000,32,32,3] uint8 = 5 severities x 10000 test images, in the
standard CIFAR-10 test order; severity s uses rows [(s-1)*10000 : s*10000].
"""
import os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
CACHE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CACHE, "data")  # CIFAR-10 auto-downloaded here (clean images + labels)
C10C_DIR = os.environ.get("CIFAR10C_DIR", os.path.join(CACHE, "CIFAR-10-C"))
SEVERITY = int(os.environ.get("CIFAR10C_SEVERITY", "5"))   # 1..5; 5 = standard hardest
dev = "cpu"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def shared_transform(x):  # x: (3,32,32) float [0,1] -> (3,224,224) normalized
    x = F.interpolate(x.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
    return (x - IMAGENET_MEAN) / IMAGENET_STD

# State name -> official CIFAR-10-C corruption file. index 0 == clean (no corruption).
# One corruption from each of the four CIFAR-10-C families (noise / blur / weather /
# digital), plus brightness as an easy anchor, so the stream spans a real difficulty range.
STATE_TO_C10C = {
    "shot_noise":   "shot_noise",     # noise
    "defocus_blur": "defocus_blur",   # blur
    "fog":          "fog",            # weather
    "jpeg":         "jpeg_compression",  # digital
    "brightness":   "brightness",     # weather (easy anchor)
}
STATES = ["clean"] + list(STATE_TO_C10C.keys())

def build_backbone():
    bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feat = nn.Sequential(*list(bb.children())[:-1]).eval()
    for p in feat.parameters(): p.requires_grad = False
    return feat

@torch.no_grad()
def extract(feat, imgs01, bs=100):
    """imgs01: list/tensor of (3,32,32) float [0,1] images -> (N,512) features."""
    out = []
    for i in range(0, len(imgs01), bs):
        batch = torch.stack([shared_transform(im) for im in imgs01[i:i+bs]])
        out.append(feat(batch).flatten(1))
    return torch.cat(out)

def _load_c10c(name, idx):
    """Official corruption `name` at SEVERITY, sliced to test indices `idx`.
    Returns a list of (3,32,32) float[0,1] tensors."""
    path = os.path.join(C10C_DIR, name + ".npy")
    if not os.path.exists(path):
        sys.exit(f"CIFAR-10-C file not found: {path}\n"
                 f"Download CIFAR-10-C.tar from Zenodo and extract it, or set CIFAR10C_DIR.")
    arr = np.load(path, mmap_mode="r")                    # [50000,32,32,3] uint8
    base = (SEVERITY - 1) * 10000
    sub = np.asarray(arr[base + idx])                     # [n,32,32,3] uint8, HWC
    t = torch.from_numpy(sub).permute(0, 3, 1, 2).float() / 255.0
    return [t[i] for i in range(t.shape[0])]

def _load_cifar10():
    """CIFAR-10 clean images in canonical order via HuggingFace datasets (torchvision's
    cs.toronto.edu mirror is unreachably slow from some networks). Returns
    (train_imgs01, train_labels, test_imgs01, test_labels) as (3,32,32) float tensors."""
    from datasets import load_dataset  # lazy
    ds = load_dataset("uoft-cs/cifar10")
    def to01(split):
        imgs = [torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
                for im in split["img"]]
        return imgs, torch.tensor(split["label"])
    tr_i, tr_l = to01(ds["train"]); te_i, te_l = to01(ds["test"])
    return tr_i, tr_l, te_i, te_l


def _verify_order(te_imgs_full):
    """We lack CIFAR-10-C's labels.npy to confirm the clean test set is in the same
    order the corruptions were generated from. Verify structurally instead: a low-severity
    corruption preserves image content, so clean[i] must correlate far more with
    corrupt[i] than with corrupt[j!=i]. brightness severity 1 is ~clean plus a small
    offset, so its diagonal correlation should be near 1."""
    arr = np.load(os.path.join(C10C_DIR, "brightness.npy"), mmap_mode="r")  # sev1 = rows 0..9999
    idx = [7, 123, 4567, 8901, 2345]
    clean = np.stack([te_imgs_full[i].permute(1, 2, 0).numpy() for i in idx])   # [k,32,32,3]
    corr1 = np.stack([np.asarray(arr[i]) for i in idx]).astype(np.float32) / 255.0
    def cc(a, b):
        a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    diag = np.mean([cc(clean[k], corr1[k]) for k in range(len(idx))])
    off = np.mean([cc(clean[k], corr1[(k + 1) % len(idx)]) for k in range(len(idx))])
    print(f"order check: diag corr(clean_i, brightness1_i)={diag:.3f}  off-diag={off:.3f}")
    assert diag > 0.9 and diag - off > 0.5, (
        "clean CIFAR-10 test order does NOT match CIFAR-10-C order -- aborting to avoid "
        "silently pairing wrong images/labels.")
    print("order alignment with official CIFAR-10-C: OK")


def main():
    out = os.path.join(CACHE, "cache.pt")
    if os.path.exists(out) and os.environ.get("FORCE", "") != "1":
        print("cache exists:", out, "(set FORCE=1 to rebuild)"); return
    feat = build_backbone()
    tr_imgs_full, tr_lab_full, te_imgs_full, te_lab_full = _load_cifar10()
    _verify_order(te_imgs_full)   # confirm clean/corrupt alignment before extracting
    rng = np.random.RandomState(SEED)
    tr_idx = rng.choice(len(tr_imgs_full), 4000, replace=False)
    te_idx = rng.choice(len(te_imgs_full), 1500, replace=False)
    tr_imgs = [tr_imgs_full[i] for i in tr_idx]; tr_lab = tr_lab_full[tr_idx]
    te_imgs = [te_imgs_full[i] for i in te_idx]; te_lab = te_lab_full[te_idx]

    t0 = time.time()
    print(f"extracting clean TRAIN features (head training) ... [severity {SEVERITY}]")
    train_feats = extract(feat, tr_imgs)
    cache = {"train_feats": train_feats, "train_labels": tr_lab,
             "test_labels": te_lab, "states": STATES, "state_feats": {},
             "severity": SEVERITY, "source": "CIFAR-10-C (Hendrycks & Dietterich, 2019)"}
    for name in STATES:
        print(f"extracting TEST features for state '{name}' ... ({time.time()-t0:.0f}s)")
        imgs = te_imgs if name == "clean" else _load_c10c(STATE_TO_C10C[name], te_idx)
        cache["state_feats"][name] = extract(feat, imgs)
    torch.save(cache, out)
    print("saved", out, "in %.0fs" % (time.time()-t0))

if __name__ == "__main__":
    main()
