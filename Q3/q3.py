import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from common.analysis import confidence_stats
from common.data import load_cifar, get_device, batches
from common.train import count_params, predict_proba
from Q2.models import CNN
from Q3.models import ConvAE

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "checkpoints")
DIMS = [16, 128, 1024]
SIGMA = 0.1
EPOCHS = 30


def add_noise(x, generator=None):
    if generator is None:
        n = torch.randn_like(x)
    else:
        n = torch.randn(x.shape, generator=generator).to(x.device)
    return (x + SIGMA * n).clamp(0, 1)


@torch.no_grad()
def recon_mse(ae, x_in, x_target, bs=1000):
    ae.eval()
    tot = 0.0
    for i in range(0, len(x_in), bs):
        tot += F.mse_loss(ae(x_in[i:i + bs]), x_target[i:i + bs], reduction="sum").item()
    return tot / x_in.numel()


@torch.no_grad()
def reconstruct(ae, x, bs=1000):
    ae.eval()
    return torch.cat([ae(x[i:i + bs]) for i in range(0, len(x), bs)])


@torch.no_grad()
def encode(ae, x, bs=1000):
    ae.eval()
    return torch.cat([ae.encode(x[i:i + bs]) for i in range(0, len(x), bs)])


def train_ae(d, raw, denoise, dev, val_noisy=None):
    tag = f"{'dae' if denoise else 'ae'}_d{d}"
    path = os.path.join(CKPT, f"{tag}.pt")
    hist_path = os.path.join(OUT, f"{tag}_history.json")
    ae = ConvAE(d).to(dev)
    if os.path.exists(path):
        ae.load_state_dict(torch.load(path, map_location=dev))
        return ae, json.load(open(hist_path))
    print(tag, "params", count_params(ae), flush=True)
    torch.manual_seed(0)
    ae = ConvAE(d).to(dev)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    x_tr, x_va = raw["train"], raw["val"]
    va_in = val_noisy if denoise else x_va
    hist, best, best_state = [], 1e9, None
    for ep in range(1, EPOCHS + 1):
        ae.train()
        tot, n = 0.0, 0
        for xb, _ in batches(x_tr, x_tr[:, 0, 0, 0], 256, generator=gen):
            inp = add_noise(xb) if denoise else xb
            loss = F.mse_loss(ae(inp), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        va = recon_mse(ae, va_in, x_va)
        hist.append({"epoch": ep, "train_mse": tot / n, "val_mse": va})
        if va < best:
            best = va
            best_state = {k: v.detach().clone() for k, v in ae.state_dict().items()}
        print(f"  ep {ep:2d} train {tot / n:.5f} val {va:.5f}", flush=True)
    ae.load_state_dict(best_state)
    torch.save(best_state, path)
    json.dump(hist, open(hist_path, "w"), indent=1)
    return ae, hist


def linear_probe(f_tr, y_tr, f_va, y_va, f_te, y_te, dev, epochs=40):
    mu, sd = f_tr.mean(0, keepdim=True), f_tr.std(0, keepdim=True) + 1e-6
    f_tr, f_va, f_te = (f_tr - mu) / sd, (f_va - mu) / sd, (f_te - mu) / sd
    results = {}
    for wd in [0.0, 1e-4, 1e-3, 1e-2]:
        torch.manual_seed(0)
        lin = torch.nn.Linear(f_tr.shape[1], 10).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
        gen = torch.Generator().manual_seed(0)
        best_va, best_state = -1, None
        for ep in range(epochs):
            for xb, yb in batches(f_tr, y_tr, 512, generator=gen):
                loss = F.cross_entropy(lin(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                va = (lin(f_va).argmax(1) == y_va).float().mean().item()
            if va > best_va:
                best_va = va
                best_state = {k: v.clone() for k, v in lin.state_dict().items()}
        lin.load_state_dict(best_state)
        with torch.no_grad():
            tr = (lin(f_tr).argmax(1) == y_tr).float().mean().item()
            te = (lin(f_te).argmax(1) == y_te).float().mean().item()
        results[wd] = {"train": tr, "val": best_va, "test": te}
    wd = max(results, key=lambda w: results[w]["val"])
    return {"selected_wd": wd, **results[wd], "all": {str(k): v for k, v in results.items()}}


@torch.no_grad()
def knn_acc(f_tr, y_tr, f_q, y_q, k=20):
    mu, sd = f_tr.mean(0, keepdim=True), f_tr.std(0, keepdim=True) + 1e-6
    a = F.normalize((f_tr - mu) / sd, dim=1)
    q = F.normalize((f_q - mu) / sd, dim=1)
    correct = 0
    for i in range(0, len(q), 500):
        sim = q[i:i + 500] @ a.T
        idx = sim.topk(k, dim=1).indices
        votes = F.one_hot(y_tr[idx], 10).sum(1)
        correct += (votes.argmax(1) == y_q[i:i + 500]).sum().item()
    return correct / len(q)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    dev = get_device()
    data = load_cifar(dev)
    mean, std = data["mean"].to(dev), data["std"].to(dev)
    raw = {s: data[f"x_{s}"] * std + mean for s in ["train", "val", "test"]}
    y = {s: data[f"y_{s}"] for s in ["train", "val", "test"]}
    noisy = {s: add_noise(raw[s], torch.Generator().manual_seed(seed))
             for s, seed in [("val", 11), ("test", 12)]}
    summary = {"sigma": SIGMA, "dims": DIMS}

    aes, table = {}, {}
    for d in DIMS:
        aes[d], hist = train_ae(d, raw, False, dev)
        table[d] = {"params": count_params(aes[d]),
                    "val_mse": recon_mse(aes[d], raw["val"], raw["val"]),
                    "best_epoch": int(np.argmin([h["val_mse"] for h in hist])) + 1}
    sel = min(DIMS, key=lambda d: table[d]["val_mse"])
    summary["a"] = {"table": table, "selected_d": sel,
                    "selected_test_mse": recon_mse(aes[sel], raw["test"], raw["test"])}
    print(json.dumps(summary["a"], indent=1), flush=True)

    dae, _ = train_ae(sel, raw, True, dev, val_noisy=noisy["val"])
    q2sel = json.load(open(os.path.join(ROOT, "Q2", "results", "selection.json")))
    cnn_lr = q2sel["lrs"]["cnn"]
    cnn = CNN().to(dev)
    cnn.load_state_dict(torch.load(os.path.join(ROOT, "Q2", "checkpoints",
                                                f"cnn_clean_lr{cnn_lr:g}_s0.pt"), map_location=dev))
    norm = lambda x: (x - mean) / std
    yte = y["test"].cpu().numpy()
    inputs = {
        "clean": raw["test"],
        "noisy": noisy["test"],
        "ae_recon_of_noisy": reconstruct(aes[sel], noisy["test"]),
        "dae_recon_of_noisy": reconstruct(dae, noisy["test"]),
        "ae_recon_of_clean": reconstruct(aes[sel], raw["test"]),
        "dae_recon_of_clean": reconstruct(dae, raw["test"]),
    }
    b = {}
    for name, x in inputs.items():
        st = confidence_stats(predict_proba(cnn, norm(x)).cpu().numpy(), yte)
        st.pop("confusion")
        st["mse_to_clean"] = F.mse_loss(x, raw["test"]).item()
        b[name] = st
        print(f"{name:22s} acc {st['acc']:.4f}  mse_to_clean {st['mse_to_clean']:.5f}", flush=True)
    summary["b"] = {"classifier": f"Q2 cnn_clean_lr{cnn_lr:g}_s0", "results": b}
    torch.save({k: v[:16].cpu() for k, v in inputs.items()}, os.path.join(OUT, "examples.pt"))

    torch.manual_seed(0)
    rand_enc = ConvAE(1024).to(dev)
    encoders = {f"ae_d{d}": aes[d] for d in DIMS}
    encoders[f"dae_d{sel}"] = dae
    encoders["random_d1024"] = rand_enc
    c = {}
    for name, enc in encoders.items():
        f = {s: encode(enc, raw[s]) for s in ["train", "val", "test"]}
        lp = linear_probe(f["train"], y["train"], f["val"], y["val"], f["test"], y["test"], dev)
        c[name] = {"code_dim": f["train"].shape[1],
                   "test_mse": recon_mse(enc, raw["test"], raw["test"]),
                   "linear_probe": lp,
                   "knn_val": knn_acc(f["train"], y["train"], f["val"], y["val"]),
                   "knn_test": knn_acc(f["train"], y["train"], f["test"], y["test"])}
        print(f"{name:14s} mse {c[name]['test_mse']:.5f} probe test {lp['test']:.4f} "
              f"knn test {c[name]['knn_test']:.4f}", flush=True)
    flat = {s: raw[s].flatten(1) for s in raw}
    c["raw_pixels"] = {"code_dim": 3072, "test_mse": 0.0,
                       "linear_probe": linear_probe(flat["train"], y["train"], flat["val"],
                                                    y["val"], flat["test"], y["test"], dev),
                       "knn_val": knn_acc(flat["train"], y["train"], flat["val"], y["val"]),
                       "knn_test": knn_acc(flat["train"], y["train"], flat["test"], y["test"])}
    print("raw pixels probe", c["raw_pixels"]["linear_probe"]["test"],
          "knn", c["raw_pixels"]["knn_test"], flush=True)
    summary["c"] = c
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
