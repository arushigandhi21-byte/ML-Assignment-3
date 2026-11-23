import json
import os

import numpy as np
import torch

from common.analysis import confidence_stats
from common.data import (load_cifar, get_device, fixed_pixel_permutation,
                         apply_permutation, block_shuffle)
from common.train import train, count_params, predict_proba
from Q1.models import build as build_mlp
from Q2.models import CNN

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "checkpoints")
EPOCHS = 30
LRS = [3e-4, 1e-3, 3e-3]
MLP_LR = 1e-3
BLOCKS = [4, 8, 16]


def make_model(kind):
    return CNN() if kind == "cnn" else build_mlp("shallow_wide")


def condition_data(base, cond):
    if cond == "clean":
        return base, None, None
    d = dict(base)
    if cond == "permuted":
        perm = fixed_pixel_permutation()
        for k in ["x_train", "x_val", "x_test"]:
            d[k] = apply_permutation(base[k], perm)
        return d, None, None
    k = int(cond.replace("block", ""))
    d["x_val"] = block_shuffle(base["x_val"], k, torch.Generator().manual_seed(1))
    d["x_test"] = block_shuffle(base["x_test"], k, torch.Generator().manual_seed(2))
    x_ev = block_shuffle(base["x_train"], k, torch.Generator().manual_seed(3))
    return d, (lambda xb: block_shuffle(xb, k)), x_ev


def run(kind, cond, lr, seed, data, tf, x_ev, dev, save_ckpt=True):
    tag = f"{kind}_{cond}_lr{lr:g}_s{seed}"
    res_path = os.path.join(OUT, f"{tag}.json")
    if os.path.exists(res_path):
        return json.load(open(res_path))
    print(tag, flush=True)
    torch.manual_seed(seed)
    model = make_model(kind).to(dev)
    hist, best = train(model, data, epochs=EPOCHS, lr=lr, seed=seed, verbose=False,
                       train_eval_n=10000, batch_transform=tf, x_train_eval=x_ev)
    best_ep = int(np.argmax([h["val_acc"] for h in hist])) + 1
    probs = predict_proba(best, data["x_test"]).cpu().numpy()
    stats = confidence_stats(probs, data["y_test"].cpu().numpy())
    np.save(os.path.join(OUT, f"{tag}_test_probs.npy"), probs.astype(np.float32))
    if save_ckpt:
        torch.save(best.state_dict(), os.path.join(CKPT, f"{tag}.pt"))
    res = {"kind": kind, "cond": cond, "lr": lr, "seed": seed,
           "params": count_params(model), "best_epoch": best_ep,
           "best_val_acc": hist[best_ep - 1]["val_acc"],
           "train_acc_at_best": hist[best_ep - 1]["train_acc"],
           "test": stats, "history": hist}
    json.dump(res, open(res_path, "w"), indent=1)
    print(f"  val {res['best_val_acc']:.4f} (ep {best_ep})  test {stats['acc']:.4f}", flush=True)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    dev = get_device()
    base = load_cifar(dev)

    d, tf, xe = condition_data(base, "clean")
    sweep = {lr: run("cnn", "clean", lr, 0, d, tf, xe, dev)["best_val_acc"] for lr in LRS}
    cnn_lr = max(LRS, key=lambda l: sweep[l])
    print("CNN LR sweep", sweep, "-> selected", cnn_lr, flush=True)
    lrs = {"cnn": cnn_lr, "mlp": MLP_LR}

    for cond in ["clean", "permuted"]:
        d, tf, xe = condition_data(base, cond)
        for kind in ["mlp", "cnn"]:
            for s in [0, 1, 2]:
                run(kind, cond, lrs[kind], s, d, tf, xe, dev)

    for k in BLOCKS:
        d, tf, xe = condition_data(base, f"block{k}")
        for kind in ["mlp", "cnn"]:
            run(kind, f"block{k}", lrs[kind], 0, d, tf, xe, dev)

    json.dump({"cnn_lr_sweep": {str(k): v for k, v in sweep.items()}, "lrs": lrs},
              open(os.path.join(OUT, "selection.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
