import json
import os

import numpy as np
import torch

from common.data import load_cifar, fixed_pixel_permutation, apply_permutation, get_device
from common.train import train, evaluate, count_params
from Q1.models import build

OUT = os.path.join(os.path.dirname(__file__), "results", "q1a")
EPOCHS = 30
LRS = [3e-4, 1e-3, 3e-3]
SEEDS = [0, 1, 2]
ARCHS = ["shallow_wide", "deep_narrow"]
INPUTS = ["original", "permuted"]


def permuted(data, perm):
    d = dict(data)
    for k in ["x_train", "x_val", "x_test"]:
        d[k] = apply_permutation(data[k], perm)
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = get_device()
    base = load_cifar(dev)
    datasets = {"original": base, "permuted": permuted(base, fixed_pixel_permutation())}

    sweep = {}
    for arch in ARCHS:
        for inp in INPUTS:
            for lr in LRS:
                tag = f"{arch}_{inp}_lr{lr:g}"
                print(tag, flush=True)
                model = build(arch).to(dev)
                hist, _ = train(model, datasets[inp], epochs=EPOCHS, lr=lr, seed=0,
                                log_path=os.path.join(OUT, f"sweep_{tag}.json"), verbose=False)
                sweep[tag] = max(h["val_acc"] for h in hist)
                print(f"  best val {sweep[tag]:.4f}", flush=True)

    final = {}
    for arch in ARCHS:
        for inp in INPUTS:
            lr = max(LRS, key=lambda l: sweep[f"{arch}_{inp}_lr{l:g}"])
            runs = []
            for s in SEEDS:
                model = build(arch).to(dev)
                hist, best = train(model, datasets[inp], epochs=EPOCHS, lr=lr, seed=s,
                                   log_path=os.path.join(OUT, f"final_{arch}_{inp}_s{s}.json"),
                                   verbose=False)
                d = datasets[inp]
                _, test_acc = evaluate(best, d["x_test"], d["y_test"])
                best_ep = int(np.argmax([h["val_acc"] for h in hist])) + 1
                runs.append({"seed": s, "best_epoch": best_ep,
                             "train_acc_at_best": hist[best_ep - 1]["train_acc"],
                             "val_acc": hist[best_ep - 1]["val_acc"],
                             "final_train_acc": hist[-1]["train_acc"],
                             "test_acc": test_acc})
                print(f"{arch} {inp} lr={lr:g} seed={s}: val {runs[-1]['val_acc']:.4f} "
                      f"test {test_acc:.4f}", flush=True)
            final[f"{arch}_{inp}"] = {
                "arch": arch, "input": inp, "hidden": build(arch).hidden,
                "params": count_params(build(arch)), "lr": lr, "runs": runs,
                "test_mean": float(np.mean([r["test_acc"] for r in runs])),
                "test_std": float(np.std([r["test_acc"] for r in runs])),
            }

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"sweep_best_val": sweep, "final": final}, f, indent=1)
    for k, v in final.items():
        print(f"{k:28s} params={v['params']:,} lr={v['lr']:g} "
              f"test={100*v['test_mean']:.2f} ± {100*v['test_std']:.2f}")


if __name__ == "__main__":
    main()
