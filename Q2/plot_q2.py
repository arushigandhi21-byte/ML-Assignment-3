import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.analysis import CLASSES

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
COL = {"mlp": "tab:blue", "cnn": "tab:red"}
NAME = {"mlp": "MLP", "cnn": "CNN"}


def load(kind, cond):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, f"{kind}_{cond}_lr*_s*.json")))]


def main():
    os.makedirs(FIG, exist_ok=True)
    lrs = json.load(open(os.path.join(RES, "selection.json")))["lrs"]
    conds = ["clean", "permuted", "block16", "block8", "block4"]
    table = {}
    for kind in ["mlp", "cnn"]:
        for cond in conds:
            runs = [r for r in load(kind, cond) if abs(r["lr"] - lrs[kind]) < 1e-12]
            acc = [r["test"]["acc"] for r in runs]
            table[f"{kind}_{cond}"] = {
                "params": runs[0]["params"], "lr": lrs[kind], "n_seeds": len(runs),
                "test_mean": float(np.mean(acc)), "test_std": float(np.std(acc)),
                "val_mean": float(np.mean([r["best_val_acc"] for r in runs])),
                "train_at_best_mean": float(np.mean([r["train_acc_at_best"] for r in runs])),
                **{k: float(np.mean([r["test"][k] for r in runs]))
                   for k in ["nll", "ece", "mean_conf", "mean_conf_correct",
                             "mean_conf_wrong", "frac_wrong_conf_gt_0.9"]},
                "per_class_acc": np.mean([r["test"]["per_class_acc"] for r in runs], 0).tolist(),
            }
    json.dump(table, open(os.path.join(RES, "table.json"), "w"), indent=1)
    print(f"{'model':18s} {'params':>9s} {'seeds':>5s} {'test acc':>14s} {'train@best':>10s} "
          f"{'NLL':>6s} {'ECE':>6s} {'conf':>5s} {'conf_ok':>7s} {'conf_err':>8s}")
    for k, v in table.items():
        print(f"{k:18s} {v['params']:9,d} {v['n_seeds']:5d} {100*v['test_mean']:7.2f}±{100*v['test_std']:4.2f}"
              f" {100*v['train_at_best_mean']:10.1f} {v['nll']:6.3f} {v['ece']:6.3f} {v['mean_conf']:5.3f}"
              f" {v['mean_conf_correct']:7.3f} {v['mean_conf_wrong']:8.3f}")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for kind in ["mlp", "cnn"]:
        r = [r for r in load(kind, "clean") if r["seed"] == 0 and abs(r["lr"] - lrs[kind]) < 1e-12][0]
        ep = [h["epoch"] for h in r["history"]]
        ax.plot(ep, [h["train_acc"] for h in r["history"]], "--", color=COL[kind], label=f"{NAME[kind]} train")
        ax.plot(ep, [h["val_acc"] for h in r["history"]], "-", color=COL[kind], label=f"{NAME[kind]} val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q2_clean_curves.pdf"))
    fig.savefig(os.path.join(FIG, "q2_clean_curves.png"), dpi=150)

    labels = ["clean", "block 16", "block 8", "block 4", "pixel perm."]
    order = ["clean", "block16", "block8", "block4", "permuted"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    x = np.arange(len(order))
    for off, kind in [(-0.2, "mlp"), (0.2, "cnn")]:
        m = [table[f"{kind}_{c}"]["test_mean"] for c in order]
        s = [table[f"{kind}_{c}"]["test_std"] for c in order]
        axes[0].bar(x + off, m, 0.4, yerr=s, color=COL[kind], label=NAME[kind])
        axes[1].plot(x, [table[f"{kind}_{c}"]["mean_conf_correct"] for c in order], "o-",
                     color=COL[kind], label=f"{NAME[kind]} correct")
        axes[1].plot(x, [table[f"{kind}_{c}"]["mean_conf_wrong"] for c in order], "s--",
                     color=COL[kind], label=f"{NAME[kind]} wrong")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=20); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test accuracy"); axes[0].legend(fontsize=7)
    axes[1].set_ylabel("Mean max-softmax"); axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q2_conditions.pdf"))
    fig.savefig(os.path.join(FIG, "q2_conditions.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for c, ls in zip(["clean", "block8", "block4", "permuted"], ["-", "--", "-.", ":"]):
        ax.plot(table[f"cnn_{c}"]["per_class_acc"], ls, marker="o", ms=3, label=c)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=60, fontsize=6)
    ax.set_ylabel("CNN test accuracy"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q2_cnn_per_class.pdf"))
    fig.savefig(os.path.join(FIG, "q2_cnn_per_class.png"), dpi=150)


if __name__ == "__main__":
    main()
