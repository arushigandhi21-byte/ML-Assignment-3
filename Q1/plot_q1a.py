import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "results", "q1a")
FIG = os.path.join(os.path.dirname(__file__), "figures")
STYLE = {"shallow_wide": "tab:blue", "deep_narrow": "tab:orange"}
LABEL = {"shallow_wide": "Shallow-wide [512]", "deep_narrow": "Deep-narrow [350x5]"}


def load(arch, inp, key):
    a = []
    for s in range(3):
        h = json.load(open(os.path.join(RES, f"final_{arch}_{inp}_s{s}.json")))
        a.append([r[key] for r in h])
    return np.array(a)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, inp in zip(axes, ["original", "permuted"]):
        for arch, c in STYLE.items():
            for key, ls in [("train_acc", "--"), ("val_acc", "-")]:
                a = load(arch, inp, key)
                ep = np.arange(1, a.shape[1] + 1)
                ax.plot(ep, a.mean(0), ls, color=c,
                        label=f"{LABEL[arch]} {'train' if key == 'train_acc' else 'val'}")
                ax.fill_between(ep, a.min(0), a.max(0), color=c, alpha=0.15)
        ax.set_title(f"{inp.capitalize()} pixels")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q1a_curves.pdf"))
    fig.savefig(os.path.join(FIG, "q1a_curves.png"), dpi=150)


if __name__ == "__main__":
    main()
