import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "results", "q1b")
FIG = os.path.join(os.path.dirname(__file__), "figures")


def col(h, k):
    return np.array([r[k] for r in h])


def main():
    os.makedirs(FIG, exist_ok=True)
    h40 = json.load(open(os.path.join(RES, "history_noise40.json")))
    h0 = json.load(open(os.path.join(RES, "history_noise0.json")))
    ep = col(h40, "epoch")
    val = col(h40, "val_acc")
    fit, true = col(h40, "noisy_fit"), col(h40, "noisy_true")
    best = int(ep[np.argmax(val)])
    cross = int(ep[np.argmax(fit > true)])

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(ep, col(h40, "clean_acc"), color="tab:green", label="Train, clean subset")
    ax.plot(ep, fit, color="tab:red", label="Train, flipped (vs noisy label)")
    ax.plot(ep, true, color="tab:red", ls=":", label="Train, flipped (vs true label)")
    ax.plot(ep, val, color="tab:blue", label="Validation (40% noise run)")
    ax.plot(col(h0, "epoch"), col(h0, "val_acc"), color="tab:blue", ls="--", alpha=0.6,
            label="Validation (0% noise control)")
    ax.axvline(best, color="k", lw=0.8, ls="--", label=f"Validation peak (epoch {best})")
    ax.axvline(cross, color="gray", lw=0.8, ls=":", label=f"Noisy fit > true (epoch {cross})")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q1b_memorisation.pdf"))
    fig.savefig(os.path.join(FIG, "q1b_memorisation.png"), dpi=150)
    print("val peak epoch", best, "crossover epoch", cross)


if __name__ == "__main__":
    main()
