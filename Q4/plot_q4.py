import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.analysis import CLASSES
from common.data import load_raw, get_split

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")


def main():
    os.makedirs(FIG, exist_ok=True)
    s = json.load(open(os.path.join(RES, "summary.json")))
    sweep = json.load(open(os.path.join(RES, "svm_sweep.json")))
    print("representation", s["representation"], "dim", s["dim"])
    print(f"{'kernel':7s} {'C':>6s} {'train':>6s} {'val':>6s} {'SV frac':>7s} {'fit s':>6s}")
    for r in sweep:
        print(f"{r['kernel']:7s} {r['C']:6g} {r['train_acc']:6.3f} {r['val_acc']:6.3f} "
              f"{r['sv_frac']:7.3f} {r['fit_seconds']:6.0f}")
    print("selected", s["a"]["selected"]["kernel"], s["a"]["selected"]["C"], "test", s["a"]["test_acc"])

    kernels = list(dict.fromkeys(r["kernel"] for r in sweep))
    fig, axes = plt.subplots(1, len(kernels), figsize=(7, 2.4), sharey=True)
    for ax, k in zip(axes, kernels):
        rs = [r for r in sweep if r["kernel"] == k]
        C = [r["C"] for r in rs]
        ax.plot(C, [r["train_acc"] for r in rs], "o--", color="tab:gray", label="train acc")
        ax.plot(C, [r["val_acc"] for r in rs], "o-", color="tab:blue", label="val acc")
        ax.plot(C, [r["sv_frac"] for r in rs], "s-", color="tab:red", label="SV fraction")
        ax.set_xscale("log"); ax.set_title(k, fontsize=8); ax.set_xlabel("C"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q4_sweep.pdf"))
    fig.savefig(os.path.join(FIG, "q4_sweep.png"), dpi=150)

    xtr, ytr, xte, yte = load_raw()
    tr_idx, _ = get_split(ytr)
    x45 = xtr[tr_idx]
    for p, b in enumerate(s["b"]):
        i, j = b["pair"]
        print(f"pair {CLASSES[i]}-{CLASSES[j]}: confusions {b['multiclass_test_confusions']}, "
              f"pair conf. rate {b['multiclass_pair_confusion_rate']:.3f}, binary acc {b['binary_test_acc']:.3f}, "
              f"binary SV frac {b['binary_sv_frac']:.3f}, test within margin {b['binary_frac_test_within_margin']:.3f}, "
              f"separation ratio {b['separation_ratio']:.3f}")
        fig, axes = plt.subplots(2, 8, figsize=(7, 2.2))
        for c in range(8):
            a = axes[0, c]
            a.imshow(x45[b["sv_train_idx"][c]]); a.axis("off")
            a.set_title(f"{CLASSES[b['sv_labels'][c]][:5]}\n{b['sv_scores'][c]:+.2f}", fontsize=6)
            a = axes[1, c]
            a.imshow(xte[b["near_test_idx"][c]]); a.axis("off")
            wrong = b["near_test_multiclass_pred"][c] != b["near_test_labels"][c]
            a.set_title(f"{CLASSES[b['near_test_labels'][c]][:5]}{' x' if wrong else ''}\n"
                        f"{b['near_test_scores'][c]:+.4f}", fontsize=6, color="red" if wrong else "black")
        axes[0, 0].text(-40, 16, "train SVs", fontsize=6, rotation=90, va="center")
        axes[1, 0].text(-40, 16, "test near\nboundary", fontsize=6, rotation=90, va="center")
        fig.suptitle(f"{CLASSES[i]} (score<0) vs {CLASSES[j]} (score>0)", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"q4_pair{p + 1}.pdf"))
        fig.savefig(os.path.join(FIG, f"q4_pair{p + 1}.png"), dpi=150)

    c = s["c"]
    print("clustering selected", c["selected"], "test ARI", round(c["test_ari"], 4), "NMI", round(c["test_nmi"], 4))
    print("contingency (rows = clusters, cols = classes)")
    print("     " + " ".join(f"{n[:5]:>5s}" for n in CLASSES))
    for k, row in enumerate(c["contingency"]):
        print(f"c{k:<3d} " + " ".join(f"{v:5d}" for v in row))
    pt = c["pair_table"]
    for r in sorted(pt, key=lambda r: r["cluster_overlap"] - r["svm_confusion_rate"], reverse=True)[:5]:
        i, j = r["pair"]
        print(f"  {CLASSES[i]}-{CLASSES[j]}: SVM confusion {r['svm_confusion_rate']:.3f}, "
              f"cluster overlap {r['cluster_overlap']:.3f}")

    cont = np.array(c["contingency"])
    fig, ax = plt.subplots(figsize=(3.4, 0.25 * cont.shape[0] + 1))
    ax.imshow(cont / cont.sum(0, keepdims=True), cmap="Blues", aspect="auto")
    for (r, k), v in np.ndenumerate(cont):
        ax.text(k, r, v, ha="center", va="center", fontsize=5)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=60, fontsize=6)
    ax.set_yticks(range(cont.shape[0])); ax.set_yticklabels([f"c{k}" for k in range(cont.shape[0])], fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q4_contingency.pdf"))
    fig.savefig(os.path.join(FIG, "q4_contingency.png"), dpi=150)


if __name__ == "__main__":
    main()
