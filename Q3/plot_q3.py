import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")


def main():
    os.makedirs(FIG, exist_ok=True)
    s = json.load(open(os.path.join(RES, "summary.json")))

    print("(a) bottleneck table")
    for d, v in s["a"]["table"].items():
        print(f"  d={d:>5s} params={v['params']:,} val_mse={v['val_mse']:.5f} best_ep={v['best_epoch']}")
    print(f"  selected d={s['a']['selected_d']} test_mse={s['a']['selected_test_mse']:.5f}")
    print("(b) fixed CNN classifier on test")
    for k, v in s["b"]["results"].items():
        print(f"  {k:22s} acc={v['acc']:.4f} mse_to_clean={v['mse_to_clean']:.5f} "
              f"conf={v['mean_conf']:.3f} nll={v['nll']:.3f}")
    print("(c) class information vs reconstruction")
    for k, v in s["c"].items():
        lp = v["linear_probe"]
        print(f"  {k:14s} dim={v['code_dim']:5d} test_mse={v['test_mse']:.5f} "
              f"probe train/val/test={lp['train']:.3f}/{lp['val']:.3f}/{lp['test']:.3f} "
              f"knn test={v['knn_test']:.3f}")

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for d in s["dims"]:
        h = json.load(open(os.path.join(RES, f"ae_d{d}_history.json")))
        ax.plot([r["epoch"] for r in h], [r["val_mse"] for r in h], label=f"d={d} val")
        ax.plot([r["epoch"] for r in h], [r["train_mse"] for r in h], "--", alpha=0.5,
                color=ax.lines[-1].get_color(), label=f"d={d} train")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q3_ae_curves.pdf"))
    fig.savefig(os.path.join(FIG, "q3_ae_curves.png"), dpi=150)

    ex = torch.load(os.path.join(RES, "examples.pt"))
    rows = ["clean", "noisy", "ae_recon_of_noisy", "dae_recon_of_noisy"]
    names = ["Clean", "Noisy", "Clean-AE(noisy)", "DAE(noisy)"]
    fig, axes = plt.subplots(len(rows), 8, figsize=(7, 3.6))
    for r, key in enumerate(rows):
        for c in range(8):
            axes[r, c].imshow(ex[key][c].permute(1, 2, 0).clamp(0, 1).numpy())
            axes[r, c].axis("off")
        axes[r, 0].set_title(names[r], fontsize=7, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q3_denoise_examples.pdf"))
    fig.savefig(os.path.join(FIG, "q3_denoise_examples.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for k, v in s["c"].items():
        if k in ("raw_pixels", "random_d1024"):
            continue
        ax.scatter(v["test_mse"], v["linear_probe"]["test"], color="tab:blue")
        ax.scatter(v["test_mse"], v["knn_test"], color="tab:orange", marker="s")
        ax.annotate(k, (v["test_mse"], v["linear_probe"]["test"]), fontsize=6,
                    xytext=(3, -9 if k == "ae_d1024" else 4), textcoords="offset points")
    raw = s["c"]["raw_pixels"]
    ax.axhline(raw["linear_probe"]["test"], color="tab:blue", ls=":", lw=1, label="raw pixels, linear")
    ax.axhline(raw["knn_test"], color="tab:orange", ls=":", lw=1, label="raw pixels, k-NN")
    ax.scatter([], [], color="tab:blue", label="linear probe")
    ax.scatter([], [], color="tab:orange", marker="s", label="k-NN (k=20)")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.003, 0.005, 0.01, 0.016])
    ax.set_xticklabels(["0.003", "0.005", "0.01", "0.016"])
    ax.set_xlabel("Test reconstruction MSE (clean input)")
    ax.set_ylabel("Test accuracy from code"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q3_recon_vs_probe.pdf"))
    fig.savefig(os.path.join(FIG, "q3_recon_vs_probe.png"), dpi=150)


if __name__ == "__main__":
    main()
