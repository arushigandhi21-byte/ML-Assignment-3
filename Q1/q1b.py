import json
import os

import numpy as np
import torch

from common.data import load_cifar, get_device, SEED
from common.train import train, evaluate, count_params
from Q1.models import MLP

OUT = os.path.join(os.path.dirname(__file__), "results", "q1b")
EPOCHS = 100
HIDDEN = [1024, 1024]
NOISE_RATES = [0.0, 0.4]


def make_noisy_labels(y, rate, seed=SEED):
    g = np.random.default_rng(seed)
    y = y.cpu().numpy().copy()
    n = len(y)
    flip = np.zeros(n, dtype=bool)
    flip[g.choice(n, int(round(rate * n)), replace=False)] = True
    offset = g.integers(1, 10, size=n)
    y_noisy = np.where(flip, (y + offset) % 10, y)
    return y_noisy, flip


@torch.no_grad()
def predict(model, x, bs=2000):
    model.eval()
    return torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)])


def run(data, rate, dev):
    y_true = data["y_train"]
    y_noisy_np, flip_np = make_noisy_labels(y_true, rate)
    y_noisy = torch.from_numpy(y_noisy_np).to(dev)
    flip = torch.from_numpy(flip_np).to(dev)

    def hook(model, ep):
        p = predict(model, data["x_train"])
        rec = {"clean_acc": (p[~flip] == y_true[~flip]).float().mean().item()}
        if flip.any():
            rec["noisy_fit"] = (p[flip] == y_noisy[flip]).float().mean().item()
            rec["noisy_true"] = (p[flip] == y_true[flip]).float().mean().item()
        return rec

    torch.manual_seed(0)
    model = MLP(HIDDEN).to(dev)
    tag = f"noise{int(rate * 100)}"
    print(tag, "params", count_params(model), "flipped", int(flip_np.sum()), flush=True)
    hist, best = train(model, data, epochs=EPOCHS, lr=1e-3, weight_decay=0.0, seed=0,
                       y_train=y_noisy, epoch_hook=hook,
                       log_path=os.path.join(OUT, f"history_{tag}.json"), verbose=False)
    best_ep = int(np.argmax([h["val_acc"] for h in hist])) + 1
    _, test_best = evaluate(best, data["x_test"], data["y_test"])
    _, test_final = evaluate(model, data["x_test"], data["y_test"])
    summary = {"noise_rate": rate, "hidden": HIDDEN, "params": count_params(model),
               "best_val_epoch": best_ep, "best_val_acc": hist[best_ep - 1]["val_acc"],
               "test_acc_best_val_ckpt": test_best, "test_acc_final": test_final,
               "at_best": hist[best_ep - 1], "final": hist[-1]}
    print(json.dumps({k: v for k, v in summary.items() if k not in ("at_best", "final")}),
          flush=True)
    return summary


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = get_device()
    data = load_cifar(dev)
    out = {f"noise{int(r * 100)}": run(data, r, dev) for r in NOISE_RATES}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
