import copy
import json
import time
import torch
import torch.nn.functional as F

from .data import batches


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    loss_sum, correct = 0.0, 0
    for xb, yb in batches(x, y, batch_size, shuffle=False):
        out = model(xb)
        loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return loss_sum / len(y), correct / len(y)


@torch.no_grad()
def predict_proba(model, x, batch_size=1000):
    model.eval()
    return torch.cat([F.softmax(model(x[i:i + batch_size]), 1)
                      for i in range(0, x.shape[0], batch_size)])


def train(model, data, epochs=30, lr=1e-3, weight_decay=5e-4, batch_size=256,
          seed=0, y_train=None, epoch_hook=None, log_path=None, verbose=True,
          train_eval_n=None, batch_transform=None, x_train_eval=None):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x_tr = data["x_train"]
    y_tr = data["y_train"] if y_train is None else y_train
    x_ev = x_tr if x_train_eval is None else x_train_eval
    y_ev = y_tr
    if train_eval_n is not None:
        sub = torch.randperm(len(y_tr), generator=torch.Generator().manual_seed(12345))
        sub = sub[:train_eval_n].to(x_tr.device)
        x_ev, y_ev = x_ev[sub], y_ev[sub]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * ((len(y_tr) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    history, best_acc, best_state = [], -1.0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for xb, yb in batches(x_tr, y_tr, batch_size, generator=gen):
            if batch_transform is not None:
                xb = batch_transform(xb)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        tr_loss, tr_acc = evaluate(model, x_ev, y_ev)
        va_loss, va_acc = evaluate(model, data["x_val"], data["y_val"])
        rec = {"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": va_loss, "val_acc": va_acc, "time": time.time() - t0}
        if epoch_hook is not None:
            rec.update(epoch_hook(model, ep))
        history.append(rec)
        if va_acc > best_acc:
            best_acc, best_state = va_acc, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  ep {ep:3d}  train {tr_acc:.3f}  val {va_acc:.3f}  "
                  f"({rec['time']:.1f}s)", flush=True)

    if log_path:
        with open(log_path, "w") as f:
            json.dump(history, f, indent=1)
    best = copy.deepcopy(model)
    best.load_state_dict(best_state)
    return history, best
