import numpy as np

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def confidence_stats(probs, y):
    probs, y = np.asarray(probs), np.asarray(y)
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == y
    nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return {
        "acc": float(correct.mean()),
        "nll": float(nll),
        "ece": ece(conf, correct),
        "mean_conf": float(conf.mean()),
        "mean_conf_correct": float(conf[correct].mean()),
        "mean_conf_wrong": float(conf[~correct].mean()),
        "frac_wrong_conf_gt_0.9": float((conf[~correct] > 0.9).mean()),
        "per_class_acc": (cm.diagonal() / cm.sum(1)).tolist(),
        "confusion": cm.tolist(),
    }
