import json
import os
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from common.data import load_cifar, get_device, SEED
from Q3.models import ConvAE
from Q3.q3 import encode

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")
CS = [0.01, 0.1, 1, 10, 100]
KERNELS = {
    "linear": dict(kernel="linear"),
    "rbf": dict(kernel="rbf", gamma="scale"),
    "poly3": dict(kernel="poly", degree=3, gamma="scale", coef0=1),
}


def get_features():
    path = os.path.join(OUT, "features.npz")
    if os.path.exists(path):
        f = np.load(path, allow_pickle=True)
        return {k: f[k] for k in f.files}
    q3 = json.load(open(os.path.join(ROOT, "Q3", "results", "summary.json")))
    cands = {k: v for k, v in q3["c"].items() if k.startswith(("ae_", "dae_"))}
    name = max(cands, key=lambda k: cands[k]["linear_probe"]["val"])
    d = int(name.split("_d")[1])
    dev = get_device()
    ae = ConvAE(d).to(dev)
    ae.load_state_dict(torch.load(os.path.join(ROOT, "Q3", "checkpoints", f"{name}.pt"),
                                  map_location=dev))
    data = load_cifar(dev)
    mean, std = data["mean"].to(dev), data["std"].to(dev)
    out = {"name": np.array(name)}
    for s in ["train", "val", "test"]:
        out[f"f_{s}"] = encode(ae, data[f"x_{s}"] * std + mean).cpu().numpy()
        out[f"y_{s}"] = data[f"y_{s}"].cpu().numpy()
    sub, _ = train_test_split(np.arange(len(out["y_train"])), train_size=10000,
                              stratify=out["y_train"], random_state=SEED)
    out["svm_idx"] = np.sort(sub)
    np.savez(path, **out)
    print("representation:", name, {k: v.shape for k, v in out.items()}, flush=True)
    return out


def fit_one(kname, C, Xtr, ytr, Xva, yva):
    t = time.time()
    clf = SVC(C=C, cache_size=2000, **KERNELS[kname]).fit(Xtr, ytr)
    return {"kernel": kname, "C": C,
            "train_acc": float((clf.predict(Xtr) == ytr).mean()),
            "val_acc": float((clf.predict(Xva) == yva).mean()),
            "sv_frac": float(clf.n_support_.sum() / len(ytr)),
            "fit_seconds": time.time() - t}


def main():
    os.makedirs(OUT, exist_ok=True)
    F = get_features()
    idx = F["svm_idx"]
    mu, sd = F["f_train"][idx].mean(0), F["f_train"][idx].std(0) + 1e-6
    Z = {s: (F[f"f_{s}"] - mu) / sd for s in ["train", "val", "test"]}
    Xtr, ytr = Z["train"][idx], F["y_train"][idx]
    Xva, yva, Xte, yte = Z["val"], F["y_val"], Z["test"], F["y_test"]
    summary = {"representation": str(F["name"]), "dim": int(Xtr.shape[1]),
               "svm_train_subset": int(len(idx)),
               "svm_subset_class_counts": np.bincount(ytr).tolist()}

    sweep_path = os.path.join(OUT, "svm_sweep.json")
    if os.path.exists(sweep_path):
        sweep = json.load(open(sweep_path))
    else:
        sweep = Parallel(n_jobs=8, verbose=5)(
            delayed(fit_one)(k, C, Xtr, ytr, Xva, yva) for k in KERNELS for C in CS)
        json.dump(sweep, open(sweep_path, "w"), indent=1)
    for r in sweep:
        print(f"{r['kernel']:7s} C={r['C']:<6g} train {r['train_acc']:.4f} "
              f"val {r['val_acc']:.4f} SV {r['sv_frac']:.3f}", flush=True)
    best = max(sweep, key=lambda r: r["val_acc"])
    clf = SVC(C=best["C"], cache_size=2000, **KERNELS[best["kernel"]]).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (yte, pred), 1)
    summary["a"] = {"selected": best, "test_acc": float((pred == yte).mean()),
                    "test_confusion": cm.tolist()}
    print("selected", best["kernel"], best["C"], "test acc", summary["a"]["test_acc"], flush=True)

    sym = cm + cm.T
    np.fill_diagonal(sym, 0)
    pairs = []
    for _ in range(3):
        i, j = np.unravel_index(sym.argmax(), sym.shape)
        pairs.append((int(min(i, j)), int(max(i, j))))
        sym[i, j] = sym[j, i] = 0
    b = []
    for i, j in pairs:
        m_tr = np.isin(ytr, [i, j])
        bc = SVC(C=best["C"], cache_size=2000, **KERNELS[best["kernel"]])
        bc.fit(Xtr[m_tr], (ytr[m_tr] == j).astype(int))
        tr_global = idx[m_tr]
        sv_local = bc.support_
        sv_scores = bc.decision_function(Xtr[m_tr][sv_local])
        sv_order = np.argsort(np.abs(sv_scores))
        m_te = np.isin(yte, [i, j])
        te_idx = np.where(m_te)[0]
        te_scores = bc.decision_function(Xte[m_te])
        te_order = np.argsort(np.abs(te_scores))
        A, B = Xtr[ytr == i], Xtr[ytr == j]
        centre_dist = float(np.linalg.norm(A.mean(0) - B.mean(0)))
        spread = float(np.sqrt(((A - A.mean(0)) ** 2).sum(1).mean() / 2
                               + ((B - B.mean(0)) ** 2).sum(1).mean() / 2))
        b.append({
            "pair": [i, j],
            "multiclass_test_confusions": int(cm[i, j] + cm[j, i]),
            "multiclass_pair_confusion_rate": float((cm[i, j] + cm[j, i]) /
                                                    (cm[i].sum() + cm[j].sum())),
            "binary_test_acc": float(((te_scores > 0).astype(int) ==
                                      (yte[m_te] == j)).mean()),
            "binary_sv_frac": float(len(sv_local) / m_tr.sum()),
            "binary_frac_test_within_margin": float((np.abs(te_scores) < 1).mean()),
            "centroid_distance": centre_dist, "rms_within_class_spread": spread,
            "separation_ratio": centre_dist / spread,
            "sv_train_idx": tr_global[sv_local[sv_order[:8]]].tolist(),
            "sv_labels": ytr[m_tr][sv_local[sv_order[:8]]].tolist(),
            "sv_scores": sv_scores[sv_order[:8]].tolist(),
            "near_test_idx": te_idx[te_order[:8]].tolist(),
            "near_test_labels": yte[te_idx[te_order[:8]]].tolist(),
            "near_test_scores": te_scores[te_order[:8]].tolist(),
            "near_test_multiclass_pred": pred[te_idx[te_order[:8]]].tolist(),
        })
        print("pair", i, j, {k: v for k, v in b[-1].items() if not isinstance(v, list)}, flush=True)
    summary["b"] = b

    Xall = Z["train"]
    sel = []
    for method in ["kmeans", "gmm"]:
        for k in [5, 10, 15, 20]:
            if method == "kmeans":
                mdl = KMeans(k, n_init=5, random_state=0).fit(Xall)
            else:
                mdl = GaussianMixture(k, covariance_type="diag", random_state=0,
                                      max_iter=200).fit(Xall)
            pv = mdl.predict(Xva)
            r = {"method": method, "k": k,
                 "val_nmi": float(normalized_mutual_info_score(yva, pv)),
                 "val_ari": float(adjusted_rand_score(yva, pv))}
            sel.append(r)
            print("cluster", r, flush=True)
    bestc = max(sel, key=lambda r: r["val_nmi"])
    if bestc["method"] == "kmeans":
        mdl = KMeans(bestc["k"], n_init=5, random_state=0).fit(Xall)
    else:
        mdl = GaussianMixture(bestc["k"], covariance_type="diag", random_state=0,
                              max_iter=200).fit(Xall)
    pc = mdl.predict(Xte)
    cont = np.zeros((bestc["k"], 10), dtype=int)
    np.add.at(cont, (pc, yte), 1)
    pcls = cont / cont.sum(0, keepdims=True)
    pair_table = []
    for i in range(10):
        for j in range(i + 1, 10):
            pair_table.append({
                "pair": [i, j],
                "svm_confusion_rate": float((cm[i, j] + cm[j, i]) / (cm[i].sum() + cm[j].sum())),
                "cluster_overlap": float(np.minimum(pcls[:, i], pcls[:, j]).sum()),
            })
    purity_cls = (cont.max(0) / cont.sum(0)).tolist()
    summary["c"] = {"selection": sel, "selected": bestc,
                    "test_ari": float(adjusted_rand_score(yte, pc)),
                    "test_nmi": float(normalized_mutual_info_score(yte, pc)),
                    "contingency": cont.tolist(),
                    "svm_per_class_recall": (cm.diagonal() / cm.sum(1)).tolist(),
                    "class_main_cluster_share": purity_cls,
                    "pair_table": pair_table}
    np.save(os.path.join(OUT, "test_clusters.npy"), pc)
    np.save(os.path.join(OUT, "test_svm_pred.npy"), pred)
    print("clustering test ARI", summary["c"]["test_ari"], "NMI", summary["c"]["test_nmi"], flush=True)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
