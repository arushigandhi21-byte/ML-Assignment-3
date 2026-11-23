import os
import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SEED = 242001
SPLIT_FILE = os.path.join(DATA_DIR, f"split_{SEED}.npz")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_raw():
    npz = os.path.join(DATA_DIR, "cifar10.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    tr = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
    te = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
    return (tr.data, np.array(tr.targets, dtype=np.int64),
            te.data, np.array(te.targets, dtype=np.int64))


def get_split(y_train):
    if os.path.exists(SPLIT_FILE):
        s = np.load(SPLIT_FILE)
        return s["train_idx"], s["val_idx"]
    idx = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        idx, test_size=5000, stratify=y_train, random_state=SEED)
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
    np.savez(SPLIT_FILE, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


def load_cifar(device=None):
    device = device or get_device()
    xtr, ytr, xte, yte = load_raw()
    train_idx, val_idx = get_split(ytr)

    def to_t(x):
        return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255.0)

    x_train, x_val, x_test = to_t(xtr[train_idx]), to_t(xtr[val_idx]), to_t(xte)
    mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
    std = x_train.std(dim=(0, 2, 3), keepdim=True)
    norm = lambda x: ((x - mean) / std).to(device)

    return {
        "x_train": norm(x_train), "y_train": torch.from_numpy(ytr[train_idx]).to(device),
        "x_val": norm(x_val), "y_val": torch.from_numpy(ytr[val_idx]).to(device),
        "x_test": norm(x_test), "y_test": torch.from_numpy(yte).to(device),
        "mean": mean, "std": std,
    }


def fixed_pixel_permutation(seed=SEED, n_pixels=32 * 32):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_pixels, generator=g)


def apply_permutation(x, perm):
    n, c, h, w = x.shape
    return x.reshape(n, c, h * w)[:, :, perm.to(x.device)].reshape(n, c, h, w)


def batches(x, y, batch_size, shuffle=True, generator=None):
    n = x.shape[0]
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    order = order.to(x.device)
    for i in range(0, n, batch_size):
        b = order[i:i + batch_size]
        yield x[b], y[b]


def block_shuffle(x, k, generator=None):
    n, c, h, w = x.shape
    g = h // k
    p = x.reshape(n, c, g, k, g, k).permute(0, 2, 4, 1, 3, 5).reshape(n, g * g, c, k, k)
    if generator is None:
        order = torch.rand(n, g * g, device=x.device).argsort(1)
    else:
        order = torch.rand(n, g * g, generator=generator).argsort(1).to(x.device)
    p = p[torch.arange(n, device=x.device)[:, None], order]
    return p.reshape(n, g, g, c, k, k).permute(0, 3, 1, 4, 2, 5).reshape(n, c, h, w)
