# ML Assignment 3: MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR-10

Arushi Gandhi (M24MA2001)

## Environment
Python 3.12.10 on macOS (Apple M5, PyTorch MPS backend). Exact package versions are in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Data and split
- CIFAR-10, downloaded with `python -m common.fetch_cifar10` into `data/cifar10.npz`.
  The official Toronto server was too slow, so the files come from the `uoft-cs/cifar10`
  mirror on Hugging Face. The training images are in the same order as the official
  python archive (checked by matching the raw bytes of `data_batch_1`), so
  `torchvision.datasets.CIFAR10` gives the same split. `common/data.py` falls back to
  torchvision if the `.npz` file is missing.
- Split: stratified 45,000 / 5,000 with `random_state=242001` (the digits of the roll
  number, in order). Saved in `data/split_242001.npz` (`train_idx`, `val_idx` index
  the 50,000 official training images). Validation has 500 images per class.
- Normalisation statistics are computed on the 45,000 training images only.
- The official test set is used only for the final evaluation of models chosen on validation.

## Reproduction commands
Run from the repository root, in order (later questions reuse earlier checkpoints).

```bash
python -m common.fetch_cifar10

python -m Q1.q1a
python -m Q1.plot_q1a
python -m Q1.q1b
python -m Q1.plot_q1b

python -m Q2.q2
python -m Q2.plot_q2

python -m Q3.q3
python -m Q3.plot_q3

python -m Q4.q4
python -m Q4.plot_q4
```

| Command | What it does |
|---|---|
| `common.fetch_cifar10` | Downloads CIFAR-10 to `data/cifar10.npz` (the split is created on first use) |
| `Q1.q1a` | Q1(a): 2x2 MLP comparison, LR sweep and 3 seeds |
| `Q1.q1b` | Q1(b): 40% label-noise memorisation run and 0% control |
| `Q2.q2` | Q2(a-c): CNN vs MLP on clean, pixel-permuted and block-shuffled images |
| `Q3.q3` | Q3(a-c): autoencoders, denoising autoencoder, probes (uses the Q2 CNN) |
| `Q4.q4` | Q4(a-c): SVM sweep, difficult pairs, clustering (uses the Q3 encoders) |
| `Q*.plot_*` | Figures and printed tables for each question |

Approximate run times on the Apple M5: Q1 about 15 min, Q2 about 90 min, Q3 about 30 min,
Q4 about 30-60 min. Results are written to `Q*/results/`, figures to `Q*/figures/`, and
the console logs of the final runs are in `Q*/*_log.txt`. Model checkpoints (`*.pt`)
are not committed; the scripts recreate them.

## Layout
| Path | Contents |
|---|---|
| `common/data.py` | loading, saved split, normalisation, pixel permutation, block shuffle |
| `common/train.py` | shared classifier training loop (AdamW, cosine LR, best-val checkpoint) |
| `common/analysis.py` | accuracy, NLL, ECE, confidence of correct/wrong predictions, confusion |
| `Q1/` | MLP definitions, Q1(a) and Q1(b) experiments |
| `Q2/` | CNN definition and all Q2 conditions |
| `Q3/` | convolutional autoencoder, denoising, linear and k-NN probes |
| `Q4/` | SVM sweep, difficult-pair analysis, clustering |

## Reuse across questions
- Pixel permutation (`common.data.fixed_pixel_permutation`): introduced in Q1(a), reused in Q2(b).
- Shallow-wide MLP and its learning rate (1e-3): introduced in Q1(a), reused in Q1(b) (LR only) and Q2.
- Clean CNN (seed 0) from Q2(a): reused as the fixed classifier in Q3(b).
- Autoencoder encoders from Q3: one is reused as the fixed representation in Q4.
