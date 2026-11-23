import io
import os
import urllib.request
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .data import DATA_DIR

URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def convert(path):
    t = pq.read_table(path).to_pandas()
    x = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
    return x.astype(np.uint8), t["label"].to_numpy().astype(np.int64)


def main():
    os.makedirs(os.path.join(DATA_DIR, "hf"), exist_ok=True)
    out = {}
    for split in ["train", "test"]:
        path = os.path.join(DATA_DIR, "hf", f"{split}.parquet")
        if not os.path.exists(path):
            urllib.request.urlretrieve(URL.format(split), path)
        out[f"x_{split}"], out[f"y_{split}"] = convert(path)
    np.savez_compressed(os.path.join(DATA_DIR, "cifar10.npz"), **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
