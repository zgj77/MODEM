from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def load_array(root, entity, suffix):
    root = Path(root)
    for ext in (".npy", ".pkl"):
        path = root / f"{entity}_{suffix}{ext}"
        if path.exists():
            if ext == ".npy":
                return np.load(path, allow_pickle=False)
            with path.open("rb") as handle:
                return np.asarray(pickle.load(handle))
    raise FileNotFoundError(f"Missing {root / (entity + '_' + suffix)}.npy (or .pkl). "
                            "Use scripts/prepare_data.py or provide local data.")


def load_entity(root, entity):
    train = np.asarray(load_array(root, entity, "train"), dtype=np.float32)
    test = np.asarray(load_array(root, entity, "test"), dtype=np.float32)
    labels = np.asarray(load_array(root, entity, "test_label")).reshape(-1)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Train and test must be [time, features], with the same features")
    if len(test) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Test labels must be binary and match the entire test series")
    if not len(train) or not len(test) or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Data must be nonempty and finite; handle missing data explicitly before training")
    return train, test, labels.astype(np.int64)


def fit_scaler(train):
    low = train.min(0)
    span = train.max(0) - low
    return {"low": low.tolist(), "span": np.where(span > 1e-8, span, 1).tolist()}


def scale(data, scaler):

    return ((data - np.asarray(scaler["low"], dtype=np.float32)) /
            np.asarray(scaler["span"], dtype=np.float32)).astype(np.float32)


class Windows(Dataset):
    def __init__(self, data, length, stride, cover_tail=False):
        if length < 2 or stride < 1 or stride > length:
            raise ValueError("Require length >= 2 and 1 <= stride <= length")
        if not len(data):
            raise ValueError("Cannot window an empty series")
        self.data = torch.from_numpy(np.ascontiguousarray(data.T))
        self.length = length
        self.starts = list(range(0, max(len(data) - length + 1, 1), stride))
        if cover_tail and self.starts[-1] + length < len(data):
            self.starts.append(len(data) - length)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        start = self.starts[i]
        x = self.data[:, start:start + self.length]
        valid = x.shape[-1]
        if valid < self.length:
            x = torch.cat([x, x[:, -1:].expand(-1, self.length - valid)], -1)
        return x, start, valid


def complementary_mask(x, block):
    if block < 1 or block >= x.shape[-1]:
        raise ValueError("mask_block must be >= 1 and < window length")
    pattern = ((torch.arange(x.shape[-1], device=x.device) // block) % 2).float()
    return pattern[None, None].expand_as(x)
