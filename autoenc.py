import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np


class ActivationDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, dataset_dir, dtype):
        super().__init__()
        self.split = split
        self.dataset_dir = dataset_dir
        self.dtype = dtype

    def __iter__(self):
        shard_filenames = sorted(glob.glob(f"{self.dataset_dir}/batch_*.npy"))
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )
        assert len(shard_filenames) > 0, "No data found"
        while True:
            random.shuffle(shard_filenames)
            for shard in shard_filenames:
                print(f"Loading shard {shard}")
                m = np.lib.format.open_memmap(shard, dtype=np.float16, mode="r")
                num_entries = len(m)
                assert num_entries > 0
                ixs = list(range(num_entries))
                random.shuffle(ixs)
                for i in ixs:
                    yield torch.from_numpy(m[i].astype(np.float16)).type(self.dtype)


class Task:
    @staticmethod
    def iter_batches(batch_size, device, **dataset_kwargs):
        ds = ActivationDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=0,
        )

        for x in dl:
            x = x.to(device, non_blocking=True)
            yield x


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, l1: float):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1 = l1
        self.bias_dec = nn.Parameter(torch.zeros(input_dim))
        self.w_enc = nn.Linear(input_dim, hidden_dim)
        self.w_dec = nn.Linear(hidden_dim, input_dim, bias=False)

        self.last_loss = None

    def encode(self, x: torch.Tensor):
        x_center = x - self.bias_dec
        f = F.relu(self.w_enc(x_center))
        return f

    def decode(self, f: torch.Tensor):
        x_pred = self.w_dec(f) + self.bias_dec
        return x_pred

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_pred = self.decode(f)

        self.last_loss = F.mse_loss(x_pred, x, reduction="mean")
        self.last_loss += self.l1 * torch.mean(torch.norm(f, p=1, dim=1))

        return x_pred
