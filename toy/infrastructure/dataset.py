from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class ArtificialDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self._x = x
        self._y = y
        self.switches = np.zeros(self._x.shape[0])
        self.prev_pred = np.zeros(self._x.shape[0]) - 1

    def __getitem__(self, idx):
        sample = {'x': self._x[idx],
                  'y': self._y[idx],
                  'idx': idx}
        return sample

    def __len__(self):
        return self._x.shape[0]