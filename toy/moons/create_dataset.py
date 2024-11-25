from typing import Tuple
from torch.nn import functional as F

import torch
import numpy as np
import sklearn.datasets
from numpy import pi
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from toy.infrastructure.dataset import ArtificialDataset


def create_moon_dataset(num_samples: int = 1000, noise: float = 2.0, plot: bool = False) -> Tuple[Dataset, np.ndarray, np.ndarray]:

    data, target = sklearn.datasets.make_moons(n_samples=num_samples, noise=noise)
    if plot:
        plt.scatter(data[:, 0], data[:, 1], c=target)
        plt.show()
    dataset = ArtificialDataset(x=torch.from_numpy(data).float(), y=F.one_hot(torch.from_numpy(target)).float())
    return dataset, data, target


if __name__ == '__main__':
    _ = create_moon_dataset(noise=0.2, plot=True)
