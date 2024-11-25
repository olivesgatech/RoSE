from typing import Tuple
from torch.nn import functional as F
import math

import torch
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from toy.infrastructure.dataset import ArtificialDataset
from toy.spirals.spirals import generate_spiral


def create_spiral(theta: np.ndarray, max_noise: float, distance: float, num_samples: int):
    r_a = 2 * theta + distance
    x_noise = max_noise * r_a[:, np.newaxis] / np.max(r_a[:, np.newaxis])
    a_noise = np.concatenate((x_noise, x_noise), axis=1)
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(num_samples, 2) * a_noise

    return x_a


def create_spiral_dataset(num_spirals: int, spiral_loops: int = 2, num_samples: int = 1000, noise: float = 2.0,
                          plot: bool = False, add_aux_data: bool = False,
                          auto_angle: bool = True, angle: float = 0) -> Tuple[Dataset, np.ndarray, np.ndarray]:
    if num_spirals <= 0:
        raise ValueError(f'Number of spirals must be larger than zero')
    theta = np.sqrt(np.random.rand(num_samples)) * spiral_loops * 2 * pi
    data = None
    target = None
    angles = [((360 / num_spirals) if auto_angle else angle) * i for i in range(num_spirals)]
    for i, ang in enumerate(angles):
        distance = i * pi
        # sgn = 1 if i % 2 == 0 else -1
        cur_spiral = generate_spiral(samples=num_samples, start=0, end=360, angle=ang, noise=noise)
        # cur_spiral = create_spiral(theta, noise, distance, num_samples)
        cur_target = np.full(num_samples, fill_value=i)
        data = np.concatenate((data, cur_spiral), axis=0) if data is not None else cur_spiral
        target = np.concatenate((target, cur_target), axis=None) if target is not None else cur_target

        if plot:
            plt.scatter(cur_spiral[:, 0], cur_spiral[:, 1])
    if add_aux_data:
        n_ring = num_samples * 5
        noise_ring = noise * 3
        max_r = np.max(data)
        r_ring = np.sqrt(max_r**2 + max_r**2) + 5
        theta_ring = np.sqrt(np.random.rand(n_ring)) * 2 * pi
        data_ring = np.array([np.cos(theta_ring) * r_ring, np.sin(theta_ring) * r_ring]).T
        data_ring = data_ring + np.random.randn(n_ring, 2) * noise_ring
        target_ring = np.full(n_ring, fill_value=num_spirals)
        data = np.concatenate((data, data_ring), axis=0)
        target = np.concatenate((target, target_ring), axis=None)
        if plot:
            plt.scatter(data_ring[:, 0], data_ring[:, 1])

    if plot:
        plt.show()
    dataset = ArtificialDataset(x=torch.from_numpy(data).float(), y=F.one_hot(torch.from_numpy(target)).float())
    return dataset, data, target


if __name__ == '__main__':
    _ = create_spiral_dataset(num_spirals=3, noise=0.8, plot=True, add_aux_data=False)
