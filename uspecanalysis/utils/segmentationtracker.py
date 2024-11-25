import os
import numpy as np
from config import BaseConfig


def diffcount(A: np.ndarray, axis: int):
    B=A.copy()
    B.sort(axis=axis)
    C=np.diff(B,axis=axis)>0
    D=C.sum(axis=axis)+1
    return D


class USPECTracker:
    def __init__(self, set_shape: tuple, num_seeds: int):
        self._num_seeds = num_seeds
        self._predictions = np.squeeze(np.zeros((num_seeds,) + set_shape))
        self._seen = {}

    def update(self, preds: np.ndarray, seed: int):
        if seed in self._seen.keys():
            raise Exception('Seed has already been run!')

        self._seen[seed] = True
        print(self._predictions.shape)
        self._predictions[seed, ...] = preds
        return

    def save_statistics(self, directory: str, ld_type: str, cfg: BaseConfig):
        path = directory + '/uspec_statistics/' + ld_type
        if not os.path.exists(path):
            os.makedirs(path)

        if cfg.uspec_configs.segmentation.save_predictions:
            np.save(path + '/predictions.npy', self._predictions)

        if cfg.uspec_configs.segmentation.save_uspec_mean:
            mean = diffcount(self._predictions, axis=0)
            np.save(path + '/uspec-sum.npy', mean)
