import os
import numpy as np
from config import BaseConfig


class Sampler:
    def __init__(self, n_pool: int, start_idxs: np.ndarray):
        # init idx list containing elements in AL pool
        self.idx_current = np.arange(n_pool)[start_idxs]

        # init list of total elements mapped to binary variables
        self.total_pool = np.zeros(n_pool, dtype=int)
        self.total_pool[self.idx_current] = 1
        self._data = {}

    def query(self, n: int, trainer):
        '''
        Pure virtual query function. Content implemented by other submodules
        Parameters:
            :param n: number of samples to be queried
            :type n: int
            :param trainer: active learning trainer object for classification or segmentation
            :type trainer: either ActiveLearningClassificationTrainer or ActiveLearningSegmentationTrainer
        '''
        pass

    def update(self, new_idx):
        '''
        Updates the current data pool with the newly queried idxs.
        Parameters:
            :param new_idx: idxs used for update
            :type new_idx: ndarray'''
        self.idx_current = np.append(self.idx_current, new_idx)
        self.total_pool[new_idx] = 1