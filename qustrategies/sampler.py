import os
import numpy as np
from config import BaseConfig


class Sampler:
    def __init__(self, n_pool: int, start_idxs: np.ndarray, cfg: BaseConfig):
        # init idx list containing elements in AL pool
        self.idx_current = np.arange(n_pool)[start_idxs]

        self._cfg = cfg

        # init list of total elements mapped to binary variables
        self.total_pool = np.zeros(n_pool, dtype=int)
        self.total_pool[self.idx_current] = 1
        self._data = {}

    def save_query_information(self, new_idxs: np.ndarray, round: int, seed: int):
        path = self._cfg.run_configs.ld_folder_name + '/round' + str(round) + '/queryidxs/'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + f'/newidxs_seed{seed}.npy', np.squeeze(new_idxs))

        if len(self._data) != 0:
            for i in range(new_idxs.shape[0]):
                if new_idxs[i] not in self._data:
                    raise ValueError(f'Idx {new_idxs[i]} is not in the dict even though it was queried!')
                for key, val in self._data[new_idxs[i]].items():
                    folder = f'{path}/{key}'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    np.save(f'{folder}/{new_idxs[i]}_{key}_seed{seed}.npy', val)

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

    def action(self, trainer):
        '''
        Action taken between model updates or epochs. ignored by most sampling strategies.
        :param trainer:
        :return:
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

    def track_statistics(self, trainer):
        if self._cfg.active_learning.save_switches:
            print('Tracking switches')
            _ = trainer.get_unlabeled_statistics(0)


    def batch_size(self, n):
        return n