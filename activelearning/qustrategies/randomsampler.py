import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


class RandomSampling(Sampler):
    '''Class for random sampling algorithm. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(RandomSampling, self).__init__(n_pool, start_idxs, cfg)

    # def action(self, trainer):
    #     print('Tracking switches')
    #     _ = trainer.get_unlabeled_statistics(0)

    def query(self, n: int, trainer):
        '''Performs random query of points'''
        inds = np.where(self.total_pool == 0)[0]
        return inds[np.random.permutation(len(inds))][:n]