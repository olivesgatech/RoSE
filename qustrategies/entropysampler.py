import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


def entropy(probabilities: np.ndarray, n: int):
    # get max entropy
    logs = np.log2(probabilities)
    mult = logs * probabilities
    entr = np.sum(mult, axis=1)
    entr = np.nan_to_num(entr)
    prob_inds = np.argsort(entr)[:n]

    return prob_inds


class EntropySampler(Sampler):
    '''Class for sampling the highest entropy. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        '''Constructor implemented in sampler'''
        super(EntropySampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        '''Returns samples with highest entropy in the output distribution.
        Parameters:
            :param probs: datastructure containing the sigmoid probabilities and the index list
            :type probs: dict
            :param n: number of samples to be queried
            :type n: int'''
        # get probabilities and their indices
        print('Sampling Entropy Probs')
        unl_dict = trainer.get_unlabeled_statistics(0)
        probabilities, indices = unl_dict['probabilities'], unl_dict['indices']

        if 'data' in unl_dict:
            self._data = unl_dict['data']

        prob_inds = entropy(probabilities, n)

        # derive final indices
        inds = indices[prob_inds]

        return inds
