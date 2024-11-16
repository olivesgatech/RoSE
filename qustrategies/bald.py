import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig
import scipy
import math


def entropy(probabilities: np.ndarray, axis=1):
    # get max entropy
    logs = np.log2(probabilities)
    mult = - logs * probabilities
    entr = np.sum(mult, axis=axis)
    entr = np.nan_to_num(entr)
    return entr


def bald(logits_b_k_c):
    mean_probs = np.mean(logits_b_k_c, axis=1)
    total_unc = entropy(mean_probs, axis=1)
    var_unc = entropy(logits_b_k_c, axis=2)
    var_unc = np.mean(var_unc, axis=1)
    mi = total_unc - var_unc
    mi = np.nan_to_num(mi)
    return mi

class BALDSampler(Sampler):
    '''Class for sampling the highest entropy. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        '''Constructor implemented in sampler'''
        super(BALDSampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        '''Returns samples with highest entropy in the output distribution.
        Parameters:
            :param probs: datastructure containing the sigmoid probabilities and the index list
            :type probs: dict
            :param n: number of samples to be queried
            :type n: int'''
        # get probabilities and their indices
        print('Sampling MCD Probs')
        unl_dict = trainer.get_unlabeled_mcd(0)
        mcd_logits, indices = unl_dict['mcd_logits'], unl_dict['indices']

        if 'data' in unl_dict:
            self._data = unl_dict['data']
        print(mcd_logits[4, :, 4])
        balent = bald(mcd_logits)
        print(balent)
        prob_inds = np.argsort(balent)[-n:]

        # derive final indices
        inds = indices[prob_inds]

        return inds
