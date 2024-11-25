import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig
import scipy
import math


def beta_function(x, y):
    return np.exp(scipy.special.loggamma(x) + scipy.special.loggamma(y) - scipy.special.loggamma(x+y)) + 1e-32


def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.

    We pass in logits.
    """
    return scipy.special.logsumexp(logits, axis=dim, keepdims=keepdim) - math.log(logits.shape[dim])


def unlogit_meanvar(logits, dim: int, keepdim: bool = False):
    r"""Computes mean & variance

    We pass in logits.
    """
    unlogit_ave = np.exp(logit_mean(logits, dim, keepdim))
    unlogit_var = np.var(np.exp(logits), axis=dim, keepdims=keepdim)

    return unlogit_ave, unlogit_var


def entropy(probabilities: np.ndarray):
    # get max entropy
    logs = np.log2(probabilities)
    mult = - logs * probabilities
    entr = np.sum(mult, axis=1)
    entr = np.nan_to_num(entr)
    return entr


def marginalized_posterior_entropy(logits_B_K_C):
    print(logits_B_K_C.shape)
    unlogits_mean_B_C, unlogits_var_B_C = unlogit_meanvar(logits_B_K_C, dim=1)
    idx = unlogits_mean_B_C < 1e-9
    unlogits_mean_B_C[idx] = 1e-9
    idx = unlogits_var_B_C < 1e-9
    unlogits_var_B_C[idx] = 1e-9  #

    all_alpha = unlogits_mean_B_C * unlogits_mean_B_C * (1 - unlogits_mean_B_C) / unlogits_var_B_C - unlogits_mean_B_C
    all_beta = (1 / unlogits_mean_B_C - 1) * all_alpha

    hpp = np.log(beta_function(all_alpha + 1, all_beta)) - all_alpha * scipy.special.digamma(all_alpha + 1) - (
                all_beta - 1) * scipy.special.digamma(all_beta) + (all_alpha + all_beta - 1) * scipy.special.digamma(
        all_alpha + all_beta + 1)
    all_beta_entropy = unlogits_mean_B_C * hpp
    idx = all_beta_entropy > 0
    all_beta_entropy[idx] = 0
    balent_information_B = np.sum(all_beta_entropy, axis=1)
    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C)
    balent_information_B += mean_entropy_B

    if np.isnan(np.sum(balent_information_B)):
        balent_information_B[np.where(np.isnan(balent_information_B))] = -99999999999
    print(balent_information_B.shape)

    return balent_information_B

def balentacq(logits_B_K_C):

    mjent = marginalized_posterior_entropy(logits_B_K_C)
    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C)

    balentacq_val = (mean_entropy_B+0.69314718056) / (mjent+mean_entropy_B)
    idx = balentacq_val < 0
    balentacq_val[idx] = 1/balentacq_val[idx]

    return balentacq_val


class BalEntSampler(Sampler):
    '''Class for sampling the highest entropy. Inherits from sampler.'''
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        '''Constructor implemented in sampler'''
        super(BalEntSampler, self).__init__(n_pool, start_idxs, cfg)

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
        print(mcd_logits.shape)

        if 'data' in unl_dict:
            self._data = unl_dict['data']
        balent = balentacq(mcd_logits)
        print(balent)
        prob_inds = np.argsort(balent)[-n:]

        # derive final indices
        inds = indices[prob_inds]

        return inds
