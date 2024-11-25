import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


def lconf(probabilities: np.ndarray, n: int):
    # get max probs
    probabilities = np.max(probabilities, axis=1)
    prob_inds = np.argsort(probabilities)[:n]

    return prob_inds


class LeastConfidenceSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(LeastConfidenceSampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Sampling Least Confidence Probs')
        unl_dict = trainer.get_unlabeled_statistics(0)
        probabilities, indices = unl_dict['probabilities'], unl_dict['indices']

        # get max probs
        prob_inds = lconf(probabilities, n)

        # derive final indices
        inds = indices[prob_inds]

        return inds
