import numpy as np
from activelearning.qustrategies.sampler import Sampler
from activelearning.qustrategies.entropysampler import entropy
from activelearning.qustrategies.marginsampler import margin
from activelearning.qustrategies.lconfsampling import lconf
from config import BaseConfig


class IdealSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(IdealSampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Sampling Statistics')
        unl_dict = trainer.get_unlabeled_statistics(0)
        probabilities, indices, recon = unl_dict['specified probabilities'], unl_dict['indices'], \
                                        unl_dict['specified recon']

        if self._cfg.active_learning.stats.secondary_samping_type == 'entropy':
            target_inds = entropy(probabilities, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'margin':
            target_inds = margin(probabilities, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'lconf':
            target_inds = lconf(probabilities, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'recon':
            target_inds = np.argsort(recon)[-n:]
        else:
            raise ValueError(f'{self._cfg.active_learning.stats.secondary_samping_type} not implemented yet!')

        # derive final indices
        inds = indices[target_inds]

        return inds
