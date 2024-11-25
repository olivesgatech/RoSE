import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


class RelaxedNFSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(RelaxedNFSampler, self).__init__(n_pool, start_idxs, cfg)

    def action(self, trainer):
        print('Tracking NFs')
        _ = trainer.get_unlabeled_statistics(0)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Sampling NFs')
        unl_dict = trainer.get_unlabeled_statistics(0)
        predictions, probs, indices, nfs = unl_dict['predictions'], unl_dict['probabilities'], unl_dict['indices'],\
                                           unl_dict['nfs']
        relaxation = self._cfg.active_learning.stats.relaxation
        new_n = n + relaxation
        # get max entropy
        if self._cfg.active_learning.stats.stat_sampling_type == 'SV':
            target_inds = np.argsort(nfs)[-new_n:]
        elif self._cfg.active_learning.stats.stat_sampling_type == 'nSV':
            target_inds = np.argsort(nfs)[:new_n]
        else:
            raise Exception('stat sampling type not implemented yet!')

        # get highest entrtopy of widdled samples
        probabilities = probs[target_inds]

        if self._cfg.active_learning.stats.secondary_samping_type == 'entropy':
            logs = np.log2(probabilities)
            mult = logs * probabilities
            entropy = np.sum(mult, axis=1)
            prob_inds = np.argsort(entropy)[:n]
        elif self._cfg.active_learning.stats.secondary_samping_type == 'margin':
            # get smallest margins
            sorted_probs = np.sort(probabilities, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            prob_inds = np.argsort(margins)[:n]
        elif self._cfg.active_learning.stats.secondary_samping_type == 'lconf':
            # get max probs
            probabilities = np.max(probabilities, axis=1)
            prob_inds = np.argsort(probabilities)[:n]
        elif self._cfg.active_learning.stats.secondary_samping_type == 'random':
            prob_inds = target_inds[np.random.choice(target_inds.shape[0], n, replace=False)]
        else:
            raise Exception('Sampling type not implemented yet')

        # derive final indices
        inds = indices[prob_inds]

        return inds
