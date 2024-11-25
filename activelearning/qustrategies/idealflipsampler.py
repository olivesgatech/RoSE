import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


class SampleStructure:
    def __init__(self, n: int, probs: np.ndarray, indices: np.ndarray):
        self.n = n
        self.probs = probs
        self.indices = indices


class IdealFlipSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(IdealFlipSampler, self).__init__(n_pool, start_idxs, cfg)
        self._prev_acc = np.full(n_pool, fill_value=-1, dtype=int)
        self._type = 'pc'
        print('Sampling from type: ' + self._cfg.active_learning.stats.flip_sampling_type)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Fetching Predictions')
        unl_dict = trainer.get_unlabeled_statistics(0)
        predictions, probs, indices, acc = unl_dict['predictions'], unl_dict['probabilities'], \
                                           unl_dict['indices'], unl_dict['sample accuracy']

        print(predictions.shape)
        print(indices.shape)
        indices = np.squeeze(indices)
        # get relevant samples
        indices_dict = {}
        indices_dict['bc_indices'] = indices[(self._prev_acc[indices] == 1) & (acc == 1)]
        indices_dict['bw_indices'] = indices[(self._prev_acc[indices] != 1) & (acc != 1)]
        indices_dict['nf_indices'] = indices[(self._prev_acc[indices] == 1) & (acc != 1)]
        indices_dict['pf_indices'] = indices[(self._prev_acc[indices] != 1) & (acc == 1)]
        probs_dict = {}
        probs_dict['bc_indices'] = probs[(self._prev_acc[indices] == 1) & (acc == 1)]
        probs_dict['bw_indices'] = probs[(self._prev_acc[indices] != 1) & (acc != 1)]
        probs_dict['nf_indices'] = probs[(self._prev_acc[indices] == 1) & (acc != 1)]
        probs_dict['pf_indices'] = probs[(self._prev_acc[indices] != 1) & (acc == 1)]

        if self._cfg.active_learning.stats.flip_sampling_type == 'bc':
            sequence = ['bc_indices', 'pf_indices', 'bw_indices', 'nf_indices']
        elif self._cfg.active_learning.stats.flip_sampling_type == 'bw':
            sequence = ['bw_indices', 'nf_indices', 'bc_indices', 'pf_indices']
        elif self._cfg.active_learning.stats.flip_sampling_type == 'pf':
            sequence = ['pf_indices', 'bc_indices', 'nf_indices', 'bw_indices']
        elif self._cfg.active_learning.stats.flip_sampling_type == 'nf':
            sequence = ['nf_indices', 'bw_indices', 'pf_indices', 'bc_indices']
        else:
            raise Exception('Flip sampling type not implemented yet')

        inds = self.restrict(indices_dict, probs_dict, sequence, n)
        self._prev_acc[indices] = acc
        return inds

    def restrict(self, ind_dict: dict, probs_dict: dict, sequence: list, n: int):
        if ind_dict[sequence[0]].shape[0] >= n:
            out = self.sample(n, probs_dict[sequence[0]], ind_dict[sequence[0]])
            return out

        out = ind_dict[sequence[0]]
        req = n - out.shape[0]
        print('Adding additional samples from:')
        for i in range(1, len(sequence)):
            print(sequence[i])
            if ind_dict[sequence[i]].shape[0] >= req:
                app = self.sample(req, probs_dict[sequence[i]], ind_dict[sequence[i]])
                out = np.append(out, app)
                break
            else:
                out = np.append(out, ind_dict[sequence[i]])
                req -= ind_dict[sequence[i]].shape[0]
        return out

    def sample(self, n, probabilities, indices):
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
        elif self._cfg.active_learning.stats.secondary_samping_type == 'rand':
            # random
            poss_inds = np.arange(probabilities.shape[0])
            prob_inds = np.random.choice(poss_inds, n, replace=False)
        else:
            raise Exception('Type not implemented yet')

        # derive final indices
        inds = indices[prob_inds]
        return inds

