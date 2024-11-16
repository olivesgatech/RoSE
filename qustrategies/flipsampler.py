import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


class SampleStructure:
    def __init__(self, n: int, probs: np.ndarray, indices: np.ndarray):
        self.n = n
        self.probs = probs
        self.indices = indices


class FlipSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(FlipSampler, self).__init__(n_pool, start_idxs, cfg)
        self._prev_prev_pred = np.full(n_pool, fill_value=-1, dtype=int)
        self._prev_pred = np.full(n_pool, fill_value=-1, dtype=int)
        self._type = 'pc'
        # TODO: Hardcoded fill value! Problematic
        self._fill_value = 69
        self._database = None
        self._n_pool = n_pool
        print('Sampling from type: ' + self._cfg.active_learning.stats.flip_sampling_type)

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Fetching Predictions')
        unl_dict = trainer.get_unlabeled_statistics(0)
        preds, probs, indices, actual_acc = unl_dict['predictions'], unl_dict['probabilities'], \
                                           unl_dict['indices'], unl_dict['sample accuracy']
        indices = np.squeeze(indices)

        if self._database is None:
            tmp = np.full(self._n_pool, fill_value=-1)
            tmp[indices] = preds
            self._database = tmp[np.newaxis, ...]
            predictions = preds
        else:
            tmp = np.full(self._n_pool, fill_value=-1)
            tmp[indices] = preds
            self._database = np.concatenate((self._database, tmp[np.newaxis, ...]))
            samples_of_interest = self._database[:, indices]
            unique, idxs = np.unique(samples_of_interest, return_inverse=True)
            predictions = unique[np.argmax(np.apply_along_axis(np.bincount, 0, idxs.reshape(samples_of_interest.shape),
                                                               None, np.max(indices) + 1),
                                           axis=0)]

        # init current
        acc = preds == predictions
        acc = acc.astype(int)
        prev_acc = self._prev_pred[indices] == predictions
        prev_acc = prev_acc.astype(int)
        # get relevant samples
        indices_dict = {}
        indices_dict['bc_indices'] = indices[(prev_acc == 1) & (acc == 1)]
        indices_dict['bw_indices'] = indices[(prev_acc != 1) & (acc != 1)]
        indices_dict['nf_indices'] = indices[(prev_acc == 1) & (acc != 1)]
        indices_dict['pf_indices'] = indices[(prev_acc != 1) & (acc == 1)]
        probs_dict = {}
        probs_dict['bc_indices'] = probs[(prev_acc == 1) & (acc == 1)]
        probs_dict['bw_indices'] = probs[(prev_acc != 1) & (acc != 1)]
        probs_dict['nf_indices'] = probs[(prev_acc == 1) & (acc != 1)]
        probs_dict['pf_indices'] = probs[(prev_acc != 1) & (acc == 1)]

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
        self._prev_prev_pred = self._prev_pred
        self._prev_pred[indices] = preds
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
            prob_inds = np.random.choice(indices, n)
        else:
            raise Exception('Type not implemented yet')

        # derive final indices
        inds = indices[prob_inds]
        return inds

