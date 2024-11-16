import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


class SampleStructure:
    def __init__(self, n: int, probs: np.ndarray, indices: np.ndarray):
        self.n = n
        self.probs = probs
        self.indices = indices


class IdealPCALSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(IdealPCALSampler, self).__init__(n_pool, start_idxs, cfg)
        self._prev_predictions = np.full(n_pool, fill_value=-1)
        self._type = 'pc'

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Fetching Predictions')
        predictions, _, probs, indices, switches = trainer.get_unlabeled_statistics(0)

        print(predictions.shape)
        print(indices.shape)
        indices = np.squeeze(indices)
        # get relevant samples
        pc_indices = indices[predictions == self._prev_predictions[indices]]
        npc_indices = indices[predictions != self._prev_predictions[indices]]

        if self._type == 'pc':
            target_inds = pc_indices
            ntarget_inds = npc_indices
            ntarget_map = predictions != self._prev_predictions[indices]
            target_map = predictions == self._prev_predictions[indices]
        else:
            target_inds = npc_indices
            ntarget_inds = pc_indices
            ntarget_map = predictions == self._prev_predictions[indices]
            target_map = predictions != self._prev_predictions[indices]

        # sample from relevant subgroup
        # FIXME!!!! Currently hardcoding pc sampling and not npc additionally
        if len(target_inds) < n:
            print('Not enough left from type: ' + self._type)
            new_n = n - len(target_inds)
            rel_probs = probs[ntarget_map]
            sampling_input = SampleStructure(new_n, rel_probs, ntarget_inds)
            rel_inds = self.sample(sampling_input)
            inds = np.append(pc_indices, rel_inds)
        else:
            rel_probs = probs[target_map]
            sampling_input = SampleStructure(n, rel_probs, target_inds)
            inds = self.sample(sampling_input)

        self._prev_predictions[indices] = predictions

        return inds

    def sample(self, query_params: SampleStructure):
        n = query_params.n
        probabilities = query_params.probs
        indices = query_params.indices
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
        else:
            raise Exception('Type not implemented yet')

        # derive final indices
        inds = indices[prob_inds]
        return inds

