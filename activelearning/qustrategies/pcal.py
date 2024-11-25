import numpy as np
from activelearning.qustrategies.sampler import Sampler
from activelearning.qustrategies.entropysampler import entropy
from activelearning.qustrategies.marginsampler import margin
from activelearning.qustrategies.lconfsampling import lconf
from activelearning.qustrategies.coreset import furthest_first
from activelearning.qustrategies.badge import init_centers
from config import BaseConfig


class SampleStructure:
    def __init__(self, n: int, probs: np.ndarray, indices: np.ndarray, labeled: np.ndarray = None,
                 unlabeled: np.ndarray = None):
        self.n = n
        self.probs = probs
        self.indices = indices
        self.embeddings_labeled = labeled
        self.embeddings_unlabeled = unlabeled


class PCALSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(PCALSampler, self).__init__(n_pool, start_idxs, cfg)
        self._prev_predictions = np.full(n_pool, fill_value=-1)
        self._type = 'pc'

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Fetching Predictions')
        unl_dict = trainer.get_unlabeled_statistics(0)
        predictions, probs, indices, switches = unl_dict['predictions'], unl_dict['probabilities'], unl_dict['indices'], \
                                                unl_dict['switches']

        print(predictions.shape)
        print(indices.shape)
        indices = np.squeeze(indices)
        # get relevant samples
        pc_indices = indices[predictions == self._prev_predictions[indices]]
        npc_indices = indices[predictions != self._prev_predictions[indices]]

        if self._cfg.active_learning.stats.pcal_sampling_type == 'pc':
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
        if len(target_inds) < n:
            print('Not enough left from type: ' + self._cfg.active_learning.stats.pcal_sampling_type)
            new_n = n - len(target_inds)
            rel_probs = probs[ntarget_map]
            rel_labeled_embeds, rel_unlabeled_embeds = self.get_embeddings(trainer, ntarget_map)
            sampling_input = SampleStructure(new_n, rel_probs, ntarget_inds, labeled=rel_labeled_embeds,
                                             unlabeled=rel_unlabeled_embeds)
            rel_inds = self.sample(sampling_input)
            if self._cfg.active_learning.stats.pcal_sampling_type == 'pc':
                inds = np.append(pc_indices, rel_inds)
            elif self._cfg.active_learning.stats.pcal_sampling_type == 'npc':
                inds = np.append(npc_indices, rel_inds)
            else:
                raise Exception('Invalid PCAL type!')
        else:
            rel_probs = probs[target_map]
            rel_labeled_embeds, rel_unlabeled_embeds = self.get_embeddings(trainer, target_map)
            sampling_input = SampleStructure(n, rel_probs, target_inds, labeled=rel_labeled_embeds,
                                             unlabeled=rel_unlabeled_embeds)
            inds = self.sample(sampling_input)

        self._prev_predictions[indices] = predictions

        return inds

    def sample(self, query_params: SampleStructure):
        n = query_params.n
        probabilities = query_params.probs
        indices = query_params.indices
        if self._cfg.active_learning.stats.secondary_samping_type == 'entropy':
            prob_inds = entropy(probabilities, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'margin':
            prob_inds = margin(probabilities, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'lconf':
            prob_inds = lconf(probabilities, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'coreset':
            prob_inds = furthest_first(query_params.embeddings_unlabeled,
                                       query_params.embeddings_labeled, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'badge':
            prob_inds = init_centers(query_params.embeddings_unlabeled, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'rand':
            # random
            poss_inds = np.arange(probabilities.shape[0])
            prob_inds = np.random.choice(poss_inds, n, replace=False)
        else:
            raise Exception('Type not implemented yet')
        # derive final indices
        inds = indices[prob_inds]
        return inds

    def get_embeddings(self, trainer, inds):
        if self._cfg.active_learning.stats.secondary_samping_type == 'coreset':
            unlabeled_embeddings = trainer.get_embeddings(loader_type='unlabeled')
            labeled_embeddings = trainer.get_embeddings(loader_type='labeled')
            unlabeled_embeddings = unlabeled_embeddings['nongrad_embeddings']
            labeled_embeddings = labeled_embeddings['nongrad_embeddings']
            unlabeled_embeddings = unlabeled_embeddings[inds]
        elif self._cfg.active_learning.stats.secondary_samping_type == 'badge':
            unlabeled_embeddings = trainer.get_embeddings(loader_type='unlabeled')
            unlabeled_embeddings = unlabeled_embeddings['embeddings']
            unlabeled_embeddings = unlabeled_embeddings[inds]
            labeled_embeddings = None
        else:
            unlabeled_embeddings = None
            labeled_embeddings = None

        return labeled_embeddings, unlabeled_embeddings

