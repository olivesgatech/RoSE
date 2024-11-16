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


class VARBSSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(VARBSSampler, self).__init__(n_pool, start_idxs, cfg)
        self._memory = 10
        self._prev_predictions = np.full((self._memory, n_pool), fill_value=-1)
        self._type = 'pc'
        self._round = 0
        self._batch_size = -1.0

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Fetching Predictions')
        unl_dict = trainer.get_unlabeled_statistics(0)
        predictions, probs, indices, switches = unl_dict['predictions'], unl_dict['probabilities'], \
                                                unl_dict['indices'], unl_dict['switches']
        indices = np.squeeze(indices)
        new_n = n
        complexity = -1.0
        if self._round > 0:
            memory = self._prev_predictions[:self._round]
            pred_switches = [predictions == memory[k, indices] for k in range(memory.shape[0])]
            complexity = 1.0 - np.sum(pred_switches) / (indices.shape[0] * memory.shape[0])
            new_n = max(int(n * complexity), 10)

        labeled_embeds, unlabeled_embeds = self.get_embeddings(trainer)

        sampling_input = SampleStructure(new_n, probs, indices, labeled=labeled_embeds,
                                         unlabeled=unlabeled_embeds)
        inds = self.sample(sampling_input)

        # update prev_predictions
        round_id = self._round % self._prev_predictions.shape[0]
        self._prev_predictions[round_id, indices] = predictions
        self._round += 1
        print(f'Default Batch Size: {n}')
        print(f'Previous Batch Size: {new_n}')
        print(f'Complexity: {complexity}')
        self._batch_size = new_n
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

    def get_embeddings(self, trainer):
        if self._cfg.active_learning.stats.secondary_samping_type == 'coreset':
            unlabeled_embeddings = trainer.get_embeddings(loader_type='unlabeled')
            labeled_embeddings = trainer.get_embeddings(loader_type='labeled')
            unlabeled_embeddings = unlabeled_embeddings['nongrad_embeddings']
            labeled_embeddings = labeled_embeddings['nongrad_embeddings']
        elif self._cfg.active_learning.stats.secondary_samping_type == 'badge':
            unlabeled_embeddings = trainer.get_embeddings(loader_type='unlabeled')
            unlabeled_embeddings = unlabeled_embeddings['embeddings']
            labeled_embeddings = None
        else:
            unlabeled_embeddings = None
            labeled_embeddings = None

        return labeled_embeddings, unlabeled_embeddings

    def batch_size(self, n):
        return self._batch_size

