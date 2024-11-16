import numpy as np
from activelearning.qustrategies.sampler import Sampler
from activelearning.qustrategies.entropysampler import entropy
from activelearning.qustrategies.marginsampler import margin
from activelearning.qustrategies.lconfsampling import lconf
from activelearning.qustrategies.coreset import furthest_first
from activelearning.qustrategies.badge import init_centers
from config import BaseConfig
from typing import List


class SampleStructure:
    def __init__(self, n: int, probs: np.ndarray, indices: np.ndarray, labeled: np.ndarray = None,
                 unlabeled: np.ndarray = None):
        self.n = n
        self.probs = probs
        self.indices = indices
        self.embeddings_labeled = labeled
        self.embeddings_unlabeled = unlabeled


def split_subsetsDEP(subsets: SampleStructure, scores: np.ndarray, threshold: float):
    correct = SampleStructure(
        n=subsets.n,
        probs=subsets.probs[scores > threshold, :],
        indices=subsets.indices[scores > threshold],
        unlabeled=subsets.embeddings_unlabeled[scores > threshold, :] if subsets.embeddings_unlabeled is not None else None,
        labeled=subsets.embeddings_labeled[scores > threshold, :] if subsets.embeddings_labeled is not None else None
    )
    wrong = SampleStructure(
        n=subsets.n,
        probs=subsets.probs[scores <= threshold, :],
        indices=subsets.indices[scores <= threshold],
        unlabeled=subsets.embeddings_unlabeled[scores <= threshold, :] if subsets.embeddings_unlabeled is not None else None,
        labeled=subsets.embeddings_labeled[scores <= threshold, :] if subsets.embeddings_labeled is not None else None
    )
    return correct, wrong


def split_subsets(subsets: SampleStructure, scores: np.ndarray, target_samples: int, target_category: str):

    num_positive_samples = target_samples if target_category == 'correct' else scores.shape[0] - target_samples
    num_neg_samples = scores.shape[0] - num_positive_samples
    print(f'Total samples {scores.shape[0]} Correct Scores {num_positive_samples} Wrong Scores {num_neg_samples}')
    sorted_score_inds = np.argsort(scores)
    # scores reflect whether sample is correct
    wrong_inds = sorted_score_inds[:num_neg_samples]
    correct_inds = sorted_score_inds[num_neg_samples:]
    # print(subsets.embeddings_unlabeled.shape)
    # print(subsets.embeddings_labeled.shape)
    # print(subsets.probs.shape)
    correct = SampleStructure(
        n=subsets.n,
        probs=subsets.probs[correct_inds, :],
        indices=subsets.indices[correct_inds],
        unlabeled=subsets.embeddings_unlabeled[correct_inds, :] if subsets.embeddings_unlabeled is not None else None,
        labeled=subsets.embeddings_labeled if subsets.embeddings_labeled is not None else None
    )
    wrong = SampleStructure(
        n=subsets.n,
        probs=subsets.probs[wrong_inds, :],
        indices=subsets.indices[wrong_inds],
        unlabeled=subsets.embeddings_unlabeled[wrong_inds, :] if subsets.embeddings_unlabeled is not None else None,
        labeled=subsets.embeddings_labeled if subsets.embeddings_labeled is not None else None
    )
    return correct, wrong


def combine_subsets(subset_list: List[SampleStructure]):
    return SampleStructure(
        n=subset_list[0].n,
        probs=np.concatenate([subset.probs for subset in subset_list], axis=0),
        indices=np.concatenate([subset.indices for subset in subset_list], axis=0),
        unlabeled=np.concatenate([subset.embeddings_unlabeled for subset in subset_list], axis=0)
        if subset_list[0].embeddings_unlabeled is not None else None,
        labeled=np.concatenate([subset.embeddings_labeled for subset in subset_list], axis=0)
        if subset_list[0].embeddings_labeled is not None else None,
    )


class UnlabeledPCALSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(UnlabeledPCALSampler, self).__init__(n_pool, start_idxs, cfg)
        self._prev_predictions = np.full(n_pool, fill_value=-1)
        self._type = 'pc'

    def query(self, n: int, trainer):
        # get probabilities and their indices
        print('Fetching Predictions')
        unl_dict = trainer.get_unlabeled_statistics(0)
        predictions, probs, indices, switches, scores = unl_dict['predictions'], unl_dict['probabilities'], \
                                                        unl_dict['indices'], unl_dict['switches'], unl_dict['scores']
        # _, acc, auc, th = trainer.validation(0)
        entr_inds = set(entropy(probs, n))

        print(predictions.shape)
        print(indices.shape)
        indices = np.squeeze(indices)
        # get relevant samples
        pc_map = predictions == self._prev_predictions[indices]
        npc_map = predictions != self._prev_predictions[indices]
        pc_embeddings_labeled, pc_embeddings_unlabeled = self.get_embeddings(trainer, pc_map)
        npc_embeddings_labeled, npc_embeddings_unlabeled = self.get_embeddings(trainer, npc_map)

        # init structures
        nf_pf = SampleStructure(n, probs=probs[npc_map], indices=indices[npc_map], labeled=npc_embeddings_labeled,
                                unlabeled=npc_embeddings_unlabeled)
        bc_bw = SampleStructure(n, probs=probs[pc_map], indices=indices[pc_map], labeled=pc_embeddings_labeled,
                                unlabeled=pc_embeddings_unlabeled)

        # pf, nf = split_subsets(nf_pf, scores[npc_map], th)
        # bc, bw = split_subsets(bc_bw, scores[pc_map], th)
        target_samples = int(self._cfg.active_learning.stats.pcal_sampling_multiplier * n)
        specifier = 'correct' if self._cfg.active_learning.stats.pcal_sampling_type in ['bc', 'pf'] else 'wrong'
        pf, nf = split_subsets(nf_pf, scores[npc_map], target_samples=target_samples, target_category=specifier)
        bc, bw = split_subsets(bc_bw, scores[pc_map], target_samples=target_samples, target_category=specifier)
        target_subsets = self._cfg.active_learning.stats.pcal_sampling_type.split(',')
        combined_subset_list = []
        priority = ['bc', 'pf', 'nf', 'bw']
        if 'nf' in target_subsets:
            combined_subset_list.append(nf)
            priority.remove('nf')
        if 'pf' in target_subsets:
            combined_subset_list.append(pf)
            priority.remove('pf')
        if 'bc' in target_subsets:
            combined_subset_list.append(bc)
            priority.remove('bc')
        if 'bw' in target_subsets:
            combined_subset_list.append(bw)
            priority.remove('bw')

        combined_samples = combine_subsets(combined_subset_list)

        # make sure combined samples contain at least n samples
        priority_idx = 0
        while len(combined_samples.indices) < n:
            if priority_idx == len(priority):
                raise ValueError(f'Number of indices in subsets < n and rest does not fill it up!')
            print(f'Not enough samples combining with: {priority[priority_idx]}\n'
                  f'Current number of samples: {len(combined_samples.indices)}')
            target_subsets.append(priority[priority_idx])
            if priority[priority_idx] == 'nf':
                combined_subset_list.append(nf)
            elif priority[priority_idx] == 'pf':
                combined_subset_list.append(pf)
            elif priority[priority_idx] == 'bc':
                combined_subset_list.append(bc)
            elif priority[priority_idx] == 'bw':
                combined_subset_list.append(bw)
            else:
                raise ValueError(f'{priority[priority_idx]} does not exist')
            combined_samples = combine_subsets(combined_subset_list)
            priority_idx += 1
        print(f'Final combined subsets: {target_subsets}\n'
              f'Current number of samples: {len(combined_samples.indices)}')

        inds = self.sample(combined_samples)

        self._prev_predictions[indices] = predictions

        intersection = entr_inds.intersection(set(inds))
        print(f'The number of intersecting indices with entropy is {len(intersection)}')

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

