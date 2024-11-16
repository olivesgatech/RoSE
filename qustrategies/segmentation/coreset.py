import numpy as np
from sklearn.metrics import pairwise_distances
from activelearning.qustrategies.sampler import Sampler
from activelearning.qustrategies.coreset import furthest_first
from config import BaseConfig


class SegmentationCoresetSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(SegmentationCoresetSampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        print('Gather unlabeled embeddings')
        unlabeled_embeddings = trainer.get_unlabeled_statistics(0)
        print('Gather labeled embeddings')
        labeled_embeddings = trainer.get_unlabeled_statistics(0, labeled=True)
        unlabeled_indices = unlabeled_embeddings['indices']

        # do coreset algorithm
        chosen = furthest_first(unlabeled_embeddings['embeddings'], labeled_embeddings['embeddings'], n)

        # derive final indices
        inds = unlabeled_indices[chosen]
        # print(f'Chosen inds {inds}')

        return inds
