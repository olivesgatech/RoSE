import numpy as np
from activelearning.qustrategies.sampler import Sampler
from activelearning.qustrategies.entropysampler import entropy
from activelearning.qustrategies.marginsampler import margin
from activelearning.qustrategies.lconfsampling import lconf
from activelearning.qustrategies.coreset import furthest_first
from activelearning.segmentation.trainer import ActiveLearningSegmentationTrainer
from config import BaseConfig


class ALPSMisPredSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(ALPSMisPredSampler, self).__init__(n_pool, start_idxs, cfg)
        self._prev = None
        self._switches = None

    def query(self, n: int, trainer: ActiveLearningSegmentationTrainer):
        # get probabilities and their indices
        print('Validating')
        miou = trainer.validation()
        print('Sampling Statistics')
        flip_spec = 'nf'
        print(f'Sampling from {flip_spec}')
        unl_dict = trainer.get_unlabeled_statistics(0, prev_predictions=self._prev, switch_images=self._switches,
                                                    miou=miou, flip_specifier=flip_spec)
        probabilities, indices, recon = unl_dict['specified probabilities'], unl_dict['indices'], \
                                        unl_dict['specified recon']
        self._data = unl_dict['data']
        self._prev = unl_dict['predictions']
        self._switches = unl_dict['switch_images']

        if self._cfg.active_learning.stats.secondary_samping_type == 'entropy':
            target_inds = entropy(probabilities, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'margin':
            target_inds = margin(probabilities, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'lconf':
            target_inds = lconf(probabilities, n)
        elif self._cfg.active_learning.stats.secondary_samping_type == 'recon':
            target_inds = np.argsort(recon)[-n:]
        elif self._cfg.active_learning.stats.secondary_samping_type == 'coreset':
            print('Gather labeled embeddings')
            labeled_embeddings = trainer.get_unlabeled_statistics(0, labeled=True)
            # do coreset algorithm
            target_inds = furthest_first(unl_dict['specified embeddings'], labeled_embeddings['specified embeddings'],
                                         n)
        else:
            raise ValueError(f'{self._cfg.active_learning.stats.secondary_samping_type} not implemented yet!')

        # derive final indices
        inds = indices[target_inds]

        return inds
