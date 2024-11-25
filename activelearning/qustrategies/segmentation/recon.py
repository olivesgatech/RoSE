import numpy as np
from activelearning.qustrategies.sampler import Sampler
from config import BaseConfig


class ReconSampler(Sampler):
    def __init__(self, n_pool, start_idxs, cfg: BaseConfig):
        super(ReconSampler, self).__init__(n_pool, start_idxs, cfg)

    def query(self, n: int, trainer):
        # get losses and their indices
        print('Sampling Recon Losses')
        unl_dict = trainer.get_unlabeled_statistics(0)
        recon, indices = unl_dict['recon'], unl_dict['indices']

        recon_inds = np.argsort(recon)[-n:]

        # derive final indices
        print(recon_inds)
        # indices = indices[0]
        inds = indices[recon_inds.astype(int)]

        return inds
