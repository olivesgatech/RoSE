import numpy as np
from activelearning.qustrategies.randomsampler import RandomSampling
from activelearning.qustrategies.entropysampler import EntropySampler
from activelearning.qustrategies.marginsampler import MarginSampler
from activelearning.qustrategies.lconfsampling import LeastConfidenceSampler
from activelearning.qustrategies.switchsampler import SwitchSampler
from activelearning.qustrategies.coreset import CoresetSampler
from activelearning.qustrategies.badge import BadgeSampler
from activelearning.qustrategies.gmmss import GauSSSampler
from activelearning.qustrategies.balent import BalEntSampler
from activelearning.qustrategies.bald import BALDSampler
from activelearning.qustrategies.power_bald import PowerBALDSampler
from activelearning.qustrategies.pcal_misprediction import PCALSampler
from activelearning.qustrategies.pcal_unlabeled import UnlabeledPCALSampler
from activelearning.qustrategies.relaxedss import RelaxedSwitchSampler
from activelearning.qustrategies.relaxednfs import RelaxedNFSampler
from activelearning.qustrategies.relaxedentropy import RelaxedEntropySampler
from activelearning.qustrategies.idealflipsampler import IdealFlipSampler
from activelearning.qustrategies.flipsampler import FlipSampler
from activelearning.qustrategies.alps import ALPSSampler
from activelearning.qustrategies.varbs import VARBSSampler

from activelearning.qustrategies.segmentation.recon import ReconSampler
from activelearning.qustrategies.segmentation.gauss import GauSSSegmentationSampler
from activelearning.qustrategies.segmentation.coreset import SegmentationCoresetSampler
from activelearning.qustrategies.segmentation.alps_misprediction import ALPSMisPredSampler
from config import BaseConfig


def get_sampler(cfg: BaseConfig, n_pool: int, start_idxs: np.ndarray):
    if cfg.active_learning.strategy == 'random':
        print('Using Random Sampler')
        sampler = RandomSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'entropy':
        print('Using Entropy Sampler')
        sampler = EntropySampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'margin':
        print('Using Margin Sampler')
        sampler = MarginSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'lconf':
        print('Using Least Confidence Sampler')
        sampler = LeastConfidenceSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'switch':
        print('Using Switch Sampler')
        sampler = SwitchSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'badge':
        print('Using BADGE Sampler')
        sampler = BadgeSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'coreset':
        print('Using Coreset Sampler')
        sampler = CoresetSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'gauss':
        print('Using GauSS Sampler')
        sampler = GauSSSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'pcal':
        print('Using PCAL Sampler')
        sampler = PCALSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'upcal':
        print('Using UPCAL Sampler')
        sampler = UnlabeledPCALSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'balent':
        print('Using BalEnt Sampler')
        sampler = BalEntSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'bald':
        print('Using BALD Sampler')
        sampler = BALDSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'power_bald':
        print('Using PowerBALD Sampler')
        sampler = PowerBALDSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'relss':
        print('Using Relaxed Switch Sampler')
        sampler = RelaxedSwitchSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'relnf':
        print('Using Relaxed NF Sampler')
        sampler = RelaxedNFSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'relentropy':
        print('Using Relaxed Entropy Sampler')
        sampler = RelaxedEntropySampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'idealflipsampling':
        print('Using Ideal Flip Sampler')
        sampler = IdealFlipSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'flipsampling':
        print('Using Flip Sampler')
        sampler = FlipSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'varbs':
        print('Using VARBS Sampler')
        sampler = VARBSSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    else:
        raise Exception('Strategy not implemented yet')

    return sampler


def get_segmentation_sampler(cfg: BaseConfig, n_pool: int, start_idxs: np.ndarray):
    if cfg.active_learning.strategy == 'random':
        print('Using Random Sampler')
        sampler = RandomSampling(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'switch':
        print('Using Switch Sampler')
        sampler = SwitchSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'entropy':
        print('Using Entropy Sampler')
        sampler = EntropySampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'margin':
        print('Using Margin Sampler')
        sampler = MarginSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'lconf':
        print('Using LConf Sampler')
        sampler = LeastConfidenceSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'recon':
        print('Using Recon Sampler')
        sampler = ReconSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'gauss':
        print('Using GauSS Sampler')
        sampler = GauSSSegmentationSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'coreset':
        print('Using Coreset Sampler')
        sampler = SegmentationCoresetSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'alps':
        print('Using ALPS Sampler')
        sampler = ALPSSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    elif cfg.active_learning.strategy == 'alps_mispred':
        print('Using ALPS Misprediction Sampler')
        sampler = ALPSMisPredSampler(n_pool=n_pool, start_idxs=start_idxs, cfg=cfg)
    else:
        raise Exception('Strategy not implemented yet')

    return sampler
