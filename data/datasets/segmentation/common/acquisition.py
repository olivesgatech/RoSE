import numpy as np

from data.datasets.segmentation.seismic.configuration import get_seismic, get_seismic_test_config
from data.datasets.segmentation.parihaka.configuration import get_parihaka, get_parihaka_test_config
from data.datasets.segmentation.common.dataobjects import LoaderObject
from config import BaseConfig


def get_dataset(cfg: BaseConfig, override: str = None, idxs: np.ndarray = None, is_train: bool = True):
    if override is not None:
        dataset = override
    else:
        dataset = cfg.data.dataset

    if dataset == 'seismic':
        return get_seismic(cfg, idxs, is_train=is_train)
    if dataset == 'parihaka':
        return get_parihaka(cfg, idxs, is_train=is_train)
    else:
        raise Exception('Dataset not implemented yet!')


def get_stat_config(loaders: LoaderObject, cfg: BaseConfig, uspec_analysis: bool = False):
    if cfg.data.dataset == 'seismic':
        out = get_seismic_test_config(loaders, cfg, uspec_analysis=uspec_analysis)
        return out
    if cfg.data.dataset == 'parihaka':
        out = get_parihaka_test_config(loaders, cfg, uspec_analysis=uspec_analysis)
        return out
    else:
        raise Exception('Dataset not implemented yet!')