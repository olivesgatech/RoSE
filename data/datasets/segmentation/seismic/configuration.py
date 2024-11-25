import glob

import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from data.datasets.segmentation.common import custom_transforms as transforms
from data.datasets.segmentation.seismic.dataobjects import SeismicLoaderObject, SeismicStructure
from data.datasets.segmentation.common.dataobjects import StatObject
from data.datasets.segmentation.seismic.dataset import SeismicLoader
from training.segmentation.segmentationtracker import SegmentationTracker
from uspecanalysis.utils.segmentationtracker import USPECTracker
from config import BaseConfig


def get_seismic_test_config(loaders: SeismicLoaderObject, cfg: BaseConfig, uspec_analysis: bool = False):
    out = []
    if cfg.data.seismic.sets.training_xline:
        set_shape = (loaders.data_config.train_xline_len, 255, 400)
        if uspec_analysis:
            tracker = USPECTracker(set_shape, cfg.uspec_configs.num_seeds)
        else:
            tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.train_xline, tracker, 'training_xline')
        out.append(stats)
    if cfg.data.seismic.sets.training_inline:
        set_shape = (loaders.data_config.train_inline_len, 255, 700)
        if uspec_analysis:
            tracker = USPECTracker(set_shape, cfg.uspec_configs.num_seeds)
        else:
            tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.train_inline, tracker, 'training_inline')
        out.append(stats)

    if cfg.data.seismic.sets.val_xline:
        set_shape = (loaders.data_config.val_xline_len, 255, 400)
        if uspec_analysis:
            tracker = USPECTracker(set_shape, cfg.uspec_configs.num_seeds)
        else:
            tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.val_xline, tracker, 'val_xline')
        out.append(stats)
    if cfg.data.seismic.sets.val_inline:
        set_shape = (loaders.data_config.val_inline_len, 255, 700)
        if uspec_analysis:
            tracker = USPECTracker(set_shape, cfg.uspec_configs.num_seeds)
        else:
            tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.val_inline, tracker, 'val_inline')
        out.append(stats)

    if cfg.data.seismic.sets.test1_inline:
        set_shape = (loaders.data_config.test1_inline_len, 255, 701)
        if uspec_analysis:
            tracker = USPECTracker(set_shape, cfg.uspec_configs.num_seeds)
        else:
            tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.test1_inline, tracker, 'test1_inline')
        out.append(stats)
    if cfg.data.seismic.sets.test1_xline:
        set_shape = (loaders.data_config.test1_xline_len, 255, 200)
        if uspec_analysis:
            tracker = USPECTracker(set_shape, cfg.uspec_configs.num_seeds)
        else:
            tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.test1_xline, tracker, 'test1_xline')
        out.append(stats)

    if cfg.data.seismic.sets.test2_inline:
        set_shape = (loaders.data_config.test2_inline_len, 255, 200)
        if uspec_analysis:
            tracker = USPECTracker(set_shape, cfg.uspec_configs.num_seeds)
        else:
            tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.test2_inline, tracker, 'test2_inline')
        out.append(stats)
    if cfg.data.seismic.sets.test2_xline:
        set_shape = (loaders.data_config.test2_xline_len, 255, 601)
        if uspec_analysis:
            tracker = USPECTracker(set_shape, cfg.uspec_configs.num_seeds)
        else:
            tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.test2_xline, tracker, 'test2_xline')
        out.append(stats)

    return out


def get_unique_sections(path: str, prefix: str = ''):
    labels = glob.glob(path + prefix + '*_label.npy')
    samples = [name[:-10] + '.npy' for name in labels]
    labels.sort(key=lambda x: int(x.split('_')[-2]))
    samples.sort(key=lambda x: int(x.split('_')[-1][:-4]))
    return samples, labels


def get_seismic(cfg: BaseConfig, idxs: np.ndarray, is_train: bool = True):
    seismic_path = os.path.expanduser(cfg.data.data_loc + 'seismic/')
    train_path = seismic_path + 'train/'
    val_path = seismic_path + 'val/'
    test_path = seismic_path + 'test/'
    test1_path = seismic_path + 'test1/'
    test2_path = seismic_path + 'test2/'

    # get all npy files for each split
    train_samples, train_labels = get_unique_sections(train_path)
    val_samples, val_labels = get_unique_sections(val_path)
    test_samples, test_labels = get_unique_sections(test_path)

    # get separate inlines and crosslines for test groups
    train_inlines_samples, train_inlines_labels = get_unique_sections(train_path, prefix='inline')
    train_crosslines_samples, train_crosslines_labels = get_unique_sections(train_path, prefix='crossline')
    val_inlines_samples, val_inlines_labels = get_unique_sections(val_path, prefix='inline')
    val_crosslines_samples, val_crosslines_labels = get_unique_sections(val_path, prefix='crossline')
    test1_inlines_samples, test1_inlines_labels = get_unique_sections(test1_path, prefix='inline')
    test1_crosslines_samples, test1_crosslines_labels = get_unique_sections(test1_path, prefix='crossline')
    test2_inlines_samples, test2_inlines_labels = get_unique_sections(test2_path, prefix='inline')
    test2_crosslines_samples, test2_crosslines_labels = get_unique_sections(test2_path, prefix='crossline')

    data_configs = SeismicStructure()

    # get specifications on inline/xline per section
    # TODO: only works for non-windows machines
    specs = [train_samples[i].split('/')[-1].split('_')[0] for i in range(len(train_samples))]
    data_configs.train_specs = specs

    data_configs.train_set = train_samples
    data_configs.train_labels = train_labels
    data_configs.train_len = len(train_samples)

    data_configs.val_set = val_samples
    data_configs.val_labels = val_labels
    data_configs.val_len = len(val_samples)

    data_configs.test_set = test_samples
    data_configs.test_labels = test_labels
    data_configs.test_len = len(test_samples)

    # specific seismic loaders
    data_configs.train_inline_set = train_inlines_samples
    data_configs.train_inline_labels = train_inlines_labels
    data_configs.train_inline_len = len(train_inlines_samples)

    data_configs.train_xline_set = train_crosslines_samples
    data_configs.train_xline_labels = train_crosslines_labels
    data_configs.train_xline_len = len(train_crosslines_samples)

    data_configs.val_inline_set = val_inlines_samples
    data_configs.val_inline_labels = val_inlines_labels
    data_configs.val_inline_len = len(val_inlines_samples)

    data_configs.val_xline_set = val_crosslines_samples
    data_configs.val_xline_labels = val_crosslines_labels
    data_configs.val_xline_len = len(val_crosslines_samples)

    data_configs.test1_inline_set = test1_inlines_samples
    data_configs.test1_inline_labels = test1_inlines_labels
    data_configs.test1_inline_len = len(test1_inlines_samples)

    data_configs.test1_xline_set = test1_crosslines_samples
    data_configs.test1_xline_labels = test1_crosslines_labels
    data_configs.test1_xline_len = len(test1_crosslines_samples)

    data_configs.test2_inline_set = test2_inlines_samples
    data_configs.test2_inline_labels = test2_inlines_labels
    data_configs.test2_inline_len = len(test2_inlines_samples)

    data_configs.test2_xline_set = test2_crosslines_samples
    data_configs.test2_xline_labels = test2_crosslines_labels
    data_configs.test2_xline_len = len(test2_crosslines_samples)
    data_configs.num_classes = 6

    data_configs.is_configured = True

    # generate datasets
    # add transforms
    train_transform = transforms.Compose([])

    if cfg.data.augmentations.random_hflip and is_train:
        train_transform.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_rotate and is_train:
        train_transform.append(transforms.RandomRotate(10))
    if cfg.data.augmentations.random_crop and is_train:
        train_transform.append(transforms.RandomCrop(255))

    # mandatory transforms
    mean = (0.5, 0.5, 0.5)
    std = (1., 1., 1.)
    train_transform.append(transforms.Normalize(mean=mean, std=std))

    # test transforms
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # configure class weights
    weights = torch.tensor([0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], requires_grad=False)

    # create loaders
    if is_train:
        batch_size = cfg.segmentation.batch_size
    else:
        train_transform = test_transform
        batch_size = 1
    train_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                            split='train',
                                            transform=train_transform, current_idxs=idxs),
                              batch_size=batch_size,
                              shuffle=True
                              )
    val_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                          split='val',
                                          transform=test_transform),
                            batch_size=1,
                            shuffle=False)
    test_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                           split='test',
                                           transform=test_transform),
                             batch_size=1,
                             shuffle=False)

    # seismic special loaders
    test1_inline_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                            split='test1_i',
                                            transform=test_transform),
                                     batch_size=cfg.segmentation.batch_size,
                                     shuffle=False)
    test1_xline_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                                  split='test1_x',
                                                  transform=test_transform),
                                    batch_size=cfg.segmentation.batch_size,
                                    shuffle=False)
    test2_inline_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                            split='test2_i',
                                            transform=test_transform),
                                     batch_size=cfg.segmentation.batch_size,
                                     shuffle=False)
    test2_xline_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                                  split='test2_x',
                                                  transform=test_transform),
                                    batch_size=cfg.segmentation.batch_size,
                                    shuffle=False)
    train_inline_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                            split='train_i',
                                            transform=test_transform),
                                     batch_size=cfg.segmentation.batch_size,
                                     shuffle=False)
    train_xline_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                                  split='train_x',
                                                  transform=test_transform),
                                    batch_size=cfg.segmentation.batch_size,
                                    shuffle=False)
    val_inline_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                                 split='val_i',
                                                 transform=test_transform),
                                   batch_size=cfg.segmentation.batch_size,
                                   shuffle=False)
    val_xline_loader = DataLoader(SeismicLoader(data_config=data_configs,
                                                split='val_x',
                                                transform=test_transform),
                                  batch_size=cfg.segmentation.batch_size,
                                  shuffle=False)

    loaders = SeismicLoaderObject(train_loader=train_loader,
                                  test_loader=test_loader,
                                  val_loader=val_loader,
                                  test1_inline=test1_inline_loader,
                                  test1_xline=test1_xline_loader,
                                  test2_inline=test2_inline_loader,
                                  test2_xline=test2_xline_loader,
                                  train_inline=train_inline_loader,
                                  train_xline=train_xline_loader,
                                  val_inline=val_inline_loader,
                                  val_xline=val_xline_loader,
                                  class_weights=weights,
                                  data_configs=data_configs)

    return loaders