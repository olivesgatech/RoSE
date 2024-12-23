import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.STL10.dataset import LoaderSTL10
from data.datasets.classification.common.custom_transforms import Cutout
from config import BaseConfig


def create_validation(data_config: ClassificationStructure, num_classes: int = None):
    val_set = None
    val_target = None
    train_set = data_config.train_set
    train_targets = data_config.train_labels
    for i in range(num_classes):
        im_idxs = np.argwhere(train_targets == i)
        num_val = int(im_idxs.shape[0] * 0.1)
        remove_idxs = im_idxs[:num_val]
        remove_idxs = np.squeeze(remove_idxs)
        val_set = np.concatenate((val_set, train_set[remove_idxs]), axis=0) if val_set is not None \
            else train_set[remove_idxs]
        val_target = np.concatenate((val_target, train_targets[remove_idxs]), axis=0) if val_target is not None \
            else train_targets[remove_idxs]

        train_set = np.delete(train_set, remove_idxs, axis=0)
        train_targets = np.delete(train_targets, remove_idxs)

        # print(f'Class {i}: {len(train_set[train_targets == i])}  training '
        #       f'{len(test_set[test_targets == i])} test')

    print(f'Total samples training {len(train_set)} validation {len(val_set)}')
    data_config.val_set = val_set
    data_config.val_labels = val_target
    data_config.val_len = len(data_config.val_labels)
    data_config.train_set = train_set
    data_config.train_labels = train_targets
    data_config.train_len = len(data_config.train_labels)

    return data_config


def rename_labels(input_arr, source, target):
    """renames all labels of input array according to source. E.g. all target labels [3, 4, 5] will be replaced with
    new source labels [8, 5, 1]"""
    if len(source) != len(target):
        raise Exception("Source and target arrays do not match")

    # iterate through all source and target labels
    for i in range(len(target)):
        input_arr[input_arr == target[i]] = source[i]

    return input_arr


def get_stl10(cfg: BaseConfig, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/STL10')
    raw_tr = datasets.STL10(path + '/STL10', split='train', download=cfg.data.download)
    raw_te = datasets.STL10(path + '/STL10', split='test', download=cfg.data.download)

    # if idxs is not None:
    #    raw_tr.data = raw_tr.data[idxs]
    #    raw_tr.targets = raw_tr.labels[idxs]
    #    raise Exception('Active learning not implemented for loader yet!')

    # rename stl labels to cifar10 labels
    sources = np.array([0, 2, 1, 3, 4, 5, 7, 8, 9])
    targets = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9])
    targets = targets + 100
    raw_te.labels = rename_labels(raw_te.labels, source=sources, target=targets)

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = raw_tr.data
    data_config.train_labels = raw_tr.labels
    data_config.test_set = raw_te.data
    data_config.test_labels = raw_te.labels
    data_config.train_len = len(data_config.train_labels)
    data_config.test_len = len(data_config.test_labels)
    data_config.num_classes = 10
    data_config.img_size = 32

    data_config.is_configured = True

    if cfg.run_configs.create_validation:
        data_config = create_validation(data_config, num_classes=10)

    # add transforms
    # TODO: STL10 reduction of resolution
    train_transform = transforms.Compose([transforms.Resize(size=(32, 32))])

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))

    # mandatory transforms
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # cutout requires tesnor inputs
    if cfg.data.augmentations.cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

    # test transforms
    test_transform = transforms.Compose([transforms.Resize(size=(32, 32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    val_loader = None
    train_loader = DataLoader(LoaderSTL10(data_config=data_config,
                                          split='train',
                                          transform=train_transform,
                                          current_idxs=idxs),
                              batch_size=cfg.classification.batch_size,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderSTL10(data_config=data_config,
                                         split='test',
                                         transform=test_transform),
                              batch_size=cfg.classification.batch_size,
                              shuffle=False)

    if cfg.run_configs.create_validation:
        val_loader = DataLoader(LoaderSTL10(data_config=data_config,
                                               split='val',
                                               transform=test_transform),
                                batch_size=cfg.classification.batch_size,
                                shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           val_loader=val_loader,
                           data_configs=data_config)

    return loaders
