import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.SVHN.dataset import LoaderSVHN
from config import BaseConfig



def create_validation(data_config: ClassificationStructure, num_classes: int = None):
    val_set = None
    val_target = None
    train_set = np.array(data_config.train_set)
    train_targets = np.array(data_config.train_labels)
    print(train_set.shape)
    print(train_targets.shape)
    for i in range(num_classes):
        im_idxs = np.argwhere(train_targets == i)
        num_val = int(im_idxs.shape[0] * 0.05)
        remove_idxs = im_idxs[:num_val]
        remove_idxs = np.squeeze(remove_idxs)
        # print(train_set[remove_idxs, ...].shape)
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


def get_svhn(cfg: BaseConfig, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/SVHN')
    raw_tr = datasets.SVHN(os.path.expanduser(path) + '/SVHN', split='train', download=cfg.data.download)
    raw_te = datasets.SVHN(os.path.expanduser(path) + '/SVHN', split='test', download=cfg.data.download)

    # if idxs is not None:
    #     raw_tr.data = raw_tr.train_data[idxs]
    #     raw_tr.targets = raw_tr.train_labels[idxs]
    #     raise Exception('Active learning not implemented for loader yet!')

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = raw_tr.data
    data_config.train_labels = torch.from_numpy(raw_tr.labels)
    data_config.test_set = raw_te.data
    data_config.test_labels = torch.from_numpy(raw_te.labels)
    data_config.train_len = len(data_config.train_labels)
    data_config.test_len = len(data_config.test_labels)
    data_config.num_classes = 10
    data_config.img_size = 32

    data_config.is_configured = True

    if cfg.run_configs.create_validation:
        data_config = create_validation(data_config, num_classes=10)

    # add transforms
    train_transform = transforms.Compose([])

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))

    # mandatory transforms
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # test transforms
    val_loader = None
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    train_loader = DataLoader(LoaderSVHN(data_config=data_config,
                                         split='train',
                                         transform=train_transform,
                                         current_idxs=idxs),
                              batch_size=cfg.classification.batch_size,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderSVHN(data_config=data_config,
                                         split='test',
                                         transform=test_transform),
                              batch_size=cfg.classification.batch_size,
                              shuffle=False)

    if cfg.run_configs.create_validation:
        val_loader = DataLoader(LoaderSVHN(data_config=data_config,
                                               split='val',
                                               transform=test_transform),
                                batch_size=cfg.classification.batch_size,
                                shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           val_loader=val_loader,
                           data_configs=data_config)

    return loaders