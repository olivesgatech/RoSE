import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.CIFAR10.dataset import LoaderCIFAR10
from data.datasets.classification.common.custom_transforms import Cutout
from config import BaseConfig



def create_validation(data_config: ClassificationStructure, num_classes: int = None):
    val_set = None
    val_target = None
    train_set = data_config.train_set
    train_targets = data_config.train_labels.cpu().numpy()
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
    data_config.val_labels = torch.from_numpy(val_target)
    data_config.val_len = len(data_config.val_labels)
    data_config.train_set = train_set
    data_config.train_labels = torch.from_numpy(train_targets)
    data_config.train_len = len(data_config.train_labels)

    return data_config


def get_cifar100(cfg: BaseConfig, idxs: np.ndarray = None, test_bs: bool = False, num_classes: int = None):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/CIFAR100')
    raw_tr = datasets.CIFAR100(path + '/CIFAR100', train=True, download=cfg.data.download)
    raw_te = datasets.CIFAR100(path + '/CIFAR100', train=False, download=cfg.data.download)

    #if idxs is not None:
    #    raw_tr.data = raw_tr.data[idxs]
    #    raw_tr.targets = np.array(raw_tr.targets)[idxs]

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = raw_tr.data
    data_config.train_labels = torch.from_numpy(np.array(raw_tr.targets))
    data_config.test_set = raw_te.data
    data_config.test_labels = torch.from_numpy(np.array(raw_te.targets))
    data_config.train_len = len(data_config.train_labels)
    data_config.test_len = len(data_config.test_labels)
    data_config.num_classes = 100
    data_config.img_size = 32

    data_config.is_configured = True

    if cfg.run_configs.create_validation:
        data_config = create_validation(data_config, num_classes=100)
    print(f'Total training samples {data_config.train_len}')

    # add transforms
    train_transform = transforms.Compose([])

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
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    #if test_bs:
    #    bs = 1
    #else:
    bs = cfg.classification.batch_size
    val_loader = None
    if cfg.run_configs.create_validation:
        val_loader = DataLoader(LoaderCIFAR10(data_config=data_config,
                                               split='val',
                                               transform=test_transform),
                                batch_size=bs,
                                shuffle=False)
    train_loader = DataLoader(LoaderCIFAR10(data_config=data_config,
                                            split='train',
                                            transform=train_transform, current_idxs=idxs),
                              batch_size=bs,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderCIFAR10(data_config=data_config,
                                           split='test',
                                           transform=test_transform),
                             batch_size=bs,
                             shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           val_loader=val_loader,
                           data_configs=data_config)

    return loaders
