import os
import glob
import torch
import argparse
import toml
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.OCT.dataset import LoaderOCT
from data.datasets.classification.common.custom_transforms import Cutout
from config import BaseConfig


def create_validation(data_config: ClassificationStructure, num_classes: int = None):
    val_set = None
    val_target = None
    train_set = data_config.train_set
    train_targets = data_config.train_labels.cpu().numpy()
    for i in range(num_classes):
        im_idxs = np.argwhere(train_targets == i)
        num_val = int(im_idxs.shape[0] * 0.05)
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


def get_oct(cfg: BaseConfig, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/oct')
    tot_path = os.path.expanduser(path) + '/oct/'

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = np.load(tot_path + 'train.npy')
    data_config.train_labels = torch.from_numpy(np.load(tot_path + 'train_labels.npy'))
    data_config.test_set = np.load(tot_path + 'test.npy')
    data_config.test_labels = torch.from_numpy(np.load(tot_path + 'test_labels.npy'))
    data_config.train_len = len(data_config.train_labels)
    data_config.test_len = len(data_config.test_labels)
    data_config.num_classes = 4
    data_config.img_size = 128

    data_config.is_configured = True

    if cfg.run_configs.create_validation:
        data_config = create_validation(data_config, num_classes=4)
    print(f'Total training samples {data_config.train_len}')

    # add transforms
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.Resize(size=(data_config.img_size, data_config.img_size)))
    train_transform.transforms.append(transforms.Grayscale(num_output_channels=3))

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        train_transform.transforms.append(transforms.RandomCrop(128, padding=16))

    # mandatory transforms
    mean = 0.19876538343581782
    std = 0.07863086417936489
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # cutout requires tesnor inputs
    if cfg.data.augmentations.cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=64))

    # test transforms
    test_transform = transforms.Compose([
        # transforms.Resize(size=(data_config.img_size, data_config.img_size)),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    bs = cfg.classification.batch_size
    val_loader = None
    train_loader = DataLoader(LoaderOCT(data_config=data_config,
                                            split='train',
                                            transform=train_transform, current_idxs=idxs),
                              batch_size=bs,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderOCT(data_config=data_config,
                                           split='test',
                                           transform=test_transform),
                             batch_size=bs,
                             shuffle=False)

    if cfg.run_configs.create_validation:
        val_loader = DataLoader(LoaderOCT(data_config=data_config,
                                               split='val',
                                               transform=test_transform),
                                batch_size=bs,
                                shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           val_loader=val_loader,
                           data_configs=data_config)

    return loaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/alnfr/example_config.toml')

    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    _ = get_xray(configs)
