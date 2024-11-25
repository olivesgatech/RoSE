import glob
import os
from typing import List
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.ImageNet.dataset import LoaderImageNet
from data.datasets.classification.common.custom_transforms import Cutout
# from data.datasets.classification.common.utils import create_validation
from config import BaseConfig
import tqdm

IMG_SIZE = 224


def create_validation(data_config: ClassificationStructure, num_classes: int = None):
    val_set = None
    val_target = None
    train_set = data_config.train_set
    train_targets = data_config.train_labels

    idx_file = '/data/ryan/tmp/img_net_val_idxs.npy'
    if os.path.exists(idx_file):
        val_idxs = np.load(idx_file)
    else:
        val_idxs = None
        tbar = tqdm.tqdm(range(num_classes))
        for i, _ in enumerate(tbar):
            im_idxs = np.argwhere(train_targets == i)
            num_val = int(im_idxs.shape[0] * 0.01)
            remove_idxs = im_idxs[:num_val]
            remove_idxs = np.squeeze(remove_idxs)
            val_idxs = np.concatenate((val_idxs, remove_idxs)) if val_idxs is not None else remove_idxs
            # val_set = np.concatenate((val_set, train_set[remove_idxs]), axis=0) if val_set is not None \
            #     else train_set[remove_idxs]
            # val_target = np.concatenate((val_target, train_targets[remove_idxs]), axis=0) if val_target is not None \
            #     else train_targets[remove_idxs]

            # train_set = np.delete(train_set, remove_idxs, axis=0)
            # train_targets = np.delete(train_targets, remove_idxs)

            # print(f'Class {i}: {len(train_set[train_targets == i])}  training '
            #       f'{len(test_set[test_targets == i])} test')
        np.save(idx_file, val_idxs)

    val_set = train_set[val_idxs]
    val_target = train_targets[val_idxs]

    train_set = np.delete(train_set, val_idxs, axis=0)
    train_targets = np.delete(train_targets, val_idxs)

    print(f'Total samples training {len(train_set)} validation {len(val_set)}')
    data_config.val_set = val_set
    data_config.val_labels = torch.from_numpy(val_target)
    data_config.val_len = len(data_config.val_labels)
    data_config.train_set = train_set
    data_config.train_labels = torch.from_numpy(train_targets)
    data_config.train_len = len(data_config.train_labels)

    return data_config


def prepare_imagenet(path: str):
    class_folders = glob.glob(os.path.join(path, '*'))
    class_list = sorted([class_folders[i].split('/')[-1] for i in range(len(class_folders))])
    class_dict = {k: v for v, k in enumerate(class_list)}

    image_paths = []
    labels = []
    for folder in class_folders:
        class_id = class_dict[folder.split('/')[-1]]
        cur_images = glob.glob(os.path.join(folder, '*.JPEG'))
        image_paths.extend(cur_images)
        labels.extend([class_id for _ in range(len(cur_images))])

    out_dict = {
        'images': image_paths,
        'labels': labels,
        'class_dict': class_dict
    }
    return out_dict


def get_train_transforms(cfg: BaseConfig):
    # add transforms
    train_transform = transforms.Compose([transforms.Resize(size=(IMG_SIZE, IMG_SIZE))])

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        train_transform.transforms.append(transforms.RandomCrop(IMG_SIZE, padding=16))

    # mandatory transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # cutout requires tesnor inputs
    if cfg.data.augmentations.cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=112))

    # test transforms
    test_transform = transforms.Compose([transforms.Resize(size=(IMG_SIZE, IMG_SIZE))])
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=mean, std=std))
    return train_transform, test_transform


def get_imagenet(cfg: BaseConfig, idxs: np.ndarray = None, test_bs: bool = False):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/ImageNet')
    new_path = os.path.expanduser(path)
    train_dict = prepare_imagenet(os.path.join(new_path, 'ImageNet', 'train'))
    test_dict = prepare_imagenet(os.path.join(new_path, 'ImageNet', 'val'))

    # init data configs
    data_config = ClassificationStructure()
    data_config.num_classes = 1000
    data_config.img_size = IMG_SIZE

    data_config.train_set = np.array(train_dict['images'])
    data_config.train_labels = np.array(train_dict['labels'])
    data_config.test_set = np.array(test_dict['images'])
    data_config.test_labels = np.array(test_dict['labels'])
    data_config.train_len = len(data_config.train_labels)
    data_config.test_len = len(data_config.test_labels)

    data_config.pretrained = True
    data_config.batch_size = cfg.classification.batch_size

    data_config.is_configured = True

    if cfg.run_configs.create_validation:
        data_config = create_validation(data_config, num_classes=1000)
        # data_config = create_test_validation(data_config)

    train_transform, test_transform = get_train_transforms(cfg)

    # create loaders
    bs = cfg.classification.batch_size
    train_loader = DataLoader(LoaderImageNet(data_config=data_config,
                                             split='train',
                                             transform=train_transform, current_idxs=idxs),
                              batch_size=bs,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderImageNet(data_config=data_config,
                                            split='test',
                                            transform=test_transform),
                             batch_size=bs,
                             shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           data_configs=data_config)

    if cfg.run_configs.create_validation:
        # val used for a loss and therefore, we use transforms
        val_loader = DataLoader(LoaderImageNet(data_config=data_config,
                                               split='val',
                                               transform=test_transform),
                                batch_size=bs,
                                shuffle=True)
        loaders.val_loader = val_loader

    return loaders
