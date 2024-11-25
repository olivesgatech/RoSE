import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

DATADIR = '~/data/parihaka'

# save locations for images
TRAINDIR = os.path.expanduser(f'{DATADIR}/dataset/train/')
TEST1DIR = os.path.expanduser(f'{DATADIR}/dataset/test1/')
TEST2DIR = os.path.expanduser(f'{DATADIR}/dataset/test2/')
TESTDIR = os.path.expanduser(f'{DATADIR}/dataset/test/')

# path to training block
TRAIN = os.path.expanduser(f'{DATADIR}/splits/v1/training/training.npy')
TRAINLABEL = os.path.expanduser(f'{DATADIR}/splits/v1/training/training_labels.npy')

# path to test1 block
TEST1 = os.path.expanduser(f'{DATADIR}/splits/v1/test/test.npy')
TEST1LABEL = os.path.expanduser(f'{DATADIR}/splits/v1/test/test_labels.npy')

# path to test2 block
# TEST2 = os.path.expanduser(f'{DATADIR}/splits/test/test2.npy')
# TEST2LABEL = os.path.expanduser(f'{DATADIR}/splits/test/test2_labels.npy')


def mkdir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_dataset(volume_dir_list: List[str], prefix_list: List[str], data_dir: str, annotation: bool = False):
    if len(volume_dir_list) != len(prefix_list):
        raise ValueError(f'Volume and prefix list length must match!')
    mkdir(data_dir)
    total_id = 0
    suffix = '_label' if annotation else ''
    print(f'WARNING: Cutting off at depth 255')
    for volume_id in range(len(volume_dir_list)):
        volume = np.load(volume_dir_list[volume_id])
        # subtract 1 to be in range 0, 5
        if annotation:
            volume -= 1
        print(f'Sourcing data from {volume_dir_list[volume_id]}')
        print(f'Processing {volume.shape[0]} xlines from shape {volume.shape}')
        for xline in range(volume.shape[0]):
            name = os.path.join(data_dir, f'{prefix_list[volume_id]}crossline_0_{xline}_{total_id}{suffix}.npy')
            section = np.swapaxes(volume[xline, :, :255], 0, 1)
            np.save(name, section)
            total_id += 1
        print(f'Processing {volume.shape[1]} inlines from shape {volume.shape}')
        for inline in range(volume.shape[1]):
            name = os.path.join(data_dir, f'{prefix_list[volume_id]}inline_0_{inline}_{total_id}{suffix}.npy')
            section = np.swapaxes(volume[:, inline, :255], 0, 1)
            np.save(name, section)
            total_id += 1


if __name__ == '__main__':
    data_list = [
        [TRAIN],
        [TEST1],
        # [TEST2],
        # [TEST1, TEST2],
    ]
    label_list = [
        [TRAINLABEL],
        [TEST1LABEL],
        # [TEST2LABEL],
        # [TEST1LABEL, TEST2LABEL],
    ]
    data_dir = [
        TRAINDIR,
        # TEST1DIR,
        # TEST2DIR,
        TESTDIR
    ]
    prefix_list = [
        [''],
        [''],
        # [''],
        # ['t1_', 't2_'],
    ]
    print(f'######################## CREATING DATA ########################')
    for i in range(len(data_list)):
        print(f'Creating dataset in {data_list[i]}')
        create_dataset(data_list[i], prefix_list[i], data_dir[i], annotation=False)
    print(f'######################## CREATING LABELS ########################')
    for i in range(len(data_list)):
        print(f'Creating dataset labels in {data_list[i]}')
        create_dataset(label_list[i], prefix_list[i], data_dir[i], annotation=True)
