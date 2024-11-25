import os

import numpy as np
import matplotlib.pyplot as plt

DATAFILE = os.path.expanduser(f'~/datasets/parihaka/volumes/parihaka_data.npy')
LABELFILE = os.path.expanduser(f'~/datasets/parihaka/volumes/parihaka_labels.npy')
TARGETFILE = os.path.expanduser(f'~/datasets/parihaka/volumes/parihaka_data_processed.npy')
TARGETLABELFILE = os.path.expanduser(f'~/datasets/parihaka/volumes/parihaka_labels_processed.npy')
SPLITDIR = os.path.expanduser(f'~/datasets/parihaka/splits/v1/')
clip_value = 1500


def pre_process(data_volume: np.ndarray, label_volume: np.ndarray):
    new_shape = (590, 782, 255)
    print(data_volume.shape)
    data_volume = np.clip(data_volume, -clip_value, clip_value)
    # data_volume = np.resize(data_volume, new_shape)
    # label_volume = np.resize(label_volume, new_shape)
    data_volume = data_volume[:, :, 200::3]
    label_volume = label_volume[:, :, 200::3]
    # data_volume = data_volume[:, :, :255]
    # label_volume = label_volume[:, :, :255]
    print(data_volume.shape)

    return data_volume, label_volume


def print_class_distribution(name: str, volume: np.ndarray):
    out_str = name
    total_pixels = volume.size
    total = 0
    for i in range(1, 7):
        ratio = np.count_nonzero(volume == i) / total_pixels
        out_str += f' {ratio:.3f}'
        total += ratio
    print(f'{out_str} Total {total}')


def split_volumes(volume: np.ndarray, dir: str, suffix: str):
    # training = volume[:, :391, :]
    # test1 = volume[:, 391:, :]
    training = volume[:400, :600, :]
    test1 = volume[400:, :600, :]
    test2 = volume[:, 600:, :]
    print_class_distribution('Training', training[:, :, :255])
    print_class_distribution('Test', test1[:, :, :255])
    print_class_distribution('Test', test2[:, :, :255])
    # np.save(os.path.join(dir, f'training/training{suffix}.npy'), training)
    # np.save(os.path.join(dir, f'test/test{suffix}.npy'), test1)
    # np.save(os.path.join(dir, f'test/test2{suffix}.npy'), test2)

    return training, test1, test2


if __name__ == '__main__':

    seismic = np.load(DATAFILE)
    labels = np.load(LABELFILE)
    seismic, labels = pre_process(seismic, labels)

    print_class_distribution()

    np.save(TARGETFILE, seismic)
    np.save(TARGETLABELFILE, labels)

    _, _, _ = split_volumes(seismic, dir=SPLITDIR, suffix='')
    tr, te1, te2 = split_volumes(labels, dir=SPLITDIR, suffix='_labels')

    print(f'Training: {tr.shape}')
    print(f'Test1: {te1.shape}')
    print(f'Test2: {te2.shape}')