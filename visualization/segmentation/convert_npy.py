import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from visualization.segmentation.seismic_utils import *


def save_as_png(filename: str, arr: np.ndarray, datatype: str):
    if filename[-3:] != 'png':
        raise ValueError(f'File {filename} must be a png file')
    if datatype in ['prediction', 'gt']:
        cmap = 'viridis'
        npy = decode_segmap(arr)
    elif datatype == 'image':
        cmap = 'seismic'
        arr = arr.astype(np.uint8)
        npy = np.moveaxis(arr, 0, -1)
    elif datatype == 'switches':
        cmap = 'jet'
        npy = np.squeeze(arr)
    else:
        raise ValueError(f'datatype {datatype} not implemented yet!')

    plt.imsave(filename, npy, cmap=cmap)


def makedirs(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


# TODO: only works on linux or macos
def save_to_dir(dir_path: str, file_list: List[str], datatype: str):
    for file in file_list:
        arr = np.load(file)
        arr_name = file.split('/')[-1][:-4]
        save_as_png(os.path.join(dir_path, f'{arr_name}.png'), arr, datatype=datatype)


def convert_predictions(pred_path: str):
    all_files = glob.glob(os.path.join(pred_path, '*.npy'))
    img_files = []
    pred_files = []
    gt_files = []
    pred_dir = makedirs(os.path.join(pred_path, 'pred_png'))
    img_dir = makedirs(os.path.join(pred_path, 'img_png'))
    gt_dir = makedirs(os.path.join(pred_path, 'gt_png'))

    for file in all_files:
        delimiter = file.split('_')[-1]
        if delimiter == 'image.npy':
            img_files.append(file)
        elif delimiter == 'gt.npy':
            gt_files.append(file)
        else:
            pred_files.append(file)

    print(f'Saving predictions')
    save_to_dir(pred_dir, pred_files, datatype='prediction')
    print(f'Saving gts')
    save_to_dir(gt_dir, gt_files, datatype='gt')
    # print(f'Saving images')
    # save_to_dir(img_dir, img_files, datatype='image')


def convert_switches(switch_path: str):
    all_files = glob.glob(os.path.join(switch_path, '*.npy'))
    switch_dir = makedirs(os.path.join(switch_path, 'switch_png'))

    print(f'Saving switches')
    save_to_dir(switch_dir, all_files, datatype='switches')


if __name__ == '__main__':
    path = os.path.expanduser('~/results/alnfr_segmentation/rcrop-rhflip/parihakav1/dlab_resnet18/nquery2/'
                              'with_preds/al_lconf/predictions/round7/test_inline/')

    convert_predictions(path)
    # convert_switches(path)