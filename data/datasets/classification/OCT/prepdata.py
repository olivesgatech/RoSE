from PIL import Image
import numpy as np
import glob
import tqdm
import os
import shutil
import cv2 as cv
from config import BaseConfig


def extract_paths_and_labels(path: str, split: str):
    encoding = {'NORMAL': 0, 'CNV': 1, 'DME': 2, 'DRUSEN': 3}
    imgs = []
    labels = []
    for label in encoding.keys():
        jpegs = sorted(glob.glob(os.path.join(path, f'{split}/{label}/*')))
        targets = [encoding[label] for img in jpegs]
        imgs.extend(jpegs)
        labels.extend(targets)
    out = {'imgs': np.array(imgs), 'labels': np.array(labels)}
    return out


def extractnpy_arr(img_paths: np.ndarray):
    out = None
    tbar = tqdm.tqdm(range(len(img_paths)))
    for i, _ in enumerate(tbar):
        im = Image.open(img_paths[i])
        arr = np.array(im)
        if len(arr.shape) == 3:
            arr = cv.cvtColor(arr, cv.COLOR_BGR2GRAY)
        arr = cv.resize(arr, (128, 128), interpolation=cv.INTER_AREA)
        if out is not None:
            out = np.append(out, arr[np.newaxis, ...], axis=0)
        else:
            out = arr[np.newaxis, ...]

    return out


def cache_extract(img_paths: np.ndarray, shard_size: int = 2000):
    cache_dir = os.path.expanduser('~/.cache')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    os.makedirs(cache_dir)
    cur_index = 0
    cache_not_full = True
    npy_files = []

    while cache_not_full:
        end_index = cur_index + shard_size
        shard_paths = img_paths[cur_index:end_index] if end_index < len(img_paths) else img_paths[cur_index:]

        cur_arr = extractnpy_arr(shard_paths)

        file_name = os.path.join(cache_dir, f'{cur_index}.npy')
        np.save(file_name, cur_arr)

        del cur_arr

        npy_files.append(file_name)

        cur_index = cur_index + shard_size
        cache_not_full = True if end_index < len(img_paths) else False

    total_arr = None
    for i in range(len(npy_files)):
        print(f'Fusing {npy_files[i]}')
        arr = np.load(npy_files[i])
        if total_arr is not None:
            total_arr = np.append(total_arr, arr, axis=0)
        else:
            total_arr = arr
    print(f'Removing cache {cache_dir}')
    shutil.rmtree(cache_dir)
    return total_arr


if __name__ == '__main__':
    img_path = os.path.expanduser('~/data/oct/')
    tr_dict = extract_paths_and_labels(img_path, 'train')
    te_dict = extract_paths_and_labels(img_path, 'test')

    raw_tr = cache_extract(tr_dict['imgs'])

    with open(os.path.join(img_path, 'train.npy'), 'wb') as f:
        np.save(f, raw_tr)

    del raw_tr

    raw_te = cache_extract(te_dict['imgs'])

    with open(os.path.join(img_path, 'test.npy'), 'wb') as f:
        np.save(f, raw_te)
    del raw_te

    y_tr = tr_dict['labels']
    y_te = te_dict['labels']

    # mean = np.mean(np.mean(raw_tr / 255.0))
    # print(mean)
    # std = np.std(np.std(raw_tr / 255.0))
    # print(std)
    with open(os.path.join(img_path, 'train_labels.npy'), 'wb') as f:
        np.save(f, y_tr)
    with open(os.path.join(img_path, 'test_labels.npy'), 'wb') as f:
        np.save(f, y_te)
