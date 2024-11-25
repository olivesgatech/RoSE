from PIL import Image
import numpy as np
import glob
import tqdm
import os
import cv2 as cv
from config import BaseConfig


def extract_paths_and_labels(path: str, split: str):
    encoding = {'NORMAL': 0, 'BACTERIA': 1, 'VIRUS': 2}
    normal_images = sorted(glob.glob(path + f'{split}/NORMAL/*'))
    normal_labels = [encoding['NORMAL'] for img in normal_images]
    sick_images = sorted(glob.glob(path + f'{split}/PNEUMONIA/*'))
    sick_labels = [encoding[sick_images[i].split('/')[-1].split('-')[0]] for i in range(len(sick_images))]
    normal_images.extend(sick_images)
    normal_labels.extend(sick_labels)
    # print(normal_labels)
    # print(normal_images)
    out = {'imgs': np.array(normal_images), 'labels': np.array(normal_labels)}
    return out


def extractnpy_arr(img_paths: np.ndarray):
    out = None
    for i in range(len(img_paths)):
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


if __name__ == '__main__':
    img_path = os.path.expanduser('~/data/xray/')
    tr_dict = extract_paths_and_labels(img_path, 'train')
    te_dict = extract_paths_and_labels(img_path, 'test')

    raw_tr = extractnpy_arr(tr_dict['imgs'])
    raw_te = extractnpy_arr(te_dict['imgs'])

    y_tr = tr_dict['labels']
    y_te = te_dict['labels']

    mean = np.mean(np.mean(raw_tr / 255.0))
    print(mean)
    std = np.std(np.std(raw_tr / 255.0))
    print(std)

    with open(os.path.join(img_path, 'train.npy'), 'wb') as f:
        np.save(f, raw_tr)
    with open(os.path.join(img_path, 'train_labels.npy'), 'wb') as f:
        np.save(f, y_tr)

    with open(os.path.join(img_path, 'test.npy'), 'wb') as f:
        np.save(f, raw_te)
    with open(os.path.join(img_path, 'test_labels.npy'), 'wb') as f:
        np.save(f, y_te)
