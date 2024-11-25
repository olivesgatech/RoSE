from PIL import Image
import numpy as np
import tqdm
import os
import cv2 as cv


def create_numpy(imgpath: str, split: str, file_list: list):
    labels = []
    imgs = None
    print(f'Creating numpy array for {split}')
    tbar = tqdm.tqdm(file_list)
    for i, fname in enumerate(tbar):
        name = os.path.join(imgpath, split, fname[0].split('-')[0], fname[0])
        img = np.array(Image.open(name))
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA)
        if imgs is not None:
            imgs = np.concatenate((imgs, img[np.newaxis, ...]), axis=0)
        else:
            imgs = img[np.newaxis, ...]
        labels.append(int(fname[1]))

    return imgs, np.array(labels)

img_path = os.path.expanduser('~/PycharmProjects/Dataset/xray/')
file_path = os.path.expanduser('~/PycharmProjects/Dataset/xray/npy/')

train_files = open(os.path.join(file_path, 'train.txt'), "r")
test_files = open(os.path.join(file_path, 'test.txt'), "r")

train_list = train_files.readlines()
test_list = test_files.readlines()

train_list = [id_.rstrip().split(',') for id_ in train_list]
test_list = [id_.rstrip().split(',') for id_ in test_list]

raw_tr, y_tr = create_numpy(img_path, 'train', train_list)
raw_te, y_te = create_numpy(img_path, 'test', test_list)

mean = np.mean(np.mean(raw_tr / 255.0))
print(mean)
std = np.std(np.std(raw_tr / 255.0))
print(std)

with open(os.path.join(file_path, 'train.npy'), 'wb') as f:
    np.save(f, raw_tr)
with open(os.path.join(file_path, 'train_labels.npy'), 'wb') as f:
    np.save(f, y_tr)

with open(os.path.join(file_path, 'test.npy'), 'wb') as f:
    np.save(f, raw_te)
with open(os.path.join(file_path, 'test_labels.npy'), 'wb') as f:
    np.save(f, y_te)
