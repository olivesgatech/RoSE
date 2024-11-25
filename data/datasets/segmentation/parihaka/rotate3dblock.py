import numpy as np
import os
import math
from glob import glob
from os.path import join as pjoin
from shutil import copyfile
import shutil
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate


# decides to save entire sections or store patches. True -> random crop with entire sections
random_crop = True
idx = 0
machine = 'linux'

DATADIR = '~/data/parihaka'

# save locations for images
train_images = os.path.expanduser(f'{DATADIR}/train/')
train_val = os.path.expanduser(f'{DATADIR}/val/')
test1_images = os.path.expanduser(f'{DATADIR}/test1/')
test2_images = os.path.expanduser(f'{DATADIR}/test2/')
test_images = os.path.expanduser(f'{DATADIR}/test/')

# path to training block
complete = os.path.expanduser(f'{DATADIR}/volumes/total/complete_seismic.npy')
complete_label = os.path.expanduser(f'{DATADIR}/volumes/total/complete_seismic_label.npy')

# path to training block
train = os.path.expanduser(f'{DATADIR}/volumes/train/train_seismic.npy')
train_label = os.path.expanduser(f'{DATADIR}/volumes/train/train_labels.npy')

# path to test1 block
test1 = os.path.expanduser(f'{DATADIR}/volumes/test_once/test1_seismic.npy')
test1_label = os.path.expanduser(f'{DATADIR}/volumes/test_once/test1_labels.npy')

# path to test2 block
test2 = os.path.expanduser(f'{DATADIR}/volumes/test_once/test2_seismic.npy')
test2_label = os.path.expanduser(f'{DATADIR}/volumes/test_once/test2_labels.npy')






if __name__ == '__main__':
    create_data()