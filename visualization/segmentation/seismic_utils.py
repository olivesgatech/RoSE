import numpy as np
import matplotlib.pyplot as plt


def get_seismic_labels():
    return np.asarray([[69, 117, 180], [145, 191, 219], [224, 243, 248], [254, 224, 144], [252, 141, 89], [215, 48, 39]])


def decode_segmap(label_mask, plot=False):
    n_classes = 6
    label_colours = get_seismic_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb