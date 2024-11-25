import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from config import BaseConfig
from visualization.utils.distances import histogram_distance


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


class VolumeViewer:
    def __init__(self, cfg: BaseConfig, target_volume: np.ndarray, training_volume: np.ndarray = None):
        self._cfg = cfg
        self._volume = target_volume
        self._training_volume = training_volume

    def multi_slice_viewer(self):
        print('Starting multi slice viewer')
        aspect = 20
        pad_fraction = 0.5
        remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()

        # volume
        ax.index = self._volume.shape[0] // 2
        ax.index = 10
        im = ax.imshow(self._volume[ax.index], cmap='jet')
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1. / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)

        # fevent histograms
        # ax[1].scatter(x[inds], y[inds])
        fig.canvas.mpl_connect('key_press_event', self._process_key)
        cbar = plt.colorbar(im, cax=cax)
        ax.cbar = cbar
        plt.show()

    def _print_distance(self, flattened_target: np.ndarray):
        if self._training_volume is not None:
            dist = histogram_distance(self._training_volume.flatten(), flattened_target,
                                      bin_param=self._cfg.visualization.hist_bin_param,
                                      dist_type=self._cfg.visualization.dist_type)
            print('Distance Metric: %f' % dist)
        else:
            pass

    def _process_key(self, event):
        fig = event.canvas.figure
        ax0 = fig.axes[0]
        if event.key == 'j':
            self._previous_slice(ax0)
        elif event.key == 'k':
            self._next_slice(ax0)
        fig.canvas.draw()
        # plt.show()
        # fig.canvas.flush_events()

    def _previous_slice(self, ax0):
        print(ax0.index)
        if ax0.index > 0:
            ax0.index = (ax0.index - 1)  # wrap around using %
            # ax0.images[0].set_array(self._volume[ax0.index])  # fevent histograms
            im = ax0.imshow(self._volume[ax0.index], cmap='jet')  # fevent histograms

    def _next_slice(self, ax0):
        print(ax0.index)
        if ax0.index < self._volume.shape[0] - 1:
            ax0.index = (ax0.index + 1)
            ax0.images[0].set_array(self._volume[ax0.index])  # fevent histograms