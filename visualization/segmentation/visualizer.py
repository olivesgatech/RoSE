import argparse
import toml
import glob
import os
from config import BaseConfig
import numpy as np
from visualization.segmentation.volumeviewer import VolumeViewer
from visualization.utils.plotting import plot_fevent_switch_event_histograms

max_range=800


class Visualizer:
    def __init__(self, cfg: BaseConfig):
        self._cfg = cfg
        if cfg.visualization.machine == 'win':
            self._separator = '\\'
        else:
            self._separator = '/'

        self._volume = np.load(os.path.expanduser(self._cfg.visualization.seismic.target_volume))
        self._volume = np.flip(np.swapaxes(self._volume, 1, 2), axis=1)
        self._training_volume = np.load(os.path.expanduser(self._cfg.visualization.seismic.training_volume))

        if self._cfg.visualization.seismic.calculate_distances:
            self._viewer = VolumeViewer(cfg, target_volume=self._volume, training_volume=self._training_volume)
        else:
            self._viewer = VolumeViewer(cfg, target_volume=self._volume)

    def visualize_volume(self):
        volume = np.load(os.path.expanduser(self._cfg.visualization.seismic.target_volume))
        print(volume.shape)
        #volume = volume/np.max(volume)
        self._viewer.multi_slice_viewer()

    def calculate_distances(self):
        data_path = os.path.expanduser(self._cfg.visualization.hist_visualization_folder)
        folders = glob.glob(data_path + '*')
        files = []
        train_dir = os.path.expanduser(self._cfg.visualization.hist_visualization_folder + 'training_inline/')
        if os.path.exists(train_dir):
            pass
        else:
            train_dir = None

        for folder in folders:
            file_list = glob.glob(folder + '/*')
            out = []
            for f in file_list:
                if f.split(self._separator)[-2].split('_')[-1] == 'inline':
                     out.append(f)
            #files.extend(file_list)
            files.extend(out)

        plot_fevent_switch_event_histograms(files=files,
                                            separator=self._separator,
                                            cfg=self._cfg,
                                            reference_folder=train_dir)

    def visualize_uspec_switches(self):
        data_path = os.path.expanduser(self._cfg.visualization.uspec_visualization_folder)
        folders = glob.glob(data_path + '*')
        files = []

        for folder in folders:
            file_list = glob.glob(folder + '/*')
            files.extend(file_list)

        for file in files:
            predictions = np.load(file).flatten()
            print('FILE: ' + file)
            print('USPEC Mean: %f' % np.mean(predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/alnfr/example_config.toml')

    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    cfg = BaseConfig(configs)
    visualizer = Visualizer(cfg)
    visualizer.visualize_volume()
    #visualizer.calculate_distances()
    #visualizer.visualize_uspec_switches()