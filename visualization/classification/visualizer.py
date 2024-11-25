import argparse
import toml
import glob
import os
from config import BaseConfig
import numpy as np
import matplotlib.pyplot as plt
from visualization.utils.plotting import plot_fevent_switch_event_histograms, plot_uspec_unique_prediction_switches


class Visualizer:
    def __init__(self, cfg: BaseConfig):
        self._cfg = cfg
        if cfg.visualization.machine == 'win':
            self._separator = '\\'
        else:
            self._separator = '/'

    def visualize_forgetting_events(self):
        '''
        Visualizes forgetting events numpy array in data_path
        Parameters:
            :param data_path: path to the example forgetting statistics
            :type data_path: str
        '''
        data_path = os.path.expanduser(self._cfg.visualization.hist_visualization_folder)
        train_dir = data_path + 'training'
        if os.path.exists(train_dir):
            pass
        else:
            train_dir = None
        folders = glob.glob(data_path + '*')
        files = []
        for folder in folders:
            file_list = glob.glob(folder + '/*')
            files.extend(file_list)
        plot_fevent_switch_event_histograms(files=files,
                                            separator=self._separator,
                                            cfg=self._cfg,
                                            reference_folder=train_dir)

    def visualize_uspec_switches(self):
        '''
        visualizes uspec switches in form of a histogram for all predicitons
        :return:
        '''
        data_path = os.path.expanduser(self._cfg.visualization.uspec_visualization_folder)
        folders = glob.glob(data_path + '*')
        files = []
        for folder in folders:
            file_list = glob.glob(folder + '/*')
            files.extend(file_list)
        plot_uspec_unique_prediction_switches(files=files, separator=self._separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/USPEC-LD/example_config.toml')

    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    cfg = BaseConfig(configs)
    visualizer = Visualizer(cfg)
    visualizer.visualize_forgetting_events()
    visualizer.visualize_uspec_switches()
