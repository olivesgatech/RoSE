import argparse
import toml
import shutil
import os
import random
import torch
import numpy as np
from config import BaseConfig
from training.segmentation.trainer import SegmentationTrainer
from data.datasets.segmentation.common.acquisition import get_stat_config


def main(cfg: BaseConfig):
    trainer = SegmentationTrainer(cfg)
    loaders = trainer.get_loaders()
    stat_track_configs = get_stat_config(loaders, cfg)

    if cfg.run_configs.train:
        epochs = cfg.segmentation.epochs
        for i in range(epochs):
            trainer.training(i, save_checkpoint=True)
            for track_elem in stat_track_configs:
                print(track_elem.name)
                output = trainer.testing(i, loader=track_elem.loader, tracker=track_elem.tracker)
                track_elem.tracker, df = output.tracker, output.statdf

        # save statistics
        if not os.path.exists(cfg.run_configs.ld_folder_name):
            os.makedirs(cfg.run_configs.ld_folder_name)
        for track_elem in stat_track_configs:
            folder = cfg.run_configs.ld_folder_name
            track_elem.tracker.save_statistics(folder, track_elem.name)

    elif cfg.run_configs.test:
        _, df = trainer.testing(0)
    else:
        raise Exception('Execution type not supported yet!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/USPEC-LD/example_config.toml')

    args = parser.parse_args()
    # set random seeds deterministicly to 0
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    main(configs)
    shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + 'parameters.toml')
