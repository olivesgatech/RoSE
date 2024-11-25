import argparse
import toml
import os
import shutil
import random
import torch
import numpy as np
from config import BaseConfig
from training.segmentation.trainer import SegmentationTrainer
from data.datasets.segmentation.common.acquisition import get_stat_config


def train_model(cfg: BaseConfig):
    trainer = SegmentationTrainer(cfg)
    epochs = cfg.segmentation.epochs
    for i in range(epochs):
        trainer.training(i, save_checkpoint=True)

    return trainer


def main(cfg: BaseConfig):
    # get relevant uspec configs
    uspec_init = False

    for i in range(cfg.uspec_configs.num_seeds):
        # set random seeds
        random.seed(i)
        torch.manual_seed(i)
        np.random.seed(i)

        trainer = train_model(cfg)
        if not uspec_init:
            loaders = trainer.get_loaders()
            stat_track_configs = get_stat_config(loaders, cfg, uspec_analysis=True)
            uspec_init = True

        for elem in stat_track_configs:
            print('Testing ' + elem.name)
            output = trainer.testing(0, loader=elem.loader, tracker=None)
            elem.tracker.update(output.prediction, i)

        del(trainer)

    for elem in stat_track_configs:
        folder = cfg.run_configs.ld_folder_name
        elem.tracker.save_statistics(directory=folder, ld_type=elem.name, cfg=cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/USPEC-LD/example_config.toml')

    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    main(configs)
    shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + 'parameters.toml')
