import argparse
import toml
import shutil
import os
import random
import numpy as np
import torch
from config import BaseConfig
from training.classification.trainer import ClassificationTrainer


def main(cfg: BaseConfig):
    trainer = ClassificationTrainer(cfg)

    if cfg.run_configs.train:
        epochs = cfg.classification.epochs
        for i in range(epochs):
            trainer.training(i, save_checkpoint=True, track_summaries=True)

    # save learning dynamics
    if not os.path.exists(cfg.run_configs.ld_folder_name):
        os.makedirs(cfg.run_configs.ld_folder_name)

    if cfg.classification.track_statistics:
        trainer.save_statistics(cfg.run_configs.ld_folder_name)


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
