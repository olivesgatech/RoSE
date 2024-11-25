import argparse
import toml
import os
import shutil
import random
import torch
import numpy as np
from config import BaseConfig
from training.classification.trainer import ClassificationTrainer
from uspecanalysis.utils.classificationtracker import USPECTracker, get_uspec_inputs


def train_model(cfg: BaseConfig):
    trainer = ClassificationTrainer(cfg)
    epochs = cfg.classification.epochs
    for i in range(epochs):
        trainer.training(i, save_checkpoint=True)

    return trainer


def main(cfg: BaseConfig):
    # get relevant uspec configs
    uspec_inputs = []

    for i in range(cfg.uspec_configs.num_seeds):
        # set random seeds
        random.seed(i)
        torch.manual_seed(i)
        np.random.seed(i)

        trainer = train_model(cfg)
        if len(uspec_inputs) == 0:
            uspec_inputs = get_uspec_inputs(cfg, trainer)

        for elem in uspec_inputs:
            print(elem[1])
            preds, _ = trainer.testing(0, mode=elem[1])
            elem[0].update(preds, i)

        del(trainer)

    for elem in uspec_inputs:
        folder = cfg.run_configs.ld_folder_name
        elem[0].save_statistics(directory=folder, ld_type=elem[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/USPEC-LD/example_config.toml')

    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    main(configs)
    shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + 'parameters.toml')
