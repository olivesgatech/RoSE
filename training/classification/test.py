import argparse
import toml
import shutil
import os
import random
import numpy as np
import torch
from config import BaseConfig
from uspecanalysis.utils.classificationtracker import get_uspec_inputs
from training.classification.trainer import ClassificationTrainer


def main(cfg: BaseConfig):
    trainer = ClassificationTrainer(cfg)
    uspec_inputs = get_uspec_inputs(cfg)

    for elem in uspec_inputs:
        print(elem[1])
        preds, cur_acc = trainer.testing(0, alternative_loader_struct=elem[2])


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
