import random
import toml
import os
import shutil
import argparse
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from config import BaseConfig
from data.datasets.segmentation.common.acquisition import get_dataset
from activelearning.segmentation.trainer import ActiveLearningSegmentationTrainer
from activelearning.qustrategies import get_segmentation_sampler
from data.datasets.segmentation.common.acquisition import get_stat_config

def makedirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def main(cfg: BaseConfig):
    # qnp.random.seed(cfg.active_learning.init_seed)

    save_predictions = False

    # get all relevant statistics for the dataset
    train_configs = get_dataset(cfg)
    n_pool = train_configs.data_config.train_len
    print(n_pool)
    nquery = cfg.active_learning.n_query
    nstart = cfg.active_learning.n_start
    nend = cfg.active_learning.n_end
    start_idxs = np.arange(n_pool)[np.random.permutation(n_pool)][:nstart]

    if nend < n_pool:
        nrounds = int((nend - nstart) / nquery)
        print('Rounds: %d' % nrounds)
    else:
        nrounds = int((n_pool - nstart) / nquery) + 1
        print('Number of end samples too large! Using total number of samples instead. Rounds: %d Total Samples: %d' %
              (nrounds, n_pool))

    for i in range(cfg.active_learning.start_seed, cfg.active_learning.end_seed):
        # set random seeds
        random.seed(i)
        torch.manual_seed(i)
        np.random.seed(i)

        sampler = get_segmentation_sampler(cfg=cfg, n_pool=n_pool, start_idxs=start_idxs)
        metrics = defaultdict(list)
        # uspec_inputs = get_uspec_inputs(cfg)

        for round in range(nrounds):
            trainer = ActiveLearningSegmentationTrainer(cfg)
            trainer.update_loader(sampler.idx_current.astype(int), np.squeeze(np.argwhere(sampler.total_pool == 0)))
            if round == 0:
                loaders = trainer.get_loaders()
                stat_track_configs = get_stat_config(loaders, cfg)

            acc = 0.0
            epoch = 1

            while acc < cfg.active_learning.convergence_acc:
                # breaking because model has converged
                if epoch > 300 and acc > 0.90:
                    print('Exiting since model has converged!')
                    break
                # reset model if not converging
                if epoch % 80 == 0 and acc < 0.2:
                    print('Model not converging! Resetting now!')
                    del(trainer)
                    trainer = ActiveLearningSegmentationTrainer(cfg)
                    trainer.update_loader(sampler.idx_current.astype(int),
                                          np.squeeze(np.argwhere(sampler.total_pool == 0)))
                    epoch = 1

                acc = trainer.training(epoch)
                print('Round: %d' % round)
                print('Seed: %d' % i)

                # perform sampling action
                sampler.action(trainer)
                epoch += 1

            new_idxs = sampler.query(nquery, trainer)
            sampler.update(new_idxs)

            # save query idxs
            if cfg.active_learning.save_query_idxs:
                sampler.save_query_information(new_idxs, round, seed=i)

            if cfg.active_learning.save_switch_images and round == nrounds - 1:
                switch_ims = trainer.get_switch_images()
                save_path = os.path.join(os.path.expanduser(cfg.run_configs.ld_folder_name),
                                         'final_switch_images')
                makedirs(save_path)
                for key, val in switch_ims.items():
                    path_name = os.path.join(save_path, f'{key}.npy')
                    np.save(path_name, val)

            cur_row = defaultdict(list)
            names = []
            for track_elem in stat_track_configs:
                print(track_elem.name)
                output = trainer.testing(i, loader=track_elem.loader, tracker=track_elem.tracker, track_stats=True)
                track_elem.tracker, df = output.tracker, output.statdf
                predictions = output.prediction
                images = output.images
                gt = output.gt

                save_interval = predictions.shape[0] // 10
                count = 0
                if cfg.segmentation.save_predictions:
                    for key_id in range(predictions.shape[0]):
                        if key_id // save_interval == 0:
                            save_path = os.path.join(os.path.expanduser(cfg.run_configs.ld_folder_name),
                                                     'predictions', f'round{round}', track_elem.name)
                            makedirs(save_path)
                            np.save(f'{save_path}/{count}.npy', predictions[key_id])
                            np.save(f'{save_path}/{count}_image.npy', images[key_id])
                            np.save(f'{save_path}/{count}_gt.npy', gt[key_id])
                            count += 1

                for col in df:
                    cur_row[col].append(df.iloc[0][col])
                names.append(track_elem.name)

            for _, key in enumerate(cur_row):
                metrics[key].append(cur_row[key])
            del trainer

            for _, metric in enumerate(metrics):
                save_path = os.path.join(os.path.expanduser(cfg.run_configs.ld_folder_name), metric)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                metric_df = pd.DataFrame(np.array(metrics[metric]), columns=names)
                print(f'{metric} \n{metric_df}')
                metric_df.to_excel(f'{save_path}/{metric}_seed{i}.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/alnfr/example_config.toml')

    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    main(configs)
    shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + 'parameters.toml')