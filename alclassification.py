import random
import toml
import os
import shutil
import argparse
import torch
import numpy as np
import pandas as pd
from config import BaseConfig
from data.datasets.classification.common.aquisition import get_dataset
from training.classification.utils import PCFocalLoss
from activelearning.classification.trainer import ActiveLearningClassificationTrainer
from activelearning.qustrategies import get_sampler
from uspecanalysis.utils.classificationtracker import get_uspec_inputs


def main(cfg: BaseConfig):
    np.random.seed(cfg.active_learning.init_seed)

    # get all relevant statistics for the dataset
    train_configs = get_dataset(cfg)
    n_pool = train_configs.data_config.train_len
    nquery = cfg.active_learning.n_query
    nstart = cfg.active_learning.n_start
    nend = cfg.active_learning.n_end
    start_idxs = np.arange(n_pool)[np.random.permutation(n_pool)][:nstart]

    # focal pc loss
    if cfg.classification.pc_training:
        pc_loss = PCFocalLoss(cfg, n_pool)
    else:
        pc_loss = None

    if nend < n_pool:
        nrounds = int((nend - nstart) / nquery)
        print('Rounds: %d' % nrounds)
    else:
        nrounds = int((n_pool - nstart) / nquery) + 1
        print('Number of end samples too large! Using total number of samples instead. Rounds: %d Total Samples: %d' %
              (nrounds, n_pool))

    # define ideal model if applicable
    ideal_model = 'none'
    if cfg.active_learning.ideal_model_path != 'none':
        print(f'Loading ideal model at {cfg.active_learning.ideal_model_path}')
        ideal_model = ActiveLearningClassificationTrainer(cfg=cfg,
                                                          ideal_model_path=cfg.active_learning.ideal_model_path)

    for i in range(cfg.active_learning.start_seed, cfg.active_learning.end_seed):
        # set random seeds
        random.seed(i)
        torch.manual_seed(i)

        sampler = get_sampler(cfg=cfg, n_pool=n_pool, start_idxs=start_idxs)
        accuracies = []
        uspec_inputs = get_uspec_inputs(cfg)
        round = 0
        while np.sum(sampler.total_pool) <= nend:
            trainer = ActiveLearningClassificationTrainer(cfg)
            trainer.update_loader(sampler.idx_current.astype(int), np.squeeze(np.argwhere(sampler.total_pool == 0)))

            if not isinstance(ideal_model, str):
                ideal_model.update_loader(sampler.idx_current.astype(int),
                                          np.squeeze(np.argwhere(sampler.total_pool == 0)))

            acc = 0.0
            epoch = 0
            target_folder = cfg.run_configs.ld_folder_name + '/round' + str(round) + '/seed' + str(i) + '/'
            prev_acc = -1.0
            acc_count = 0

            while epoch < cfg.classification.epochs:
                if acc > cfg.active_learning.convergence_acc:
                    print('Breaking since convergence accuracy reached')
                    break
                # if cfg.run_configs.create_validation:
                    # _, val_acc, _, _ = trainer.validation(0)
                    # if val_acc < prev_acc:
                    #     acc_count += 1
                    # else:
                    #     acc_count = 0
                    # if acc_count == 3:
                    #     print('Breaking due to overfitting!')
                    #     break
                    # prev_acc = val_acc
                # breaking because model has converged
                if epoch > cfg.active_learning.max_epochs and acc > 90.0:
                    print('Exiting since model has converged!')
                    break
                # reset model if not converging -> set to epoch +1 s.t. it does not set off in the first epoch
                if (epoch + 1) % 400 == 0 and acc < 13.0:
                    print('Model not converging! Resetting now!')
                    del(trainer)
                    trainer = ActiveLearningClassificationTrainer(cfg)
                    trainer.update_loader(sampler.idx_current.astype(int),
                                          np.squeeze(np.argwhere(sampler.total_pool == 0)))
                    epoch = 1
                if cfg.classification.pc_training and round != 0:
                    acc = trainer.training(epoch, pc_loss=pc_loss)
                else:
                    acc = trainer.training(epoch)
                print('Round: %d' % round)
                print('Seed: %d' % i)

                # track learning dynamics on holdout sets
                if cfg.classification.track_statistics:
                    _ = trainer.get_unlabeled_statistics(0)
                    for elem in uspec_inputs:
                        print(elem[1])
                        _, _, _ = trainer.testing(0, alternative_loader_struct=elem[2])

                # perform sampling action
                if isinstance(ideal_model, str):
                    relevant_trainer = trainer
                else:
                    relevant_trainer = ideal_model
                sampler.action(relevant_trainer)
                sampler.track_statistics(relevant_trainer)
                epoch += 1

            if isinstance(ideal_model, str):
                relevant_trainer = trainer
            else:
                relevant_trainer = ideal_model
            new_idxs = sampler.query(nquery, relevant_trainer)
            sampler.update(new_idxs)

            # save query idxs
            if cfg.active_learning.save_query_idxs:
                path = cfg.run_configs.ld_folder_name + '/round' + str(round) + '/queryidxs/'
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/newidxs_seed' + str(i) + '.npy', np.squeeze(new_idxs))

            # save query switches
            if cfg.active_learning.save_switches:
                print('Sampling Switches')
                unl_dict = trainer.get_unlabeled_statistics(0)
                switches, indices, accs = unl_dict['switches'], unl_dict['full inds'], unl_dict['sample accuracy']
                path = cfg.run_configs.ld_folder_name + '/round' + str(round) + '/switches/'
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/switches_seed' + str(i) + '.npy', np.squeeze(switches))
                np.save(path + '/indices_seed' + str(i) + '.npy', np.squeeze(indices))
                np.save(path + '/accs_seed' + str(i) + '.npy', np.squeeze(accs))

            # save forgetting statistics
            folder = target_folder
            if cfg.classification.track_statistics:
                trainer.save_train_statistics(folder)
                trainer.save_unlabeled_statistics(folder)
                for elem in uspec_inputs:
                    print(elem[1])
                    loader_struct = elem[2]
                    tracker = loader_struct[1]
                    tracker.save_statistics(folder, elem[1])

            cur_row = []
            names = []
            for elem in uspec_inputs:
                print(elem[1])
                cur_out_dict, cur_acc, auc = trainer.testing(0, alternative_loader_struct=elem[2])
                cur_row.append(cur_acc)
                cur_row.append(auc)
                names.append(elem[1])
                names.append(f'{elem[1]} AUROC')
                path = cfg.run_configs.ld_folder_name + '/round' + str(round) + '/uspec_statistics/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/predictions_seed' + str(i) + '.npy', np.squeeze(cur_out_dict['predictions']))
                if i == cfg.active_learning.start_seed:
                    np.save(path + '/gts.npy', np.squeeze(cur_out_dict['gt']))
                np.save(path + '/scores_seed' + str(i) + '.npy', np.squeeze(cur_out_dict['scores']))
                # elem[0].update(preds, i)
            names.append('Batch Size')
            cur_row.append(sampler.batch_size(nquery))
            accuracies.append(cur_row)

            if cfg.classification.pc_training:
                trainer.update_loader(sampler.idx_current.astype(int), np.squeeze(np.argwhere(sampler.total_pool == 0)))
                assert pc_loss is not None
                pc_loss = trainer.update_pc_loss(pc_loss)
            del trainer
            round += 1

            acc_df = pd.DataFrame(np.array(accuracies), columns=names)
            print(acc_df)
            acc_df.to_excel(cfg.run_configs.ld_folder_name + '/al_seed' + str(i) + '.xlsx')

    # for round in range(nrounds):
    #    for elem in trackers[round]:
    #        folder = cfg.run_configs.ld_folder_name + '/round' + str(round) + '/'
    #        name = elem[1]
    #        elem[0].save_statistics(directory=folder, ld_type=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/alnfr/example_config.toml')

    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    main(configs)
    shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + 'parameters.toml')