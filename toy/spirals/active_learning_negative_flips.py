from toy.infrastructure.nfr import ModelBilinear, train_model, eval_on_single_set
from toy.infrastructure.dataset import ArtificialDataset
from toy.spirals.create_dataset import create_spiral_dataset
from torch.utils.data import Dataset, DataLoader
import models.classification.torchmodels as models
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch


def run_al_round(ds_train: ArtificialDataset, batch_size: int, num_classes: int, num_epochs: int,
                 dataloader_test: DataLoader,
                 nfs: np.ndarray, prev_acc: np.ndarray):
    dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    model = ModelBilinear(20, num_classes)

    nfs, prev_acc = train_model(dataloader_train, dataloader_test, nfs, prev_acc, model, epochs=num_epochs)
    return nfs, prev_acc, model


def get_training_set(idxs: np.ndarray, data: np.ndarray, target: np.ndarray):
    dataset = ArtificialDataset(x=torch.from_numpy(data[idxs]).float(),
                                y=F.one_hot(torch.from_numpy(target[idxs])).float())
    return dataset

def main():
    num_spirals = 8
    batch_size = 32
    n_al_rounds = 40
    num_epochs = 100
    num_samples = 500
    al_bs = num_samples*num_spirals // n_al_rounds
    noise = 0.3
    aux_data = False
    num_classes = num_spirals + int(aux_data)
    ds_train, d_train, y_train = create_spiral_dataset(num_spirals=num_spirals, noise=noise, add_aux_data=aux_data,
                                                       num_samples=num_samples)
    ds_test, d_test, y_test = create_spiral_dataset(num_spirals=num_spirals, noise=noise, add_aux_data=aux_data,
                                                    num_samples=num_samples)
    # model = models.VGG(type=16, num_classes=num_classes)

    # domain = 10
    # x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
    # y_lin = np.linspace(-domain, domain, 100)

    # xx, yy = np.meshgrid(x_lin, y_lin)

    # x_grid = np.column_stack([xx.flatten(), yy.flatten()])
    # y_grid = np.zeros(x_grid.shape[0], dtype=int)

    # ds_test = ArtificialDataset(x=torch.from_numpy(x_grid).float(), y=F.one_hot(torch.from_numpy(y_grid),
    #                                                                             num_classes).float())
    dataloader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

    prev_acc = np.zeros(d_test.shape[0])
    nfs = np.zeros(d_test.shape[0])
    idx_order = np.zeros(num_samples*num_spirals, dtype=int)
    total_idxs = np.arange(num_samples*num_spirals)
    # idx_order = np.arange(num_samples*num_spirals)[np.random.permutation(num_samples * num_spirals)]
    rand_perm = total_idxs[np.random.permutation(num_samples * num_spirals)]
    unlabeled_nfs = np.zeros(num_samples * num_spirals)
    unlabeled_prev_acc = np.zeros(num_samples * num_spirals)

    for round in range(n_al_rounds):
        unlabeled_idx = idx_order[idx_order == 0]
        if round == 0:
            rel_idx = total_idxs[np.random.permutation(num_samples * num_spirals)][:al_bs]
        else:
            rel_idx = np.argsort(cur_unl_nfs.astype(int))[:al_bs]
        # rel_idx = rand_perm[:round*al_bs + al_bs]
        idx_order[rel_idx] = 1
        cur_trainset = get_training_set(total_idxs[idx_order == 1], d_train, y_train)
        nfs, prev_acc, model = run_al_round(cur_trainset, batch_size, num_classes, num_epochs, dataloader_test, nfs, prev_acc)

        try:
            unlabeled_set = DataLoader(get_training_set(idx_order == 0, d_train, y_train), batch_size=batch_size,
                                       shuffle=False, drop_last=True)
            cur_unl_nfs, cur_unl_prev_acc = eval_on_single_set(unlabeled_set,
                                                               unlabeled_nfs[idx_order == 0],
                                                               unlabeled_prev_acc[idx_order == 0],
                                                               model)
            unlabeled_nfs[idx_order == 0] += cur_unl_nfs
            unlabeled_prev_acc[idx_order == 0] = cur_unl_prev_acc
        except:
            break

    # z = switches.reshape(xx.shape)

    plt.figure()
    # plt.contourf(x_lin, y_lin, z)
    plt.scatter(d_test[:, 0], d_test[:, 1], c=nfs)
    # plt.scatter(d_test[nfs > 0, 0], d_test[nfs > 0, 1], c=nfs[nfs > 0])
    plt.show()


if __name__ == '__main__':
    main()
