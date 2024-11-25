from toy.infrastructure.nfr import ModelBilinear, train_model
from toy.infrastructure.dataset import ArtificialDataset
from toy.moons.create_dataset import create_moon_dataset
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
    return nfs, prev_acc


def get_training_set(idxs: np.ndarray, data: np.ndarray, target: np.ndarray):
    dataset = ArtificialDataset(x=torch.from_numpy(data[idxs]).float(),
                                y=F.one_hot(torch.from_numpy(target[idxs])).float())
    return dataset


def main():
    num_spirals = 2
    batch_size = 32
    n_al_rounds = 10
    num_epochs = 100
    num_samples = 2000
    al_bs = num_samples // n_al_rounds
    noise = 0.1
    aux_data = False
    num_classes = num_spirals + int(aux_data)
    ds_train, d_train, y_train = create_moon_dataset(num_samples=num_samples, noise=noise)
    ds_test, d_test, y_test = create_moon_dataset(num_samples=num_samples, noise=noise)
    dataloader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

    prev_acc = np.zeros(d_test.shape[0])
    nfs = np.zeros(d_test.shape[0])
    idx_order = np.arange(num_samples)[np.random.permutation(num_samples)]

    for round in range(n_al_rounds):
        rel_idx = idx_order[:round*al_bs + al_bs]
        cur_trainset = get_training_set(rel_idx, d_train, y_train)
        nfs, prev_acc = run_al_round(cur_trainset, batch_size, num_classes, num_epochs, dataloader_test, nfs, prev_acc)

    # z = switches.reshape(xx.shape)

    plt.figure()
    # plt.contourf(x_lin, y_lin, z)
    plt.scatter(d_test[:, 0], d_test[:, 1], c=nfs)
    # plt.scatter(d_test[nfs > 0, 0], d_test[nfs > 0, 1], c=nfs[nfs > 0])
    plt.show()


if __name__ == '__main__':
    main()
