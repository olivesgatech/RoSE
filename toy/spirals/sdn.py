from toy.infrastructure.sdn import ModelBilinear, train_model
from toy.infrastructure.dataset import ArtificialDataset
from toy.spirals.create_dataset import create_spiral_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch


def entropy(probs: np.ndarray, axis: int):
    logs = np.log2(probs)
    mult = logs * probs
    entropy = -1 * np.sum(mult, axis=axis)
    entropy[np.isnan(entropy)] = 0
    return np.squeeze(entropy)


def main():
    num_spirals = 3
    batch_size = 32
    num_epochs = 800
    num_ensembles = 1
    num_sdns = 2
    noise = 0.5
    lr = 0.001 # adam
    # lr = 0.1 # sgd
    aux_data = False
    num_classes = num_spirals + int(aux_data)
    ds_train, d_train, y_train = create_spiral_dataset(num_spirals=num_spirals, noise=noise, add_aux_data=aux_data)
    dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

    num_points = 2
    ood_data = np.random.multivariate_normal(mean=[15, 3.3], cov=[[1.0, 0.0], [0.0, 1.0]], size=num_points)
    ood_data = None

    domain = 10
    lim = None
    x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
    y_lin = np.linspace(-domain, domain, 100)

    xx, yy = np.meshgrid(x_lin, y_lin)

    x_grid = np.column_stack([xx.flatten(), yy.flatten()])
    y_grid = np.zeros(x_grid.shape[0], dtype=int)

    ds_test = ArtificialDataset(x=torch.from_numpy(x_grid).float(), y=F.one_hot(torch.from_numpy(y_grid),
                                                                                num_classes).float())
    dataloader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

    total_probs = np.zeros((x_grid.shape[0], num_sdns + 1, num_classes))
    total_ood = None
    for seed in range(num_ensembles):
        if ood_data is not None and num_spirals != 3:
            raise ValueError(f'Num spirals must be == 3 if you want to use ood data!')
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ModelBilinear(features=5, num_sdns=num_sdns, num_output_classes=num_classes)
        probs = np.zeros((x_grid.shape[0], num_sdns + 1, num_classes))

        probs, ood_probs = train_model(dataloader_train, dataloader_test, probs, model, epochs=num_epochs,
                                       ensemble=seed, lr=lr,
                                       ood_data=ood_data)
        if ood_probs is not None:
            total_ood = np.concatenate((total_ood, ood_probs[np.newaxis, ...]), axis=0) if total_ood is not None else \
                ood_probs[np.newaxis, ...]
        total_probs += probs
        del model
    mean_probs = np.mean(total_probs, axis=1)
    total_entropy = entropy(mean_probs, axis=1)
    var_entropy = entropy(total_probs, axis=2)
    mutual_info = total_entropy - np.mean(var_entropy, axis=1)

    z = mutual_info.reshape(xx.shape)

    plt.figure()
    plt.contourf(x_lin, y_lin, z)
    plt.scatter(d_train[:, 0], d_train[:, 1], c=y_train*0.5, cmap='winter')
    plt.show()


'''
    fig, axs = plt.subplots(1, 2)
    axs[0].contourf(xx, yy, z)
    axs[1].scatter(d_train[:, 0], d_train[:, 1], c=y_train + 5, cmap='Blues')
    if ood_data is not None:
        axs[1].scatter(ood_data[:, 0], ood_data[:, 1], color='red')
    if lim is not None:
        plt.xlim((-lim, lim))
        plt.ylim((-lim, lim))
    # plt.scatter(d_train[:, 0], d_train[:, 1], c=y_train, cmap='Blues')
    plt.show()

    if total_ood is not None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(num_ensembles):
            ax.scatter(total_ood[i, :, 0], total_ood[i, :, 1], total_ood[i, :, 2])
        ax.set_xlim3d((0.0, 1.0))
        ax.set_ylim3d((0.0, 1.0))
        ax.set_zlim3d((0.0, 1.0))
        plt.show()
'''





if __name__ == '__main__':
    main()
