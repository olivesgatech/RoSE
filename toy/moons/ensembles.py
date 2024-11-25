from toy.infrastructure.ensembles import ModelBilinear, train_model
from toy.infrastructure.dataset import ArtificialDataset
from toy.moons.create_dataset import create_moon_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch


def entropy(probs: np.ndarray):
    logs = np.log2(probs)
    mult = logs * probs
    entropy = -1 * np.sum(mult, axis=1)
    return np.squeeze(entropy)


def main():
    num_samples = 1000
    batch_size = 32
    num_epochs = 40
    num_ensembles = 10
    noise = 0.1
    aux_data = True
    num_classes = 2
    ds_train, d_train, y_train = create_moon_dataset(num_samples=num_samples, noise=noise)
    dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

    domain = 5
    x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
    y_lin = np.linspace(-domain, domain, 100)

    xx, yy = np.meshgrid(x_lin, y_lin)

    x_grid = np.column_stack([xx.flatten(), yy.flatten()])
    y_grid = np.zeros(x_grid.shape[0], dtype=int)

    ds_test = ArtificialDataset(x=torch.from_numpy(x_grid).float(), y=F.one_hot(torch.from_numpy(y_grid),
                                                                                num_classes).float())
    dataloader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

    total_probs = np.zeros((x_grid.shape[0], num_classes))

    for seed in range(num_ensembles):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ModelBilinear(20, num_classes)
        probs = np.zeros((x_grid.shape[0], num_classes))

        probs = train_model(dataloader_train, dataloader_test, probs, model, epochs=num_epochs, ensemble=seed)
        total_probs += probs
        del model
    total_probs /= num_ensembles
    total_entropy = entropy(total_probs)
    z = total_entropy.reshape(xx.shape)

    plt.figure()
    plt.contourf(x_lin, y_lin, z)
    # plt.scatter(d_train[:, 0], d_train[:, 1], c=y_train, cmap='Greens')
    plt.show()


if __name__ == '__main__':
    main()
