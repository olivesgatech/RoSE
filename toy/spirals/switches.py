from toy.infrastructure.switches import ModelBilinear, train_model
from toy.infrastructure.dataset import ArtificialDataset
from toy.spirals.create_dataset import create_spiral_dataset
from torch.utils.data import Dataset, DataLoader
import models.classification.torchmodels as models
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch


def main():
    num_spirals = 4
    batch_size = 32
    num_epochs = 100
    noise = 1.0
    aux_data = False
    num_classes = num_spirals + int(aux_data)
    ds_train, d_train, y_train = create_spiral_dataset(num_spirals=num_spirals, noise=noise, add_aux_data=aux_data)
    dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    model = ModelBilinear(20, num_classes)
    # model = models.VGG(type=16, num_classes=num_classes)

    domain = 40
    x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
    y_lin = np.linspace(-domain, domain, 100)

    xx, yy = np.meshgrid(x_lin, y_lin)

    x_grid = np.column_stack([xx.flatten(), yy.flatten()])
    y_grid = np.zeros(x_grid.shape[0], dtype=int)

    ds_test = ArtificialDataset(x=torch.from_numpy(x_grid).float(), y=F.one_hot(torch.from_numpy(y_grid),
                                                                                num_classes).float())
    dataloader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

    prev_pred = np.zeros(x_grid.shape[0]) - 1
    switches = np.zeros(x_grid.shape[0])

    switches = train_model(dataloader_train, dataloader_test, switches, prev_pred, model, epochs=num_epochs,
                           track_switches=True)

    z = switches.reshape(xx.shape)

    plt.figure()
    plt.contourf(x_lin, y_lin, z)
    # plt.scatter(d_train[:, 0], d_train[:, 1], c=y_train)
    plt.show()


if __name__ == '__main__':
    main()
