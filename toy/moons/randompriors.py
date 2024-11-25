from toy.infrastructure.randompriors import ModelBilinear, train_model
from toy.infrastructure.dataset import ArtificialDataset
from toy.moons.create_dataset import create_moon_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch


def main():
    num_spirals = 4
    batch_size = 32
    num_epochs = 50
    num_features = 100
    num_priors = 100
    noise = 0.2
    aux_data = False
    num_classes = num_spirals + int(aux_data)
    ds_train, d_train, y_train = create_moon_dataset(noise=noise)
    dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    model = ModelBilinear(num_features)
    # model = models.VGG(type=16, num_classes=num_classes)

    domain = 100
    x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
    y_lin = np.linspace(-domain, domain, 100)

    xx, yy = np.meshgrid(x_lin, y_lin)

    x_grid = np.column_stack([xx.flatten(), yy.flatten()])
    y_grid = np.zeros(x_grid.shape[0], dtype=int)

    ds_test = ArtificialDataset(x=torch.from_numpy(x_grid).float(), y=F.one_hot(torch.from_numpy(y_grid),
                                                                                num_classes).float())
    dataloader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

    uncertainty = np.zeros(x_grid.shape[0])

    uncertainty = train_model(dataloader_train, dataloader_test, model=model, uncertainty=uncertainty,
                              epochs=num_epochs, num_features=num_features, num_priors=num_priors)

    z = uncertainty.reshape(xx.shape)

    plt.figure()
    plt.contourf(x_lin, y_lin, z)
    plt.scatter(d_train[:, 0], d_train[:, 1], c=y_train)
    plt.show()


if __name__ == '__main__':
    main()
