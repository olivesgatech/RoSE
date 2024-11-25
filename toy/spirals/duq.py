import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
import seaborn as sns
from toy.infrastructure.duq import ModelBilinear, calc_gradient_penalty, output_transform_acc, \
    output_transform_gp, output_transform_bce
from toy.spirals.create_dataset import create_spiral_dataset

sns.set()
# code modified from https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/two_moons.ipynb


def step(engine, batch):
    model.train()
    optimizer.zero_grad()

    x, y = batch['x'], batch['y']
    x.requires_grad_(True)

    z, y_pred = model(x)

    loss1 = F.binary_cross_entropy(y_pred, y)
    loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)

    loss = loss1 + loss2

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.update_embeddings(x, y)

    return loss.item()


def eval_step(engine, batch):
    model.eval()

    x, y = batch['x'], batch['y']

    x.requires_grad_(True)

    z, y_pred = model(x)

    return y_pred, y, x, z


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    num_spirals = 3
    batch_size = 32
    num_epochs = 150
    num_ensembles = 20
    domain = 20
    noise = 0.8
    lr = 0.004
    aux_data = False
    num_classes = num_spirals + int(aux_data)

    l_gradient_penalty = 0.8
    ds_train, d_train, y_train = create_spiral_dataset(num_spirals=num_spirals, noise=noise, add_aux_data=aux_data)
    ds_test, d_test, y_test = create_spiral_dataset(num_spirals=num_spirals, noise=noise, add_aux_data=aux_data)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True, drop_last=True)

    batch_size = 64

    model = ModelBilinear(20, num_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Accuracy(output_transform=output_transform_acc)
    metric.attach(evaluator, "accuracy")

    metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
    metric.attach(evaluator, "bce")

    metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
    metric.attach(evaluator, "gp")

    # ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
    #                                           F.one_hot(torch.from_numpy(y_train)).float())
    # dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

    # ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(),
    #                                          F.one_hot(torch.from_numpy(y_test)).float())
    # dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        evaluator.run(dl_test)
        metrics = evaluator.state.metrics

        print("Test Results - Epoch: {} Acc: {:.4f} BCE: {:.2f} GP {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['bce'], metrics['gp']))


    trainer.run(dl_train, max_epochs=150)
    x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
    y_lin = np.linspace(-domain, domain, 100)

    xx, yy = np.meshgrid(x_lin, y_lin)

    X_grid = np.column_stack([xx.flatten(), yy.flatten()])

    with torch.no_grad():
        output = model(torch.from_numpy(X_grid).float())[1]
        confidence = output.max(1)[0].numpy()

    z = confidence.reshape(xx.shape)

    plt.figure()
    plt.contourf(x_lin, y_lin, -1*z)
    plt.show()
