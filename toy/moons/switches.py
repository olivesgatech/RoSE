import torch
import torch.utils.data
from typing import Dict
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# code modified from https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/two_moons.ipynb


class ModelBilinear(nn.Module):
    def __init__(self, features, num_output_classes):
        super().__init__()

        self.gamma = 0.99
        self.sigma = 0.3

        embedding_size = 10

        self.fc1 = nn.Linear(2, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, num_output_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x


class MoonDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self._x = x
        self._y = y
        self.switches = np.zeros(self._x.shape[0])
        self.prev_pred = np.zeros(self._x.shape[0]) - 1

    def __getitem__(self, idx):
        sample = {'x': self._x[idx],
                  'y': self._y[idx],
                  'idx': idx}
        return sample

    def __len__(self):
        return self._x.shape[0]

np.random.seed(0)
torch.manual_seed(0)

l_gradient_penalty = 1.0

# Moons
noise = 0.1
X_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
X_test, y_test = sklearn.datasets.make_moons(n_samples=1000, noise=noise)

num_classes = 2
batch_size = 64

model = ModelBilinear(20, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)


def output_transform_acc(output):
    y_pred, y, x = output

    y = torch.argmax(y, dim=1)

    return y_pred, y


def output_transform_bce(output):
    y_pred, y, x = output

    return y_pred, y


def step(batch):
    global model
    global optimizer
    model.train()
    optimizer.zero_grad()

    x, y = batch
    x.requires_grad_(True)

    y_pred = model(x)

    loss1 = F.binary_cross_entropy(y_pred, y)

    loss = loss1

    loss.backward()
    optimizer.step()

    # evaluator.run(dl_test)

    return loss.item()


def eval_step(batch: Dict[str, torch.Tensor], switches: np.ndarray, prev_pred: np.ndarray):
    model.eval()

    x, y, idx = batch['x'], batch['y'], batch['idx']

    x.requires_grad_(True)

    y_pred = model(x)

    pred = torch.argmax(y_pred, dim=1)
    pred = pred.cpu().numpy()
    acc = prev_pred[idx] != pred
    switches[idx] = switches[idx] + acc.astype(int)
    prev_pred[idx] = pred

    target = torch.argmax(y, dim=1)
    return torch.from_numpy(pred), target, x, switches, prev_pred


def main(trainset: DataLoader, testset: DataLoader, switches: np.ndarray, prev_pred: np.ndarray,
         model: nn.Module,
         epochs: int = 100):

    for epoch in range(epochs):
        for i, batch in enumerate(trainset):
            model.train()
            optimizer.zero_grad()

            x, y = batch
            x.requires_grad_(True)

            y_pred = model(x)

            loss1 = F.binary_cross_entropy(y_pred, y)

            loss = loss1

            loss.backward()
            optimizer.step()

        num_samples = 0
        correct_samples = 0
        for k, batch in enumerate(testset):
            pred, target, sample, switches, prev_pred = eval_step(batch, switches, prev_pred)
            acc = pred.eq(target)
            correct_samples += torch.sum(acc)
            num_samples += target.shape[0]

        print(f"Test Results - Epoch: {epoch} Acc: {correct_samples / num_samples} Loss: {loss} ")

    return switches

domain = 3
x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
y_lin = np.linspace(-domain, domain, 100)

xx, yy = np.meshgrid(x_lin, y_lin)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])
y_grid = np.zeros(X_grid.shape[0], dtype=int)


ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = MoonDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)

prev_pred = np.zeros(X_test.shape[0]) - 1
switches = np.zeros(X_test.shape[0])

switches = main(dl_train, dl_test, switches, prev_pred, model)

# X_vis, y_vis = sklearn.datasets.make_moons(n_samples=1000, noise=noise)
X_vis, y_vis = X_test, y_test
mask = y_vis.astype(np.bool)

with torch.no_grad():
    output = model(torch.from_numpy(X_grid).float())
    confidence = output.max(1)[0].numpy()

# z = switches.reshape(xx.shape)

# plt.figure()
# plt.contourf(x_lin, y_lin, z, cmap='cividis')

# plt.scatter(X_vis[mask, 0], X_vis[mask, 1], c=switches[mask])
# plt.scatter(X_vis[~mask, 0], X_vis[~mask, 1], c=switches[~mask])
# plt.scatter(X_vis[mask, 0], X_vis[mask, 1])
# plt.scatter(X_vis[~mask, 0], X_vis[~mask, 1])
print(switches)
# plt.show()


fig, ax = plt.subplots(1, 2)

ax[0].scatter(X_vis[mask, 0], X_vis[mask, 1], c=switches[mask])
ax[0].scatter(X_vis[~mask, 0], X_vis[~mask, 1], c=switches[~mask])
ax[1].scatter(X_vis[mask, 0], X_vis[mask, 1])
ax[1].scatter(X_vis[~mask, 0], X_vis[~mask, 1])

plt.show()