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


def eval_step(batch: Dict[str, torch.Tensor], switches: np.ndarray, prev_pred: np.ndarray, model: nn.Module):
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


def train_model(trainset: DataLoader, testset: DataLoader, switches: np.ndarray, prev_pred: np.ndarray,
                model: nn.Module, epochs: int = 100, track_switches: bool = False):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    for epoch in range(epochs):
        for i, batch in enumerate(trainset):
            model.train()
            optimizer.zero_grad()

            x, y = batch['x'], batch['y']
            x.requires_grad_(True)

            y_pred = model(x)

            loss1 = F.binary_cross_entropy(y_pred, y)

            loss = loss1

            loss.backward()
            optimizer.step()

        if track_switches:
            num_samples = 0
            correct_samples = 0
            for k, batch in enumerate(testset):
                pred, target, sample, switches, prev_pred = eval_step(batch, switches, prev_pred, model)
                acc = pred.eq(target)
                correct_samples += torch.sum(acc)
                num_samples += target.shape[0]

            print(f"Test Results - Epoch: {epoch} Acc: {correct_samples / num_samples} Loss: {loss} ")
        else:
            print(f"Train - Epoch: {epoch} Loss: {loss} ")

    return switches
