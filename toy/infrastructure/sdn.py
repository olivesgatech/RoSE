import torch
import torch.utils.data
from typing import Dict, List
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

class LayerBlock(nn.Module):
    def __init__(self, in_features, out_features, num_classes):
        super(LayerBlock, self).__init__()
        self._init_layer = nn.Linear(in_features, out_features)
        self._bn = nn.BatchNorm1d(out_features)
        self._internal_classifier = nn.Linear(in_features, num_classes)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        classification_output = self._softmax(self._internal_classifier(input))
        output = F.relu(self._init_layer(input))
        return output, classification_output


class ModelBilinear(nn.Module):
    def __init__(self, features, num_sdns, num_output_classes):
        super().__init__()

        self.gamma = 0.99
        self.sigma = 0.3

        self._fc0 = nn.Linear(2, features)
        self._fc1 = nn.Linear(features, features)
        layers = []
        for i in range(num_sdns):
            layers.append(LayerBlock(features, features, num_output_classes))
        self._layers = layers
        self._final_layer = nn.Linear(features, num_output_classes)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = []
        x = F.relu(self._fc0(x))
        x = F.relu(self._fc1(x))
        for i in range(len(self._layers)):
            x, output = self._layers[i](x)
            outputs.append(output)
        x = self._softmax(self._final_layer(x))
        outputs.append(x)

        return outputs


class ModelBilinearDEP(nn.Module):
    def __init__(self, features, num_sdns, num_output_classes):
        super().__init__()

        self.gamma = 0.99
        self.sigma = 0.3

        embedding_size = 10

        self.fc1 = nn.Linear(2, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, num_output_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)

        return [x, x]


def output_transform_acc(output):
    y_pred, y, x = output

    y = torch.argmax(y, dim=1)

    return y_pred, y


def output_transform_bce(output):
    y_pred, y, x = output

    return y_pred, y


def sdn_loss(outputs: List[torch.Tensor], target: torch.Tensor):
    loss = 0.0
    temperature = 2
    norm = np.sum([np.exp(- temperature * k) for k in range(len(outputs))])
    weight = [np.exp(- temperature * k) / norm for k in range(len(outputs))]
    # weight = [1 / len(outputs) for k in range(len(outputs))]
    weight.reverse()
    # print(weight)
    for i in range(len(outputs)):
        loss += weight[i] * F.binary_cross_entropy(outputs[i], target)
    # loss = F.binary_cross_entropy(outputs[-1], target)
    return loss


def step(batch):
    global model
    global optimizer
    model.train()
    optimizer.zero_grad()

    x, y = batch
    x.requires_grad_(True)

    y_pred = model(x)

    loss1 = sdn_loss(y_pred, y)

    loss = loss1

    loss.backward()
    optimizer.step()

    # evaluator.run(dl_test)

    return loss.item()


def process_output(y_pred: List[torch.Tensor]):
    out_array = None
    for i in range(len(y_pred)):
        cur_array = y_pred[i].detach().cpu().numpy()
        out_array = np.concatenate((out_array, cur_array[:, np.newaxis, ...]), axis=1) if out_array is not None \
            else cur_array[:, np.newaxis, ...]
    if out_array is not None:
        return out_array
    else:
        raise ValueError('Model output is a list of length zero!')


def eval_step(batch: Dict[str, torch.Tensor], probs: np.ndarray, model: nn.Module):
    model.eval()
    x, y, idx = batch['x'], batch['y'], batch['idx']

    x.requires_grad_(True)

    y_pred = model(x)

    pred = torch.argmax(y_pred[-1], dim=1)
    pred = pred.cpu().numpy()

    probs[idx] = process_output(y_pred)

    target = torch.argmax(y, dim=1)
    return torch.from_numpy(pred), target, x, probs


def train_model(trainset: DataLoader, testset: DataLoader, probs: np.ndarray,
                model: nn.Module, epochs: int = 100, ensemble: int = 0, lr: float = 0.01,
                ood_data: np.ndarray = None):
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    gamma = 0.01
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[300, 500, 750, 1000], # adam
                                                     # milestones=[30, 60, 90, 100, 500, 1000],
                                                     gamma=gamma)
    for epoch in range(epochs):
        for i, batch in enumerate(trainset):
            model.train()
            optimizer.zero_grad()

            x, y = batch['x'], batch['y']
            x.requires_grad_(True)

            y_pred = model(x)

            loss1 = sdn_loss(y_pred, y)

            loss = loss1

            loss.backward()
            optimizer.step()
        scheduler.step(epoch)
        print(f"Train - Epoch: {epoch} Loss: {loss} Ensemble: {ensemble}")

    num_samples = 0
    correct_samples = 0
    for k, batch in enumerate(testset):
        pred, target, sample, probs = eval_step(batch, probs, model)
        acc = pred.eq(target)
        correct_samples += torch.sum(acc)
        num_samples += target.shape[0]

    print(f"Test Results Acc: {correct_samples / num_samples} Loss: {loss} ")
    model.eval()
    with torch.no_grad():
        ood_out = process_output(model(torch.from_numpy(ood_data).float())) if ood_data is not None else None

    return probs, ood_out
