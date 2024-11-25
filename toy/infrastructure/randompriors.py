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


class ModelBilinear(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.gamma = 0.99
        self.sigma = 0.3

        embedding_size = 10

        self.fc1 = nn.Linear(2, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class RandomPrior(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.gamma = 0.99
        self.sigma = 0.3
        mult = 1

        embedding_size = 10

        self.fc1 = nn.Linear(2, features)
        self.fc2 = nn.Linear(features, features*mult)
        self.fc3 = nn.Linear(features*mult, features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def output_transform_acc(output):
    y_pred, y, x = output

    y = torch.argmax(y, dim=1)

    return y_pred, y


def output_transform_bce(output):
    y_pred, y, x = output

    return y_pred, y


def eval_step(batch: Dict[str, torch.Tensor], uncertainty: np.ndarray, model: nn.Module, priors: List[nn.Module],
              num_features: int):
    model.eval()
    x, y, idx = batch['x'], batch['y'], batch['idx']
    norms = torch.zeros(x.shape[0])
    spreads = torch.zeros(x.shape[0])
    loss = 0.0
    with torch.no_grad():
        y_pred = model(x)

    for l in range(len(priors)):
        random_prior = priors[l]
        random_prior.eval()

        with torch.no_grad():
            target = random_prior(x)
            diff = y_pred - target
            norms += torch.sum(diff * diff, dim=1) / (num_features * len(priors))
            loss += F.mse_loss(y_pred, target) / len(priors)

    for l in range(len(priors)):
        random_prior = priors[l]
        random_prior.eval()

        with torch.no_grad():
            target = random_prior(x)
            diff = y_pred - target
            cur_norm = torch.sum(diff * diff, dim=1) / num_features
            diff = (norms - cur_norm)**2
            spreads += diff / len(priors)

    spreads = torch.sqrt(spreads)

    uncertainty[idx] = norms.squeeze().detach().cpu().numpy() + spreads.squeeze().detach().cpu().numpy()
    return uncertainty, loss


def train_model(trainset: DataLoader, testset: DataLoader, uncertainty: np.ndarray,
                model: nn.Module, num_features: int, epochs: int = 100, num_priors: int = 5):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    model.train()
    priors = []
    for l in range(num_priors):
        random_prior = RandomPrior(features=num_features)
        random_prior.eval()
        priors.append(random_prior)

    for epoch in range(epochs):
        for i, batch in enumerate(trainset):

            optimizer.zero_grad()

            x, y = batch['x'], batch['y']
            x.requires_grad_(True)

            y_pred = model(x)
            loss = 0.0
            for l in range(num_priors):
                with torch.no_grad():
                    target = priors[l](x)

                loss += F.mse_loss(y_pred, target) / num_priors

            loss.backward()
            optimizer.step()
        print(f"Train - Epoch: {epoch} Loss: {loss} ")

    test_loss = 0.0
    for k, batch in enumerate(testset):
        uncertainty, cur_loss = eval_step(batch, uncertainty=uncertainty, model=model, priors=priors,
                                          num_features=num_features)
        test_loss += cur_loss / len(testset)
    print(f'Test loss {test_loss}')
    # uncertainty = np.clip(uncertainty - test_loss.item(), a_min=0, a_max=None)
    return uncertainty
