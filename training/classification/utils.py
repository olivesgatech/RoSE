import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from config import BaseConfig


def get_auroc_scores(scores: np.ndarray, pred: np.ndarray, gt: np.ndarray):
    targets = np.equal(pred, gt).astype(int)
    auc = roc_auc_score(targets, scores, average=None)
    fpr, tpr, ths = roc_curve(targets, scores)
    return auc


def get_optimal_auroc_th(scores: np.ndarray, pred: np.ndarray, gt: np.ndarray):
    targets = np.equal(pred, gt).astype(int)
    fpr, tpr, ths = roc_curve(targets, scores)

    # Calculate Youden's J statistic for each threshold
    youden_j = tpr - fpr

    # Find the optimal threshold that maximizes Youden's J statistic
    optimal_threshold_index = np.argmax(youden_j)
    optimal_threshold = ths[optimal_threshold_index]

    return optimal_threshold


def torch_entropy(probs: torch.tensor):
    logs = torch.log(probs)
    mult = logs * probs
    entr = torch.sum(mult, dim=1)
    entr = torch.nan_to_num(entr)
    return entr


class PCFocalLoss:
    def __init__(self, cfg: BaseConfig, num_samples: int):
        self._cfg = cfg
        self._alpha = cfg.classification.pctraining.alpha
        self._beta = cfg.classification.pctraining.beta
        self._lambda = 0.5
        self._num_samples = num_samples

        self._previous_output = None

        self.mse = torch.nn.MSELoss()

    def update_output(self, output: np.ndarray, idxs: np.ndarray):
        if self._previous_output is None:
            self._previous_output = np.zeros((self._num_samples, output.shape[1]), dtype=float)
        self._previous_output[idxs] = output

    def __call__(self, output: torch.Tensor, target: torch.Tensor, idxs: np.ndarray):

        if self._previous_output is None:
            raise Exception('PC focal loss not updated! Make sure a previous output is available')

        prev_output = torch.from_numpy(self._previous_output[idxs])
        if self._cfg.run_configs.cuda:
            prev_output = prev_output.to(output.get_device()).float()

        _, pred = torch.max(prev_output.data, 1)
        acc = pred.eq(target.data)

        correct = self.mse(output[acc], prev_output[acc]) * (self._alpha + self._beta)
        incorrect = self.mse(output[~acc], prev_output[~acc]) * self._alpha

        total = correct + incorrect
        return total * self._lambda
