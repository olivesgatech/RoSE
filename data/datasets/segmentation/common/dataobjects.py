import torch
from torch.utils.data import DataLoader
from data.datasets.shared.utils import DatasetStructure
from training.segmentation.segmentationtracker import SegmentationTracker


class StatObject:
    def __init__(self, loader: DataLoader, tracker, name: str):
        self.tracker = tracker
        self.loader = loader
        self.name = name


class SegmentationStructure(DatasetStructure):
    def __init__(self):
        super(SegmentationStructure, self).__init__()

        self.val_set = None
        self.val_labels = None
        self.val_len = None


class LoaderObject:
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader,
                 class_weights: torch.tensor, data_configs: DatasetStructure, val_loader: DataLoader = None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.data_config = data_configs
        self.class_weights = class_weights
