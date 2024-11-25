import torch
from torch.utils.data import DataLoader
from data.datasets.segmentation.common.dataobjects import SegmentationStructure, LoaderObject
from data.datasets.shared.utils import DatasetStructure


class SeismicStat:
    def __init__(self):
        pass


class SeismicStructure(SegmentationStructure):
    def __init__(self):
        super(SegmentationStructure, self).__init__()

        self.train_shape = None

        self.val_set = None
        self.val_labels = None
        self.val_len = None

        self.val_inline_set = None
        self.val_inline_labels = None
        self.val_inline_len = None

        self.val_xline_set = None
        self.val_xline_labels = None
        self.val_xline_len = None

        self.train_inline_set = None
        self.train_inline_labels = None
        self.train_inline_len = None

        self.train_xline_set = None
        self.train_xline_labels = None
        self.train_xline_len = None

        self.test1_inline_set = None
        self.test1_inline_labels = None
        self.test1_inline_len = None

        self.test1_xline_set = None
        self.test1_xline_labels = None
        self.test1_xline_len = None

        self.test2_inline_set = None
        self.test2_inline_labels = None
        self.test2_inline_len = None

        self.test2_xline_set = None
        self.test2_xline_labels = None
        self.test2_xline_len = None

        self.train_specs = None


class SeismicLoaderObject(LoaderObject):
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 train_inline: DataLoader,
                 train_xline: DataLoader,
                 test1_inline: DataLoader,
                 test1_xline: DataLoader,
                 test2_inline: DataLoader,
                 test2_xline: DataLoader,
                 class_weights: torch.tensor, data_configs: SeismicStructure,
                 val_inline: DataLoader = None,
                 val_xline: DataLoader = None,):
        super(SeismicLoaderObject, self).__init__(train_loader=train_loader,
                                                  test_loader=test_loader,
                                                  val_loader=val_loader,
                                                  class_weights=class_weights,
                                                  data_configs=data_configs)
        # seismic loaders
        self.train_inline = train_inline
        self.train_xline = train_xline
        self.val_inline = val_inline
        self.val_xline = val_xline
        self.test1_inline = test1_inline
        self.test1_xline = test1_xline
        self.test2_inline = test2_inline
        self.test2_xline = test2_xline

