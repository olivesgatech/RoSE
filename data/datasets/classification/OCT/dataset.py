import PIL.Image as Image
import numpy as np
from torch.utils.data import Dataset
from data.datasets.classification.common.dataobjects import DatasetStructure


class LoaderOCT(Dataset):
    def __init__(self, data_config: DatasetStructure, split: str, current_idxs: list = None, transform=None):
        if transform is None:
            raise Exception(f'This dataset only works with resize, grayscale channel extension, and normalization!')
        # make sure split is in correct format
        if split != 'train' and split != 'test' and split != 'val':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'train\', \'val\','
                            ' or \'test\'!!!')

        # initialize data
        if split == 'train':
            self._X = data_config.train_set
            self._Y = data_config.train_labels
        elif split == 'val':
            self._X = data_config.val_set
            self._Y = data_config.val_labels
        else:
            self._X = data_config.test_set
            self._Y = data_config.test_labels

        self._current_idxs = current_idxs
        if current_idxs is not None:
            self._X = self._X[current_idxs]
            self._Y = self._Y[current_idxs]
        self.transform = transform

    def __getitem__(self, index):
        x, y = self._X[index], self._Y[index]
        if self.transform is not None:
            # im = Image.open(x)
            im = Image.fromarray(x)
            x = self.transform(im)
        else:
            raise Exception('This dataset needs transforms!')

        if self._current_idxs is not None:
            global_idx = self._current_idxs[index]
        else:
            global_idx = index

        sample = {'data': x, 'label': y, 'idx': index, 'global_idx': global_idx}
        return sample

    def __len__(self):
        return len(self._X)
