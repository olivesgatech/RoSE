import numpy as np
from torch.utils.data import Dataset
from data.datasets.segmentation.seismic.dataobjects import SeismicStructure
from data.datasets.segmentation.common.custom_transforms import ToTensor


def replicate_image_to_mutlichannels(image, num_channels=3):
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=0)
        image = image.repeat(num_channels, 0)
    return image


class SeismicLoader(Dataset):
    def __init__(self, data_config: SeismicStructure, split: str, current_idxs: np.ndarray = None, transform=None):
        if not data_config.is_configured:
            raise Exception('Dataset not configured yet!')
        # initialize data
        if split == 'train':
            self._X = data_config.train_set
            self._Y = data_config.train_labels
        elif split == 'val':
            self._X = data_config.val_set
            self._Y = data_config.val_labels
        elif split == 'test':
            self._X = data_config.test_set
            self._Y = data_config.test_labels
        elif split == 'test1_i':
            self._X = data_config.test1_inline_set
            self._Y = data_config.test1_inline_labels
        elif split == 'test1_x':
            self._X = data_config.test1_xline_set
            self._Y = data_config.test1_xline_labels
        elif split == 'test2_i':
            self._X = data_config.test2_inline_set
            self._Y = data_config.test2_inline_labels
        elif split == 'test2_x':
            self._X = data_config.test2_xline_set
            self._Y = data_config.test2_xline_labels
        elif split == 'train_i':
            self._X = data_config.train_inline_set
            self._Y = data_config.train_inline_labels
        elif split == 'train_x':
            self._X = data_config.train_xline_set
            self._Y = data_config.train_xline_labels
        elif split == 'val_i':
            self._X = data_config.val_inline_set
            self._Y = data_config.val_inline_labels
        elif split == 'val_x':
            self._X = data_config.val_xline_set
            self._Y = data_config.val_xline_labels
        else:
            raise Exception('Dataset handling is not correct!!! Split name should be either \'train\', \'val\', or'
                            ' \'test1/2\' with _i or _x specification!!!')

        self._transform = transform
        self._totensor = ToTensor()

        self._X = np.array(self._X)
        self._Y = np.array(self._Y)

        self._current_idxs = current_idxs
        if current_idxs is not None:
            self._X = self._X[current_idxs]
            self._Y = self._Y[current_idxs]

    def __getitem__(self, index):
        x, y = self._X[index], self._Y[index]
        raw_sample = {'data': np.load(x), 'label': np.load(y)}

        if self._transform is not None:
            tr_sample = self._transform(raw_sample)
        else:
            tr_sample = raw_sample

        if self._current_idxs is not None:
            global_idx = self._current_idxs[index]
        else:
            global_idx = index

        # convert image to three channels
        tr_sample['data'] = replicate_image_to_mutlichannels(tr_sample['data'], num_channels=3)
        tr_sample = self._totensor(tr_sample)

        output = {'data': tr_sample['data'], 'label': tr_sample['label'], 'idx': index, 'global_idx': global_idx}

        return output

    def __len__(self):
        return len(self._X)


if __name__ == '__main__':
    split = 'train'
    revolve = SeismicLoader(split=split)
    print(len(revolve))



