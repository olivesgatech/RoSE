import torch
import random
import numpy as np
import numbers
from PIL import Image, ImageOps, ImageFilter
from skimage.transform import resize
from scipy import ndimage

cval=-1


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def append(self, transform):
        self.augmentations.append(transform)

    def __call__(self, sample):
        if sample['data'].shape != sample['label'].shape:
            raise Exception('Sample and label should have the same shape!')
        for a in self.augmentations:
            sample = a(sample)
        return sample


class Normalize(object):
    """Normalize a tensor data with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['data']
        mask = sample['label']
        # img = np.array(img).astype(np.float32)
        # mask = np.array(mask).astype(np.float32)
        img = (img - img.mean()) / (img.max() - img.min()) + 0.5
        return {'data': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['data']
        mask = sample['label']
        img = np.array(img).astype(np.float32)#.transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'data': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['data']
        mask = sample['label']
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
        return {'data': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['data']
        mask = sample['label']
        if random.random() < 0.5:
            choice = np.random.choice(a=[-1, 1], size=1)[0]
            img = ndimage.rotate(input=img, angle=choice*self.degree, reshape=True, order=0, mode='constant', cval=cval)
            mask = ndimage.rotate(input=mask, angle=choice*self.degree, reshape=True, order=0, mode='constant', cval=cval)

        return {'data': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = Image.fromarray(sample['data'])
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            out_img = np.asarray(img)
        return {'data': out_img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['data']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = resize(img, (ow, oh), order=0, cval=cval)
        mask = resize(mask, (ow, oh), order=0, cval=cval)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'data': img,
                'label': mask}


class RandomCrop(object):
    def __init__(self, size, pad=0):
        if isinstance(size, numbers.Number):
            self.shape = (int(size), int(size))
        else:
            self.shape = size
        self.pd = pad

    def __call__(self, sample):
        img, mask = sample['data'], sample['label']
        if self.pd > 0:
            img = np.pad(img, ((self.pd, self.pd), (self.pd, self.pd)), constant_values=(cval, cval, cval, cval))
            mask = np.pad(mask, ((self.pd, self.pd), (self.pd, self.pd)), constant_values=(cval, cval, cval, cval))
        if len(img.shape) == 2:
            w, h = img.shape
        else:
            w, h, _ = img.shape
        th, tw = self.shape
        if w == tw and h == th:
            return sample
        if w < tw or h < th:
            sample['data'] = resize(img, (th, tw), order=0, mode='constant', cval=cval)
            sample['label'] = resize(mask, (th, tw), order=0, mode='constant', cval=cval)
            return sample
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        sample['data'] = img[x1:x1+tw, y1:y1+th]
        sample['label'] = mask[x1:x1+tw, y1:y1+th]
        return sample
