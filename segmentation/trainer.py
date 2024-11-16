import copy
import tqdm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import DataLoader
from training.segmentation.utils import TestOutput
from training.segmentation.trainer import SegmentationTrainer
from training.segmentation.segmentationtracker import SegmentationTracker
from data.datasets.segmentation.common.acquisition import get_dataset
from config import BaseConfig

def torch_energy(logits: torch.Tensor, temperature: float = 0.0001) -> torch.Tensor:
    out = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return out

class ActiveLearningSegmentationTrainer(SegmentationTrainer):
    def __init__(self, cfg: BaseConfig):
        super(ActiveLearningSegmentationTrainer, self).__init__(cfg=cfg)

        self._train_pool = np.arange(self._loaders.data_config.train_len)
        self.n_pool = self._loaders.data_config.train_len
        self._unlabeled_loader = None
        self._labeled_test_loader = None
        self._unlabeled_statistics = None
        self._switches = None
        self._prev_predictions = {}
        self._switch_image = {}
        self._prev_acc = {}
        self._softmax = torch.nn.Softmax(dim=1)

    def update_loader(self, idxs: np.ndarray, unused_idxs: np.array):
        self._loaders = get_dataset(self._cfg, idxs=idxs)
        self._unlabeled_loader = get_dataset(self._cfg, idxs=unused_idxs, is_train=False)
        self._labeled_test_loader = get_dataset(self._cfg, idxs=idxs, is_train=False)
        # self._unlabeled_statistics = SegmentationTracker(self._unlabeled_loader.data_config.train_len)
        self._switches = np.zeros(self.n_pool)

        # self._prev_predictions = {}

    def get_switch_images(self):
        return self._switch_image

    def get_unlabeled_statistics(self, epoch: int, prev_predictions: np.ndarray = None,
                                 switch_images: Dict[int, np.ndarray] = None, labeled: bool = False,
                                 miou: float = None, flip_specifier: str = 'nf'):
        if prev_predictions is not None:
            self._prev_predictions = prev_predictions
        if switch_images is not None:
            self._switch_image = switch_images
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        if labeled:
            tbar = tqdm.tqdm(self._labeled_test_loader.train_loader)
            num_img_tr = len(self._labeled_test_loader.train_loader)
        else:
            tbar = tqdm.tqdm(self._unlabeled_loader.train_loader)
            num_img_tr = len(self._unlabeled_loader.train_loader)

        # init statistics parameters
        test_loss = 0.0
        total = 0

        # reset evaluator
        self._evaluator.reset()

        # init predictions array
        usp_predictions = None
        indices = np.zeros(self.n_pool)
        recon_scores = np.zeros(self.n_pool) - 1
        spec_recon = np.zeros(self.n_pool) - 1
        total_probs = np.zeros((self.n_pool, self._num_classes))
        total_embeddings = np.zeros((self.n_pool, self._penultimate_dim))
        spec_probs = np.zeros((self.n_pool, self._num_classes))
        spec_embeddings = np.zeros((self.n_pool, self._penultimate_dim))
        data = {}

        mse_loss = nn.MSELoss(reduction='none')

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            image, target, idxs = sample['data'], sample['label'], sample['global_idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                output, recon, embeddings = self._model(image)

                probs = self._softmax(output)
                mean_probs = self._softmax(torch.mean(output, dim=(2, 3)))
                mean_embeddings = torch.mean(embeddings, dim=(2, 3))
                total_embeddings[idxs.long().numpy()[0]] = mean_embeddings.cpu().numpy()
                total_probs[idxs.long().numpy()[0]] = mean_probs.cpu().numpy()

                pred = torch.argmax(output, dim=1)
                total += target.size(0)

                # collect forgetting events
                acc = pred.eq(target.data)

                # check if prediction has changed
                predicted = pred.cpu().numpy()
                indices[idxs.long().numpy()] = 1
                # TODO: bs is currently always 1
                data[idxs.long().numpy()[0]] = {
                    'output': output.cpu().numpy(),
                    'image': image.cpu().numpy(),
                    'gt': target.cpu().numpy(),
                }
                if idxs.long().numpy()[0] in self._prev_predictions:
                    change = self._prev_predictions[idxs.long().numpy()[0]] != predicted
                    data[idxs.long().numpy()[0]]['change'] = change

                    if idxs.long().numpy()[0] not in self._switch_image:
                        self._switch_image[idxs.long().numpy()[0]] = change.astype(int)
                    else:
                        self._switch_image[idxs.long().numpy()[0]] += change.astype(int)

                    data[idxs.long().numpy()[0]]['switches'] = self._switch_image[idxs.long().numpy()[0]]

                    # probs
                    perm_probs = torch.squeeze(torch.permute(output, (0, 2, 3, 1)))
                    perm_probs = perm_probs[change]
                    if miou is not None:
                        unc_scores = torch_energy(perm_probs)
                        sorted_inds = torch.argsort(unc_scores)
                        num_pos_samples = int(miou * unc_scores.shape[0])
                        perm_probs = perm_probs[sorted_inds[num_pos_samples:]] if flip_specifier == 'nf' else \
                            perm_probs[sorted_inds[:num_pos_samples]]
                    perm_probs = torch.unsqueeze(torch.mean(perm_probs, dim=0), dim=0)
                    spec_probs[idxs.long().numpy()[0]] = self._softmax(perm_probs).cpu().numpy()

                    # embeddings
                    perm_embeddings = torch.squeeze(torch.permute(embeddings, (0, 2, 3, 1)))
                    perm_embeddings = perm_embeddings[change]
                    perm_embeddings = torch.mean(perm_embeddings, dim=0)
                    spec_embeddings[idxs.long().numpy()[0]] = perm_embeddings.cpu().numpy()

                    # switches
                    num_pixels = change.shape[0]
                    self._switches[idxs.long().numpy()[0]] += np.sum(change) / num_pixels if num_pixels > 0 else 0

                    if recon is not None:
                        filtered_recon = recon[(predicted >= 3) & change]
                        filtered_image = image[(predicted >= 3) & change]

                        filtered_mse = mse_loss(filtered_recon, filtered_image)
                        filtered_mse = torch.mean(filtered_mse, dim=(1, 2, 3))
                        filtered_mse = filtered_mse.cpu().numpy()
                        spec_recon[idxs.long().numpy()[0]] = filtered_mse

                self._prev_predictions[idxs.long().numpy()[0]] = predicted

                # Perform model update
                # calculate loss
                loss = self._loss(output, target.long())

                if recon is not None:
                    mse = mse_loss(recon, image)
                    mse = torch.mean(mse, dim=(1, 2, 3))
                    mse = mse.cpu().numpy()
                    recon_scores[idxs.long().numpy()[0]] = mse

            self._evaluator.add_batch(target.cpu().numpy(), predicted)
            # extract loss value as float and add to train_loss
            test_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            self._writer.add_scalar('test/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        # calculate accuracy
        self._writer.add_scalar('test/total_loss_epoch', test_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.segmentation.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % test_loss)
        acc = self._evaluator.Pixel_Accuracy()
        acc_class, acc_all_class = self._evaluator.Pixel_Accuracy_Class()
        mIoU = self._evaluator.Mean_Intersection_over_Union()
        df = pd.DataFrame(columns=self._cols)
        df.loc[0, 'CA'] = acc
        df.loc[0, 'MCA'] = acc_class
        df.loc[0, 'mIoU'] = mIoU
        for i in range(self._num_classes):
            key = 'class' + str(i)
            df.loc[0, key] = acc_all_class[i]

        print(df)
        inds = (indices == 1).nonzero()
        # define output
        ret = {
            'switches': self._switches[indices == 1],
            'recon': recon_scores[indices == 1],
            'specified recon': spec_recon[indices == 1],
            'indices': inds[0],
            'predictions': self._prev_predictions,
            'probabilities': total_probs[indices == 1],
            'specified probabilities': spec_probs[indices == 1],
            'embeddings': total_embeddings[indices == 1],
            'specified embeddings': spec_embeddings[indices == 1],
            'switch_images': self._switch_image,
            'acc df': df,
            'data': data
        }

        return ret

    def get_unlabeled_statisticsDEP(self, epoch: int, prev_predictions: np.ndarray = None):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        if prev_predictions is not None:
            self._prev_predictions = prev_predictions
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._unlabeled_loader.train_loader)
        num_img_tr = len(self._unlabeled_loader.train_loader)

        # init statistics parameters
        test_loss = 0.0
        total = 0

        # reset evaluator
        self._evaluator.reset()

        # init predictions array
        usp_predictions = None
        indices = np.zeros(self.n_pool)
        recon_scores = np.zeros(self.n_pool) - 1
        spec_recon = np.zeros(self.n_pool) - 1
        total_probs = np.zeros((self.n_pool, self._num_classes))
        spec_probs = np.zeros((self.n_pool, self._num_classes))
        data = {}

        mse_loss = nn.MSELoss(reduction='none')

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            image, target, idxs = sample['data'], sample['label'], sample['global_idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                output, recon = self._model(image)

                probs = self._softmax(output)
                mean_probs = torch.mean(probs, dim=(2, 3))
                total_probs[idxs.long().numpy()[0]] = mean_probs.cpu().numpy()

                pred = torch.argmax(output, dim=1)
                total += target.size(0)

                # collect forgetting events
                acc = pred.eq(target.data)

                # check if prediction has changed
                predicted = pred.cpu().numpy()
                indices[idxs.long().numpy()] = 1
                # TODO: bs is currently always 1
                data[idxs.long().numpy()[0]] = {
                    'probs': probs.cpu().numpy(),
                    'image': image.cpu().numpy(),
                    'gt': target.cpu().numpy(),
                }
                if idxs.long().numpy()[0] in self._prev_predictions:
                    change = self._prev_predictions[idxs.long().numpy()[0]] != predicted
                    data[idxs.long().numpy()[0]]['change'] = change

                    if idxs.long().numpy()[0] not in self._switch_image:
                        self._switch_image[idxs.long().numpy()[0]] = change.astype(int)
                    else:
                        self._switch_image[idxs.long().numpy()[0]] += change.astype(int)

                    data[idxs.long().numpy()[0]]['switches'] = self._switch_image[idxs.long().numpy()[0]]
                    # change = change[predicted >= 3]
                    perm_probs = torch.squeeze(torch.permute(probs, (0, 2, 3, 1)))
                    perm_probs = perm_probs[change]
                    spec_probs[idxs.long().numpy()[0]] = torch.mean(perm_probs, dim=0).cpu().numpy()
                    num_pixels = change.shape[0]
                    self._switches[idxs.long().numpy()[0]] += np.sum(change) / num_pixels if num_pixels > 0 else 0

                    if recon is not None:
                        filtered_recon = recon[(predicted >= 3) & change]
                        filtered_image = image[(predicted >= 3) & change]

                        filtered_mse = mse_loss(filtered_recon, filtered_image)
                        filtered_mse = torch.mean(filtered_mse, dim=(1, 2, 3))
                        filtered_mse = filtered_mse.cpu().numpy()
                        spec_recon[idxs.long().numpy()[0]] = filtered_mse

                self._prev_predictions[idxs.long().numpy()[0]] = predicted

                # Perform model update
                # calculate loss
                loss = self._loss(output, target.long())

                if recon is not None:
                    mse = mse_loss(recon, image)
                    mse = torch.mean(mse, dim=(1, 2, 3))
                    mse = mse.cpu().numpy()
                    recon_scores[idxs.long().numpy()[0]] = mse

            self._evaluator.add_batch(target.cpu().numpy(), predicted)
            # extract loss value as float and add to train_loss
            test_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            self._writer.add_scalar('test/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        # calculate accuracy
        self._writer.add_scalar('test/total_loss_epoch', test_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.segmentation.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % test_loss)
        acc = self._evaluator.Pixel_Accuracy()
        acc_class, acc_all_class = self._evaluator.Pixel_Accuracy_Class()
        mIoU = self._evaluator.Mean_Intersection_over_Union()
        df = pd.DataFrame(columns=self._cols)
        df.loc[0, 'CA'] = acc
        df.loc[0, 'MCA'] = acc_class
        df.loc[0, 'mIoU'] = mIoU
        for i in range(self._num_classes):
            key = 'class' + str(i)
            df.loc[0, key] = acc_all_class[i]

        print(df)
        inds = (indices == 1).nonzero()
        # define output
        ret = {
            'switches': self._switches[indices == 1],
            'recon': recon_scores[indices == 1],
            'specified recon': spec_recon[indices == 1],
            'indices': inds[0],
            'predictions': self._prev_predictions,
            'probabilities': total_probs[indices == 1],
            'specified probabilities': spec_probs[indices == 1],
            'acc df': df,
            'data': data
        }

        return ret

