import os
import tqdm
import torch
import torch.nn as nn
import numpy as np
from typing import List
from models.architectures import build_architecture
from training.common.trainutils import determine_multilr_milestones
from data.datasets.classification.common.aquisition import get_dataset
from training.classification.classificationtracker import ClassifcationTracker
from training.classification.utils import PCFocalLoss, torch_entropy, get_auroc_scores, get_optimal_auroc_th
from training.common.saver import Saver
from training.common.summaries import TensorboardSummary
from config import BaseConfig


class SDNLoss:
    def __init__(self, num_ics: int, coeffs: List[int] = None):
        self._num_ics = num_ics
        self._loss = nn.CrossEntropyLoss()
        self._mse = nn.MSELoss()

        if coeffs is not None:
            if len(coeffs) != num_ics:
                raise ValueError(f'Number of coefficients must be equal to number of ics!')
            self._coeffs = coeffs
        else:
            self._coeffs = [float(1/num_ics) for _ in range(num_ics)]
            temperature = 1.8
            norm = np.sum([np.exp(- temperature * k) for k in range(num_ics)])
            self._coeffs = [np.exp(- temperature * k) / norm for k in range(num_ics)]
            self._coeffs.reverse()
            print(f'Coeffs: {self._coeffs}')

    def __call__(self, outputs: List[torch.Tensor], target: torch.Tensor):
        loss = 0.0

        for i in range(self._num_ics):
            loss += self._coeffs[i]*self._loss(outputs[i], target)
        # loss += 100.0*self.focal_distillation(outputs, target)
        return loss


def torch_energy(logits: torch.Tensor, temperature: float = 0.0001) -> torch.Tensor:
    out = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return out

class ClassificationTrainer:
    def __init__(self, cfg: BaseConfig):
        self._cfg = cfg
        self._epochs = cfg.classification.epochs
        self._device = cfg.run_configs.gpu_id

        # Define Saver
        self._saver = Saver(cfg)

        # Define Tensorboard Summary
        self._summary = TensorboardSummary(self._saver.experiment_dir)
        self._writer = self._summary.create_summary()

        self._loaders = get_dataset(cfg)
        n_samples = self._loaders.data_config.train_len

        self._model, num_sdn = build_architecture(self._cfg.classification.model, self._loaders.data_config, cfg)
        self._softmax = torch.nn.Softmax(dim=1)
        self._num_sdn = num_sdn

        if self._cfg.classification.loss == 'ce' and num_sdn < 0:
            self._loss = nn.CrossEntropyLoss()
        elif num_sdn > 0:
            self._loss = SDNLoss(num_sdn)
        else:
            raise Exception('Loss not implemented yet!')

        if self._cfg.classification.optimization.optimizer == 'adam':
            self._optimizer = torch.optim.Adam(self._model.parameters(),
                                               lr=self._cfg.classification.optimization.lr)
        elif self._cfg.classification.optimization.optimizer == 'sgd':
            self._optimizer = torch.optim.SGD(self._model.parameters(),
                                              lr=self._cfg.classification.optimization.lr,
                                              momentum=0.9,
                                              nesterov=True,
                                              weight_decay=5e-4)
        else:
            raise Exception('Optimizer not implemented yet!')

        if self._cfg.classification.optimization.scheduler == 'multiLR':
            milestones = determine_multilr_milestones(self._epochs, self._cfg.classification.optimization.multiLR_steps)
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   milestones=milestones,
                                                                   gamma=self._cfg.classification.optimization.gamma)
        elif self._cfg.classification.optimization.scheduler == 'none':
            pass
        else:
            raise Exception('Scheduler not implemented yet!')

        # Using cuda
        if self._cfg.run_configs.cuda:
            # use multiple GPUs if available
            self._model = torch.nn.DataParallel(self._model,
                                                device_ids=[self._cfg.run_configs.gpu_id])
        else:
            self._device = torch.device('cpu')

        # LD stats
        self._train_statistics = ClassifcationTracker(self._loaders.data_config.train_len)
        self._test_statistics = ClassifcationTracker(self._loaders.data_config.test_len)
        self._val_statistics = ClassifcationTracker(self._loaders.data_config.val_len) if \
            cfg.run_configs.create_validation else None

        if self._cfg.run_configs.resume != 'none':
            resume_file = self._cfg.run_configs.resume
            self._load_model(resume_file)

    def _load_model(self, resume_file: str):
        # we have a checkpoint
        if not os.path.isfile(resume_file):
            raise RuntimeError("=> no checkpoint found at '{}'".format(resume_file))
        # load checkpoint
        checkpoint = torch.load(resume_file)
        # minor difference if working with cuda
        if self._cfg.run_configs.cuda:
            self._model.load_state_dict(checkpoint['state_dict'])
        else:
            self._model.load_state_dict(checkpoint['state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))

    def training(self, epoch, save_checkpoint=False, track_summaries=False, pc_loss: PCFocalLoss = None):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.train()
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._loaders.train_loader)
        num_img_tr = len(self._loaders.train_loader)

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            image, target, idxs, global_idxs = sample['data'], sample['label'], sample['idx'], sample['global_idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            # computes output of our model
            output = self._model(image)

            # Perform model update
            # calculate loss
            loss = self._loss(output, target.long())
            if self._num_sdn > 0:
                output = output[-1]

            _, pred = torch.max(output.data, 1)
            total += target.size(0)

            # collect forgetting events
            acc = pred.eq(target.data)

            # check if prediction has changed
            predicted = pred.cpu().numpy()
            self._train_statistics.update(acc.cpu().numpy(), predicted, idxs.cpu().numpy())

            # add pc loss if set
            if pc_loss is not None:
                assert self._cfg.classification.pc_training
                loss += pc_loss(output, target, global_idxs.cpu().numpy())
            # perform backpropagation
            loss.backward()

            # update params with gradients
            self._optimizer.step()

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            if track_summaries:
                self._writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            correct_samples += acc.cpu().sum()

        # Update optimizer step
        if self._cfg.classification.optimization.scheduler != 'none':
            self._scheduler.step(epoch)

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        if track_summaries:
            self._writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.classification.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Training Accuracy: %.3f' % acc)

        # save checkpoint
        if save_checkpoint:
            self._saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            })
        return acc

    def update_pc_loss(self, pc_loss: PCFocalLoss):
        """
        tests the model on a given holdout set. Provide an alterantive loader structure if you do not want to test on
        the test set.
        :return:
        """
        # assert if loss should be updated
        assert self._cfg.classification.pc_training
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._loaders.train_loader)
        print('Updating focal pc loss...')

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            image, target, idxs, global_idxs = sample['data'], sample['label'], sample['idx'], sample['global_idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                output = self._model(image)
                if self._num_sdn > 0:
                    output = output[-1]

                pc_loss.update_output(output.cpu().numpy(), global_idxs.cpu().numpy())

        return pc_loss

    def testing(self, epoch, alternative_loader_struct=None):
        """
        tests the model on a given holdout set. Provide an alterantive loader structure if you do not want to test on
        the test set.
        :param epoch:
        :param alternative_loader_struct:
        :return:
        """
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        if alternative_loader_struct is None:
            num_samples = self._loaders.data_config.test_len
            tbar = tqdm.tqdm(self._loaders.test_loader)
            num_img_tr = len(self._loaders.test_loader)
            tracker = self._test_statistics
        else:
            alternative_loader = alternative_loader_struct[0]
            tracker = alternative_loader_struct[1]
            num_samples = alternative_loader.data_config.test_len
            tbar = tqdm.tqdm(alternative_loader.test_loader)
            num_img_tr = len(alternative_loader.test_loader)

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # iterate over all samples in each batch i
        predictions = torch.zeros(num_samples, dtype=torch.long, device=self._device)
        gts = torch.zeros(num_samples, dtype=torch.long, device=self._device)
        scores = torch.zeros(num_samples, dtype=torch.float, device=self._device)
        for i, sample in enumerate(tbar):
            image, target, idxs = sample['data'], sample['label'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                output = self._model(image)

                # Perform model update
                # calculate loss
                loss = self._loss(output, target.long())

                if self._num_sdn > 0:
                    output = output[-1]

                _, pred = torch.max(output.data, 1)
                total += target.size(0)
                predictions[idxs] = pred
                gts[idxs] = target.long()
                probs = self._softmax(output)
                entropy = torch_entropy(probs)
                scores[idxs] = entropy

                # collect forgetting events
                acc = pred.eq(target.data)

                # check if prediction has changed
                predicted = pred.cpu().numpy()
                tracker.update(acc.cpu().numpy(), predicted, idxs.cpu().numpy())

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Test loss: %.3f' % (train_loss / (i + 1)))
            self._writer.add_scalar('test/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            correct_samples += acc.cpu().sum()

        # Update optimizer step
        if self._cfg.classification.optimization.scheduler != 'none':
            self._scheduler.step()

        # calculate accuracy
        auc = get_auroc_scores(scores=scores.cpu().numpy(), pred=predictions.cpu().numpy(), gt=gts.cpu().numpy())
        acc = 100.0 * correct_samples.item() / total
        self._writer.add_scalar('test/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.classification.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Test Accuracy: %.3f' % acc)
        print('Test AUROC: %.3f' % auc)

        output_dict = {
            'predictions': predictions.cpu().numpy(),
            'gt': gts.cpu().numpy(),
            'scores': scores.cpu().numpy()
        }

        return output_dict, acc, auc

    def validation(self, epoch):
        """
        tests the model on a given holdout set. Provide an alterantive loader structure if you do not want to test on
        the test set.
        :param epoch:
        :param alternative_loader_struct:
        :return:
        """
        if not self._cfg.run_configs.create_validation:
            raise ValueError(f'Validation not enabled in config!')
        print(f'Validating to get the optimal threshold')
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        num_samples = self._loaders.data_config.val_len
        tbar = tqdm.tqdm(self._loaders.val_loader)
        num_img_tr = len(self._loaders.val_loader)
        # tracker = self._test_statistics

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # iterate over all samples in each batch i
        predictions = torch.zeros(num_samples, dtype=torch.long, device=self._device)
        gts = torch.zeros(num_samples, dtype=torch.long, device=self._device)
        scores = torch.zeros(num_samples, dtype=torch.float, device=self._device)
        for i, sample in enumerate(tbar):
            image, target, idxs = sample['data'], sample['label'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                output = self._model(image)

                # Perform model update
                # calculate loss
                loss = self._loss(output, target.long())

                if self._num_sdn > 0:
                    output = output[-1]

                _, pred = torch.max(output.data, 1)
                total += target.size(0)
                predictions[idxs] = pred
                gts[idxs] = target.long()
                probs = self._softmax(output)
                entropy = torch_entropy(probs)
                scores[idxs] = torch_energy(output)

                # collect forgetting events
                acc = pred.eq(target.data)

                # check if prediction has changed
                predicted = pred.cpu().numpy()
                # tracker.update(acc.cpu().numpy(), predicted, idxs.cpu().numpy())

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Test loss: %.3f' % (train_loss / (i + 1)))
            self._writer.add_scalar('test/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            correct_samples += acc.cpu().sum()

        # Update optimizer step
        if self._cfg.classification.optimization.scheduler != 'none':
            self._scheduler.step()

        # calculate accuracy
        auc = get_auroc_scores(scores=scores.cpu().numpy(), pred=predictions.cpu().numpy(), gt=gts.cpu().numpy())
        th = get_optimal_auroc_th(scores=scores.cpu().numpy(), pred=predictions.cpu().numpy(), gt=gts.cpu().numpy())
        acc = 100.0 * correct_samples.item() / total
        self._writer.add_scalar('test/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.classification.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Test Accuracy: %.3f' % acc)
        print('Test AUROC: %.3f' % auc)
        print('Optimal Threshhold: %.3f' % th)

        output_dict = {
            'predictions': predictions.cpu().numpy(),
            'gt': gts.cpu().numpy(),
            'scores': scores.cpu().numpy(),
        }

        return output_dict, acc, auc, th

    def save_train_statistics(self, folder: str):
        self._train_statistics.save_statistics(folder, 'train')
