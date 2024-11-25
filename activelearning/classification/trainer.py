import copy
import tqdm
import torch
import numpy as np
from training.classification.trainer import ClassificationTrainer, torch_energy
from training.classification.classificationtracker import ClassifcationTracker
from data.datasets.classification.common.aquisition import get_dataset
from config import BaseConfig

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            print('Enabling dropout')
            m.train()

class ActiveLearningClassificationTrainer(ClassificationTrainer):
    def __init__(self, cfg: BaseConfig, ideal_model_path: str = 'none'):
        super(ActiveLearningClassificationTrainer, self).__init__(cfg=cfg)

        self._train_pool = np.arange(self._loaders.data_config.train_len)
        self.n_pool = self._loaders.data_config.train_len
        self._unlabeled_loader = None
        self._unlabeled_statistics = None
        self._mcd_softmax = torch.nn.Softmax(dim=2)

        if ideal_model_path != 'none':
            self._load_model(ideal_model_path)

    def update_loader(self, idxs: np.ndarray, unused_idxs: np.array):
        self._loaders = get_dataset(self._cfg, idxs=idxs)
        if self._cfg.active_learning.strategy == 'badge':
            self._unlabeled_loader = get_dataset(self._cfg, idxs=unused_idxs, test_bs=True)
        else:
            self._unlabeled_loader = get_dataset(self._cfg, idxs=unused_idxs, test_bs=False)
        self._unlabeled_statistics = ClassifcationTracker(self._unlabeled_loader.data_config.train_len)

    def get_unlabeled_statistics(self, epoch: int):
        # sets model into eval mode -> important for dropout batchnorm. etc.
        self._model.eval()
        # initializes cool bar for visualization
        num_samples = self._unlabeled_loader.data_config.train_len
        tbar = tqdm.tqdm(self._unlabeled_loader.train_loader)
        num_img_tr = len(self._unlabeled_loader.train_loader)
        tracker = self._unlabeled_statistics

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # iterate over all samples in each batch i
        predictions = torch.zeros(num_samples, dtype=torch.long, device=self._device)
        # probs = torch.full((self.n_pool, self._unlabeled_loader.data_config.num_classes), 2.5, dtype=torch.float)
        logits = torch.full((self.n_pool, self._unlabeled_loader.data_config.num_classes), 2.5, dtype=torch.float)
        scores = torch.zeros(num_samples, device=self._device, dtype=torch.float)
        indices = torch.zeros(self.n_pool)
        accs = torch.zeros(self.n_pool, dtype=int)
        switches = np.zeros(self.n_pool)
        nfs = np.zeros(self.n_pool)
        for i, sample in enumerate(tbar):
            image, target, idxs, local_idx = sample['data'], sample['label'], sample['global_idx'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)
                logits, indices, idxs = logits.to(self._device), indices.to(self._device), idxs.to(self._device)
                accs = accs.to(self._device)
                local_idx = local_idx.to(self._device)

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

                # convert sdn output to normal output if applicable
                if self._num_sdn > 0:
                    output = output[-1]

                scores[idxs] = torch_energy(output)
                logits[idxs.long()] = output

                _, pred = torch.max(output.data, 1)
                total += target.size(0)
                predictions[idxs] = pred
                probs_output = self._softmax(output)

                # insert to probs array
                # probs[idxs.long()] = probs_output
                indices[idxs.long()] = 1

                # collect forgetting events
                acc = pred.eq(target.data)
                accs[idxs.long()] = acc.long()

                # check if prediction has changed
                predicted = pred.cpu().numpy()
                switches[idxs.long().cpu().numpy()] = tracker.get_stats(local_idx.cpu().numpy(), tracking_type='SE')
                nfs[idxs.long().cpu().numpy()] = tracker.get_stats(local_idx.cpu().numpy(), tracking_type='FE')
                tracker.update(acc.cpu().numpy(), predicted, local_idx.cpu().numpy())

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
        acc = 100.0 * correct_samples.item() / total
        self._writer.add_scalar('test/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.classification.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Test Accuracy: %.3f' % acc)
        probs = self._softmax(logits)
        probs = probs[indices == 1].cpu().numpy()
        logits = logits[indices == 1].cpu().numpy()
        scores = scores[indices == 1].cpu().numpy()
        predictions = predictions[indices == 1]
        accs = accs[indices == 1]
        inds = (indices == 1).nonzero().cpu().numpy()
        switches = switches[indices.cpu().numpy() == 1]
        nfs = nfs[indices.cpu().numpy() == 1]
        full_inds = indices.cpu().numpy()

        output_dict = {
            'predictions': predictions.cpu().numpy(),
            'scalar accuracy': acc,
            'indices': inds,
            'probabilities': probs,
            'logits': logits,
            'switches': switches,
            'nfs': nfs,
            'sample accuracy': accs.cpu().numpy(),
            'scores': scores,
            'full inds': full_inds
        }

        return output_dict

    def get_unlabeled_mcd(self, epoch: int):
        # sets model into eval mode -> important for dropout batchnorm. etc.
        self._model.eval()
        enable_dropout(self._model)
        # initializes cool bar for visualization
        num_samples = self._unlabeled_loader.data_config.train_len
        tbar = tqdm.tqdm(self._unlabeled_loader.train_loader)
        num_img_tr = len(self._unlabeled_loader.train_loader)
        tracker = self._unlabeled_statistics

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0
        num_mcd_samples = 20

        # iterate over all samples in each batch i
        predictions = torch.zeros(num_samples, dtype=torch.long, device=self._device)
        # probs = torch.full((self.n_pool, self._unlabeled_loader.data_config.num_classes), 2.5, dtype=torch.float)
        mcd_logits = torch.full((self.n_pool, num_mcd_samples, self._unlabeled_loader.data_config.num_classes), 2.5,
                                dtype=torch.float)
        scores = torch.zeros(num_samples, device=self._device, dtype=torch.float)
        indices = torch.zeros(self.n_pool)
        accs = torch.zeros(self.n_pool, dtype=int)
        switches = np.zeros(self.n_pool)
        nfs = np.zeros(self.n_pool)
        for i, sample in enumerate(tbar):
            image, target, idxs, local_idx = sample['data'], sample['label'], sample['global_idx'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)
                mcd_logits, indices, idxs = mcd_logits.to(self._device), indices.to(self._device), idxs.to(self._device)
                accs = accs.to(self._device)
                local_idx = local_idx.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                output = None
                for k in range(num_mcd_samples):
                    cur_output = self._model(image)
                    output = torch.cat((cur_output.unsqueeze(dim=1), output), dim=1) if output is not None else \
                        cur_output.unsqueeze(dim=1)

                mcd_logits[idxs.long()] = output
                output = torch.mean(output, dim=1)

                # Perform model update
                # calculate loss
                loss = self._loss(output, target.long())

                # convert sdn output to normal output if applicable
                if self._num_sdn > 0:
                    output = output[-1]

                scores[idxs] = torch_energy(output)
                # logits[idxs.long()] = output

                _, pred = torch.max(output.data, 1)
                total += target.size(0)
                predictions[idxs] = pred

                # insert to probs array
                # probs[idxs.long()] = probs_output
                indices[idxs.long()] = 1

                # collect forgetting events
                acc = pred.eq(target.data)
                accs[idxs.long()] = acc.long()

                # check if prediction has changed
                predicted = pred.cpu().numpy()
                switches[idxs.long().cpu().numpy()] = tracker.get_stats(local_idx.cpu().numpy(), tracking_type='SE')
                nfs[idxs.long().cpu().numpy()] = tracker.get_stats(local_idx.cpu().numpy(), tracking_type='FE')
                tracker.update(acc.cpu().numpy(), predicted, local_idx.cpu().numpy())

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
        acc = 100.0 * correct_samples.item() / total
        self._writer.add_scalar('test/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.classification.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Test Accuracy: %.3f' % acc)
        probs = self._softmax(torch.mean(mcd_logits, dim=1))
        mcd_logits = self._mcd_softmax(mcd_logits)
        probs = probs[indices == 1].cpu().numpy()
        mcd_logits = mcd_logits[indices == 1].cpu().numpy()
        scores = scores[indices == 1].cpu().numpy()
        predictions = predictions[indices == 1]
        accs = accs[indices == 1]
        inds = (indices == 1).nonzero().cpu().numpy()
        switches = switches[indices.cpu().numpy() == 1]
        nfs = nfs[indices.cpu().numpy() == 1]
        full_inds = indices.cpu().numpy()

        output_dict = {
            'predictions': predictions.cpu().numpy(),
            'scalar accuracy': acc,
            'indices': inds,
            'probabilities': probs,
            'mcd_logits': mcd_logits,
            'switches': switches,
            'nfs': nfs,
            'sample accuracy': accs.cpu().numpy(),
            'scores': scores,
            'full inds': full_inds
        }

        return output_dict

    def get_embeddings(self, loader_type: str = 'unlabeled'):
        # set model to evaluation mode
        self._model.eval()

        # get embed dim
        if self._cfg.run_configs.cuda:
            embedDim = self._model.module.get_penultimate_dim()
        else:
            embedDim = self._model.get_penultimate_dim()

        if loader_type == 'labeled':
            tbar = tqdm.tqdm(self._loaders.train_loader, desc='\r')
        elif loader_type == 'unlabeled':
            tbar = tqdm.tqdm(self._unlabeled_loader.train_loader, desc='\r')
        else:
            raise Exception('You can only load labeled and unlabeled pools!')

        # init softmax layer
        softmax = torch.nn.Softmax(dim=1)

        # init embedding tesnors and indices for tracking
        embeddings = torch.zeros((self.n_pool, embedDim * self._unlabeled_loader.data_config.num_classes),
                                 dtype=torch.float)
        nongrad_embeddings = torch.zeros((self.n_pool, embedDim),
                                         dtype=torch.float)
        indices = torch.zeros(self.n_pool)

        # if self._cfg.run_configs.cuda:
        #     indices = indices.to(self._device)

        with torch.no_grad():
            # iterate over all sample batches
            for i, sample in enumerate(tbar):
                image, target, idxs, local_idx = sample['data'], sample['label'], sample['global_idx'], sample['idx']
                # assign each image and target to GPU
                if self._cfg.run_configs.cuda:
                    image, target = image.to(self._device), target.to(self._device)

                # convert image to suitable dims
                image = image.float()

                # computes output of our model
                output = self._model(image)

                if self._num_sdn > 0:
                    output = output[-1]

                # get penultimate embedding
                if self._cfg.run_configs.cuda:
                    penultimate = self._model.module.penultimate_layer
                else:
                    penultimate = self._model.penultimate_layer

                nongrad_embeddings[idxs] = penultimate.cpu()

                # get softmax probs
                probs_output = softmax(output)

                _, pred = torch.max(output.data, 1)

                # insert to embediing array
                for j in range(target.shape[0]):
                    for c in range(self._unlabeled_loader.data_config.num_classes):
                        if c == pred[j].item():
                            tmp = copy.deepcopy(penultimate[j]) * (1 - probs_output[j, c].item())
                            embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = tmp.cpu()
                        else:
                            tmp = copy.deepcopy(penultimate[j]) * (-1 * probs_output[j, c].item())
                            embeddings[idxs[j], embedDim * c: embedDim * (c + 1)] = tmp.cpu()
                indices[idxs.long()] = 1

        # sort idxs
        # embeddings = embeddings.cpu()
        # nongrad_embeddings = nongrad_embeddings.cpu()
        output_structure = {}
        output_structure['embeddings'] = embeddings[indices == 1].numpy()
        ind_list = (indices == 1).nonzero().cpu().numpy()
        output_structure['indices'] = ind_list
        output_structure['nongrad_embeddings'] = nongrad_embeddings[indices == 1].numpy()

        return output_structure

    def get_norms(self, loader_type: str = 'unlabeled'):
        print('Calculating norms')
        # set model to evaluation mode
        self._model.eval()

        # get embed dim
        if self._cfg.run_configs.cuda:
            embedDim = self._model.module.get_penultimate_dim()
        else:
            embedDim = self._model.get_penultimate_dim()

        if loader_type == 'labeled':
            tbar = tqdm.tqdm(self._loaders.train_loader, desc='\r')
        elif loader_type == 'unlabeled':
            tbar = tqdm.tqdm(self._unlabeled_loader.train_loader, desc='\r')
        else:
            raise Exception('You can only load labeled and unlabeled pools!')

        # init softmax layer
        softmax = torch.nn.Softmax(dim=1)

        # init embedding tesnors and indices for tracking
        # embeddings = torch.zeros((self.n_pool, embedDim * self._unlabeled_loader.data_config.num_classes),
        #                          dtype=torch.float)
        nongrad_embeddings = torch.zeros((self.n_pool, embedDim),
                                         dtype=torch.float)
        norms = torch.zeros(self.n_pool, dtype=torch.float, device=self._device)
        indices = torch.zeros(self.n_pool)

        # if self._cfg.run_configs.cuda:
        #     indices = indices.to(self._device)

        with torch.no_grad():
            # iterate over all sample batches
            for i, sample in enumerate(tbar):
                image, target, idxs, local_idx = sample['data'], sample['label'], sample['global_idx'], sample['idx']
                # assign each image and target to GPU
                if self._cfg.run_configs.cuda:
                    image, target = image.to(self._device), target.to(self._device)

                # convert image to suitable dims
                image = image.float()
                embed = torch.zeros((target.shape[0],
                                     embedDim * self._unlabeled_loader.data_config.num_classes),
                                    dtype=torch.float, device=self._device)

                # computes output of our model
                output = self._model(image)

                if self._num_sdn > 0:
                    output = output[-1]

                # get penultimate embedding
                if self._cfg.run_configs.cuda:
                    penultimate = self._model.module.penultimate_layer
                else:
                    penultimate = self._model.penultimate_layer

                nongrad_embeddings[idxs] = penultimate.cpu()

                # get softmax probs
                probs_output = softmax(output)

                _, pred = torch.max(output.data, 1)

                # insert to embediing array
                for j in range(target.shape[0]):

                    for c in range(self._unlabeled_loader.data_config.num_classes):
                        if c == pred[j].item():
                            tmp = penultimate[j].detach() * (1 - probs_output[j, c].item())
                            embed[j, embedDim * c: embedDim * (c + 1)] = tmp
                        else:
                            tmp = penultimate[j] * (-1 * probs_output[j, c].item())
                            embed[j, embedDim * c: embedDim * (c + 1)] = tmp
                norms[idxs] = torch.norm(embed, p=2, dim=1)
                indices[idxs.long()] = 1

        # sort idxs
        # embeddings = embeddings.cpu()
        # nongrad_embeddings = nongrad_embeddings.cpu()
        output_structure = {}
        output_structure['norms'] = norms[indices == 1].cpu().numpy()
        ind_list = (indices == 1).nonzero().cpu().numpy()
        output_structure['indices'] = ind_list
        output_structure['nongrad_embeddings'] = nongrad_embeddings[indices == 1].numpy()

        return output_structure

    def save_unlabeled_statistics(self, folder: str):
        self._train_statistics.save_statistics(folder, 'unlabeled')
