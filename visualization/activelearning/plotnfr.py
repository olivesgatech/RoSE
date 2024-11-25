import os
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import auc
from visualization.utils.plotting import *
from data.datasets.classification.CURETSR.utils import *
from data.datasets.classification.CINIC10.dataset import LoaderCINIC10
from data.datasets.classification.TinyImageNet.configuration import prepare_testimagenet, prepare_imagenet
from data.datasets.classification.common.dataobjects import ClassificationStructure

# This code was coded by me before working at plato


def get_nfr_dataframe(target_path, plot_type: str = 'nfr', nstart: int = 128, nquery: int = 1024,
                      target_class: int = None, num_classes: int = None):
    """
    Returns aggregated dataframe with all runs stored with and additional column id.
    NOTE: This script only works for active learning experiments trained on CIFAR10. In order to make it run for other
    things, you have to change the line
    ...
    else:
        # change this line to mnist etc.
        target = cifar10

    Sorry for the inconvenience.

    :param plot_type: type you would like to plot. Possibilities are average fr, fr and nfr.
    :param target_path: path to folder with target excel files
    :type target_path: str
    """

    # get all excel files
    NFRCINIC_FLAG = False
    folder_list = glob.glob(target_path + 'round*')
    data_dict = {}
    prev_acc = {}
    prev_pred = {}
    dataset_path = os.path.expanduser('~/datasets/')
    if not os.path.exists(dataset_path):
        raise Exception('Plug in the dataset Folder!')
    path = dataset_path
    if NFRCINIC_FLAG:
        raw_te = datasets.ImageFolder(os.path.expanduser(path) + '/CINIC10/test')
        data_config = ClassificationStructure()
        data_config.test_set = raw_te
        data_config.test_len = len(raw_te)
        test_tr = transforms.Compose([transforms.ToTensor()])
        cinic_loader = DataLoader(LoaderCINIC10(data_config=data_config,
                                                split='test',
                                                transform=test_tr),
                                  batch_size=len(raw_te),
                                  shuffle=False)
        sample = next(iter(cinic_loader))
        d = 0
        cinic10 = sample['label'].cpu().numpy()
        print('loaded CINIC10')
    else:
        test_dir = os.path.expanduser(path) + '/CURE-TSR/Real_Test/ChallengeFree'
        curetsr_te_data = make_dataset(test_dir)
        cifar10 = np.array(datasets.CIFAR10(path + '/CIFAR10', train=False, download=False).targets)
        cifar100 = np.array(datasets.CIFAR100(path + '/CIFAR100', train=False, download=False).targets)
        mnist = datasets.MNIST(os.path.expanduser(path) + '/MNIST', train=False, download=False).test_labels
        stl10 = datasets.STL10(path + '/STL10', split='test', download=False).labels
        svhn = datasets.SVHN(os.path.expanduser(path) + '/SVHN', split='test', download=False).labels
        cure_tsr = np.array([obj[1] for obj in curetsr_te_data])
        xray = np.load(os.path.expanduser(path) + '/xray/test_labels.npy')
        timgnet_path = os.path.expanduser(os.path.join(path, 'tinyimagenet'))
        train_dict = prepare_imagenet(os.path.join(timgnet_path, 'train'))
        test_dict = prepare_testimagenet(os.path.join(timgnet_path, 'val'), train_dict['class_dict'])
        timgnet = np.array(test_dict['labels'])
        x = 0

    for i in range(len(folder_list)):
        set_path = target_path + 'round' + str(i) + '/uspec_statistics/'
        set_data_paths = glob.glob(set_path + '*')
        k = 0

        for path in set_data_paths:
            pred_paths = glob.glob(path + '/predictions*.npy')
            pred_paths.sort()
            predictions = None
            target_cum = None
            name = path.split('/')[-1]
            if name == 'train':
                continue
            elif NFRCINIC_FLAG:
                target = cinic10
            elif name == 'MNIST':
                target = mnist
            elif name == 'STL10':
                continue
                target = stl10
            elif name == 'SVHN':
                target = svhn
            else:
                # change this line if you want to look at different datasets
                # target = mnist
                # target = cifar100
                # target = cifar10
                # target = xray
                # target = cinic10
                target = timgnet
                # target = stl10
                # target = svhn
                # target = cure_tsr
            if num_classes is not None:
                target = target[target < num_classes]

                reduce_test = False
                if reduce_test:
                    for cl in range(num_classes):
                        if cl >= num_classes // 2:
                            im_idxs = np.argwhere(target == cl)
                            remove_idxs = im_idxs[:90]
                            remove_idxs = np.squeeze(remove_idxs)

                            target = np.delete(target, remove_idxs)

            count = 0
            for file in pred_paths:
                pred = np.load(file)
                if target_class is not None:
                    pred = pred[target == target_class]
                    target_cum = target[target == target_class]
                else:
                    target_cum = target
                count += 1
                if predictions is not None:
                    predictions = np.concatenate((predictions, pred[np.newaxis, ...]))
                    targets = np.concatenate((targets, target_cum[np.newaxis, ...]))
                else:
                    predictions = pred[np.newaxis, ...]
                    targets = target_cum[np.newaxis, ...]

            if plot_type == 'entropy':
                value = average_entropy(predictions, num_classes=10)
            elif plot_type == 'afr':
                value = calculate_average_fr(predictions)
            elif plot_type == 'nfr':
                if name not in prev_acc.keys():
                    prev_acc[name] = np.zeros(targets.shape)
                try:
                    value, prev_acc[name] = calculate_nfr(predictions, prev_acc[name], targets, name)
                    value = 100*value
                except:
                    continue
                ids = np.arange(len(pred_paths))
            elif plot_type == 'internal_nfr':
                if name not in prev_acc.keys():
                    prev_acc[name] = np.zeros(targets.shape)
                value = internal_nfr(predictions, targets, name)
                value = value
                ids = np.arange(len(pred_paths))
            elif plot_type == 'fr':
                if name not in prev_pred.keys():
                    prev_pred[name] = np.zeros(targets.shape)
                try:
                    value, prev_pred[name] = calculate_fr(predictions, prev_pred[name], targets, name)
                    if i == 0:
                        value = np.zeros(value.shape)
                except:
                    continue
                ids = np.arange(len(pred_paths))
            else:
                raise Exception('Plot type not implemented yet!')
            if not plot_type in ['nfr', 'fr', 'internal_nfr']:
                if name not in data_dict.keys():
                    data_dict[name] = [value]
                else:
                    data_dict[name].append(value)
            else:
                if name not in data_dict.keys():
                    data_dict[name] = value
                    data_dict['ID'] = ids
                    data_dict['Round'] = np.full(ids.shape, fill_value=i)
                    data_dict['Samples'] = np.full(ids.shape, fill_value=nstart + i*nquery)
                else:
                    data_dict[name] = np.append(data_dict[name], value)
                    if data_dict[name].shape[0] > data_dict['ID'].shape[0]:
                        data_dict['ID'] = np.append(data_dict['ID'], ids)
                        data_dict['Round'] = np.append(data_dict['Round'], np.full(ids.shape, fill_value=i))
                        data_dict['Samples'] = np.append(data_dict['Samples'], np.full(ids.shape,
                                                                                       fill_value=nstart + i*nquery))

            k += 1

    total = pd.DataFrame(data=data_dict)

    return total


def plot_scatter_lines(df, algs, plot_col):
    for alg in algs:
        rel_df = df.loc[df['Algorithm'] == alg]
        rel_df = rel_df.groupby('Round').mean()
        plt.plot(rel_df['Rounds'].values, rel_df[plot_col].values, label=alg, marker='.', markersize=10)

def plot_lc(path_list, names, plotting_col='Test Acc', xlim=None, ylim=None, plot_type='nfr', nstart: int = 128,
            nquery: int = 1024, target_class: int = None, num_classes: int = None):
    if len(path_list) != len(names):
        raise Exception("Each element in the pathlist must have a corresponding name")

    # iterate over all paths and add to plot dataframe
    total = pd.DataFrame([])
    seen = {}
    for i in range(len(path_list)):
        df = get_nfr_dataframe(target_path=path_list[i], plot_type=plot_type, nstart=nstart, nquery=nquery,
                               target_class=target_class, num_classes=num_classes)
        df['Algorithm'] = names[i]
        if names[i] in seen.keys():
            raise Exception("Names cannot appear more than once")
        else:
            seen[names[i]] = True
        total = pd.concat([total, df])

    total = total.rename_axis('Rounds')
    if plot_type == 'nfr' or plot_type == 'fr' or plot_type == 'internal_nfr':
        x_axis = 'Samples'
    else:
        x_axis = 'Rounds'
    total.to_excel('blah.xlsx')
    total = total.reset_index()
    # total = total.group_by('Round').mean()
    # plot_scatter_lines(total, names, plotting_col)
    sns.lineplot(data=total, x=x_axis, y=plotting_col, hue='Algorithm', ci=90)

                 # ci=99)
    plt.grid()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(loc="upper left", prop={'size': 10})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel(x_axis, fontsize=13)
    # plt.ylabel(plotting_col, fontsize=13)
    plt.ylabel('PSR', fontsize=13)
    plt.show()


def average_df(subtotal: pd.DataFrame, plotting_type: str = 'STL10'):
    subtotal['Avg ' + plotting_type] = subtotal.groupby(['Samples'])[plotting_type].transform('mean')

    return subtotal


def generate_acc_diffs(path_list, names, plotting_col='Test Acc', nstart: int = 128, nquery: int = 1024, xmax: int = 20000):

    # iterate over all paths and add to plot dataframe
    total = pd.DataFrame([])
    seen = {}
    for i in range(len(path_list)):
        df = get_nfr_dataframe(target_path=path_list[i], nstart=nstart, nquery=nquery)
        df = average_df(df, plotting_col)
        df['Algorithm'] = names[i]

        if names[i] in seen.keys():
            raise Exception("Names cannot appear more than once")
        else:
            seen[names[i]] = True
        total = pd.concat([total, df])


    avg_total = total
    avg_total = avg_total.loc[avg_total['ID'] == 0]
    avg_total = avg_total[avg_total['Samples'] < xmax]
    false_df = avg_total.loc[avg_total['Algorithm'] == 'Original']
    ref_vals = false_df.loc[:, 'Avg ' + plotting_col].values
    samples = false_df.loc[:, 'Samples'].values
    samples = (samples - nstart) / nquery

    for name in names:
        cur_df = avg_total.loc[avg_total['Algorithm'] == name]
        cur_vals = cur_df.loc[:,  'Avg ' + plotting_col].values
        diff = cur_vals - ref_vals
        area = auc(samples, diff)
        print(name + f' Area: {area}')


if __name__ == "__main__":
    precursor = '~/results/gauss/CIFAR100/al/rcrop-rhflip/incremental/'
    precursor = '~/results/alnfrv1/rhflip/svhn/resnet18/nquery256/'
    precursor = os.path.expanduser('~/results/alnfrv1/no-augmentations/cifar10/')
    precursor = '~/results/alnfrv1/rcrop-rhflip-cutout/cifar10/resnet18/nquery1024/varbs/'
    # precursor = '~/results/alnfrv1/no-augmentations/oct/resnet18/nquery128/'
    # precursor = '~/results/alnfrv1/no-augmentations/cifar10/resnet18/nf-vs-switch/'
    # precursor = '~/PycharmProjects/Results/alnfrv1/no-augmentations/xray/'
    precursor = '~/results/alnfrv1/no-augmentations/cifar10/'
    precursor = '~/results/alnfrv1/rcrop-rhflip-cutout/cifar100/resnet18/nquery128/fixed_epochs/'
    precursor = '~/results/alnfrv1/rcrop-rhflip-cutout/cifar10/resnet18/nquery1024/fixed_epochs/'
    precursor = '~/results/alnfrv1/rcrop-rhflip-cutout/cifar10/resnet18/nquery128/fixed_epochs/pcal_experiments/'
    precursor = '~/results/alnfrv1/rcrop-rhflip-cutout/cinic10/resnet18/nquery1024/fixed_epochs/pcal_experiments/'
    precursor = '~/results/alnfrv1/no-augmentations/tinyimagenet/resnet18/nquery1024/pcal_experiments/'
    precursor = '~/results/alnfrv1/no-augmentations/tinyimagenet/resnet18/nquery4096/fixed_epochs/pcal_experiments/'
    nstart = 128
    nquery = 1024
    xmax = 20000
    num_classes = None
    addons = [
              'al_badge_pretrained/',
              'al_badge_pretrained_pcal_nf/',
              # 'al_entropy_pretrained_pcal_bc/',


              ]
    # addons = ['al_entropy_densenet121/', 'al_pcalentropy_densenet121/', 'al_relssentropy1x1024_densenet121/']
    paths = [os.path.expanduser(precursor + addon) for addon in addons]
    alg_names = [
                 'Entropy',
                 'Entropy NF',
                 # 'Entropy PF',
                 ]
    col = 'test'
    metric = 'nfr'
    # generate_acc_diffs(path_list=paths, names=alg_names, plotting_col=col,
    #                    nstart=nstart,
    #                    nquery=nquery,
    #                    xmax=xmax)
    plot_lc(path_list=paths, names=alg_names, plotting_col=col,
            # xlim=(32, 300),
            # ylim=(0, 30),
            nstart=nstart,
            nquery=nquery,
            plot_type=metric, num_classes=num_classes)
