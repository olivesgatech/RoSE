import os
import seaborn as sns
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from typing import List

# This code was coded by me before working at plato


def get_acc_dataframe(target_path, nstart: int =128, nquery: int = 1024):
    """
    Returns aggregated dataframe with all runs stored with and additional column id
    Parameters:
        :param target_path: path to folder with target excel files
        :type target_path: str
    """

    # get all excel files
    path_list = glob.glob(target_path + '*.xlsx')
    total = pd.DataFrame([])

    # add excel files into complete dataframe
    for i in range(len(path_list)):
        df = pd.read_excel(path_list[i], index_col=0, engine='openpyxl')
        rounds = len(df)
        samples = [nstart + nquery * x for x in range(rounds)]
        df['Samples'] = np.array(samples)
        # code to display difference between samples and not the actual accuracy
        # df = df.set_index('Samples').diff()
        # df = df.fillna(0)
        df = df.reset_index()
        df['id'] = i
        total = pd.concat([total, df])

    return total


def plot_scatter_lines(df, algs, plot_col):
    for alg in algs:
        rel_df = df.loc[df['Algorithm'] == alg]
        rel_df = rel_df.groupby('index').mean()
        plt.plot(rel_df['Rounds'].values, rel_df[plot_col].values, label=alg, marker='.', markersize=10)
    # plt.legend(loc="upper left")


def plot_lc(path_list, names, plotting_col='Test Acc', xlim=None, ylim=None, nstart: int = 128, nquery: int = 1024,
            invert: bool = False):
    if len(path_list) != len(names):
        raise Exception("Each element in the pathlist must have a corresponding name")

    # iterate over all paths and add to plot dataframe
    total = pd.DataFrame([])
    seen = {}
    for i in range(len(path_list)):
        df = get_acc_dataframe(target_path=path_list[i], nstart=nstart, nquery=nquery)
        df['Algorithm'] = names[i]

        if names[i] in seen.keys():
            raise Exception("Names cannot appear more than once")
        else:
            seen[names[i]] = True
        total = pd.concat([total, df])

    total = total.rename_axis('Rounds')
    total = total.reset_index()
    # if invert:
    #     total[plotting_col] = 1 - total[plotting_col].values

    total.to_excel('blah2.xlsx')
    sns.lineplot(data=total, x='Samples', y=plotting_col, hue='Algorithm')
    plt.grid()
    # plot_scatter_lines(total, names, plotting_col)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(loc="upper left", prop={'size': 10}, framealpha=1.0)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Samples', fontsize=13)
    # plt.ylabel(plotting_col, fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    plt.show()
def print_accs(arr, inp_str: str):
    mean = np.mean(arr)
    std = np.std(arr)
    print(f'{inp_str} Acc {mean:.1f}±{std:.1f}')
def print_round_accs(df: pd.DataFrame, rounds: np.ndarray, plotting_col: str):
    seeds = list(set(df['id'].values))
    accs = []
    for seed in seeds:
        seed_df = df.loc[df['id'] == seed]
        accs.append(seed_df[plotting_col].values[rounds])
    accs = np.array(accs)
    for i in range(len(rounds)):
        in_str = f'Round {i}'
        print_accs(accs[:, i], in_str)

def wrapper_round_accs(path: str, rounds: List[int], plotting_col: str, nstart: int = 128, nquery: int = 1024):
    df = get_acc_dataframe(target_path=path, nstart=nstart, nquery=nquery)
    print_round_accs(df, rounds, plotting_col)

def average_df(subtotal: pd.DataFrame, plotting_type: str = 'STL10'):
    subtotal['Avg ' + plotting_type] = subtotal.groupby(['Samples'])[plotting_type].transform('mean')

    return subtotal


def generate_acc_diffs(path_list, names, plotting_col='Test Acc', nstart: int = 128, nquery: int = 1024, xmax: int = 20000):

    # iterate over all paths and add to plot dataframe
    total = pd.DataFrame([])
    seen = {}
    for i in range(len(path_list)):
        df = get_acc_dataframe(target_path=path_list[i], nstart=nstart, nquery=nquery)
        df = average_df(df, plotting_col)
        df['Algorithm'] = names[i]

        if names[i] in seen.keys():
            raise Exception("Names cannot appear more than once")
        else:
            seen[names[i]] = True
        total = pd.concat([total, df])


    avg_total = total
    avg_total = avg_total.loc[avg_total['id'] == 0]
    avg_total = avg_total[avg_total['Samples'] < xmax]
    false_df = avg_total.loc[avg_total['Algorithm'] == 'Original']
    ref_vals = false_df.loc[:, 'Avg ' + plotting_col].values
    samples = false_df.loc[:, 'Samples'].values
    samples = (samples - nstart) / nquery
    areas = []
    for name in names:
        cur_df = avg_total.loc[avg_total['Algorithm'] == name]
        cur_vals = cur_df.loc[:,  'Avg ' + plotting_col].values
        diff = cur_vals - ref_vals
        area = auc(samples, diff)
        areas.append(area)
        print(name + f' Area: {area}')

    return area

cifar10c = {
    'brightness': True,
    'contrast': True,
    'defocus_blur': True,
    'elastic_transform': True,
    'fog': True,
    'frost': True,
    'gaussian_blur': True,
    'gaussian_noise': True,
    'glass_blur': True,
    'impulse_noise': True,
    'jpeg_compression': True,
    # 'labels': True,
    'motion_blur': True,
    'pixelate': True,
    'saturate': True,
    # 'shot_noise': True,
    'snow': True,
    'spatter': True,
    # 'speckle_noise': True,
    'zoom_blur': True,
    # 'STL10' : True
}

if __name__ == "__main__":
    # precursor = '~/results/gauss/CIFAR100/al/incremental/resnet18/'
    # precursor = '/Volumes/shannon/Results/alnfr/active learning/CIFAR10/no augmentations/10its/'
    # precursor = '/Volumes/shannon/Results/alnfr/active learning/CIFAR10/no augmentations/cifar10_vgg16/'
    precursor = '~/results/alnfrv1/rhflip/svhn/resnet18/nquery256/'
    precursor = '~/results/alnfrv1/no-augmentations/svhn/resnet18/nquery256/'
    precursor = '~/results/alnfrv1/rcrop-rhflip-cutout/cifar10/resnet18/nquery128/fixed_epochs/pcal_experiments/'
    precursor = '~/results/alnfrv1/no-augmentations/tinyimagenet/resnet18/nquery2048/pcal_experiments/'
    # precursor = '~/results/alnfrv1/rcrop-rhflip-cutout/cifar10/resnet18/nquery384/idealexperiments/'
    # precursor = '~/results/alnfrv1/no-augmentations/cifar10/'
    # precursor = '~/PycharmProjects/Results/alnfrv1/no-augmentations/xray/'
    nstart = 128
    nquery = 256
    xmax = 20000
    addons = [
              # 'al_lconf/',
              # 'al_entropy/',
              # 'al_entropy_pcal_nf/',
              # 'al_entropy_pcal_bc/',
              'al_lconf_pretrained/',
              'al_lconf_pretrained_pcal_nf/',
              # 'al_lconf/',
              # 'al_margin/',
              # 'al_lconf_spectral/',
              # 'al_entropy_spectral/',
              # 'al_lconf_spectral/',
              # 'al_coreset/',
              # 'al_ideal_entropy/',
              # 'al_ideal_margin/',
              # 'al_ideal_lconf/',
              # 'al_ideal_coreset/',
              # 'al_ideal_badge/',
              # 'al_entropy/',
              # 'al_lconf/',
              # 'al_margin/',


              ]
    # addons = ['al_entropy_densenet121/', 'al_pcalentropy_densenet121/', 'al_relssentropy1x1024_densenet121/']
    paths = [os.path.expanduser(precursor + addon) for addon in addons]
    alg_names = [
                 # 'Switch Events',
                 # 'Random',
                 # 'Random SN',
                 'Entropy',
                 'Entropy NF',
                 # 'Coreset',
                 # 'Least Conf.',
                 # 'Coreset',
                 # 'Spectral Entropy',
                 # 'Spectral Least Conf.',
                 # 'Entropy',
                 # 'Spectral Entropy',
                 # 'Entropy SN',
                 # 'Margin',
                 # 'Least Conf.',
                 # 'Coreset',
                 # 'BADGE',
                 # 'RELSS 256',
                 # 'Least Conf.',
                 # 'Margin',
                 ]
    col = 'test'
    gen_acc = False
    invert = False
    plotting_types = [col]
    # for corr in cifar10c.keys():
    #     plotting_types.append(corr + '2')
    #     plotting_types.append(corr + '5')
    if gen_acc:
        data = {'Algorithms': alg_names}
        for plot_type in plotting_types:
            area = generate_acc_diffs(path_list=paths, names=alg_names, plotting_col=plot_type,
                                      nstart=nstart,
                                      nquery=nquery,
                                      xmax=xmax)
            data[plot_type] = area

        xlsx = pd.DataFrame(data=data)
        print(xlsx)
        xlsx.to_excel('acc_results.xlsx')
    plot_lc(path_list=paths, names=alg_names, plotting_col=col,
            # xlim=(0, 12000),
            # ylim=(10, 70),
            nstart=nstart,
            nquery=nquery, invert=invert)

