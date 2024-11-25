import os
from typing import List
import seaborn as sns
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import auc

# This code was coded by me before working at plato


def get_acc_dataframe(target_path: str, metric: str, nstart: int =128, nquery: int = 1024):
    """
    Returns aggregated dataframe with all runs stored with and additional column id
    Parameters:
        :param target_path: path to folder with target excel files
        :type target_path: str
    """

    # get all excel files
    path_list = glob.glob(os.path.join(target_path, metric, f'*.xlsx'))
    total = pd.DataFrame([])

    # add excel files into complete dataframe
    for i in range(len(path_list)):
        df = pd.read_excel(path_list[i], index_col=0, engine='openpyxl')
        df['id'] = i
        rounds = df['id'].count()
        samples = [nstart + nquery * x for x in range(rounds)]
        df['Samples'] = np.array(samples)
        total = pd.concat([total, df])

    return total


def plot_lc(path_list, names, metric: str = 'mIoU', plotting_col='Test Acc', xlim=None, ylim=None, nstart: int = 128,
            nquery: int = 1024, pallete: List[str] = None):
    if len(path_list) != len(names):
        raise Exception("Each element in the pathlist must have a corresponding name")

    # iterate over all paths and add to plot dataframe
    total = pd.DataFrame([])
    seen = {}
    for i in range(len(path_list)):
        df = get_acc_dataframe(target_path=path_list[i], metric=metric, nstart=nstart, nquery=nquery)
        df['Algorithm'] = names[i]

        if names[i] in seen.keys():
            raise Exception("Names cannot appear more than once")
        else:
            seen[names[i]] = True
        total = pd.concat([total, df])

    total = total.rename_axis('Rounds')
    total = total.reset_index()
    total.to_excel('blah2.xlsx')
    sns.lineplot(data=total, x='Samples', y=plotting_col, hue='Algorithm', ci=90)
    plt.grid()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    samples = total['Samples'].values
    new_list = np.arange(start=math.floor(xlim[0]), stop=xlim[1] + 1, step=2)
    plt.legend(loc="lower right", prop={'size': 10}, framealpha=1.0)
    plt.xticks(new_list, fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Sections', fontsize=13)
    # plt.ylabel(plotting_col, fontsize=13)
    plt.ylabel(metric, fontsize=13)
    # plt.title('F3 Test Volume Inline')
    plt.show()


if __name__ == "__main__":
    precursor = '~/results/alnfr_segmentation/rcrop-rhflip/seismic/dlab_resnet18/nquery2/pcal_experiments/'
    # precursor = '~/results/alnfr_segmentation/rcrop-rhflip/seismic/'
    nstart = 2
    nquery = 2
    xmax = 20000
    metric = 'mIoU'
    # metric = 'class3'
    addons = [
              # 'dlab_recon_resnet18/al_random/',
              # 'dlab_resnet18/al_random/',
              # 'al_recon/',
              # 'nfr_analysis/al_random/',
              # 'redos_mean_latent_wo_crop/al_alps_margin/',
              # 'redos_mean_latent_wo_crop/al_alps_margin/',
              'al_alps_mispred_entropy_nf/',
              'al_entropy/',
              # 'redos_mean_latent_wo_crop/al_margin/',
              # 'al_lconf/',
              # 'redos_mean_latent_wo_crop/al_lconf_redo/',
              # 'nfr_analysis/al_margin/',
              # 'al_lconf/',
              # 'al_recon/',
              ]
    # addons = ['al_entropy_densenet121/', 'al_pcalentropy_densenet121/', 'al_relssentropy1x1024_densenet121/']
    paths = [os.path.expanduser(precursor + addon) for addon in addons]
    alg_names = [
                 # 'Random w. Recon.',
                 # 'Random',
                 # 'Entropy + Ours',
                 'ATLAS Least Conf.',
                 # 'Least Conf. + Ours',
                 # 'Entropy',
                 # 'Entropy',
                 'Least Conf.',
                 # 'Least Conf.',
                 # 'Mustafa et. al.',
                 ]
    col = 'test1_inline'
    plot_lc(path_list=paths, names=alg_names, plotting_col=col,
            xlim=(0, 50),
            # ylim=(0.48, 0.70),
            metric=metric,
            nstart=nstart,
            nquery=nquery)