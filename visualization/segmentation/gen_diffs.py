import os
from typing import List
import seaborn as sns
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import scipy

from visualization.segmentation.plotlcs import get_acc_dataframe


def generate_diff(path_list, names, plotting_col=['Test Acc'], nstart: int = 128, nquery: int = 1024,
                  xmax: int = 20000, ref_name: str = 'none', metric: str = 'mIoU'):
    if len(path_list) != len(names) != 2:
        raise Exception("Each element in the pathlist must have a corresponding name")
    # if 'Random' not in names:
    #     raise Exception('Random must be in the names list!')
    if ref_name == 'none':
        ref_name = names[0]
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

    avg_total = total
    total.to_excel('blah3.xlsx')
    # avg_total = avg_total.loc[avg_total['id'] == 0]
    avg_total = avg_total[avg_total['Samples'] < xmax]

    ref_df = avg_total.loc[avg_total['Algorithm'] == ref_name]
    count = 0
    for name in names:
        name_df = avg_total.loc[avg_total['Algorithm'] == name]
        seeds = list(set(name_df.loc[:, 'id'].values))
        aucs = []
        for seed in seeds:
            cur_df = name_df.loc[name_df['id'] == seed]
            cur_ref_df = ref_df.loc[ref_df['id'] == seed]
            samples = cur_df.loc[:, 'Samples'].values
            samples = (samples - nstart) / nquery
            # samples = samples / np.max(samples)
            cur_vals = np.zeros(len(cur_df))
            ref_vals = np.zeros(len(cur_ref_df))
            if len(cur_vals) != len(ref_vals):
                print(f'Ref and target have different length! ID {seed} cur shape {cur_vals.shape} '
                      f'ref shape {ref_vals.shape}')
                continue
            for col in plotting_col:
                cur_vals += cur_df.loc[:, col].values / len(plotting_col)
                ref_vals += cur_ref_df.loc[:, col].values / len(plotting_col)
            area = np.sum(cur_vals) / len(cur_vals)
            # area = auc(samples, diff) / divisor
            aucs.append(area)
        mean = np.mean(aucs)
        std = np.std(aucs)
        if count % 2 != 0:
            pvalues = scipy.stats.ttest_ind(ref, aucs).pvalue
            stat_sig = int(pvalues <= 0.05)
        else:
            stat_sig = 'Baseline'
        print(name + f' Area: {mean:.3f} ± {std:.3f} Stat. Sig.: {stat_sig}')
        count += 1
        ref = aucs


if __name__ == "__main__":
    precursor = '~/results/alnfr_segmentation/rcrop-rhflip/seismic/dlab_resnet18/nquery2/redos_mean_latent_wo_crop/'
    precursor = '~/results/alnfr_segmentation/rcrop-rhflip/parihakav1/dlab_resnet18/nquery2/'
    # precursor = '~/results/alnfr_segmentation/rcrop-rhflip/seismic/'
    nstart = 2
    nquery = 2
    xmax = 20
    metric = 'mIoU'
    metric = 'class5'
    addons = [
              'al_entropy/',
              'al_alps_entropy/',
              ]
    # addons = ['al_entropy_densenet121/', 'al_pcalentropy_densenet121/', 'al_relssentropy1x1024_densenet121/']
    paths = [os.path.expanduser(precursor + addon) for addon in addons]
    alg_names = [
                 'Entropy',
                 'Entropy + Ours',
        ]
    # col = ['test2_inline', 'test2_xline']
    # col = ['test1_inline', 'test1_xline']
    # col = ['test1_inline', 'test1_xline']
    col = ['test_inline', 'test_xline']
    generate_diff(path_list=paths, names=alg_names, plotting_col=col,
            # xlim=(0, 50),
            # ylim=(0, 0.1),
            metric=metric,
            nstart=nstart,
            nquery=nquery)
