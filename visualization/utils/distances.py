import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# calculate the js divergence
def js_divergence(p: np.ndarray, q: np.ndarray):
    print(np.sum(p))
    print(np.sum(q))
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def histogram_distance(flattened_data1: np.ndarray,
                       flattened_data2: np.ndarray,
                       bin_param: str = 'sturges',
                       dist_type: str = 'js'):
    training_hist, tr_bins = np.histogram(flattened_data1, density=True, bins=bin_param)
    cur_hist, cur_bins = np.histogram(flattened_data2, density=True, bins=bin_param)
    tr_bins = tr_bins.astype(int)
    cur_bins = cur_bins.astype(int)
    tr_max = np.max(tr_bins)
    cur_max = np.max(cur_bins)
    tr_hist = np.zeros(int(max(tr_max, cur_max)), dtype=float)
    tr_hist[tr_bins[:-1]] = training_hist * np.diff(tr_bins)
    cr_hist = np.zeros(max(tr_max, cur_max), dtype=float)
    cr_hist[cur_bins[:-1]] = cur_hist * np.diff(cur_bins)
    if dist_type == 'js':
        distance = js_divergence(tr_hist, cr_hist)
    else:
        raise Exception('Distance type not implemented yet!')

    return distance