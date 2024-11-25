import numpy as np
import matplotlib.pyplot as plt
from config import BaseConfig
from visualization.utils.distances import histogram_distance


def calculate_average_fr(predictions: np.ndarray):
    num_seeds = predictions.shape[0] - 1
    flips = np.diff(predictions, axis=0)
    flips = np.count_nonzero(flips != 0, axis=0)
    fr = flips/num_seeds

    return np.mean(fr)


def calculate_nfr(predictions: np.ndarray, prev_acc: np.ndarray, targets: np.ndarray, name):
    curr_acc = predictions == targets
    tmp1 = prev_acc.astype(int)
    tmp2 = curr_acc.astype(int)
    nf = np.clip(prev_acc - tmp2, a_min=0, a_max=1)
    nfr = np.sum(nf, axis=1)/nf.shape[1]
    # nfr = np.sum(nf, axis=1) / np.sum(curr_acc, axis=1)
    # nfr = np.sum(curr_acc, axis=1) / np.sum(nf, axis=1)

    return nfr, curr_acc


def internal_nfr(predictions: np.ndarray, targets: np.ndarray, name):
    curr_acc = predictions == targets
    out = []
    while len(out) < curr_acc.shape[0]:
        k = 1
        for i in range(k, curr_acc.shape[0]):
            prev_acc = curr_acc[k - 1, :].astype(int)
            nf = np.clip(prev_acc - curr_acc[i, :].astype(int), a_min=0, a_max=1)
            nfr = np.sum(nf)/nf.shape[0]
            out.append(nfr)
            if len(out) == curr_acc.shape[0]:
                break
        k += 1

    return np.array(out)


def calculate_fr(predictions: np.ndarray, prev_pred: np.ndarray, targets: np.ndarray, name):
    fl = predictions.astype(int) != prev_pred.astype(int)
    fr = np.sum(fl, axis=1)/fl.shape[1]

    return fr, predictions


def pred_entropy(predictions: np.array, num_classes: int):
    num_seeds = predictions.shape[0]
    output = np.zeros(predictions.shape[1:], dtype=float)

    for i in range(num_classes):
        cur_class_prob = np.count_nonzero(predictions == i, axis=0) / num_seeds

        output[cur_class_prob != 0] -= cur_class_prob[cur_class_prob != 0]*np.log2(cur_class_prob[cur_class_prob != 0])

    return output


def average_entropy(predictions: np.array, num_classes: int):
    out = np.mean(pred_entropy(predictions, num_classes))
    return out


def unique_means(predictions: np.array):
    uniques = np.array([len(np.unique(predictions[:, i])) for i in range(predictions.shape[1])])
    return np.mean(uniques)

def plot_uspec_unique_prediction_switches(files: list, separator: str):
    fig, axs = plt.subplots(1, int(len(files)), sharex=False)
    count = 0
    for file in files:
        data_type = file.split(separator)[-2]
        predicitons = np.load(file)
        im = np.zeros(predicitons.shape)
        uniques = np.array([len(np.unique(predicitons[:, i])) for i in range(predicitons.shape[1])])
        for i in range(uniques.shape[0]):
            im[uniques[i] - 1, i] = 1
        print('FILE: ' + file)
        print('USPEC Mean: %f' % np.mean(uniques))
        print('AFR: %f' % calculate_average_fr(predicitons))
        print('AE: %f' % average_entropy(predicitons, 10))
        # axs[count].scatter(np.arange(uniques.shape[0]), uniques)
        axs[count].imshow(im, aspect='auto')
        axs[count].title.set_text(data_type)
        count += 1
    plt.show()


def plot_fevent_switch_event_histograms(files: list, separator: str, cfg: BaseConfig,
                                        reference_folder: str = None):
    # TODO: Assuming we are only tracking forgetting and switch events
    fig, axs = plt.subplots(2, int(len(files) / 2), sharex=False)
    fig.tight_layout()

    cols = {}
    col_nums = 0
    for file in files:
        # get names
        forgetting_name = file

        event_type = file.split(separator)[-1][:-4]
        data_type = file.split(separator)[-2]
        if event_type == 'forgetting_events':
            row = 0
        elif event_type == 'switch_events':
            row = 1
        else:
            raise Exception('Visualization for event type not implemented yet!')

        if data_type not in cols.keys():
            cols[data_type] = col_nums
            col_nums += 1

        # load files
        forgetting_events = np.load(forgetting_name).flatten()

        fraction = float(np.count_nonzero(forgetting_events == 0)) / float(forgetting_events.shape[0])
        print('FILE: ' + file)
        print('Fraction Unforgettable: %f' % fraction)
        print('Mean events: %f' % np.mean(forgetting_events))

        # include kl if training is included
        if reference_folder is not None:
            flattened_training = np.load(reference_folder + '/' + event_type + '.npy').flatten()
            kl = histogram_distance(flattened_training, forgetting_events,
                                    bin_param=cfg.visualization.hist_bin_param,
                                    dist_type=cfg.visualization.dist_type)
            print('KL-Divergence/training: %f' % kl)
        # plot histogram
        # hist, _ = np.histogram(forgetting_events, density=True, bins=cfg.visualization.hist_bin_param)
        axs[row, cols[data_type]].hist(forgetting_events, density=True, bins=cfg.visualization.hist_bin_param)
        axs[row, cols[data_type]].set(xlabel='#Events', ylabel='Number of samples')
        if row == 0:
            axs[row, cols[data_type]].title.set_text(data_type)
    plt.show()