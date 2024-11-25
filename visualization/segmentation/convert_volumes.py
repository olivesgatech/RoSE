import os
import glob
import tqdm
import shutil
import numpy as np
from visualization.utils.plotting import pred_entropy


def slice_volume(prediction_volume: np.ndarray, folder: str):
    for i in range(prediction_volume.shape[1]):
        np.save(folder + str(i) + '.npy', prediction_volume[:, i, ...])


def slice_preds(files: list):
    print('Slicing Volumes')
    for file in files:
        print(file)
        folder = file[:-len('predictions.npy')]
        slice_folder = folder + '/slices/'
        if os.path.exists(slice_folder):
            shutil.rmtree(slice_folder[:-1])
        os.mkdir(slice_folder)
        predictions = np.load(file)
        slice_volume(predictions, slice_folder)


def generate_entropy_volumes(files: list):

    for file in files:
        print(file)
        folder = file[:-len('predictions.npy')]
        slice_folder = folder + '/slices/'
        if not os.path.exists(slice_folder):
            raise Exception('volume slicing has not been run yet! You must slice the volume first!')
        slice_files = glob.glob(slice_folder + '*')

        tmp_folder = folder + '/tmp/'
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)

        os.mkdir(tmp_folder)
        print('Generating entropy slices. Saving in ' + tmp_folder)
        tbar = tqdm.tqdm(slice_files)
        entropy_files = []
        for i, slice_file in enumerate(tbar):
            slice = np.load(slice_file)
            entropy = np.squeeze(pred_entropy(slice, 6))
            file_name = tmp_folder + slice_file.split('/')[-1]
            np.save(file_name, entropy)
            tbar.set_description(file_name + '->' + str(entropy.shape))
            entropy_files.append(file_name)

        print('Reassembling volume...')
        entropy_files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
        tbar = tqdm.tqdm(entropy_files)
        volume = None
        for i, entropy_file in enumerate(tbar):
            entropy_slice = np.load(entropy_file)
            if volume is not None:
                volume = np.concatenate((volume, entropy_slice[np.newaxis, ...]), axis=0)
            else:
                volume = entropy_slice[np.newaxis, ...]
            tbar.set_description(str(volume.shape))

        print('Saving volume')
        vol_file_name = folder + 'uspec-entropy.npy'
        print(vol_file_name)
        np.save(vol_file_name, volume)

        print('Cleaning up...')
        shutil.rmtree(tmp_folder)
        #entropy = pred_entropy(predictions, 6)
        #print(entropy.shape)

    return


if __name__ == '__main__':
    folder = '~/results/segmentation_seismic_USPECAnalysis/uspec_statistics/'
    folder = '~/results/segmentation_seismic_USPECAnalysis/uspec_statistics/'
    data_path = os.path.expanduser(folder)
    folders = glob.glob(data_path + '*')
    files = []
    for folder in folders:
        file_list = glob.glob(folder + '/predictions*')
        files.extend(file_list)

    # slice_preds(files)
    generate_entropy_volumes(files)