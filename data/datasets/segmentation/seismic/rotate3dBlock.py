import numpy as np
import os
import math
from glob import glob
from os.path import join as pjoin
from shutil import copyfile
import shutil
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate


# decides to save entire sections or store patches. True -> random crop with entire sections
random_crop = True
idx = 0
machine = 'linux'

if machine == 'linux':
    # save locations for images
    train_images = os.path.expanduser('~/data/seismic/train/')
    train_val = os.path.expanduser('~/data/seismic/val/')
    test1_images = os.path.expanduser('~/data/seismic/test1/')
    test2_images = os.path.expanduser('~/data/seismic/test2/')
    test_images = os.path.expanduser('~/data/seismic/test/')

    # path to training block
    complete = os.path.expanduser('~/data/seismic/volumes/total/complete_seismic.npy')
    complete_label = os.path.expanduser('~/data/seismic/volumes/total/complete_seismic_label.npy')

    # path to training block
    train = os.path.expanduser('~/data/seismic/volumes/train/train_seismic.npy')
    train_label = os.path.expanduser('~/data/seismic/volumes/train/train_labels.npy')

    # path to test1 block
    test1 = os.path.expanduser('~/data/seismic/volumes/test_once/test1_seismic.npy')
    test1_label = os.path.expanduser('~/data/seismic/volumes/test_once/test1_labels.npy')

    # path to test2 block
    test2 = os.path.expanduser('~/data/seismic/volumes/test_once/test2_seismic.npy')
    test2_label = os.path.expanduser('~/data/seismic/volumes/test_once/test2_labels.npy')
elif machine == 'ws':
    # save locations for images
    train_images = os.path.expanduser('/data/ryan/seismic/train/')
    train_val = os.path.expanduser('/data/ryan/seismic/val/')
    test1_images = os.path.expanduser('/data/ryan/seismic/test1/')
    test2_images = os.path.expanduser('/data/ryan/seismic/test2/')
    test_images = os.path.expanduser('/data/ryan/seismic/test/')

    # path to training block
    complete = os.path.expanduser('/data/ryan/facies_classification_benchmark/data/complete/complete_seismic.npy')
    complete_label = os.path.expanduser('/data/ryan/facies_classification_benchmark/data/complete/complete_seismic_label.npy')

    # path to training block
    train = os.path.expanduser('/data/ryan/facies_classification_benchmark/data/original/train/train_seismic.npy')
    train_label = os.path.expanduser('/data/ryan/facies_classification_benchmark/data/original/train/train_labels.npy')

    # path to test1 block
    test1 = os.path.expanduser('/data/ryan/facies_classification_benchmark/data/original/test_once/test1_seismic.npy')
    test1_label = os.path.expanduser('/data/ryan/facies_classification_benchmark/data/original/test_once/test1_labels.npy')

    # path to test2 block
    test2 = os.path.expanduser('/data/ryan/facies_classification_benchmark/data/original/test_once/test2_seismic.npy')
    test2_label = os.path.expanduser('/data/ryan/facies_classification_benchmark/data/original/test_once/test2_labels.npy')

else:
    # target files
    train_images = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/seismic/train/')
    train_val = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/seismic/val/')
    test1_images = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/seismic/test1/')
    test2_images = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/seismic/test2/')
    test_images = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/seismic/test/')

    # path to data files
    train = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/data/train/train_seismic.npy')
    train_label = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/data/train/train_labels.npy')
    test1 = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/data/test_once/test1_seismic.npy')
    test1_label = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/data/test_once/test1_labels.npy')
    test2 = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/data/test_once/test2_seismic.npy')
    test2_label = os.path.expanduser('~/PycharmProjects/Dataset/facies_classification_benchmark/data/test_once/test2_labels.npy')


def rotate_block(block, angle,  order, cval, mode='constant', reshape=True):
    # if no rotation needed return the block else rotate
    if angle == 0 or angle == 90:
        x = block
    else:
        print('rotating block to angle: {}, interpolation: {}'.format(angle, order))
        # rotates and performs spline interpolation
        x = rotate(input=block, angle=angle, reshape=reshape, order=order, mode=mode, cval=cval)
    print('done...')
    return x


def store_patches(section, save_loc, dtype, angle, i, patch_sz=(120, 120), stride=(64, 64), label=False, plot=False):
    """
    :param filename:
    :param patch_sz:
    :param intervals:
    :param cutoff:  a negative value of cutoff means mean we remove data from the end. A positive value means the opposite (xline, inline, depth)
    :return:
    """
    store_patches = {}
    index = {}
    h_steps = int(np.floor((section.shape[1] - patch_sz[1]) / stride[1] + 1))
    v_steps = int(np.floor((section.shape[0] - patch_sz[0]) / stride[0] + 1))
    if plot:
        print(h_steps, v_steps)

        print('Stats')
        print('section size = ', section.shape)
        print('number of patches per section = {} by {}'.format(h_steps, v_steps))

    count = 0
    for m in range(v_steps):
        for n in range(h_steps):
            store_patches[count] = section[m*stride[0]:m*stride[0]+patch_sz[0], n*stride[1]:n*stride[1]+patch_sz[1]]

            if plot:
                print(store_patches[count].shape)
                pos = plt.imshow(store_patches[count], aspect='auto', cmap='seismic')
                plt.title('section patch: {}, at angle: {} degree'.format(count, angle))
                plt.colorbar(mappable=pos)
                plt.show()

            if label:
                np.save(os.path.expanduser(save_loc) + '{}_{}_{}_{}_{}_label'.format(dtype, angle, i, count, idx),
                        store_patches[count].astype(np.int))
            else:
                np.save(os.path.expanduser(save_loc) + '{}_{}_{}_{}_{}'.format(dtype, angle, i, count, idx),
                        store_patches[count])

            index[count] = (m*stride[0], n*stride[1])
            count += 1
            idx += 1
    if plot:
        print('total patches: ', len(store_patches))
    return


def save_sections(cube, save_loc, points, angle, cval, dtype, desired_width, data, train_block, plot, rc=True):
    """
    Saves the sections of a cube specified by points. Furthermore, it stores the information in the format :
    inline/crossline_angle_sectionnumber.
    Parameters:
        :param cube: cube to be cut and saved off
        :type cube: ndarray
        :param save_loc: saving directory for the sections
        :type save_loc: str
        :param points: list of sections to be saved. Usually all using an interval of one.
        :type points: list
        :param angle: angle the cube was rotated with
        :type angle: int
        :param cval: value of undefined points generated through rotation. They are skipped
        :type cval: int
        :param dtype: inline/crossline specification
        :type dtype: str
        :param desired_width: desired width of the saved section.
        :type desired_width: int
        :param data: boolean to distinguish between data and labels
        :type data: bool
        :param train_block: UNCLEAR. ASK JOSEPH
        :type train_block: ndarray
        :param plot: plot parameter for debugging
        :type plot: bool
        :param rc: optional parameter to configure different parameters
        :type rc: bool
    """
    global idx
    count = 0
    # iterate over all points
    for i in range(0, len(points)):
        if i % 50 == 0:
            print('slicing {} section {}'.format(dtype, i))
        # if label block is not none
        if train_block is not None:
            if dtype == 'inline':
                section = cube[points[i], :, :].T
                iter_sec = train_block[points[i], :, :].T
            elif dtype == 'crossline':
                section = cube[:, points[i], :].T
                iter_sec = train_block[:, points[i], :].T
        # label is normal data in other case
        else:
            if dtype == 'inline':
                section = cube[points[i], :, :].T
                iter_sec = section
            elif dtype == 'crossline':
                section = cube[:, points[i], :].T
                iter_sec = section

        # make sure we get the desired width
        if desired_width is not None:
            pos1 = 0
            # ensure the sction fits by checking cval
            while (iter_sec[10, pos1] == cval) and (pos1 < iter_sec.shape[1] - 1):
                pos1 += 1
            pos2 = iter_sec.shape[1] - 1
            while (iter_sec[10, pos2] == cval) and (pos2 > 0):
                pos2 -= 1

            if (pos2 - pos1) > desired_width-1:
                section = section[:, pos1:pos2]
                if plot:
                    print(section.shape)
                    pos = plt.imshow(section, aspect='auto', cmap='seismic')
                    plt.title('section slice: {}, at angle: {} degree'.format(i, angle))
                    plt.colorbar(mappable=pos)
                    plt.show()
                if data:
                    if random_crop:
                        np.save(os.path.expanduser(save_loc) + '{}_{}_{}_{}'.format(dtype, angle, i, idx), section)
                        # increment id
                        idx += 1
                    else:
                        if rc:
                            store_patches(section, save_loc, dtype, angle, i, label=False, plot=plot)
                        else:
                            store_patches(section, save_loc, dtype, angle, i, stride=(120, 120), label=False, plot=plot)
                else:
                    if random_crop:
                        np.save(os.path.expanduser(save_loc) + '{}_{}_{}_{}_label'.format(dtype, angle, i, idx),
                                section.astype(np.int))
                        # increment id
                        idx += 1
                    else:
                        if rc:
                            store_patches(section, save_loc, dtype, angle, i, label=True, plot=plot)
                        else:
                            store_patches(section, save_loc, dtype, angle, i, stride=(120, 120), label=True, plot=plot)
                count += 1
        else:
            if data:
                if random_crop:
                    np.save(os.path.expanduser(save_loc) + '{}_{}_{}_{}'.format(dtype, angle, i, idx), section)
                    # increment id
                    idx += 1
                else:
                    if rc:
                        store_patches(section, save_loc, dtype, angle, i, label=False, plot=plot)
                    else:
                        store_patches(section, save_loc, dtype, angle, i, stride=(120, 120), label=False, plot=plot)
            else:
                if random_crop:
                    np.save(os.path.expanduser(save_loc) + '{}_{}_{}_{}_label'.format(dtype, angle, i, idx),
                            section.astype(np.int))
                    # increment id
                    idx += 1
                else:
                    if rc:
                        store_patches(section, save_loc, dtype, angle, i, label=True, plot=plot)
                    else:
                        store_patches(section, save_loc, dtype, angle, i, stride=(120, 120), label=True, plot=plot)


# slices the rotated block/cube
def slice(cube, save_loc, angle, intervals, plot, cval, data, train_block, desired_width=None, rc=True):
    x = cube.shape[0]
    y = cube.shape[1]
    # define the areas to cut. We do two angles at once (inline, crossline)
    points_x = range(intervals, x, intervals)
    points_y = range(intervals, y, intervals)
    save_sections(cube, save_loc, points_x, angle, cval, dtype='inline', desired_width=desired_width, data=data,
                  train_block=train_block, plot=plot, rc=rc)
    save_sections(cube, save_loc, points_y, angle, cval, dtype='crossline', desired_width=desired_width, data=data,
                  train_block=train_block, plot=plot, rc=rc)


def create_validation_set(data_loc, data_dest, plot=False, corrupt=False, corruption_type='patch',
                          trace_corruption_type='random'):
    '''
    Creates the validation set for a seismic volume. Since we want the implementation to be predictable, we are
    sampling 20% of all inlines/crosslines with fixed distances. Moreover, it corrupts every 10 sections with a
    gaussian function.
    Parameters:
        :param data_loc: location of the entire dataset
        :param data_dest: destination for the validation set
        :param plot: debug parameter to plot the corrupted vs. non corrupted image
        :param corrupt: specifies wether the validation set is to be partially corrupted (every 10th sample)
        :param corruption_type: type of corruption. 'patch' type results in a deterministic patch being corrupted.
        'trace' type results in a traces being corrupted
        :param trace_corruption_type: Only important if corruption_type='trace'. Specifies whether the trace choice is
        random choice or deterministic (corrupt left section)
        :return:
    '''
    val = []
    print("Including training images in validation..")

    # give corruption info
    if corrupt:
        print('Corrupting validation images')

    # get indices for inline and xline
    val_inline_numbers = np.arange(start=0, stop=400, step=5)
    val_xnline_numbers = np.arange(start=0, stop=700, step=5)

    # add inline to validation set
    count = 0
    for inline in val_inline_numbers:
        data_name = 'inline_0_' + str(inline) + '.npy'
        label_name = 'inline_0_' + str(inline) + '_label.npy'

        if count % 10 == 0 and corrupt:
            # create corrupted image
            image = np.load(pjoin(data_loc, data_name))
            mask = np.load(pjoin(data_loc, label_name))

            # check corruption type
            if corruption_type == 'patch':
                cor_image = corrupt_patch(image, mask)
            elif corruption_type == 'trace':
                cor_image = corrupt_trace(image, dtype=trace_corruption_type)
            else:
                raise Exception('Corrupt Validation: Corruption type must be patch or trace!')

            # replace image with corrupted image
            os.remove(pjoin(data_loc, data_name))
            np.save(pjoin(data_loc, data_name), cor_image)
            if plot:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.imshow(image, cmap='Greys')
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.imshow(cor_image, cmap='Greys')
        val.append(pjoin(data_loc, data_name))
        val.append(pjoin(data_loc, label_name))
        count += 1

    # add xline to validation set
    count = 0
    for xline in val_xnline_numbers:
        data_name = 'crossline_0_' + str(xline) + '.npy'
        label_name = 'crossline_0_' + str(xline) + '_label.npy'
        if count % 10 == 0 and corrupt:
            # create corrupted image
            image = np.load(pjoin(data_loc, data_name))
            mask = np.load(pjoin(data_loc, label_name))

            # check corruption type
            if corruption_type == 'patch':
                cor_image = corrupt_patch(image, mask)
            elif corruption_type == 'trace':
                cor_image = corrupt_trace(image, dtype=trace_corruption_type)
            else:
                raise Exception('Corrupt Validation: Corruption type must be patch or trace!')

            # replace image with corrupted image
            os.remove(pjoin(data_loc, data_name))
            np.save(pjoin(data_loc, data_name), cor_image)
            if plot:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.imshow(image, cmap='Greys')
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.imshow(cor_image, cmap='Greys')
        val.append(pjoin(data_loc, data_name))
        val.append(pjoin(data_loc, label_name))
        count += 1

    # move images to validation folder
    for item in val:
        shutil.move(item, data_dest)


def create_deterministic_validation_set(data_loc, data_dest):
    angles = {}
    val =[]
    label = []
    # fmap for forgetting events
    fmap = []
    total_len = 0
    print("Including training images in validation..")

    # get indices for inline and xline
    val_inline_numbers = np.arange(start=0, stop=400, step=5)
    val_xnline_numbers = np.arange(start=0, stop=700, step=5)

    # add inline to validation set
    for inline in val_inline_numbers:
        data = glob(pjoin(data_loc, 'inline_0_' + str(inline) + '_*_label.npy'))
        if len(data) > 1:
            raise Exception('Duplicate ids!!!!!')
        id = data[0].split('_')[-2]
        data_name = pjoin(data_loc, 'inline_0_' + str(inline) + '_' + id + '.npy')
        label_name = data[0]
        val.append(data_name)
        val.append(label_name)

    # add xline to validation set
    for xline in val_xnline_numbers:
        data = glob(pjoin(data_loc, 'crossline_0_' + str(xline) + '_*_label.npy'))
        if len(data) > 1:
            raise Exception('Duplicate ids!!!!!')
        id = data[0].split('_')[-2]
        data_name = pjoin(data_loc, 'crossline_0_' + str(xline) + '_' + id + '.npy')
        label_name = data[0]
        val.append(data_name)
        val.append(label_name)

    # move images to validation folder
    for item in val:
        shutil.move(item, data_dest)


def norm_pdf_multivariate(x1, x2, mu, sigma):
    size = 2
    # x = np.array([x1, x2])
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ (math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2))
        xm1 = x1 - mu[0]
        xm2 = x2 - mu[1]
        inv = np.linalg.inv(sigma)
        result = np.exp(-0.5 * ((xm1**2)*inv[0, 0] + xm1*xm2*(inv[0, 1] + inv[1, 0]) + (xm2**2)*inv[1, 1]))
        return result
    else:
        raise NameError("The dimensions of the input don't match")


def corrupt_patch(image, mask, corruption_window=(150, 150), stride=20, class_list=[3, 4, 5], var=1500.0, plot=True):
    # get cropped image width and height
    tw = corruption_window[0]
    th = corruption_window[1]
    # init best coordinates
    x1 = 0
    y1 = 0
    best_count = 0

    # get corruption window based on class list
    w, h = image.shape
    w_steps = int(np.floor((w - tw) / stride + 1))
    h_steps = int(np.floor((h - th) / stride + 1))

    # iterate over all possible crops
    for m in range(w_steps):
        for n in range(h_steps):
            # get current image crop
            cur_patch = mask[m * stride:m * stride + tw, n * stride:n * stride + th]

            # iterate over all classes and count class occurances
            cur_count = 0
            for j in class_list:
                num_pixels_in_patch = np.count_nonzero(cur_patch == j)
                cur_count = cur_count + num_pixels_in_patch

            # save coordinates if patch is the best match
            if cur_count > best_count:
                best_count = cur_count
                x1 = m * stride
                y1 = n * stride

    # get statisitics from range -> init mean and covariance matrix
    mean = np.array([x1 + tw/2, y1 + th/2])
    variance_sample = var
    cov = np.diag(np.array([variance_sample, variance_sample]))

    # x and y
    x = np.arange(x1, x1 + tw)
    y = np.arange(y1, y1 + th)
    # init corruption array
    gaussian = norm_pdf_multivariate(x[:, None], y[None, :], mean, cov)
    corruption_array = np.zeros(image.shape)
    corruption_array[x1:x1 + tw, y1:y1 + th] = gaussian

    # subtract from real image
    out = image - image*corruption_array

    if plot:
        print('Current Variance: %f' %variance_sample)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(out, cmap='Greys')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        im = plt.imshow(corruption_array, cmap='jet')
        # plt.colorbar(im)
        plt.axis('off')
        plt.show()

    return out


def corrupt_trace(image, corruption_percentage=0.3, dtype='random', plot=False):
    # get height and width of image
    w = image.shape[0]
    h = image.shape[1]

    # init org image for comparison
    org = image.copy()

    # number of corrupted traces
    num_corrupted = int(corruption_percentage*h)

    if dtype == 'random':
        # get random indeces and corrupt
        corruption_indices = np.random.choice(h, num_corrupted, replace=False)
        for index in list(corruption_indices):
            image[:, index] = np.zeros(w)
    else:
        image[:, :num_corrupted] = np.zeros((w, num_corrupted))

    # plot if specified
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(org, cmap='Greys')
        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap='Greys')
        plt.show()

    return image


def run(load_loc, save_loc, data, train_block, order, cval=-1,  plot=False, intervals=1, angles=(0, 30, 45, 60),
        desired_width=None, diff_loc=None, rc=True):
    # rotate over all angles
    for i in angles:
        print('rotating angle {}'.format(i))
        # load data array
        x = np.load(os.path.expanduser(load_loc)).astype(np.float)
        # if we are not cutting over inline/crossline
        if i != 0:
            # get rid of diff pixels if specified
            if diff_loc is not None:
                print('removing boundary infractions...')
                diff = np.load(os.path.expanduser(diff_loc.format(i))).astype(np.float)
                x[diff.nonzero()] = -1.0
            # rotate the volume
            x = rotate_block(x, angle=i, order=order, cval=cval)
            if train_block is not None:
                block = np.load(os.path.expanduser(train_label)).astype(np.float)
                print('rotating training label block...')
                block = rotate_block(block, angle=i, order=order, cval=cval)
            else:
                block = None
        else:
            block = None
        slice(x, save_loc, angle=i, intervals=intervals, plot=plot, cval=cval, data=data, train_block=block,
              desired_width=desired_width, rc=rc)


def create_data():
    global idx
    # init required parameters
    cval1 = -1
    cval2 = -1
    plot = False
    intervals = 1
    # train_angles = [0, 30, 45, 60]
    train_angles = [0]

    # param to create images
    data_create = True

    # param for partitioning to validation
    create_validation = True
    deterministic_validation = True

    # param for corrupting test and validation
    corrupt_test = False
    corrupt_validation = False
    corrupt_training = False

    # specify corruption type -> choices trace and random
    corruption_type = 'patch'
    # only important if trace is specified! Choice of corrupting random traces or the left section
    trace_corruption_type = 'det'

    if data_create:
        print('Creating cutted images....')
        # remove directorz specified by train_images
        if os.path.exists(train_images):
            shutil.rmtree(os.path.expanduser(train_images))
        # make new dir at that place
        os.makedirs(os.path.expanduser(train_images))
        # run the necessary data scripts
        print("Generating train rotated")
        if random_crop:
            run(load_loc=train, save_loc=train_images, data=True, train_block='train_block', order=0, cval=cval1,
                plot=plot,
                intervals=intervals, angles=train_angles, desired_width=255, rc=True)
            idx = 0
            run(load_loc=train_label, save_loc=train_images, data=False, train_block='train_block', order=0, cval=cval2,
                plot=plot, intervals=intervals, angles=train_angles, desired_width=255, rc=True)
            idx = 0
        else:
            run(load_loc=train, save_loc=train_images, data=True, train_block='train_block', order=0, cval=cval1,
                plot=plot,
                intervals=intervals, angles=train_angles, desired_width=255, rc=False)
            idx = 0
            run(load_loc=train_label, save_loc=train_images, data=False, train_block='train_block', order=0, cval=cval2,
                plot=plot, intervals=intervals, angles=train_angles, desired_width=255, rc=False)
            idx = 0

        print("Generating test1 not rotated")
        if os.path.exists(test1_images):
            shutil.rmtree(os.path.expanduser(test1_images))
        os.makedirs(os.path.expanduser(test1_images))
        run(load_loc=test1, save_loc=test1_images, data=True, train_block=None, order=0, cval=cval1, plot=plot,
            intervals=intervals, angles=[0], desired_width=None, rc=False)
        idx = 0
        run(load_loc=test1_label, save_loc=test1_images, data=False, train_block=None, order=0, cval=cval2, plot=plot,
            intervals=intervals, angles=[0], desired_width=None, rc=False)
        idx = 0
        print("Generating test2 not rotated")
        # do not reset id for test 2
        if os.path.exists(test2_images):
            shutil.rmtree(os.path.expanduser(test2_images))
        os.makedirs(os.path.expanduser(test2_images))
        run(load_loc=test2, save_loc=test2_images, data=True, train_block=None, order=0, cval=cval1, plot=plot,
            intervals=intervals, angles=[0], desired_width=None, rc=False)
        idx = 0
        run(load_loc=test2_label, save_loc=test2_images, data=False, train_block=None, order=0, cval=cval2, plot=plot,
            intervals=intervals, angles=[0], desired_width=None, rc=False)

        # copying into one test folder
        print('copying all test sets into one folder')
        if os.path.exists(test_images):
            shutil.rmtree(os.path.expanduser(test_images))
        os.makedirs(os.path.expanduser(test_images))
        files_t1 = glob(os.path.expanduser(test1_images) + '*.npy')
        files_t2 = glob(os.path.expanduser(test2_images) + '*.npy')
        # files_t3 = glob(os.path.expanduser(test3_images) + '*.npy')
        for file in files_t1:
            filename = file.split('/')[-1]
            if machine == 'win':
                filename = filename[6:]
            copyfile(file, pjoin(os.path.expanduser(test_images), 't1_' + filename))

        for file in files_t2:
            filename = file.split('/')[-1]
            if machine == 'win':
                filename = filename[6:]
            copyfile(file, pjoin(os.path.expanduser(test_images), 't2_' + filename))

        # corrupt test images
        if corrupt_test:
            print('Corrupting Test images...')
            files_test = glob(os.path.expanduser(test_images) + '*.npy')
            count = 0
            for file in files_test:
                if file.split('_')[-1][:-4] != 'label' and count % 10 == 0:
                    # load and corrupt image
                    img = np.load(file)
                    mask = np.load(file[:-4] + '_label.npy')

                    # check corruption type
                    if corruption_type == 'patch':
                        cor = corrupt_patch(img, mask)
                    elif corruption_type == 'trace':
                        cor = corrupt_trace(img, dtype=trace_corruption_type)
                    else:
                        raise Exception('Corrupt Test: Corruption type must be patch or trace!')

                    # replace image with corrupted image
                    os.remove(file)
                    np.save(file, cor)

                    # inc counter
                    count += 1
                elif file.split('_')[-1][:-4] != 'label':
                    count += 1

        if os.path.exists(train_val):
            shutil.rmtree(os.path.expanduser(train_val))
        os.makedirs(os.path.expanduser(train_val))
    if create_validation:
        print("Creating validation set")
        if deterministic_validation:
            create_deterministic_validation_set(data_loc=train_images, data_dest=train_val)
        else:
            create_validation_set(data_loc=train_images, data_dest=train_val, corrupt=corrupt_validation,
                                  corruption_type=corruption_type, trace_corruption_type=trace_corruption_type)

    if corrupt_training:
        print('Corrupting Training images...')
        files_training = glob(os.path.expanduser(train_images) + '*.npy')
        count = 0
        for file in files_training:
            if file.split('_')[-1][:-4] != 'label' and count % 100 == 0:
                # load and corrupt image
                img = np.load(file)
                mask = np.load(file[:-4] + '_label.npy')

                # check corruption type
                if corruption_type == 'patch':
                    cor = corrupt_patch(img, mask)
                elif corruption_type == 'trace':
                    cor = corrupt_trace(img, dtype=trace_corruption_type)
                else:
                    raise Exception('Corrupt Training: Corruption type must be patch or trace!')

                # replace image with corrupted image
                os.remove(file)
                np.save(file, cor)

                # inc counter
                count += 1
            elif file.split('_')[-1][:-4] != 'label':
                count += 1
    print('Finished with the Blocks!!!!!')


if __name__ == '__main__':
    create_data()