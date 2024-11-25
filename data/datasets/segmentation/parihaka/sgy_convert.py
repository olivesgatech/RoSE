import segyio
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = os.path.expanduser(f'~\datasets\parihaka\TrainingData_Image.segy')
    seismic = segyio.tools.cube(filename)

    plt.imshow(seismic[:, :, 15], cmap='gray_r')
    plt.show()

