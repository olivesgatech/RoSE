import plotly.graph_objects as go
import numpy as np
import os


TARGETFILE = os.path.expanduser(f'~/datasets/parihaka/parihaka_data_processed.npy')
TARGETLABELFILE = os.path.expanduser(f'~/datasets/parihaka/volumes/parihaka_labels_processed.npy')
# TARGETLABELFILE = os.path.expanduser(f'~/datasets/parihaka/splits/training/training_labels.npy')
# TARGETLABELFILE = os.path.expanduser(f'~/datasets/parihaka/splits/test/test2_labels.npy')


if __name__ == '__main__':
    # seismic = np.load(TARGETFILE)
    labels = np.load(TARGETLABELFILE)
    labels = labels[::6, ::6, :]
    X, Y, Z = np.mgrid[:labels.shape[0], :labels.shape[1], :labels.shape[2]]
    # values = np.sin(np.pi * X) * np.cos(np.pi * Z) * np.sin(np.pi * Y)

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=labels.flatten(),
        # isomin=-0.1,
        # isomax=0.8,
        opacity=0.3,  # needs to be small to see through all surfaces
        surface_count=6,  # needs to be a large number for good volume rendering
    ))
    fig.show()