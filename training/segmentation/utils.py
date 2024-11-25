import pandas as pd
import numpy as np
from training.segmentation.segmentationtracker import SegmentationTracker


class TestOutput:
    def __init__(self, out_df: pd.DataFrame, out_tracker: SegmentationTracker, prediction: np.ndarray,
                 images: np.ndarray, gt: np.ndarray):
        self.statdf = out_df
        self.tracker = out_tracker
        self.prediction = prediction
        self.images = images
        self.gt = gt
