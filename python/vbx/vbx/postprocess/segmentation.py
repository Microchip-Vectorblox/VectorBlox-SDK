import argparse
import numpy as np
from .yolo import voc_colors

voc_colors0 = np.asarray([[0, 0, 0]] + voc_colors, dtype="uint8")

def segmentation_mask(arr):
    return voc_colors0[np.argmax(arr, axis=0)]
