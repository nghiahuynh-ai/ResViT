import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

def gen_label(image, polygons):
        
    h, w, _ = image.shape

    gt = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(gt, polygons, 1)
        
    return gt