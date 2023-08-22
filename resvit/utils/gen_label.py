import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

def gen_label(image, polygons):
        
    h, w, _ = image.shape

    gt = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(gt, polygons, 1)
        
    return gt


class MakeSegDetectionData:
    
    def __init__(self, min_text_size=8, shrink_ratio=0.4):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        
    def __call__(self, image, polygons):
       
        h, w = image.shape[:2]
        gt = np.zeros((h, w), dtype=np.float32)
        thres = np.zeros((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            
            if min(height, width) < self.min_text_size:
                cv2.fillPoly(gt, polygons, 1)
                cv2.fillPoly(thres, polygons, 1)
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(thres, polygon.astype(np.int32), 1)
                    cv2.fillPoly(gt, polygon.astype(np.int32), 1)
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(thres, polygon.astype(np.int32), 1)
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)
        
        return gt, thres