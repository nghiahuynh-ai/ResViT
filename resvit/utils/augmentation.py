# from PIL import Image
import numpy as np
from imgaug import augmenters as augmentor


class ImgAugTransform:
  def __init__(self):
    sometimes = lambda aug: augmentor.Sometimes(0.3, aug)

    self.aug = augmentor.Sequential(augmentor.SomeOf((1, 5), 
    [
        # blur
        sometimes(augmentor.OneOf([
            augmentor.GaussianBlur(sigma=(0, 1.0)),
            augmentor.MotionBlur(k=3),
            augmentor.AverageBlur(k=3),
            augmentor.MedianBlur(k=3)    
        ])),
        
        # contrast
        sometimes(augmentor.OneOf([
            augmentor.GammaContrast((0.5, 2.0)),
            augmentor.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
            augmentor.LogContrast(gain=(0.6, 1.4)),
            augmentor.LinearContrast((0.4, 1.6)),
        ])),
        
        # color
        sometimes(augmentor.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
        sometimes(augmentor.Invert(0.25, per_channel=0.5)),
        sometimes(augmentor.Solarize(0.5, threshold=(32, 128))),
        sometimes(augmentor.Dropout2d(p=0.5)),
        sometimes(augmentor.Multiply((0.5, 1.5), per_channel=0.5)),
        sometimes(augmentor.Add((-40, 40), per_channel=0.5)),
        sometimes(augmentor.JpegCompression(compression=(5, 80))),
    ],
        random_order=True),
    random_order=True)
      
  def __call__(self, img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    img = self.aug.augment_image(img)
    # img = Image.fromarray(img)
    return img