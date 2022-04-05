import random
import numpy as np
from albumentations import ImageOnlyTransform


class RandomCrop(ImageOnlyTransform):
    """
    Crop the given image at a random location.
    """
    def __init__(self, height, width, padding, always_apply=False, p=0.5):
        super(RandomCrop, self).__init__(always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.padding = padding

    def apply(self, img, **params):
        h, w, _ = img.shape
        img_mask = np.zeros((h+2*self.padding, w+2*self.padding, 3), dtype='uint8')
        img_mask[self.padding: self.padding+h, self.padding: self.padding+w, :] = img
        h_start, w_start = random.randint(0, 2*self.padding), random.randint(0, 2*self.padding)
        img = img_mask[h_start: h_start+h, w_start: w_start+w, :]
        return img



