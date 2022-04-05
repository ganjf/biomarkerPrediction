import numpy as np
from albumentations import ImageOnlyTransform

class Cutout(ImageOnlyTransform):
    """
    Randomly mask out one or more patches from an image. https://arxiv.org/abs/1708.04552
    Args:
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, size, n_holes, always_apply=False, p=0.5):
        super(Cutout, self).__init__(always_apply=always_apply, p=p)
        self.size = size
        self.n_holes = n_holes

    def apply(self, img, **param):
        h, w, _ = img.shape
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            x = np.random.randint(w)
            y = np.random.randint(h)

            y1 = np.clip(y - self.size // 2, 0, h)
            y2 = np.clip(y + self.size // 2, 0, h)
            x1 = np.clip(x - self.size // 2, 0, w)
            x2 = np.clip(x + self.size // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
        mask = mask[:, :, np.newaxis].repeat(3, axis=2)
        img = img * mask
        return img

    def get_transform_init_args_names(self):
        return ()