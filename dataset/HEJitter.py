import random
import numpy as np
from skimage import color
from albumentations import ImageOnlyTransform

class HEJitter(ImageOnlyTransform):
    """
        Decomposing the RGB color of the tiles into HED color space, followed by multiplying
        the magnitude of H and E of every pixel by two i.i.d. Gaussian random variables
        with expectation equal to one.
        H = 1.88 * R - 0.77 * G - 0.6 * B.
        E = -1.02 * R - 1.13 * G - 0.48 * B.
        D = -0.55 * R - 0.13 * G + 1.57 * B.
    """

    def __init__(
        self,
        hematoxylin_limit=(0, 0),
        eosin_limit=(0, 0),
        dab_limit = (0, 0),
        always_apply=False,
        p=0.5,
    ):
        super(HEJitter, self).__init__(always_apply=always_apply, p=p)
        self.hematoxylin_limit = hematoxylin_limit
        self.eosin_limit = eosin_limit
        self.dab_limit = dab_limit

    def get_params(self):
        if isinstance(self.hematoxylin_limit, tuple):
            alpha = random.uniform(1 + self.hematoxylin_limit[0], 1 + self.hematoxylin_limit[1])
        else:
            alpha = random.gauss(mu=self.hematoxylin_limit, sigma=0.5)
        if isinstance(self.eosin_limit, tuple):
            beta = random.uniform(1 + self.eosin_limit[0], 1 + self.eosin_limit[1])
        else:
            beta = random.gauss(mu=self.eosin_limit, sigma=0.5)
        if isinstance(self.dab_limit, tuple):
            gamma = random.uniform(1 + self.dab_limit[0], 1 + self.dab_limit[1])
        else:
            gamma = random.gauss(mu=self.dab_limit, sigma=0.5)

        return {"alpha":alpha, "beta":beta, "gamma":gamma}

    def apply(self, img, alpha, beta, gamma, **params):
        # print('alpha: {}, beta: {}, gamma:{}'.format(alpha, beta, gamma))
        hed_img = color.rgb2hed(img)
        h, e, d = hed_img[:, :, 0], hed_img[:, :, 1], hed_img[:, :, 2]
        h *= alpha
        e *= beta
        d *= gamma
        hed_img = np.concatenate([h[:, :, np.newaxis], e[:, :, np.newaxis], d[:, :, np.newaxis]], axis=2)
        img = color.hed2rgb(hed_img)
        imin, imax = img.min(), img.max()
        img = (255 * (img - imin) / (imax - imin)).astype('uint8')
        return img

    def get_transform_init_args_names(self):
        return ()