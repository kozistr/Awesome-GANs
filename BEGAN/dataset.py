import os
import numpy as np
import scipy.misc as scm


class Dataset:

    def __init__(self, dataset_path, random_rate=42, dataset_name="Celeb-A"):
        self.path = dataset_path
        self.rand_rate = random_rate
        self.name = dataset_name

        self.load_image()

    def load_image(self):
        if self.name in "Celeb-A":
            # load image
            img = scm.imread(self.path)
            img = img / 255. - 0.5

            # rgb to bgr
            img = img[..., ::-1]

            return img
        else:
            pass
