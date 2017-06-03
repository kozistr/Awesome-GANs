import os
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.misc import imread, imresize


def get_image(path, w, h):
    img = imread(path).astype(np.float)

    orig_h, orig_w = img.shape[:2]
    new_h = int(orig_h * w / orig_w)

    img = imresize(img, (new_h, w))
    margin = int(round((new_h - h) / 2))

    return img[margin:margin + h]


class Dataset:

    def __init__(self, dataset_path, input_height=64, input_width=64, input_channel=3,
                 random_rate=42, dataset_name="Celeb-A"):
        self.path = dataset_path
        self.h5_path = '/home/zero/celeba/celeba.h5'
        self.rand_rate = random_rate
        self.name = dataset_name

        self.filenames = glob(os.path.join("img_align_celeba", "*.jpg"))
        self.filenames = np.sort(self.filenames)

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel

        self.data = np.zeros((len(self.filenames),
                              self.input_height * self.input_width * self.input_channel), dtype=np.uint8)

        for n, fname in tqdm(enumerate(self.filenames)):
            image = get_image(fname, self.input_width, self.input_height)
            self.data[n] = image.flatten()

        with h5py.File(''.join([self.h5_path]), 'w') as f:
            f.create_dataset("images", data=self.data)

        self.images = self.load_data()

    def load_data(self, size=202599, offset=0):
        '''
            From great jupyter notebook by Tim Sainburg:
            http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
        '''
        with h5py.File(self.h5_path, 'r') as hf:
            faces = hf['images']

            full_size = len(faces)
            if size is None:
                size = full_size

            n_chunks = int(np.ceil(full_size / size))
            if offset >= n_chunks:
                print("[*] Looping back to start.")
                offset = offset % n_chunks

            if offset == n_chunks - 1:
                print("[-] Not enough data available, clipping to end.")
                faces = faces[offset * size:]

            else:
                faces = faces[offset * size:(offset + 1) * size]

            faces = np.array(faces, dtype=np.float16)

        return faces / 255

