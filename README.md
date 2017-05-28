# Awesome-GANs in Tensorflow [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)[![Build Status](https://travis-ci.org/dwyl/esta.svg?branch=master)](https://travis-ci.org/)
Tensorflow implementation of GANs(Generative Adversarial Networks)

## Prerequisites
* Python 2.7+ or 3.3+
* Tensorflow r1.1
* SciPy
* pillow

## Usage
    (before running train.py, make sure downloading dataset & changing dataset directory in train.py)
    just download it and run train.py
    $ python3 train.py

## Dataset
Now supporting(?) cifar-10 and cifar-100 (code is in dataset.py)
(more dataset will be added soon!)

## Paper
BEGAN     : https://arxiv.org/abs/1703.10717
CGAN      :
DCGAN     : https://arxiv.org/abs/1511.06434
DiscoGAN  :
EnergyGAN :
GAN       :
InfoGAN   :
LAPGAN    :
LSGAN     :
SalGAN    :
SeqGAN    :
WGAN      :

## Results
### DCGAN
#### global step : 0
![Alt text](/DCGAN/DCGAN/train_0_0.png)
#### global step : 150k
![Alt text](/DCGAN/DCGAN/train_199_149250.png)

## Author
Hyeongchan Kim / @kozistr
