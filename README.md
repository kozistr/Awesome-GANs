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

## Datasets
Now supporting(?) cifar-10 and cifar-100 (code is in dataset.py)
(more dataset will be added soon!)

## Papers
* BEGAN     : https://arxiv.org/abs/1703.10717
* CGAN      : https://arxiv.org/abs/1411.1784
* DCGAN     : https://arxiv.org/abs/1511.06434
* DiscoGAN  : https://arxiv.org/abs/1703.05192
* EnergyGAN : https://arxiv.org/abs/1609.03126
* GAN       : https://arxiv.org/abs/1406.2661
* InfoGAN   : https://arxiv.org/abs/1606.03657
* LAPGAN    : https://arxiv.org/abs/1506.05751
* LSGAN     : https://arxiv.org/abs/1701.06264
* SalGAN    : https://arxiv.org/abs/1701.01081
* SeqGAN    : https://arxiv.org/abs/1609.05473
* WGAN      : https://arxiv.org/abs/1701.07875

## Results
### BEGAN
#### global step : 0
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/BEGAN/train_0_0.png)
#### global step : 15k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/BEGAN/train_0_0.png)

### DCGAN
#### global step : 0
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/DCGAN/DCGAN/train_0_0.png)
#### global step : 14.1k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/DCGAN/DCGAN/train_199_140250.png)

### GAN
#### global step : 0
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/GAN/train_0.png)
#### glob al step : 100k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/GAN/train_100000.png)

## Author
Hyeongchan Kim / @kozistr, [@zer0day](http://zer0day.tistory.com)
