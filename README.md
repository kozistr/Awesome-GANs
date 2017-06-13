# Awesome-GANs with Tensorflow [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)[![Build Status](https://travis-ci.org/dwyl/esta.svg?branch=master)](https://travis-ci.org/)
Tensorflow implementation of GANs(Generative Adversarial Networks)

## Prerequisites
* python 3.5+
* tensorflow 1.1.0+
* scipy
* pillow
* h5py
* pickle
* glob, tqdm
* (sklearn for train_test_split)

## Usage
    (before running train.py, make sure run after downloading dataset & changing dataset directory in train.py)
    just download it and run train.py
    $ python3 train.py

## Datasets
Now supporting(?) DataSets are... (code is in /datasets.py)
* MNIST
* Cifar-10
* Cifar-100
* Celeb-A
* pix2pix shoes
* pix2pix bags
* (more DataSets will be added soon!)

## Papers & Codes
* ACGAN        : Auxiliary Classifier Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1610.09585)
* AdaGAN       : Boosting Generative Models [[arXiv]](https://arxiv.org/abs/1701.02386)
* BEGAN        : Boundary Equilibrium Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1703.10717) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/began.py)
* BSGAN        : Boundary-Seeking Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1702.08431)
* CGAN         : Conditional Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1411.1784) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/CGAN/cgan.py)
* CoGAN        : Coupled Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1606.07536)
* DCGAN        : Deep Convolutional Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1511.06434) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/DCGAN/dcgan.py)
* DiscoGAN     : Discover Cross-Domain Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1703.05192) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/DiscoGAN/discogan.py)
* EnergyGAN    : Energy-based Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1609.03126)
* f-GAN        : Training Generative Neural Samplers using Variational Divergence Minimization [[arXiv]](https://arxiv.org/abs/1606.00709)
* GAN          : Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1406.2661) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/gan.py)
* Softmax GAN  : Generative Adversarial Networks with Softmax [[arXiv]](https://arxiv.org/pdf/1704.06191.pdf) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/gan.py)
* GAP          : Generative Adversarial Parallelization [[arXiv]](https://arxiv.org/abs/1612.04021)
* InfoGAN      : Interpretable Representation Learning by Information Maximizing Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1606.03657)
* LAPGAN       : Laplacian Pyramid Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1506.05751)
* LSGAN        : Loss-Sensitive Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1701.06264)
* MAGAN        : Margin Adaptation for Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1704.03817)
* MRGAN        : Mode Regularized Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1612.02136)
* SalGAN       : Visual Saliency Prediction Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1701.01081)
* SeqGAN       : Sequence Generative Adversarial Networks with Policy Gradient [[arXiv]](https://arxiv.org/abs/1609.05473)
* SGAN         : Stacked Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1612.04357)
* WGAN         : Wasserstein Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1701.07875)
* ImprovedWGAN : Improved Training of Wasserstein Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1704.00028)

## Results
### BEGAN
#### global step : 0
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/BEGAN/train_0_0.png)
#### global step : 15k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/BEGAN/train_0_0.png)

### CGAN
#### global step : 0
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CGAN/CGAN/train_00000000.png)
#### global step : 440k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CGAN/CGAN/train_00440000.png)

### DCGAN
#### global step : 0
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/DCGAN/DCGAN/train_0_0.png)
#### global step : 14.1k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/DCGAN/DCGAN/train_199_140250.png)

### DiscoGAN
#### global step : 0

#### global step : 300k


### GAN (with Softmax)
#### global step : 0
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/GAN/train_00000000.png)
#### global step : 250k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/GAN/train_00250000.png)

## Author
Hyeongchan Kim / @kozistr, [@zer0day](http://zer0day.tistory.com)
