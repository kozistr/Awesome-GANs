# Awesome-GANs with Tensorflow [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)[![Build Status](https://travis-ci.org/dwyl/esta.svg?branch=master)](https://travis-ci.org/)
Tensorflow implementation of GANs(Generative Adversarial Networks)

## Test Environments
* OS : Linux Ubuntu 16.04 x86-64
* CPU : i7-7700K, GPU : GTX 1060 6GB
* Tensorflow 1.4.0 with CUDA 8.0 + cuDNN 7.0
* Python 3.5

## Prerequisites
* python 3.5+
* tensorflow 1.4.0+ (renewal)
* scipy
* pillow
* h5py
* pickle
* glob, tqdm
* sklearn
* Internet :)

## Usage
    (before running train.py, make sure run after downloading dataset & changing dataset directory in train.py)
    just download it and run train.py
    $ python3 xxx_train.py

## DataSets
Now supporting(?) DataSets are... (code is in /datasets.py)
* MNIST
* CiFar-10
* CiFar-100
* Celeb-A
* pix2pix shoes
* pix2pix bags
* (more DataSets will be added soon!)

## Repo Tree
> [GAN Name/] <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-- [gen_img/...] (generated images) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-- [..._train.py] (training) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-- [..._model.py] (gan model) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-- [....md] (explain text) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-- [dataset.py] (dataset loader) <br/>

## Papers & Codes
* ACGAN        : Auxiliary Classifier Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1610.09585)
* AdaGAN       : Boosting Generative Models [[arXiv]](https://arxiv.org/abs/1701.02386)
* BEGAN        : Boundary Equilibrium Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1703.10717) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/began_model.py)
* BSGAN        : Boundary-Seeking Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1702.08431)
* CGAN         : Conditional Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1411.1784) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/CGAN/cgan_model.py)
* CoGAN        : Coupled Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1606.07536)
* CycleGAN     : Unpaired img2img translation using Cycle-consistent Adversarial Networks [[arXiv]](https://arxiv.org/pdf/1703.10593.pdf)
* DCGAN        : Deep Convolutional Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1511.06434) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/DCGAN/dcgan_model.py)
* DiscoGAN     : Discover Cross-Domain Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1703.05192) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/DiscoGAN/discogan.py)
* EnergyGAN    : Energy-based Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1609.03126)
* f-GAN        : Training Generative Neural Samplers using Variational Divergence Minimization [[arXiv]](https://arxiv.org/abs/1606.00709)
* GAN          : Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1406.2661) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/gan_model.py)
* Softmax GAN  : Generative Adversarial Networks with Softmax [[arXiv]](https://arxiv.org/pdf/1704.06191.pdf) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/gan_model.py)
* 3D GAN       : 3D Generative Adversarial Networks [[MIT]](http://3dgan.csail.mit.edu/)
* GAP          : Generative Adversarial Parallelization [[arXiv]](https://arxiv.org/abs/1612.04021)
* GEGAN        : Generalization and Equilibrium in Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1703.00573)
* InfoGAN      : Interpretable Representation Learning by Information Maximizing Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1606.03657)
* LAPGAN       : Laplacian Pyramid Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1506.05751)
* LSGAN        : Loss-Sensitive Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1701.06264)
* MAGAN        : Margin Adaptation for Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1704.03817)
* MRGAN        : Mode Regularized Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1612.02136)
* SalGAN       : Visual Saliency Prediction Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1701.01081)
* SeqGAN       : Sequence Generative Adversarial Networks with Policy Gradient [[arXiv]](https://arxiv.org/abs/1609.05473)
* SGAN         : Stacked Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1612.04357)
* WGAN         : Wasserstein Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1701.07875) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/WGAN/wgan_model.py)
* ImprovedWGAN : Improved Training of Wasserstein Generative Adversarial Networks [[arXiv]](https://arxiv.org/abs/1704.00028) [[code]](https://github.com/kozistr/Awesome-GANs/blob/master/WGAN/wgan_model.py)

## Results
### BEGAN
#### global step 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 150k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/gen_img/train_0_0.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/gen_img/train_0_0.png)

### CGAN
#### global step 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 225k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CGAN/gen_img/train_00000000.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CGAN/gen_img/train_00225000.png)

### DCGAN
#### global step 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 270k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/DCGAN/gen_img/train_0_0.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/DCGAN/gen_img/train_43_27000.png)

### DiscoGAN
#### global step 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 300k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/DiscoGAN/gen_img/train_0_0.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/DiscoGAN/gen_img/train_0_0.png)

### GAN
#### global step 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 250k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/gen_img/train_00000000.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/GAN/gen_img/train_00250000.png)

### InfoGAN
#### global step 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 250k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/InfoGAN/gen_img/train_00000000.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/InfoGAN/gen_img/train_00250000.png)

### LAPGAN
#### global step 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 250k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/LAPGAN/gen_img/train_00000000.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/LAPGAN/gen_img/train_00250000.png)

### MRGAN
#### global step 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 250k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/MRGAN/gen_img/train_00000000.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/MRGAN/gen_img/train_00250000.png)

### WGAN & Improved-WGAN
#### global step 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 90k &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; global step 80k
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/WGAN/gen_img/train_00000000.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/WGAN/gen_img/train_00090000_1.png)
![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/WGAN/gen_img/train_00080000_2.png)

WGAN took so much times(?) so i just stopped at 90k. <br/>
If you want better results, maybe you should increase global step or adjust the d_lambda, a little bit.

## Author
HyeongChan Kim / [@kozistr](https://kozistr.github.io), [@zer0day](http://zer0day.tistory.com)
