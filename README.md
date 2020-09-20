# Awesome-GANs with Tensorflow

Tensorflow implementation of GANs (Generative Adversarial Networks)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
[![Total alerts](https://img.shields.io/lgtm/alerts/g/kozistr/Awesome-GANs.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/Awesome-GANs/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/kozistr/Awesome-GANs.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/Awesome-GANs/context:python)

**WIP** : this repo is about to be refactored & supporting `tf 2.x`.

## Environments

### Preferred Environment
* OS  : Windows 10 / Linux Ubuntu x86-64 ~
* CPU : any (quad core ~)
* GPU : GTX 1060 6GB ~
* RAM : 16GB ~
* Library : TF 1.x with CUDA 9.0~ + cuDNN 7.0~
* Python 3.x

Because of the image and model size, (especially **BEGAN**, **SRGAN**, **StarGAN**, ... using high resolution images as input),
if you want to train them comfortably, you need a GPU which has more than 8GB.

But, of course, the most of the implementations use MNIST or CiFar-10, 100 DataSets.
Meaning that we can handle it with EVEN lower spec GPU than 'The Preferred' :).

## Prerequisites

* python 3.x
* tensorflow 1.x
* numpy
* scipy (some features are about to **deprecated**, it'll be replaced to OpenCV SOON!)
* scikit-image
* opencv-python
* pillow
* h5py
* tqdm
* Internet :)

## Usage

### Dependency Install
    $ sudo python3 -m pip install -r requirements.txt

### Training GAN
    (Before running train.py, MAKE SURE run after downloading DataSet & changing DataSet's directory in xxx_train.py)
    just after it, RUN train.py
    $ python3 xxx_train.py

## DataSets

Now supporting(?) DataSets are... (code is in /datasets.py)
* MNIST / ~~Fashion MNIST~~
* CiFar-10 / 100
* CelebA/CelebA-HQ
* pix2pix DataSets
* DIV2K DataSets
* ~~ImageNet DataSets~~
* ~~UrbanSound8K~~
* ~~3DShapeNet DataSet~~
* (more DataSets will be added soon!)

## Repo Tree

```
│
├── xxGAN
│    ├──gan_img (generated images)
│    │     ├── train_xxx.png
│    │     └── train_xxx.png
│    ├── model  (model)
│    │     └── model.txt (google-drive link for pre-trained model)
│    ├── gan_model.py (gan model)
│    ├── gan_train.py (gan trainer)
│    ├── gan_tb.png   (Tensor-Board result)
│    └── readme.md    (results & explains)
├── tfutil.py         (useful TF util)
├── image_utils.py    (image processing)
└── datasets.py       (DataSet loader)
```

## Pre-Trained Models

Here's a **google drive link**. 
You can download pre-trained models from [here](https://drive.google.com/open?id=1XUiCC_q7bkSA8uQBFgn6vexVJqaMw9tA)

## Papers & Codes

Here's the list-up for tons of GAN papers. all papers are sorted by alphabetic order.

### Start

Here's the beginning of the **GAN**.

| *Name* | *Summary* | *Paper* | *Code* |
| :---: | :---: | :---: | :---: |
| **GAN** | *Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1406.2661) | [[code]](./GAN)

### Theory & Concept

Here for the theories & concepts of the GAN.

| *Name* | *Summary* | *Paper* | *Code* |
| :---: | :---: | :---: | :---: |
| **ACGAN**        | *Auxiliary Classifier Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1610.09585) | [[code]](./ACGAN) |
| **AdaGAN**       | *Boosting Generative Models* | [[arXiv]](https://arxiv.org/abs/1701.02386) | [[~~code~~]]() |
| **BEGAN**        | *Boundary Equilibrium Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1703.10717) | [[code]](./BEGAN) |
| **BGAN**         | *Boundary-Seeking Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1702.08431) | [[code]](./BGAN) |
| **BigGAN**       | *Large Scale GAN Training for High Fidelity Natural Image Synthesis* | [[arXiv]](https://arxiv.org/abs/1809.11096) | [[~~code~~]]() |
| **CGAN**         | *Conditional Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1411.1784) | [[code]](./CGAN) |
| **CoGAN**        | *Coupled Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1606.07536) | [[code]](./CoGAN) |
| **DCGAN**        | *Deep Convolutional Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1511.06434) | [[code]](./DCGAN) |
| **DRAGAN**       | *On Convergence and Stability of Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1705.07215) | [[code]](./DRAGAN) |
| **EBGAN**        | *Energy-based Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1609.03126) | [[code]](./EBGAN) |
| **f-GAN**        | *Training Generative Neural Samplers using Variational Divergence Minimization* | [[arXiv]](https://arxiv.org/abs/1606.00709) | [[code]](./FGAN) |
| **GP-GAN**       | *Towards Realistic High-Resolution Image Blending* | [[arXiv]](https://arxiv.org/abs/1703.07195) | [[~~code~~]]() |
| **Softmax GAN**  | *Generative Adversarial Networks with Softmax* | [[arXiv]](https://arxiv.org/abs/1704.06191) | [[code]](./GAN) |
| **GAP**          | *Generative Adversarial Parallelization* | [[arXiv]](https://arxiv.org/abs/1612.04021) | [[~~code~~]]() |
| **GEGAN**        | *Generalization and Equilibrium in Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1703.00573) | [[~~code~~]]() |
| **InfoGAN**      | *Interpretable Representation Learning by Information Maximizing Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1606.03657) | [[code]](./InfoGAN) |
| **LAPGAN**       | *Laplacian Pyramid Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1506.05751) | [[code]](./LAPGAN) |
| **LSGAN**        | *Loss-Sensitive Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1701.06264) | [[code]](./LSGAN) |
| **MAGAN**        | *Margin Adaptation for Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1704.03817) | [[code]](./MAGAN) |
| **MRGAN**        | *Mode Regularized Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1612.02136) | [[code]](./MRGAN) |
| **MSGGAN**       | *Multi-Scale Gradients for Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1903.06048) | [[~~code~~]]() |
| **PGGAN**        | *Progressive Growing of GANs for Improved Quality, Stability, and Variation* | [[arXiv]](https://arxiv.org/abs/1710.10196) | [[~~code~~]]() |
| **RaGAN**        | *The relativistic discriminator: a key element missing from standard GAN* | [[arXiv]](https://arxiv.org/pdf/1807.00734v3.pdf) | [[~~code~~]]() |
| **SeAtGAN**      | *Self-Attention Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1805.08318) | [[code]](./SAGAN) |
| **SphereGAN**    | *Sphere Generative Adversarial Network Based on Geometric Moment Matching* | [[CVPR2019]](http://cau.ac.kr/~jskwon/paper/SphereGAN_CVPR2019.pdf) | [[~~code~~]]() |
| **SGAN**         | *Stacked Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1612.04357) | [[~~code~~]](https://github.com/kozistr/Awesome-GANs/blob/master/SGAN) |
| **SGAN++**       | *Realistic Image Synthesis with Stacked Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1710.10916) | [[~~code~~]](https://github.com/kozistr/Awesome-GANs/blob/master/SGAN) |
| **SinGAN**       | *Learning a Generative Model from a Single Natural Image* | [[arXiv]](https://arxiv.org/abs/1905.01164) | [[~~code~~]]() |
| **StableGAN**    | *Stabilizing Adversarial Nets With Prediction Methods* | [[arXiv]](https://arxiv.org/abs/1705.07364) | [[~~code~~]]() |
| **StyleGAN**     | *A Style-Based Generator Architecture for Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1812.04948) | [[~~code~~]]() |
| **StyleGAN V2**  | *Analyzing and Improving the Image Quality of StyleGAN* | [[arXiv]](http://arxiv.org/abs/1912.04958) | [[~~code~~]]() |
| **TripleGAN**    | *Triple Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1703.02291) | [[~~code~~]]() |
| **UGAN**         | *Unrolled Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1611.02163) | [[~~code~~]]() |
| **WGAN**         | *Wasserstein Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1701.07875) | [[code]](./WGAN) |
| **WGAN-GP**      | *Improved Training of Wasserstein Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1704.00028) | [[code]](./WGAN) |

### Applied Vision

Here for the GAN applications on Vision domain, 
like image-to-image translation, image in-painting, single image super resolution , etc.

| *Name* | *Summary* | *Paper* | *Code* |
| :---: | :---: | :---: | :---: |
| **3D GAN**       | *3D Generative Adversarial Networks* | [[MIT]](http://3dgan.csail.mit.edu/) | [[~~code~~]]() |
| **CycleGAN**     | *Unpaired img2img translation using Cycle-consistent Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1703.10593) | [[code]](./CycleGAN) |
| **DAGAN**        | *Instance-level Image Translation by Deep Attention Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1802.06454) | [[~~code~~]]() |
| **DeblurGAN**    | *Blind Motion Deblurring Using Conditional Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1711.07064) | [[~~code~~]]() |
| **DualGAN**      | *Unsupervised Dual Learning for Image-to-Image Translation* | [[arXiv]](https://arxiv.org/abs/1704.02510) | [[~~code~~]]()
| **ESRGAN**       | *Enhanced Super-Resolution Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1809.00219) | [[~~code~~]]() |
| **HiFaceGAN**    | *Face Renovation via Collaborative Suppression and Replenishment* | [[arXiv]](https://arxiv.org/abs/2005.05005v1) | [[~~code~~]]() |
| **SpAtGAN**      | *Generative Adversarial Network with Spatial Attention for Face Attribute Editing* | [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/html/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.html) | [[~~code~~]]() |
| **SalGAN**       | *Visual Saliency Prediction Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1701.01081) | [[~~code~~]]() |
| **SRGAN**        | *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network* | [[arXiv]](https://arxiv.org/abs/1609.04802) | [[code]](./SRGAN) |
| **StarGAN**      | *Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation* | [[arXiv]](https://arxiv.org/abs/1711.09020) | [[code]](./StarGAN) |
| **StarGAN V2**   | *Diverse Image Synthesis for Multiple Domains* | [[arXiv]](https://arxiv.org/abs/1912.01865) | [[~~code~~]]() |
| **TecoGAN**      | *Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation* | [[arXiv]](https://arxiv.org/abs/1811.09393) | [[~~code~~]]() |
| **TextureGAN**   | *Controlling Deep Image Synthesis with Texture Patches* | [[arXiv]](https://arxiv.org/abs/1706.02823) | [[~~code~~]]() |
| **TwinGAN**      | *Cross-Domain Translation fo Human Portraits* | [[github]](https://github.com/jerryli27/TwinGAN) | [[~~code~~]]() |
| **XGAN**         | *Unsupervised Image-to-Image Translation for Many-to-Many Mappings* | [[arXiv]](https://arxiv.org/abs/1711.05139) | [[~~code~~]]() |

### Applied Audio

Here for the GAN applications on Audio domain, 
like wave generation, wave to wave translation, etc.

| *Name* | *Summary* | *Paper* | *Code* |
| :---: | :---: | :---: | :---: |
| *AAS*                | *Adversarial Audio Synthesis* | [[arXiv]](https://arxiv.org/abs/1802.04208) | [[~~code~~]]() |
| **BeatGAN**          | *Generating Drum Loops via GANs* | [[arXiv]](https://github.com/NarainKrishnamurthy/BeatGAN2.0) | [[~~code~~]]() |
| **GANSynth**         | *Adversarial Neural Audio Synthesis* | [[arXiv]](https://arxiv.org/abs/1902.08710) | [[~~code~~]]() |
| **SEGAN**            | *Speech Enhancement Generative Adversarial Network* | [[arXiv]](https://arxiv.org/abs/1703.09452) | [[~~code~~]]() |
| **TempoGAN**         | *A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow* | [[arXiv]](https://arxiv.org/abs/1801.09710) | [[~~code~~]]() |
| **Parallel WaveGAN** | *A fast waveform generation model based on GAN with multi-resolution spectrogram* | [[arXiv]](https://arxiv.org/abs/1910.11480) | [[~~code~~]]() |
| **WaveGAN**          | *Synthesizing Audio with Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1802.04208) | [[~~code~~]]() |

## Applied Others

Here for the GAN applications on other domains, 
like nlp, tabular, etc.

| *Name* | *Summary* | *Paper* | *Code* |
| :---: | :---: | :---: | :---: |
| **AnoGAN**       | *Unsupervised Anomaly Detection with Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1703.05921) | [[~~code~~]](./AnoGAN)
| **CipherGAN**    | *Unsupervised Cipher Cracking Using Discrete GANs* | [[github]](https://arxiv.org/abs/1801.04883) | [[~~code~~]]() |
| **DiscoGAN**     | *Discover Cross-Domain Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1703.05192) | [[~~code~~]]() |
| **eCommerceGAN** | *A Generative Adversarial Network for E-commerce* | [[arXiv]](https://arxiv.org/abs/1801.03244) | [[~~code~~]]() |
| **PassGAN**      | *A Deep Learning Approach for Password Guessing* | [[arXiv]](https://arxiv.org/abs/1709.00440) | [[~~code~~]]() |
| **SeqGAN**       | *Sequence Generative Adversarial Networks with Policy Gradient* | [[arXiv]](https://arxiv.org/abs/1609.05473) | [[~~code~~]]() |
| **TAC-GAN**      | *Text Conditioned Auxiliary Classifier Generative Adversarial Network* | [[arXiv]](https://arxiv.org/abs/1703.06412.pdf) | [[~~code~~]]() |

## To-Do

1. updating `worth a try` GAN papers
2. refactoring the whole codes
3. supporting tensorflow 2.x
4. linking to the official implementations, if not, unofficial implementations

## ETC

**Any suggestions and PRs and issues are WELCOME :)**

## Author

HyeongChan Kim / [@kozistr](http://kozistr.tech)
