# Awesome-GANs with Tensorflow

Tensorflow implementation of GANs (**Generative Adversarial Networks**)

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/kozistr/Awesome-GANs.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/Awesome-GANs/context:python)

## **WIP** : This repo is about to be refactored & supporting `tf 2.x`.

Maybe some codes wouldn't work on master branch, because i'm just working on the branch.

## Environments

Because of the image and model size, (especially **BEGAN**, **SRGAN**, **StarGAN**, ... using high resolution images as input),
if you want to train them comfortably, you need a GPU which has more than `8GB`.

But, of course, the most of the implementations use `MNIST` or `CIFAR-10, 100` DataSets.
Meaning that we can handle it with EVEN lower spec GPU than 'The Preferred' :).

## Usage

Now on **refactoring**... All GAN training script can be run module-wisely like below. (**WIP**)

### Install dependencies

You can also use *conda*, *virtualenv* environments.

```shell script
$ python3 -m pip install -r requirements.txt
```

### Train GANs

Before running the model, make sure that 

1. downloading the dataset like *CelebA*, *MNIST*, etc what you want
2. In `awesome_gans/config.py`, there are several configurations, customize with your flavor!
3. running the model like below

```shell script
$ python3 -m awesome_gans.acgan
```

## DataSets

Supporting datasets are ... (code is in `/awesome_gans/datasets.py`)

* MNIST / ~~Fashion MNIST~~
* CIFAR10 / 100
* CelebA/CelebA-HQ
* Pix2Pix
* DIV2K
* (more DataSets will be added soon!)

## Repo Tree

```
│
├── awesome_gans (source codes & eplainations & results & models) 
│        │
│        ├── acgan
│        │    ├──gen_img (generated images)
│        │    │     ├── train_xxx.png
│        │    │     └── train_xxx.png
│        │    ├── model  (pre-trained model file)
│        │    │     └── model.txt (google-drive link)
│        │    ├── __init__.py
│        │    ├── __main__.py
│        │    ├── model.py (gan model)
│        │    ├── train.py (gan trainer)
│        │    ├── gan_tb.png   (tensorboard loss plot)
│        │    └── readme.md    (results & explainations)
│        ├── config.py         (configurations)
│        ├── modules.py        (networks & operations)
│        ├── utils.py          (auxiliary utils)
│        ├── image_utils.py    (image processing)
│        └── datasets.py       (dataset loader)
├── CONTRIBUTING.md
├── Makefile   (for linting the codes)
├── LICENSE
├── README.md  (Usage & GAN paper list-up)
└── requirements.txt
```

## Pre-Trained Models

Here's a **google drive link**. 
You can download pre-trained models from [here](https://drive.google.com/open?id=1XUiCC_q7bkSA8uQBFgn6vexVJqaMw9tA)

## Papers & Codes

Here's the list-up for tons of GAN papers. all papers are sorted by alphabetic order.

### Start

Here's the beginning of the **GAN**.

| *Name*  |             *Summary*             |                  *Paper*                   |            *Code*            |
|:-------:|:---------------------------------:|:------------------------------------------:|:----------------------------:|
| **GAN** | *Generative Adversarial Networks* | [[arXiv]](https://arxiv.org/abs/1406.2661) | [[code]](./awesome_gans/GAN) |

### Theory & Concept

Here for the theories & concepts of the GAN.

|      *Name*       |                                                         *Summary*                                 |                                                                               *Paper*                                                                           |                                   *Code*                               |                            *Official Code*                           |
|:-----------------:|:-------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|     **ACGAN**     |                      *Auxiliary Classifier Generative Adversarial Networks*                       |                                                           [[arXiv]](https://arxiv.org/abs/1610.09585)                                                           |                     [[code]](./awesome_gans/ACGAN)                     |
|    **AdaGAN**     |                                   *Boosting Generative Models*                                    |                                                           [[arXiv]](https://arxiv.org/abs/1701.02386)                                                           |                             [[~~code~~]]()                             |
|      **bCR**      |                          *Improved Consistency Regularization for GANs*                           |                                                           [[arXiv]](https://arxiv.org/abs/2002.04724)                                                           |                             [[~~code~~]]()                             |
|     **BEGAN**     |                      *Boundary Equilibrium Generative Adversarial Networks*                       |                                                           [[arXiv]](https://arxiv.org/abs/1703.10717)                                                           |                     [[code]](./awesome_gans/BEGAN)                     |
|     **BGAN**      |                        *Boundary-Seeking Generative Adversarial Networks*                         |                                                           [[arXiv]](https://arxiv.org/abs/1702.08431)                                                           |                     [[code]](./awesome_gans/BGAN)                      |
|    **BigGAN**     |               *Large Scale GAN Training for High Fidelity Natural Image Synthesis*                |                                                           [[arXiv]](https://arxiv.org/abs/1809.11096)                                                           |                             [[~~code~~]]()                             |
|     **CGAN**      |                           *Conditional Generative Adversarial Networks*                           |                                                           [[arXiv]](https://arxiv.org/abs/1411.1784)                                                            |                     [[code]](./awesome_gans/CGAN)                      |
|     **CoGAN**     |                             *Coupled Generative Adversarial Networks*                             |                                                           [[arXiv]](https://arxiv.org/abs/1606.07536)                                                           |                     [[code]](./awesome_gans/CoGAN)                     |
|   **ConSinGAN**   |                       *Improved Techniques for Training Single-Image GANs*                        |          [[WACV21]](https://openaccess.thecvf.com/content/WACV2021/papers/Hinz_Improved_Techniques_for_Training_Single-Image_GANs_WACV_2021_paper.pdf)          |                             [[~~code~~]]()                             |          [[official]](https://github.com/tohinz/ConSinGAN)           |
|     **DCGAN**     |                       *Deep Convolutional Generative Adversarial Networks*                        |                                                           [[arXiv]](https://arxiv.org/abs/1511.06434)                                                           |                     [[code]](./awesome_gans/DCGAN)                     |
|    **DRAGAN**     |                 *On Convergence and Stability of Generative Adversarial Networks*                 |                                                           [[arXiv]](https://arxiv.org/abs/1705.07215)                                                           |                    [[code]](./awesome_gans/DRAGAN)                     |
|     **EBGAN**     |                          *Energy-based Generative Adversarial Networks*                           |                                                           [[arXiv]](https://arxiv.org/abs/1609.03126)                                                           |                     [[code]](./awesome_gans/EBGAN)                     |
|     **f-GAN**     |          *Training Generative Neural Samplers using Variational Divergence Minimization*          |                                                           [[arXiv]](https://arxiv.org/abs/1606.00709)                                                           |                     [[code]](./awesome_gans/FGAN)                      |
|    **GP-GAN**     |                        *Towards Realistic High-Resolution Image Blending*                         |                                                           [[arXiv]](https://arxiv.org/abs/1703.07195)                                                           |                             [[~~code~~]]()                             |
|  **Softmax GAN**  |                          *Generative Adversarial Networks with Softmax*                           |                                                           [[arXiv]](https://arxiv.org/abs/1704.06191)                                                           |                      [[code]](./awesome_gans/GAN)                      |
|      **GAP**      |                             *Generative Adversarial Parallelization*                              |                                                           [[arXiv]](https://arxiv.org/abs/1612.04021)                                                           |                             [[~~code~~]]()                             |
|     **GEGAN**     |                *Generalization and Equilibrium in Generative Adversarial Networks*                |                                                           [[arXiv]](https://arxiv.org/abs/1703.00573)                                                           |                             [[~~code~~]]()                             |
|     **G-GAN**     |                                          *Geometric GAN*                                          |                                                           [[arXiv]](https://arxiv.org/abs/1705.02894)                                                           |                             [[~~code~~]]()                             | 
|    **InfoGAN**    | *Interpretable Representation Learning by Information Maximizing Generative Adversarial Networks* |                                                           [[arXiv]](https://arxiv.org/abs/1606.03657)                                                           |                    [[code]](./awesome_gans/InfoGAN)                    |
|    **LAPGAN**     |                        *Laplacian Pyramid Generative Adversarial Networks*                        |                                                           [[arXiv]](https://arxiv.org/abs/1506.05751)                                                           |                    [[code]](./awesome_gans/LAPGAN)                     |
|     **LSGAN**     |                         *Loss-Sensitive Generative Adversarial Networks*                          |                                                           [[arXiv]](https://arxiv.org/abs/1701.06264)                                                           |                     [[code]](./awesome_gans/LSGAN)                     |
|     **MAGAN**     |                      *Margin Adaptation for Generative Adversarial Networks*                      |                                                           [[arXiv]](https://arxiv.org/abs/1704.03817)                                                           |                     [[code]](./awesome_gans/MAGAN)                     |
|     **MRGAN**     |                        *Mode Regularized Generative Adversarial Networks*                         |                                                           [[arXiv]](https://arxiv.org/abs/1612.02136)                                                           |                     [[code]](./awesome_gans/MRGAN)                     |
|    **MSGGAN**     |                    *Multi-Scale Gradients for Generative Adversarial Networks*                    |                                                           [[arXiv]](https://arxiv.org/abs/1903.06048)                                                           |                             [[~~code~~]]()                             |
|     **PGGAN**     |           *Progressive Growing of GANs for Improved Quality, Stability, and Variation*            |                                                           [[arXiv]](https://arxiv.org/abs/1710.10196)                                                           |                             [[~~code~~]]()                             | [[official]](https://github.com/tkarras/progressive_growing_of_gans) |
|     **RaGAN**     |             *The relativistic discriminator: a key element missing from standard GAN*             |                                                        [[arXiv]](https://arxiv.org/pdf/1807.00734v3.pdf)                                                        |                             [[~~code~~]]()                             |
|    **SeAtGAN**    |                         *Self-Attention Generative Adversarial Networks*                          |                                                           [[arXiv]](https://arxiv.org/abs/1805.08318)                                                           |                     [[code]](./awesome_gans/SAGAN)                     |
|   **SphereGAN**   |            *Sphere Generative Adversarial Network Based on Geometric Moment Matching*             |                                               [[CVPR2019]](http://cau.ac.kr/~jskwon/paper/SphereGAN_CVPR2019.pdf)                                               |                             [[~~code~~]]()                             |
|     **SGAN**      |                             *Stacked Generative Adversarial Networks*                             |                                                           [[arXiv]](https://arxiv.org/abs/1612.04357)                                                           | [[~~code~~]](https://github.com/kozistr/Awesome-GANs/blob/master/SGAN) |
|    **SGAN++**     |             *Realistic Image Synthesis with Stacked Generative Adversarial Networks*              |                                                           [[arXiv]](https://arxiv.org/abs/1710.10916)                                                           | [[~~code~~]](https://github.com/kozistr/Awesome-GANs/blob/master/SGAN) |
|    **SinGAN**     |                     *Learning a Generative Model from a Single Natural Image*                     |                                                           [[arXiv]](https://arxiv.org/abs/1905.01164)                                                           |                             [[~~code~~]]()                             |           [[official]](https://github.com/tamarott/SinGAN)           |
|   **StableGAN**   |                      *Stabilizing Adversarial Nets With Prediction Methods*                       |                                                           [[arXiv]](https://arxiv.org/abs/1705.07364)                                                           |                             [[~~code~~]]()                             |
|   **StyleCLIP**   |                          *Text-Driven Manipulation of StyleGAN Imagery*                           |                                                           [[arXiv]](https://arxiv.org/abs/2103.17249)                                                           |                             [[~~code~~]]()                             |        [[official]](https://github.com/orpatashnik/StyleCLIP)        | 
|   **StyleGAN**    |            *A Style-Based Generator Architecture for Generative Adversarial Networks*             |                                                           [[arXiv]](https://arxiv.org/abs/1812.04948)                                                           |                             [[~~code~~]]()                             |           [[official]](https://github.com/NVlabs/stylegan)           |
|   **StyleGAN2**   |                      *Analyzing and Improving the Image Quality of StyleGAN*                      |                                                           [[arXiv]](http://arxiv.org/abs/1912.04958)                                                            |                             [[~~code~~]]()                             |          [[official]](https://github.com/NVlabs/stylegan2)           |
| **StyleGAN2 ADA** |                       *StyleGAN2 with adaptive discriminator augmentation*                        |                                                           [[arXiv]](https://arxiv.org/abs/2006.06676)                                                           |                             [[~~code~~]]()                             |        [[official]](https://github.com/NVlabs/stylegan2-ada)         |
|   **StyleGAN3**   |                           *Alias-Free Generative Adversarial Networks*                            |                                                           [[arXiv]](https://arxiv.org/abs/2106.12423)                                                           |                             [[~~code~~]]()                             |          [[official]](https://github.com/NVlabs/stylegan3)           |
|  **StyleGAN-XL**  |                           *Scaling StyleGAN to Large Diverse Datasets*                            |                                                           [[arXiv]](https://arxiv.org/abs/2202.00273)                                                           |                             [[~~code~~]]()                             |    [[official]](https://github.com/autonomousvision/stylegan_xl)     |
|   **TripleGAN**   |                             *Triple Generative Adversarial Networks*                              |                                                           [[arXiv]](https://arxiv.org/abs/1703.02291)                                                           |                             [[~~code~~]]()                             |
|     **UGAN**      |                            *Unrolled Generative Adversarial Networks*                             |                                                           [[arXiv]](https://arxiv.org/abs/1611.02163)                                                           |                             [[~~code~~]]()                             |
|   **U-Net GAN**   |                 *A U-Net Based Discriminator for Generative Adversarial Networks*                 | [[CVPR20]](https://openaccess.thecvf.com/content_CVPR_2020/html/Schonfeld_A_U-Net_Based_Discriminator_for_Generative_Adversarial_Networks_CVPR_2020_paper.html) |                             [[~~code~~]]()                             |        [[official]](https://github.com/boschresearch/unetgan)        | 
|     **WGAN**      |                           *Wasserstein Generative Adversarial Networks*                           |                                                           [[arXiv]](https://arxiv.org/abs/1701.07875)                                                           |                     [[code]](./awesome_gans/WGAN)                      |
|    **WGAN-GP**    |                *Improved Training of Wasserstein Generative Adversarial Networks*                 |                                                           [[arXiv]](https://arxiv.org/abs/1704.00028)                                                           |                     [[code]](./awesome_gans/WGAN)                      |

### Applied Vision

Here for the GAN applications on Vision domain, 
like image-to-image translation, image in-painting, single image super resolution , etc.

|     *Name*      |                                                 *Summary*                                     |                                                                              *Paper*                                                                            |                *Code*             |                              *Official Code*                    |
|:---------------:|:---------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------:|:---------------------------------------------------------------:|
|   **3D GAN**    |                             *3D Generative Adversarial Networks*                              |                                                              [[MIT]](http://3dgan.csail.mit.edu/)                                                               |          [[~~code~~]]()           |
| **AnycostGAN**  |                  *Anycost GANs for Interactive Image Synthesis and Editing*                   |                                                           [[arXiv]](https://arxiv.org/abs/2103.03243)                                                           |          [[~~code~~]]()           |    [[official]](https://github.com/mit-han-lab/anycost-gan)     |
|  **CycleGAN**   |          *Unpaired img2img translation using Cycle-consistent Adversarial Networks*           |                                                           [[arXiv]](https://arxiv.org/abs/1703.10593)                                                           | [[code]](./awesome_gans/CycleGAN) |
|    **DAGAN**    |     *Instance-level Image Translation by Deep Attention Generative Adversarial Networks*      |                                                           [[arXiv]](https://arxiv.org/abs/1802.06454)                                                           |          [[~~code~~]]()           |
|  **DeblurGAN**  |               *Blind Motion Deblurring Using Conditional Adversarial Networks*                |                                                           [[arXiv]](https://arxiv.org/abs/1711.07064)                                                           |          [[~~code~~]]()           |
|   **DualGAN**   |                  *Unsupervised Dual Learning for Image-to-Image Translation*                  |                                                           [[arXiv]](https://arxiv.org/abs/1704.02510)                                                           |          [[~~code~~]]()           |
|   **DRIT/++**   |             *Diverse Image-to-Image Translation via Disentangled Representations*             |                                                           [[arXiv]](https://arxiv.org/abs/1905.01270)                                                           |          [[~~code~~]]()           |        [[official]](https://github.com/HsinYingLee/DRIT)        |
| **EdgeConnect** |                 *Generative Image Inpainting with Adversarial Edge Learning*                  |                                                           [[arXiv]](https://arxiv.org/abs/1901.00212)                                                           |          [[~~code~~]]()           |      [[official]](https://github.com/knazeri/edge-connect)      |
|   **ESRGAN**    |                  *Enhanced Super-Resolution Generative Adversarial Networks*                  |                                                           [[arXiv]](https://arxiv.org/abs/1809.00219)                                                           |          [[~~code~~]]()           |
|   **FastGAN**   |    *Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis*    |                                                           [[arXiv]](https://arxiv.org/abs/2101.04775)                                                           |          [[~~code~~]]()           |  [[official]](https://github.com/odegeasslbc/FastGAN-pytorch)   |
|    **FUNIT**    |                      *Few-Shot Unsupervised Image-to-Image Translation*                       |                                                           [[arXiv]](https://arxiv.org/abs/1905.01723)                                                           |          [[~~code~~]]()           |          [[official]](https://github.com/NVlabs/FUNIT)          |
|   **CA & GA**   |           *Generative Image Inpainting w/ Contextual Attention & Gated Convolution*           |                                 [[CVPR2018]](https://arxiv.org/abs/1801.07892), [[ICCV2019]](https://arxiv.org/abs/1806.03589)                                  |          [[~~code~~]]()           | [[official]](https://github.com/JiahuiYu/generative_inpainting) |
|  **HiFaceGAN**  |               *Face Renovation via Collaborative Suppression and Replenishment*               |                                                          [[arXiv]](https://arxiv.org/abs/2005.05005v1)                                                          |          [[~~code~~]]()           |
|    **MUNIT**    |                     *Multimodal Unsupervised Image-to-Image Translation*                      |                                                           [[arXiv]](https://arxiv.org/abs/1804.04732)                                                           |          [[~~code~~]]()           |          [[official]](https://github.com/NVlabs/MUNIT)          |
|  **NICE-GAN**   |                             *Reusing Discriminators for Encoding*                             |                                                           [[arXiv]](https://arxiv.org/abs/2003.00273)                                                           |          [[~~code~~]]()           |    [[official]](https://github.com/alpc91/NICE-GAN-pytorch)     |
|    **PSGAN**    |        *Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer*        |                                                           [[arXiv]](https://arxiv.org/abs/1909.06956)                                                           |          [[~~code~~]]()           |        [[official]](https://github.com/wtjiang98/PSGAN)         |
|   **SpAtGAN**   |      *Generative Adversarial Network with Spatial Attention for Face Attribute Editing*       |                [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/html/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.html)                 |          [[~~code~~]]()           |
|   **SalGAN**    |                 *Visual Saliency Prediction Generative Adversarial Networks*                  |                                                           [[arXiv]](https://arxiv.org/abs/1701.01081)                                                           |          [[~~code~~]]()           |
|   **SRFlow**    |                           *Super-Resolution using Normalizing Flow*                           |                                                           [[arXiv]](https://arxiv.org/abs/2006.14200)                                                           |          [[~~code~~]]()           |       [[official]](https://github.com/andreas128/SRFlow)        |
|    **SRGAN**    |    *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*     |                                                           [[arXiv]](https://arxiv.org/abs/1609.04802)                                                           |  [[code]](./awesome_gans/SRGAN)   |
|  **SRResCGAN**  | *Deep Generative Adversarial Residual Convolutional Networks for Real-World Super-Resolution* |                                                           [[arXiv]](https://arxiv.org/abs/2005.00953)                                                           |          [[~~code~~]]()           |       [[official]](https://github.com/RaoUmer/SRResCGAN)        |
|   **StarGAN**   |     *Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation*     |                                                           [[arXiv]](https://arxiv.org/abs/1711.09020)                                                           | [[code]](./awesome_gans/StarGAN)  |         [[official]](https://github.com/yunjey/stargan)         |
| **StarGAN V2**  |                        *Diverse Image Synthesis for Multiple Domains*                         |                                                           [[arXiv]](https://arxiv.org/abs/1912.01865)                                                           |          [[~~code~~]]()           |       [[official]](https://github.com/clovaai/stargan-v2)       |
| **StyleGAN-V**  |      *A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2*      |                                         [[arXiv]](https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf)                                          |          [[~~code~~]]()           |     [[official]](https://github.com/universome/stylegan-v)      |
|   **TecoGAN**   |       *Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation*       |                                                           [[arXiv]](https://arxiv.org/abs/1811.09393)                                                           |          [[~~code~~]]()           |         [[official]](https://github.com/thunil/TecoGAN)         |
| **TextureGAN**  |                    *Controlling Deep Image Synthesis with Texture Patches*                    |                                                           [[arXiv]](https://arxiv.org/abs/1706.02823)                                                           |          [[~~code~~]]()           |
|    **TUNIT**    |                *Rethinking the Truly Unsupervised Image-to-Image Translation*                 |                                                           [[arXiv]](https://arxiv.org/abs/2006.06500)                                                           |          [[~~code~~]]()           |         [[official]](https://github.com/clovaai/tunit)          |
|   **TwinGAN**   |                         *Cross-Domain Translation fo Human Portraits*                         |                                                        [[github]](https://github.com/jerryli27/TwinGAN)                                                         |          [[~~code~~]]()           |
|    **UNIT**     |                      *Unsupervised Image-to-Image Translation Networks*                       |                                                           [[arXiv]](https://arxiv.org/abs/1703.00848)                                                           |          [[~~code~~]]()           |        [[official]](https://github.com/mingyuliutw/UNIT)        |
|    **XGAN**     |              *Unsupervised Image-to-Image Translation for Many-to-Many Mappings*              |                                                           [[arXiv]](https://arxiv.org/abs/1711.05139)                                                           |          [[~~code~~]]()           |
|  **Zero-DCE**   |            *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*             | [[CVPR20]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf) |          [[~~code~~]]()           |      [[official]](https://github.com/Li-Chongyi/Zero-DCE)       |

### Applied Audio

Here for the GAN applications on Audio domain, 
like wave generation, wave to wave translation, etc.

|        *Name*        |                                       *Summary*                                        |                           *Paper*                            |     *Code*     | *Official Code* |
|:--------------------:|:--------------------------------------------------------------------------------------:|:------------------------------------------------------------:|:--------------:| :---: |
|       **AAS**        |                             *Adversarial Audio Synthesis*                              |           [[arXiv]](https://arxiv.org/abs/1802.04208)        | [[~~code~~]]() | |
|     **BeatGAN**      |                            *Generating Drum Loops via GANs*                            | [[arXiv]](https://github.com/NarainKrishnamurthy/BeatGAN2.0) | [[~~code~~]]() | |
|     **GANSynth**     |                          *Adversarial Neural Audio Synthesis*                          |         [[arXiv]](https://arxiv.org/abs/1902.08710)          | [[~~code~~]]() | |
|     **MuseGAN**      |     *Multi-track Sequential GANs for Symbolic Music Generation and Accompaniment*      |         [[arXiv]](https://arxiv.org/abs/1709.06298)          | [[~~code~~]]() | |
|      **SEGAN**       |                  *Speech Enhancement Generative Adversarial Network*                   |         [[arXiv]](https://arxiv.org/abs/1703.09452)          | [[~~code~~]]() | |
|    **StarGAN-VC**    | *Non-parallel many-to-many voice conversion with star generative adversarial networks* |         [[arXiv]](https://arxiv.org/abs/1806.02169)          | [[~~code~~]]() | |
|     **TempoGAN**     |        *A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow*         |         [[arXiv]](https://arxiv.org/abs/1801.09710)          | [[~~code~~]]() | |
| **Parallel WaveGAN** |   *A fast waveform generation model based on GAN with multi-resolution spectrogram*    |         [[arXiv]](https://arxiv.org/abs/1910.11480)          | [[~~code~~]]() | |
|     **WaveGAN**      |               *Synthesizing Audio with Generative Adversarial Networks*                |         [[arXiv]](https://arxiv.org/abs/1802.04208)          | [[~~code~~]]() | |

## Applied Others

Here for the GAN applications on other domains, 
like nlp, tabular, etc.

|      *Name*      |                                 *Summary*                              |                     *Paper*                     |         *Code*         | *Official Code* |
|:----------------:|:----------------------------------------------------------------------:|:-----------------------------------------------:|:----------------------:|:---------------:|
|   **AnoGAN**     | *Unsupervised Anomaly Detection with Generative Adversarial Networks*  |   [[arXiv]](https://arxiv.org/abs/1703.05921)   | [[~~code~~]](./AnoGAN) |
|  **CipherGAN**   |           *Unsupervised Cipher Cracking Using Discrete GANs*           |  [[github]](https://arxiv.org/abs/1801.04883)   |     [[~~code~~]]()     |
|   **DiscoGAN**   |        *Discover Cross-Domain Generative Adversarial Networks*         |   [[arXiv]](https://arxiv.org/abs/1703.05192)   |     [[~~code~~]]()     |
| **eCommerceGAN** |           *A Generative Adversarial Network for E-commerce*            |   [[arXiv]](https://arxiv.org/abs/1801.03244)   |     [[~~code~~]]()     |
|   **PassGAN**    |            *A Deep Learning Approach for Password Guessing*            |   [[arXiv]](https://arxiv.org/abs/1709.00440)   |     [[~~code~~]]()     |
|    **SeqGAN**    |    *Sequence Generative Adversarial Networks with Policy Gradient*     |   [[arXiv]](https://arxiv.org/abs/1609.05473)   |     [[~~code~~]]()     |
|   **TAC-GAN**    | *Text Conditioned Auxiliary Classifier Generative Adversarial Network* | [[arXiv]](https://arxiv.org/abs/1703.06412.pdf) |     [[~~code~~]]()     |

## Useful Resources

Here for the useful resources when you try to train and stable a gan model.

|  *Name*   |                  *Summary*                   |                    *Link*                     |
|:---------:|:--------------------------------------------:|:---------------------------------------------:|
| GAN Hacks | a bunch of tips & tricks to train GAN stable | [github](https://github.com/soumith/ganhacks) |

## Note

Any suggestions and PRs and issues are WELCOME :)

## Author

HyeongChan Kim / [@kozistr](http://kozistr.tech)
