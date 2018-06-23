# Auxiliary Classifier Generative Adversarial Networks

## Loss Function

* used ``sce_loss`` with D/G/C nets

## Architecture Networks

* Same with the AC-GAN paper.
* But, i just used hyper-parameters like weight initializer, etc...

*DIFFS* | *AC-GAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **weight initializer** | `Isotropic Gaussian` | ``HE Initializer`` |
 **z-noise size** | 110 | 128 |
 **Activation noise std** | ``0 ~ 0.2`` | ``None`` |

> Isotropic Gaussian parameters : (µ = 0, σ = 0.02) <br/>
> HE Initializer parameters     : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](https://github.com/kozistr/Awesome-GANs/blob/master/ACGAN/acgan_tb.png)

## Result

*Name* | *Global Step 50k* | *Global Step 100k* | *Global Step 200k*
:---: | :---: | :---: | :---:
**ACGAN**     | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/ACGAN/gen_img/train_00050000.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/ACGAN/gen_img/train_00100000.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/ACGAN/gen_img/train_00200000.png)

## To-Do
* 