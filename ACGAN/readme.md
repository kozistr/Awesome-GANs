# Auxiliary Classifier Generative Adversarial Networks

## Loss Function

* used ``sce_loss`` with D/G nets
* used ``softce_loss`` with C nets

## Architecture Networks

* Same with the AC-GAN paper.
* But, i just used hyper-parameters like weight initializer, etc...

*DIFFS* | *AC-GAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **weight initializer** | `Isotropic Gaussian` | ``HE Initializer`` |
 **Activation noise std** | ``0 ~ 0.2`` | ``None`` |

> Isotropic Gaussian parameters : (µ = 0, σ = 0.02) <br/>
> HE Initializer parameters     : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./acgan_tb.png)

## Result

*Name* | *Global Step 10k* | *Global Step 25k* | *Global Step 50k*
:---: | :---: | :---: | :---:
**ACGAN**     | ![img](./gen_img/train_00010000.png) | ![img](./gen_img/train_00025000.png) | ![img](./gen_img/train_00050000.png)

## To-Do
* 