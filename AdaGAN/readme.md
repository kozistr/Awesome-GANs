# Boosting Generative Adversarial Models

## Loss Function

* used ``GAN loss`` at D/G nets.

## Architecture Networks

* Same with the AdaGAN paper.

*DIFFS* | *AdaGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **image size** | ``128`` | ``64`` |
 **loss** | ``hinge loss`` | ``GAN loss`` |

## Tensorboard Result

![result](./adagan_tb.png)

> Elapsed Time : s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 10k* | *Global Step 25k* | *Global Step 50k*
:---: | :---: | :---: | :---:
**AdaGAN**     | ![img](./gen_img/train_00010000.png) | ![img](./gen_img/train_00025000.png) | ![img](./gen_img/train_00050000.png)

## To-Do
* 
