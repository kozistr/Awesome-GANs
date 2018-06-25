# Deep Convolutional Generative Adversarial Networks

## Loss Function

* used ``sce loss`` at D/G nets.

## Architecture Networks

* Same as DCGAN paper.

*DIFFS* | *DCGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |
 **z noise** | ``100`` | ``128`` |

> Normal Distribution Initializer : (µ = 0, σ = 0.02) <br/>
> HE Initializer parameters       : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./dcgan_tb.png)

> Elapsed Time : s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 5k* | *Global Step 10k* | *Global Step 25k*
:---: | :---: | :---: | :---:
**DCGAN**      | ![img](./gen_img/train_00025000.png) | ![img](./gen_img/train_00050000.png) | ![img](./gen_img/train_00100000.png)

## To-Do
* 