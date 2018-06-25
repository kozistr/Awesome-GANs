# Coupled Generative Adversarial Networks

## Loss Function

* used ``sce loss`` at D/G nets.

## Architecture Networks

* Same as CoGAN paper.

*DIFFS* | *CoGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Pooling** | ``max_pooling2d`` | ``conv2d pooling`` |
 **G net**   | ``5 fc layers``   | ``3 fc layers`` |
 **conv2d filters** | ``D[20, 50]`` | ``D[32, 64]`` |
 **fc units** | ``D[500]`` | ``D[512]`` |

> HE Initializer parameters     : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./cogan_tb.png)

> Elapsed Time : s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 25k* | *Global Step 50k* | *Global Step 100k*
:---: | :---: | :---: | :---:
**Gen 1**      | ![img](./gen_img/train_1_00025000.png) | ![img](./gen_img/train_1_00050000.png) | ![img](./gen_img/train_1_00100000.png)
**Gen 2**      | ![img](./gen_img/train_2_00025000.png) | ![img](./gen_img/train_2_00050000.png) | ![img](./gen_img/train_2_00100000.png)

## To-Do
* 