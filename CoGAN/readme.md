# Coupled Generative Adversarial Networks

## Loss Function

* used ``sce loss`` at D/G nets.

## Architecture Networks

* Same as CoGAN paper.

*DIFFS* | *CoGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Pooling** | ``max_pooling2d`` | ``conv2d pooling`` |
 **G net**   | ``5 fc layers``   | ``2 fc + 3 deconv2d layers`` |
 **conv2d filters** | ``D[20, 50]`` | ``D[32, 64]`` |
 **fc units** | ``D[500]`` | ``D[512]`` |

> HE Initializer parameters     : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./cogan_tb.png)

> Elapsed Time : s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 2.5k* | *Global Step 5k* | *Global Step 12.5k*
:---: | :---: | :---: | :---:
**Gen 1 (original)**      | ![img](./gen_img/train_1_00002500.png) | ![img](./gen_img/train_1_00005000.png) | ![img](./gen_img/train_1_00012500.png)
**Gen 2 (90° rotated)**    | ![img](./gen_img/train_2_00002500.png) | ![img](./gen_img/train_2_00005000.png) | ![img](./gen_img/train_2_00012500.png)

## To-Do
* 