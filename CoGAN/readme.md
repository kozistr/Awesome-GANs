# Coupled Generative Adversarial Networks

## Loss Function

* used ``sce loss`` at D/G nets.

## Architecture Networks

* Similar as CoGAN paper.

*DIFFS* | *CoGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Pooling** | ``max_pooling2d`` | ``conv2d pooling`` |
 
> Learning Rate : 1e-1 ~ to 1e-6, factor = 1 + 4e-5 <br/>
> HE Initializer parameters     : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./cogan_tb.png)

> Elapsed Time : s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 50k* | *Global Step 100k* | *Global Step 200k*
:---: | :---: | :---: | :---:
**Gen 1**      | ![img](./gen_img/train_1_00050000.png) | ![img](./gen_img/train_1_00100000.png) | ![img](./gen_img/train_1_00200000.png)
**Gen 2**      | ![img](./gen_img/train_2_00050000.png) | ![img](./gen_img/train_2_00100000.png) | ![img](./gen_img/train_2_00200000.png)

## To-Do
* 