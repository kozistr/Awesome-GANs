# Margin Adaptation Generative Adversarial Networks

## Loss Function

* used ``mse loss`` at D/G nets.

## Architecture Networks

* Same as MAGAN paper.

*DIFFS* | *MAGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |
 **z noise (MNIST)** | ``50`` | `` `` |
 **z noise (cifar-10)** | ``320`` | `` `` |
  
> HE Initializer parameters       : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./magan_tb.png)

> Elapsed Time : 5h 45m 51s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 25k* | *Global Step 50k* | *Global Step 75k*
:---: | :---: | :---: | :---:
**MAGAN**      | ![img](./gen_img/train_00025000.png) | ![img](./gen_img/train_00050000.png) | ![img](./gen_img/train_00075000.png)

> Initial pre-trained margin value : about 3.0585415484215974

## To-Do
* 