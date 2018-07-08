# Mode Regularized Generative Adversarial Networks

## Loss Function

* used `` loss`` at D/G nets.

## Architecture Networks

* Same as MRGAN paper.

*DIFFS* | *MRGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |
  
> HE Initializer parameters       : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./mrgan_tb.png)

> Elapsed Time : 5h 45m 51s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 25k* | *Global Step 50k* | *Global Step 75k*
:---: | :---: | :---: | :---:
**MRGAN**      | ![img](./gen_img/train_00025000.png) | ![img](./gen_img/train_00050000.png) | ![img](./gen_img/train_00075000.png)

## To-Do
* 