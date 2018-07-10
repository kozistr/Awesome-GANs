# Mode Regularized Generative Adversarial Networks

## Loss Function

* used ``GAN + (Enc) loss`` at D/G nets.
* used ``Adv + MSE loss`` at Encoder.

## Architecture Networks

* Same as MRGAN paper.

*DIFFS* | *MRGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |
 **z dim** | ``100`` | ``128`` |
  
> HE Initializer parameters       : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./mrgan_tb.png)

> Elapsed Time : 2h 18m s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 4k* | *Global Step 8k* | *Global Step 16k*
:---: | :---: | :---: | :---:
**MRGAN**      | ![img](./gen_img/train_4000.png) | ![img](./gen_img/train_8000.png) | ![img](./gen_img/train_16000.png)

## To-Do
* 