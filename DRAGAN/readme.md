# On Convergence and Stability of Generative Adversarial Networks

## Loss Function

* using ```sce loss``` instead of ```adv loss``` on ```D/G net``` with ```gradient panelty```.

## Architecture Networks

* Same as DrAGAN paper.

*DIFFS* | *DRAGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``xavier initializer`` | ``HE initializer`` |
 **z noise** | ``100`` | ``128`` |
 **image scale** | ``-1,1`` | ``0,1`` |
 
> HE Initializer parameters       : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./dragan_tb.png)

> Elapsed Time : s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 5k* | *Global Step 10k* | *Global Step 25k*
:---: | :---: | :---: | :---:
**DCGAN**      | ![img](./gen_img/train_8000.png) | ![img](./gen_img/train_16000.png) | ![img](./gen_img/train_40000.png)

## To-Do
* 