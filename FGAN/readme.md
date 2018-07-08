# Training Generative Neural Samplers using Variational Divergence Minimization

## Loss Function

* used ``f-divergences familiy`` at D/G nets.

## Architecture Networks

* Same as f-GAN paper.

*DIFFS* | *f-GAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |
 **z dim** | ``100`` | ``128`` |
 
> Normal Distribution Initializer : (µ = 0, σ = 0.91) <br/>
> HE Initializer parameters       : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./fgan_tb.png)

> Elapsed Time : h m s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 5k* | *Global Step 10k* | *Global Step 20k*
:---: | :---: | :---: | :---:
**f-GAN**      | ![img](./gen_img/train_8000.png) | ![img](./gen_img/train_16000.png) | ![img](./gen_img/train_32000.png)

## To-Do
* Add f-divergences
  * KL
  * Reverse-KL
  * JS
  * Squared-Hellinger
  * Pearson χ^2
  * Neyman χ^2
  * Jeffrey