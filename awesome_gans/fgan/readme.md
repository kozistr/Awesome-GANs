# Training Generative Neural Samplers using Variational Divergence Minimization

## Loss Function

* used ``f-divergences familiy`` at D/G nets.

## Architecture Networks

* Same as f-GAN paper.

*DIFFS* | *f-GAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |
 **z dim** | ``100`` | ``128`` |
 **fc unit** | ``D/G[240,1200]`` | ``D/G[256,1024]`` |
 **img scaling** | ``-1,1`` | ``0,1`` |
 
> Normal Distribution Initializer : (µ = 0, σ = D/G(0.005,0.05)) <br/>
> HE Initializer parameters       : (factor = 1, FAN_AVG, uniform)

## Tensorboard

*Name* | *graph*
:---: | :---:
**GAN** | ![result](./tb/gan_tb.png)
**KL** | ![result](./tb/kl_tb.png)
**Reverse-KL** | ![result](./tb/reverse-kl_tb.png)
**JS** | ![result](./tb/js_tb.png)
**JS-Weighted** | ![result](./tb/js-weighted_tb.png)
**Squared-hellinger** | ![result](./tb/squared-hellinger_tb.png)
**Pearson** | ![result](./tb/pearson_tb.png)
**Neyman** | ![result](./tb/neyman_tb.png)
**Jeffrey** | ![result](./tb/jeffrey_tb.png)
**Total-Variation** | ![result](./tb/total-variation_tb.png)

> Elapsed time : about 20m with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 5k* | *Global Step 10k* | *Global Step 20k*
:---: | :---: | :---: | :---:
**f-GAN (GAN)**                | ![img](./gen_img/GAN/train_5000.png) | ![img](./gen_img/GAN/train_10000.png) | ![img](./gen_img/GAN/train_20000.png)
**f-GAN (KL)**                 | ![img](./gen_img/KL/train_5000.png) | ![img](./gen_img/KL/train_10000.png) | ![img](./gen_img/KL/train_20000.png)
**f-GAN (Reverse-KL)**         | ![img](./gen_img/Reverse-KL/train_5000.png) | ![img](./gen_img/Reverse-KL/train_10000.png) | ![img](./gen_img/Reverse-KL/train_20000.png)
**f-GAN (JS)**                 | ![img](./gen_img/JS/train_5000.png) | ![img](./gen_img/JS/train_10000.png) | ![img](./gen_img/JS/train_20000.png)
**f-GAN (JS-Weighted)**        | ![img](./gen_img/JS-Weighted/train_5000.png) | ![img](./gen_img/JS-Weighted/train_10000.png) | ![img](./gen_img/JS-Weighted/train_20000.png)
**f-GAN (Squared-Hellinger)**  | ![img](./gen_img/Squared-Hellinger/train_5000.png) | ![img](./gen_img/Squared-Hellinger/train_10000.png) | ![img](./gen_img/Squared-Hellinger/train_20000.png)
**f-GAN (Pearson)**            | ![img](./gen_img/Pearson/train_5000.png) | ![img](./gen_img/Pearson/train_10000.png) | ![img](./gen_img/Pearson/train_20000.png)
**f-GAN (Neyman)**             | ![img](./gen_img/Neyman/train_5000.png) | ![img](./gen_img/Neyman/train_10000.png) | ![img](./gen_img/Neyman/train_20000.png)
**f-GAN (Jeffrey)**            | ![img](./gen_img/Jeffrey/train_5000.png) | ![img](./gen_img/Jeffrey/train_10000.png) | ![img](./gen_img/Jeffrey/train_20000.png)
**f-GAN (Total-Variation)**    | ![img](./gen_img/Total-Variation/train_5000.png) | ![img](./gen_img/Total-Variation/train_10000.png) | ![img](./gen_img/Total-Variation/train_20000.png)

> GAN with ``Reverse-KL divergence`` seems like *model-collapsed*. But after `10k steps`, it just works well.

## To-Do
* Add f-divergences
  * GAN - **done**
  * KL - **done**
  * Reverse-KL - **done**
  * JS - **done**
  * JS-Weighted - **done**
  * Squared-Hellinger - **done**
  * Pearson χ^2 - **done**
  * Neyman χ^2 - **done**
  * Jeffrey - **done**
  * Total-Variation - **done**
* Add α-divergences
  * α-divergence (α < 0, α is not 0)
  * α-divergence (α > 1)