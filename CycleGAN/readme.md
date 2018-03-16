# Unpaired img2img translation using Cycle-consistent Adversarial Networks

## Loss Function

* Using ```l1 loss``` for *cycle loss* and ```adv loss``` for *gen* and ```wgan-gp loss``` for *disc*.

## Architecture Networks

* Maybe, same as in the **CycleGAN paper**.

## Tensorboard

![result](https://github.com/kozistr/Awesome-GANs/blob/master/CycleGAN/cyclegan_tb.png)

## Result

*Name* | *Global Step 50k* | *Global Step 100k* | *Global Step 200k*
:---: | :---: | :---: | :---:
**CycleGAN**     | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CycleGAN/gen_img/train_00050000.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CycleGAN/gen_img/train_00100000.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CycleGAN/gen_img/train_00200000.png)

## To-Do
* 