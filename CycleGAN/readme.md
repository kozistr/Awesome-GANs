# Unpaired img2img translation using Cycle-consistent Adversarial Networks

## Loss Function

* Using ```l1 loss``` for *cycle loss* and ```adv loss``` for *gen* and ```wgan-gp loss``` for *disc*.

## Architecture Networks

* Same Architectures as in the **CycleGAN paper**.

## Tensorboard

![result](https://github.com/kozistr/Awesome-GANs/blob/master/CycleGAN/cyclegan_tb.png)

## Result

After ```10k steps```.

*Name* | *Valid A* | *Valid B* | *A to B* | *B to A*
:---: | :---: | :---: | :---:
**CycleGAN** | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CycleGAN/gen_img/valid_a.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CycleGAN/gen_img/valid_b.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CycleGAN/gen_img/train_a2b_9900.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/CycleGAN/gen_img/train_b2a_9900.png) 

## To-Do
* 