# Super-Resolution Generative Adversarial Networks

## Loss Function

* I just use 3 losses for ```G loss```

1. VGG19 Bottle Neck feature loss (content loss)
2. Adversarial loss (sigmoid loss)
3. MSE loss (with generated HR img, real HR img)

* And for ```D loss```

1. Adversarial loss (sigmoid loss)
(2. maybe, change adv loss to MSE loss could be good as well i think...)

## Architecture Networks

* Same as in the SRGAN paper

## Tensorboard

![result](https://github.com/kozistr/Awesome-GANs/blob/master/SRGAN/srgan_tb.png)

## Result

*Name* | *Valid HR image* | *Global Step 1k* | *Global Step 40k*
:---: | :---: | :---: | :---:
**SRGAN**  | ![generated_image](https://github.com/kozistr/Awesome-GANs/blob/master/SRGAN/gen_img/valid_hr.png) | ![generated_image](https://github.com/kozistr/Awesome-GANs/blob/master/SRGAN/gen_img/train_00001000.png) | ![generated_image](https://github.com/kozistr/Awesome-GANs/blob/master/SRGAN/gen_img/train_00040000.png)

## To-Do
* Results are not good as i expected... So, maybe it needs to be fixed some way soon...
