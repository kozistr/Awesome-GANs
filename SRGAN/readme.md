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

*Name* | *Global Step 50k* | *Global Step 100k* | *Global Step 200k*
:---: | :---: | :---: | :---:
**SRGAN**  | ![generated_image](https://github.com/kozistr/Awesome-GANs/blob/master/SRGAN/train_50000.png) | ![generated_image](https://github.com/kozistr/Awesome-GANs/blob/master/SRGAN/train_100000.png) | ![generated_image](https://github.com/kozistr/Awesome-GANs/blob/master/SRGAN/train_200000.png)

## To-Do
* Add Loss Function & Explains