# Progressive Growing of GANs for Improved Quality, Stability, and Variation

## Loss Function

* 

## Architecture Networks

* similar as PGGAN paper

## Tensorboard

![result](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/pggan_tb.png)

## Result

*Name* | *Global Step 4k* | *Global Step 8k* | *Global Step 12k*
:---: | :---: | :---: | :---:
**64x64**     | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes004000-64.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes008000-64.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes012000-64.png)
**128x128**   | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes004000-128.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes008000-128.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes012000-128.png)

## To-Do
* Add label-penalty for G/D nets
* Add Equalized Learning Rate
* Add Loss Function & Explains