# Progressive Growing of GANs for Improved Quality, Stability, and Variation

## Loss Function

* 

## Architecture Networks

* similar as PGGAN paper

## Tensorboard

![result](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/pggan_tb.png)

## Result

*Name* | *Global Step 8k* | *Global Step 12k*
:---: | :---: | :---:
**64x64**     | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes008000-64.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes012000-64.png)
**128x128**   | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes008000-128.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/fakes012000-128.png)

> `64x64` takes almost 5 days with `GTX 1060 6GB x 2`,
> `128x128` takes almost 6 days with single `GTX 1080 8GB` <br/>
> means that it's hard to run PGGAN at home :(...

## To-Do
* Add label-penalty for G/D nets
* Add Equalized Learning Rate
* Add Loss Function & Explains