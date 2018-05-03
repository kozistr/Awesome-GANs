# Progressive Growing of GANs for Improved Quality, Stability, and Variation

## Loss Function

* 

## Architecture Networks

* similar as PGGAN paper

## Tensorboard

![result](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/pggan_tb.png)

## Result

*Name* | *Global Step 50k* | *Global Step 100k* | *Global Step 300k*
:---: | :---: | :---: | :---:
**PGGAN**     | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/train_50000.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/train_100000.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/PGGAN/gen_img/train_300000.png)

> I just used Celeb-A instead of Celeb-A-HQ because Celeb-A-HQ DataSet's images are too bigggg to handle... :(
> Later, i'll re-model original PGGAN repo and use pre-trained so that working properly on my style :)

## To-Do
* Add Loss Function & Explains