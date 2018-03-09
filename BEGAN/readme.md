# Boundary Equilibrium Generative Adversarial Networks

## Loss Function

* using ```l1 loss```. You can see the details in ```began_model.py line 233~236```.

## Architecture Networks

* Maybe, same as in the *BEGAN paper*. But *different *hyper-parameters* :).

## Tensorboard

![result](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/began_tb.png)

## Result

*Name* | *Global Step 50k* | *Global Step 100k* | *Global Step 300k*
:---: | :---: | :---: | :---:
**BEGAN**     | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/gen_img/train_50000.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/gen_img/train_100000.png) | ![Generated Image](https://github.com/kozistr/Awesome-GANs/blob/master/BEGAN/gen_img/train_300000.png)

> ```Took about 28800 seconds on GTX 1080.```

## To-Do
* Add Loss Function & Explains