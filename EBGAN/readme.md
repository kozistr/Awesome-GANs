# Energy-based Generative Adversarial Networks

## Loss Function

* used ``mse loss`` at D net
* used ``mse_loss / pt_loss`` at G net

## Architecture Networks

* Same with the EBGAN paper.
* But, i just used hyper-parameters like activation function, etc...

*DIFFS* | *EBGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **activation function** | ``ReLU`` | ``Leaky_ReLU`` |
 **weight initializer**  | ``N(0, 2e-3)`` | ``HE initializer`` |
 **score function**      | ``modified inception score`` | ``just sce``

> Gaussian Normal dist parameters : D(µ = 0, σ = 0.002), G(µ = 0, σ = 0.0002) <br/>
> HE Initializer parameters     : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./ebgan_tb.png)

## Result

*Name* | *Global Step 50k* | *Global Step 100k* | *Global Step 200k*
:---: | :---: | :---: | :---:
**EBGAN**     | ![img](./gen_img/train_00050000.png) | ![img](./gen_img/train_00100000.png) | ![img](./gen_img/train_00200000.png)

## To-Do
*
