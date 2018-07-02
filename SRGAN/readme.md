# Super-Resolution Generative Adversarial Networks

## Loss Function

* For ``Gen loss``

1. VGG19 BottleNeck feature loss (content loss)
2. [optional] MSE loss (content loss) with sigmoid
3. Adversarial GAN loss

* For ``Disc loss``

1. Adversarial GAN loss with sigmoid

## Architecture Networks

* Same as SRGAN paper

*DIFFS* | *SRGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |
 **image scaling** | ``LR[0,1] HR[-1,1]`` | ``L/HR[-1,1]`` |
 **global steps** | ``2e5`` | ``1e5`` |

## Tensorboard

![result](./srgan_tb.png)

> Elapsed time : 1d 9h 30m 52s with ``GTX Titan X 12GB x 1 (maxwell)``

## Result

*VALID HR image* | *VALID LR image* |
:---: | :---: |
![img](./gen_img/valid_hr.png) | ![img](./gen_img/valid_lr.png) |

*Global Step 10k* | *Global Step 25k* | *Global Step 55k*
:---: | :---: | :---:
![img](./gen_img/train_00010000.png) | ![img](./gen_img/train_00025000.png) | ![img](./gen_img/train_00055000.png)

## To-Do
* Not good performance...
