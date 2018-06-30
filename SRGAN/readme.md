# Super-Resolution Generative Adversarial Networks

## Loss Function

* For ``Gen loss``

1. VGG19 BottleNeck feature loss or MSE loss (content loss)
2. Adversarial GAN loss

* For ``Disc loss``

1. Adversarial GAN loss

## Architecture Networks

* Same as SRGAN paper

*DIFFS* | *SRGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |

## Tensorboard

![result](./srgan_tb.png)

> Elapsed time : s with ``GTX Titan X 12GB x 1 (maxwell)``

## Result

*Name* | *Valid HR image* | *Global Step 1k* | *Global Step 40k*
:---: | :---: | :---: | :---:
**SRGAN**  | ![img](./gen_img/valid_hr.png) | ![img](./gen_img/train_00001000.png) | ![img](./gen_img/train_00040000.png)

## To-Do
* 
