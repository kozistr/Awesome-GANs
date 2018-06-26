# Super-Resolution Generative Adversarial Networks

## Loss Function

* For ``Gen loss``

1. VGG19 BottleNeck feature loss (content loss)
2. Adversarial loss (sigmoid loss)
3. MSE loss (with generated HR img, real HR img)

* For ``Disc loss``

1. Adversarial loss (sigmoid loss)
(2. maybe, change adv loss to MSE loss could be good as well i think...)

## Architecture Networks

* Same as SRGAN paper

*DIFFS* | *SRGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |
 **LR decay iters** | ``1e5 iters`` | ``100 epochs`` |

## Tensorboard

![result](./srgan_tb.png)

> Elapsed time : s with ``GTX Titan X 12GB (maxwell)``

## Result

*Name* | *Valid HR image* | *Global Step 1k* | *Global Step 40k*
:---: | :---: | :---: | :---:
**SRGAN**  | ![img](./gen_img/valid_hr.png) | ![img](./gen_img/train_00001000.png) | ![img](./gen_img/train_00040000.png)

## To-Do
* Add De-Noise network
