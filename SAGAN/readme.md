# Self-Attention Generative Adversarial Networks

## Loss Function

* used ``sce_loss`` with D/G nets

## Architecture Networks

* Same with the SAGAN paper.
* But, i just used hyper-parameters like weight initializer, etc...

*DIFFS* | *SAGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **image size** | ``128`` | ``64`` |
 **loss** | ``hinge loss`` | ``GAN loss`` |

## Tensorboard

![result](./sagan_tb.png)

> Elapsed Time : s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 10k* | *Global Step 25k* | *Global Step 50k*
:---: | :---: | :---: | :---:
**ACGAN**     | ![img](./gen_img/train_00010000.png) | ![img](./gen_img/train_00025000.png) | ![img](./gen_img/train_00050000.png)

## To-Do
* 
