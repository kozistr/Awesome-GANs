# Training Generative Neural Samplers using Variational Divergence Minimization

## Loss Function

* used ``sce loss`` at D/G nets.

## Architecture Networks

* Same as FGAN paper.

*DIFFS* | *FGAN Paper* | *ME*  |
 :---:  |     :---:      | :---: |
 **Weight initializer** | ``normal dist`` | ``HE initializer`` |

> Normal Distribution Initializer : (µ = 0, σ = 0.5) <br/>
> HE Initializer parameters       : (factor = 1, FAN_AVG, uniform)

## Tensorboard

![result](./fgan_tb.png)

> Elapsed Time : h m s with ``GTX 1060 6GB x 1``

## Result

*Name* | *Global Step 5k* | *Global Step 10k* | *Global Step 20k*
:---: | :---: | :---: | :---:
**FGAN**      | ![img](./gen_img/train_8000.png) | ![img](./gen_img/train_16000.png) | ![img](./gen_img/train_32000.png)

## To-Do
* 