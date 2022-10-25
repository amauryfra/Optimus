# Optimus Project
This project is an attempt to produce a __deep learning-based random prime number generator__. The approach consists in performing online continual training of generative adversarial networks while using __Elastic Weight Consolidation__.

We are seeking to find out if a certain probability distribution of prime numbers over a _finite range of integers_ can be uncovered by deep neural network architectures. As understanding such distribution is a highly ill-conditioned problem, this project is more effectively intended to provide an interesting framework for implementing and testing the limits of new machine learning methods in a stiff context.  

## Methodology
The current tested methodology consists in using generative adversarial networks (GANs) trained online using Elastic Weight Consolidation (EWC).

#### GAN framework

* A first fully-connected deep neural network acts as the generator, taking _latent integers_ as input and producing strictly positive real numbers. Said reals are rounded, doubled, and incremented by one, to produce odd integers that are targeted to be primes.
* A second fully-connected deep neural network subject to dropout acts as the discriminator, taking generated reals and computing a probability of having their odd integer transformation to be prime. Therefore, the discriminator network attempts to continuously approximate the `isPrime` function.

The GAN approach used here differs from the usual procedure, as the discriminator does not attempt to separate fake data from domain entries. The generator is encouraged to generate as many primes as possible, while the discriminator learns to perform primality tests, supervised by a pre-implemented primality testing function.

#### Elastic Weight Consolidation and Continual training

Online training is performed where batches of random latent integers are continuously fed to the generator/discriminator adversarial system. However, online learning using conventional training methods arises with several significant drawbacks. In particular neural networks cannot generally learn tasks sequentially (referred to as _Catastrophic Forgetting_). To overcome this obstacle we use the Elastic Weight Consolidation method proposed in [[1]](#1).

Accordingly, we compute the Fisher information matrix and add a specific penalty term to the chosen base loss at each newly encountered batch. The objective is to minimize the change in weights that were useful in previous prime generations.

#### Additional settings

In addition, we propose the use of several techniques to improve the training process.

* Since there is a natural class imbalance, we perform __undersampling__ to obtain batches where prime and non-prime numbers are equally represented.
* We use __Focal Crossentropy__ as the generator's base loss to mitigate any residual class imbalance [[2]](#2).
* To prevent the generator from producing mostly small integers (since the concentration of primes is higher in smaller ranges), we implement an __opposite L2 regularization__, using a negative coefficient to penalize small weights.

## Project structure and Main usage
<ins>The main directory contains</ins> :
* A general utilities script `utils.py`
* A testing script `test.py` to test the pre-trained generator and discriminator on a batch of random latent integers

<ins>The `training` directory contains</ins> :
* A script `model.py` providing the neural networks architectures
* The `training_utils.py` script holding the principal building blocks of the training process
* The main training script `train.py` that performs the online training of newly initialized models, or that trains upon previously created models if available in the directory
* Eventually weight files `Optimus_Generator.h5`, `Optimus_Discriminator.h5` and NumPy array files `fisher_generator.npy`, `fisher_discriminator.npy` of previously used Fisher information matrices

This project uses the number theory subpackage of the `sympy` library for general prime processing. The deep learning architectures are implemented using the `keras` API with a `tensorflow` based backend.

To start working on the project first create a virtual environment, then clone the repository and install all dependencies by running :

```
$ git clone https://github.com/amauryfra/Optimus.git
$ cd Optimus
$ python3 pip install -r requirements.txt
```

To initiate online training run :

```
$ cd training
$ python3 train.py
```

## Future work

The next machine learning techniques to be implemented and tested in this prime generation setting may include the use of Gated Linear Networks (GLNs) as described in [[3]](#3). The methodology could comprehend a Bernoulli-GLN as a discriminator to approximate primality classification in a continuous learning fashion.

## Credits

Parts of the code developed in this project have made use of relevant implementation ideas available in the `stijani/elastic-weight-consolidation-tf2` repository.

## References
<a id="1">[1]</a>
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13), 3521-3526.

<a id="2">[2]</a>
Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).

<a id="3">[3]</a>
Veness, J., Lattimore, T., Budden, D., Bhoopchand, A., Mattern, C., Grabska-Barwinska, A., ... & Hutter, M. (2021, May). Gated linear networks. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 11, pp. 10015-10023).
