# GAN Path Finder
### Generative  Adversarial  Networks  for  Path  Planning  in  2D

[GAN Path Finder Arxiv](https://arxiv.org/abs/1908.01499) (early version)

Implementation of GAN-Finder - path planning approach build on Generative Adversarial Networks (GANs).

## Description

In this repository we present a learnable method that automatically predicts instance-dependent heuristic in the form of *path probability map* that maps each graph vertex to a likelihood that this vertex is on a path from start to the goal.

![argmax](https://user-images.githubusercontent.com/17624024/95698777-5ef30f00-0c4b-11eb-8cff-e5ed2a220baf.png)

### Implementation Details

Training is performed on synthetic dataset of 2D grid maps with randomly generated obstacles.  Dataset generation includes generation of grid environments with obstacles and ground truth generation with A^*. Parameters for dataset generation are: `grid_size`, `density`, `max_number_of_obstacles`, `number_of_maps`, `tasks_per_map`.


Implementation provides a variety of models, all based on image-to-image GANs, namely [pix2pix](https://phillipi.github.io/pix2pix/) with specific modifications. There are variant to train:
* plain pix2pix/Unet
* loss function CE instead of L1
* Wasserstein loss
* Path Discriminator (see [GAN Path Finder Arxiv](https://arxiv.org/abs/1908.01499))
* using spectral norm layer
* Self-Attention GAN

See below performance of different models on synthetic datasets.

![2_20den](https://user-images.githubusercontent.com/17624024/95699471-2e13d980-0c4d-11eb-867b-0f2e36f43c79.png)
*20 % density dataset*

![6_25den](https://user-images.githubusercontent.com/17624024/95699524-526fb600-0c4d-11eb-8266-b616aa16e826.png)
*30 % density dataset*

All models are located in `model.py`.

There are numerous types of test, that can be performed, instead of plain validation. See `astar_test.py`.


## Getting Started

Required libraries and versions are listed in `requirements.txt`. For the connivence folder `scripts/` contain scripts for data generation, training and validation.

```
./scripts/create_dataset.sh
```
- use to create folder with input and ground truth images. Folder, size of the dataset and other parameters can be specified inside the script.


#### Train model


```
./scripts/start_training.sh
```

#### Validate model

```
./scripts/validate.sh
```
