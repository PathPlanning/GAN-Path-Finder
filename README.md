# GAN Path Finder
### Generative  Adversarial  Networks  for  Path  Planning  in  2D

[GAN Path Finder Arxiv](https://arxiv.org/abs/1908.01499) (early version)

Implementation of GAN-Finder - path planning approach build on Generative Adversarial Networks (GANs).

## Description

In this repository we present a learnable method that automatically predicts instance-dependent heuristic in the form of *path probability map* that maps each graph vertex to a likelihood that this vertex is on a path from start to the goal.

![argmax](https://user-images.githubusercontent.com/17624024/95698777-5ef30f00-0c4b-11eb-8cff-e5ed2a220baf.png)

![фдд](https://user-images.githubusercontent.com/17624024/96375936-cbf33100-1184-11eb-864f-6a0d89057e0c.jpg)

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

Required libraries and versions are listed in `requirements.txt`. For the convenience folder `scripts/` contain scripts for data generation, training and validation.

#### Generate Dataset

The simplest way to generate our default dataset, which contains 100 000 of images (50 000 input images, i.e. images without the path and 50 000 ground-truth images, i.e. images with the A* path depicted) is to run the following command
```
./scripts/create_dataset.sh
```
The resultant maps (images) will be of the size 64 by 64. Each map will be filled with randomly-sized rectangular obstacles. The obstacle density of each map will be 20% (~20% of pixels will be blocked, 80% of pixels will be free). The maximal number of obstacles on each map will be 5. For each map 10 different start-goal pairs will be chosen. Start pixel will be located close to left border of the map, goal -- close to the right border.

Resultant images (with obstacles and with the start and goal pixels highlighted) will be saved to the folder `./size_{field_size}/{density * 100}_den/`, _default_ `./size_64/20_den/`.
An examples of such a pair of images (map size 64, rectangular obstacles, density 20%) is shown here: [examples](https://github.com/PathPlanning/GAN-Path-Finder/tree/master/examples/size_64/20_den).

It is possible to tweak the dataset generation (e.g. create maps of different size, with more tasks per map etc.). In this case follow the steps, described in [data_generation](https://github.com/PathPlanning/GAN-Path-Finder/tree/master/data_generation) folder.

*Train/Validation/Test split* is happening on the fly, depending on weather you train or validate the model. Our default split is 75% train, 15% validation and 10% test. When training the model is using train batch and validation batch to see progress and catch overfitting. When final evaluation is taken place, test batch of the original data is used. 
Proportions are hardcoded in file [datasets.py](https://github.com/PathPlanning/GAN-Path-Finder/blob/master/datasets.py) and works by simply sorting input file names and splitting indices into train/val/test. Parameters could be changed directly inside the code or by specifying another dataloader - ex. if you have other maps generating procedure, that results in different structure of file names. 

#### Train model
The simplest way to train our default network (the best performing one) is to run the following command

```
./scripts/start_training.sh
```
Or customize training procedure

```
python ./train.py --img_size 64 --channels 1 --num_classes 3 --dataset_dir ./size_64/20_den/ --results_dir ./size_64/20_den/results/ --batch_size 32 --epoch_count 1 --number_of_epochs 100
```

Change parameters to change the training: 
+ img_size - size of the input and output images/field_size of the grid.
+ channels - number of channels in the input image (in case you want to change input images to accommodate multi-channel input).
+ num_classes - output number of channels/classes (3 classes: free space, obstacles and path pixels).
+ dataset_dir - path to the dataset with images.
+ results_dir - where all the results/model weights will be saved.
+ batch_size - batch size for training.
+ epoch_count - from which epoch to start (1 to start from the beginning, any other epoch to continue training from mentioned epoch and last checkpoint in results folder).
+ number_of_epochs - number of epochs to train.

As it was mentioned in [GAN Path Finder Arxiv](https://arxiv.org/abs/1908.01499) we experimented with a lot of model variations, that supports change of loss functions, addition of wasserstein training component, attention blocks and etc. All those variations are supported in the code, but has to be modified in `train.py` file directly. 


#### Validate model

?? questions ??
```
./scripts/validate.sh
```
