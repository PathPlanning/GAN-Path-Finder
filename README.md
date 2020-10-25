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

#### Generate Synthetic dataset

The simplest way to generate our default dataset, which is 50000 images in size, has 64 by 64 field size and 20 % obstacle density is to run the following command
```
./scripts/create_dataset.sh
```
However we support generation of other datasets, where you can change field sizes, densities, etc.
Maps and tasks to feed to the model as an input are generated using `maps_gen.py` (for rectangle obstacles) or `random_maps_gen.py` (for random obstacles, square, rectangle, round) files. 

```
python ./data_generation/maps_gen.py --field_size 64 --density 0.2 --obstacles_num 5 --indent 3 --dataset_size 5000 --tasks_num 10
```
Change parameters to change the dataset: 
+ field_size - the size of the square grid (_default_ 64x64).
+ density - value from 0 to 1, the proportion of the obstacles on the grid (0.2 means ~20% of the map is obstacles).
+ obstacles_num - maximal number of obstacle (obstacles are always rectangular in proportion 2x3, but their actual sizes are calculated according to maximal number of obstacles and density).
+ indent - number of grid cells near the edge, where obstacles are not generated. 
+ dataset_size - number of unique grids/maps with unique obstacles.
+ tasks_num - number of tasks (start/goal positions for each grid, ex. 5000 maps and 10 tasks per map makes 50000 images in the dataset).

Dataset is saved in the folder `./size_{field_size}/{density * 100}_den/`, _default_ `./size_64/20_den/`.
To generate maps with random obstacles, use `random_maps_gen.py`, most of the parameters are the same, the shape of each obstacle is chosen randomly. 

Now, in order to generate ground truth images with A* paths use `path_build.py` file. 

```
python ./data_generation/path_build.py --field_size 64 --density 0.2 --obstacles_num 5 --indent 3
```

See examples of size 64 and density 20 data in [examples](https://github.com/PathPlanning/GAN-Path-Finder/tree/master/examples/size_64/20_den).

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
