#!/bin/bash
source ~/anaconda3/bin/activate pytorch
conda install -c conda-forge imageio
cd GAN-PathFinder
python3 maps_gen.py
python3 path_build.py
