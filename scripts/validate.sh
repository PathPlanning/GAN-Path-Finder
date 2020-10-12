#!/bin/bash
source ~/anaconda3/bin/activate pytorch
pip install bresenham
cd GAN-PathFinder
python3 val_results.py
#python3 train_s2.py
