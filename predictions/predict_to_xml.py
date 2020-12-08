from __future__ import print_function
import argparse
import os
import numpy as np
from math import log10
import imageio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchvision.utils import save_image
from torchvision import transforms

from datasets import ImageDataset
from model import define_D, define_G, get_scheduler, GANLoss, update_learning_rate

cudnn.benchmark = True

import collections


def get_prediction(img_size, channels, num_classes, model_dir, dataset_dir, result_folder):

    device = torch.device("cuda:0")
    os.makedirs(result_folder, exist_ok=True)

    result_folders = model_dir

    model = define_G(channels, num_classes, 64, 'batch', False, 'normal', 0.02,
                     device=device, use_ce=False, use_attn=True, context_encoder=False, unet=False)
    
    model.load_state_dict(torch.load(model_dir + 'generator.pt'))
    
    transform = transforms.Compose([
                                    transforms.Normalize([0.5], [0.5])
                                    ])

    val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='eval', img_size=128, transform=transform),
                                 batch_size=1, shuffle=False, num_workers=1)

    for i, batch in enumerate(val_data_loader):
        if i > 5000: break
        input, target = batch[0].to(device), batch[1].to(device) # .to(device), batch[1].to(device)
        
        output = model(input)
        ppm = -output[0, 0, ...].detach().cpu().numpy()
        ppm = np.round((ppm - ppm.min()) / (ppm.max() - ppm.min()), 2)
        
        cells = np.zeros_like(input[0, 0, ...].detach().cpu().numpy()).astype(int)
        cells[input[0, 0, ...].detach().cpu().numpy() == 0] = 1
        
        ppm[cells == 1] = -1
        
        (si, sj), (fi, fj) = np.argwhere(input[0, 0, ...].detach().cpu().numpy() == -1)

        sx, sy = sj, si
        fx, fy = fj, fi
        field_size = img_size

        fout = open(result_folder + '%d.xml' % i, 'w')
        fout.write('<?xml version="1.0" encoding="UTF-8" ?>\n<root>\n    <map>\n        <width>' + \
                   str(field_size) + '</width>\n        <height>' + \
                   str(field_size) + '</height>\n        <startx>' + \
                   str(sx) + '</startx>\n        <starty>' + \
                   str(sy) + '</starty>\n        <finishx>' + \
                   str(fx) + '</finishx>\n        <finishy>' + \
                   str(fy) + '</finishy>\n        <grid>1\n')

        for row in cells.astype('str'):
            fout.write('            <row>' + ' '.join(row.tolist()) + '</row>\n')

        fout.write('        </grid>\n')
        fout.write('        <grid_pred>1\n')
        for row in ppm:
            row = ["{0:0.2f}".format(h) for h in row]
            fout.write('            <row>' + ' '.join(row) + '</row>\n')
        fout.write('        </grid_pred>\n')
        fout.write('    </map>\n')
        alg = '    <algorithm>\n        <searchtype>astar</searchtype>\n        <metrictype>diagonal</metrictype>\n' +\
              '        <breakingties>g-max</breakingties>\n        <hweight>1</hweight>\n        <allowdiagonal>true</allowdiagonal>\n' +\
              '        <cutcorners>false</cutcorners>\n         <allowsqueeze>false</allowsqueeze>\n    </algorithm>\n' +\
              '    <options>\n        <loglevel>1</loglevel>\n        <logpath />\n        <logfilename />\n    </options>\n'
        fout.write(alg)
        fout.write('</root>\n')
        fout.close()
        
        gt = -target[0, 0, ...].detach().cpu().numpy()
        ppm_gt = np.round((gt - gt.min()) / (gt.max() - gt.min()), 2)
        ppm_gt[cells == 1] = -1
        

        fout = open(result_folder + '%d_gt.xml' % i, 'w')
        fout.write('<?xml version="1.0" encoding="UTF-8" ?>\n<root>\n    <map>\n        <width>' + \
                   str(field_size) + '</width>\n        <height>' + \
                   str(field_size) + '</height>\n        <startx>' + \
                   str(sx) + '</startx>\n        <starty>' + \
                   str(sy) + '</starty>\n        <finishx>' + \
                   str(fx) + '</finishx>\n        <finishy>' + \
                   str(fy) + '</finishy>\n        <grid>1\n')

        for row in cells.astype('str'):
            fout.write('            <row>' + ' '.join(row.tolist()) + '</row>\n')

        fout.write('        </grid>\n')
        fout.write('        <grid_pred>1\n')
        for row in ppm_gt:
            row = ["{0:0.2f}".format(h) for h in row]
            fout.write('            <row>' + ' '.join(row) + '</row>\n')
        fout.write('        </grid_pred>\n')
        fout.write('    </map>\n')
        alg = '    <algorithm>\n        <searchtype>astar</searchtype>\n        <metrictype>diagonal</metrictype>\n' +\
              '        <breakingties>g-max</breakingties>\n        <hweight>1</hweight>\n        <allowdiagonal>true</allowdiagonal>\n' +\
              '        <cutcorners>false</cutcorners>\n         <allowsqueeze>false</allowsqueeze>\n    </algorithm>\n' +\
              '    <options>\n        <loglevel>1</loglevel>\n        <logpath />\n        <logfilename />\n    </options>\n'
        fout.write(alg)
        fout.write('</root>\n')
        fout.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input/output grid.')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels in the input image.')
    parser.add_argument('--num_classes', type=int, default=3, help='Output number of channels/classes.')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Path to the dataset with images.')
    parser.add_argument('--model_dir', type=str, default='./model', help='Path to the model checkpoint.')
    parser.add_argument('--result_folder', type=str, default='./results',
                        help='Where all the results/weights will be saved.')


    parsed_args = parser.parse_args()
    get_prediction(parsed_args.img_size, parsed_args.channels, parsed_args.num_classes, 
                   parsed_args.model_dir, parsed_args.dataset_dir, parsed_args.result_folder)


