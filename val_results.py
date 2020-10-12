
from __future__ import print_function
import argparse
import os
import numpy as np
from math import log10
#from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

#import matplotlib.pyplot as plt
from torchvision.utils import save_image

from datasets import ImageDataset
from model import define_D, define_G, get_scheduler, GANLoss, update_learning_rate

cudnn.benchmark = True

from bresenham import bresenham
import collections

def check_connection(img):
    indent = 0
    success = True
    #new_prediction = img
    #print(np.unique(img.float().round().detach().cpu().numpy()))
    inds = np.where(np.array(img.float().round().detach().tolist()).astype('uint8') == 0)
    if len(inds) < 0:
        return False, 1
    prev_x, prev_y = sorted(zip(inds[1], inds[0]))[0]
    for x, y in sorted(zip(inds[1], inds[0]))[1:]:
        if abs(prev_x - x) <= 1 and abs(prev_y - y) <= 1:
           prev_x = x
           prev_y = y 
           continue
        if prev_x == x:
            prev_y = y
            continue

        indent += 1
       # print(list(bresenham(prev_x, prev_y, x, y)))
        for cell in list(bresenham(prev_x, prev_y, x, y))[1:-1]:
            print(img[cell[1], cell[0]])
            if img[cell[1], cell[0]] == 1:
                success = False
            else:
                img[cell[1], cell[0]] = 0
        prev_x = x
        prev_y = y 

    return success, indent

img_size = 64
channels = 1
num_classes = 3
result_folder = './size_64/val_results/'
dataset_dir = './size_64/20_den/'
device = torch.device("cuda:3")

os.makedirs(result_folder, exist_ok=True)

result_folders = ['./size_64/pix2pix/',
                  './size_64/pix2pix_softmax/',
                  #'./size_64/Wpix2pix/',
                  #'./size_64/Wpix2pix_softmax/',
                  './size_64/WUnet_softmax/',
                  './size_64/Wpix2pix_path/',
                 # './size_64/Wpix2pix_ppath/',
                 # './size_64/Wpix2pix_ppath_len/',
                  
                 ]

models = [define_G(channels, channels, 64, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=False, unet=False),
          define_G(channels, num_classes, 64, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=True, unet=False),
          #define_G(channels, channels, 64, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=False, unet=False),
          #define_G(channels, num_classes, 64, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=True, unet=False),
          define_G(channels, num_classes, 64, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=True, unet=True),
          define_G(channels, num_classes, 64, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=True, unet=False),
         # define_G(channels, num_classes, 64, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=True, unet=False),
         # define_G(channels, num_classes, 64, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=True, unet=False),
         ]

for model, path in zip(models, result_folders):
    model.load_state_dict(torch.load(path + 'generator.pt'))

batch_size = 6

val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='val', img_size=img_size),
                             batch_size=1, shuffle=True, num_workers=1)

#criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
#criterionCE = nn.CrossEntropyLoss().to(device)
avg_psnr = [0] * len(models)
number_of_indents = [0] * len(models)
success_rate = [0] * len(models)
'''
for i, batch in enumerate(val_data_loader):
    #if i > 10: break
    input, target = batch[0].to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        if num == 0:
            input_ = input - torch.ones_like(input)
            prediction_ = model(input_)
            prediction = prediction_ + torch.ones_like(prediction_)
        else:
            output = model(input)
            _, prediction = torch.max(output, 1, keepdim=True)
        
        #print(np.unique(prediction.round().detach().cpu().numpy()))
        mse = criterionMSE(prediction.float(), target.float())
        #psnr = 10 * log10(1 / (mse.item() + 1e-16))
        avg_psnr[num] += mse
        success, indent, _ = check_connection(prediction.detach().cpu())
        number_of_indents[num] += indent
        success_rate[num] += success
    if i > 10:
        break
        #if i % 50 == 0:
        #    predictions += [prediction.float().data]
    #if i % 100  == 0:
       # sample = torch.cat((input.data, target.data, *predictions))
       # save_image(torch.transpose(sample, 0, 1), result_folder  + ('%d.png' % i), nrow=1, normalize=True, pad_value=0)
print(i, number_of_indents, success_rate)
with open(result_folder + 'val.txt', 'w') as f:
    f.writelines(["%s " % (item / i)  for item in avg_psnr])
    f.writelines(["%s " % (item / i)  for item in number_of_indents])
    f.writelines(["%s " % (item / i)  for item in success_rate])

'''
for i, batch in enumerate(val_data_loader):
    input, target = batch[0].to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        if num == 0:
            input_ = input - torch.ones_like(input)
            prediction_ = model(input_)
            prediction = prediction_ + torch.ones_like(prediction_)
        else:
            output = model(input)
            _, prediction = torch.max(output, 1, keepdim=True)
        prediction = torch.where(target.float() == 1, torch.ones_like(target).float(), prediction.float())
        predictions += [prediction.float().data]
        if num == len(models) -1:
            success, indent = check_connection(prediction.detach().cpu())
            predictions += [new_pred.float().data]
    sample = torch.cat((input.data, target.data, *predictions), 0)
    print(sample.size())
    save_image(sample, result_folder  + ('%d.png' % i), nrow=7, normalize=True, pad_value=255)
    if i > 10:
        break
'''
dataset_dir = './size_64/all_den/'

val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='val', img_size=img_size),
                             batch_size=6, shuffle=True, num_workers=1)
avg_psnr = [0] * len(models)

for i, batch in enumerate(val_data_loader):
    input, target = batch[0].to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        if num == 0:
            input_ = input - torch.ones_like(input)
            prediction_ = model(input_)
            prediction = prediction_ + torch.ones_like(prediction_)
        else:
            output = model(input)
            _, prediction = torch.max(output, 1, keepdim=True)
        mse = criterionMSE(prediction.float(), target.float())
        psnr = 10 * log10(1 / (mse.item() + 1e-16))
        avg_psnr[num] += psnr
        if i % 450 == 0:
            predictions += [prediction.float().data]
    if i % 450 == 0:
        sample = torch.cat((input.data, target.data, *predictions), -1)
        save_image(sample, result_folder  + 'all_' + ('%d.png' % i), nrow=1, normalize=True)
with open(result_folder + 'all_val.txt', 'w') as f:
    f.writelines(["%s " % (item / i)  for item in avg_psnr])

dataset_dir = './size_64/round/'

val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='val', img_size=img_size),
                             batch_size=6, shuffle=True, num_workers=1)
avg_psnr = [0] * len(models)

for i, batch in enumerate(val_data_loader):
    input, target = batch[0].to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        if num == 0:
            input_ = input - torch.ones_like(input)
            prediction_ = model(input_)
            prediction = prediction_ + torch.ones_like(prediction_)
        else:
            output = model(input)
            _, prediction = torch.max(output, 1, keepdim=True)
        mse = criterionMSE(prediction.float(), target.float())
        psnr = 10 * log10(1 / (mse.item() + 1e-16))
        avg_psnr[num] += psnr
        predictions += [prediction.float().data]
    if i % 450 == 0:
        sample = torch.cat((input.data, target.data, *predictions), -1)
        save_image(sample, result_folder  + 'round_' + ('%d.png' % i), nrow=1, normalize=True)
with open(result_folder + 'round_val.txt', 'w') as f:
    f.writelines(["%s " % (item / i)  for item in avg_psnr])
'''

