
from __future__ import print_function
import argparse
import os
import numpy as np
from math import log10
#from IPython.display import clear_output
import imageio
import matplotlib.pyplot as plt

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

def check_connection(out, img):
    img_size = out.shape[1]
    grid = out.reshape(img_size, img_size)
    img = img.reshape(img_size, img_size)
    grid[np.array(img.tolist()).astype('uint8') == 0] = 0
    grid[np.array(img.tolist()).astype('uint8') == 1] = 1


    inds = np.where(np.array(grid.reshape(img_size, img_size).tolist()).astype('uint8') == 0)

    path_cells = {}
    for x, y in zip(inds[0], inds[1]):
        path_cells[x * img_size + y] = True

    startgoal = np.array(np.where(np.array(img.tolist()).astype('uint8') == 0))
    start_x, start_y = startgoal[:,0]
    goal_x, goal_y = startgoal[:,1]

    success = True
    succes_w_bresenham = True
    final_path = []

    path_cells[start_x * img_size + start_y] = False
    while not (start_x == goal_x and start_y == goal_y):
        final_path += [(start_x, goal_x)]
        neighbor = False
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            x, y = start_x + dx, start_y + dy
            if x < 0 or y < 0 or x >= img_size or y >= img_size:
                continue
            if x * img_size + y in path_cells and path_cells[x * img_size + y]:
                path_cells[x * img_size + y] = False
                length_true += np.sqrt((start_x - x) ** 2 + (start_y - y) ** 2)
                start_x, start_y = x, y
                neighbor = True
                break
        if neighbor:
            continue

        success = False
        min_dist = np.inf
        closest = None
        for x, y in zip(inds[0], inds[1]):
            if (start_x - x) ** 2 + (start_y - y) ** 2 < min_dist and path_cells[x * img_size + y]:
                min_dist = (start_x - x) ** 2 + (start_y - y) ** 2
                closest = (x, y)
        for cell in list(bresenham(start_x, start_y, closest[0], closest[1]))[1:-1]:
            print(cell[0], cell[1])
            final_path += [cell]
            if out[cell[0], cell[1]] == 1:
                succes_w_bresenham = False
            else:
                out[cell[0], cell[1]] = 0
        if succes_w_bresenham:
            start_x, start_y = closest
            path_cells[closest[0] * img_size + closest[1]] = False
        else:
            break
    return success, succes_w_bresenham, final_path

def get_path_greedy(ppm, img):
    img_size = ppm.shape[1]
    img = img.reshape(img_size, img_size)
    ppm = ppm.reshape(img_size, img_size)
    startgoal = np.array(np.where(np.array(img.tolist()).astype('uint8') == 0))
    start_x, start_y = startgoal[:,0]
    goal_x, goal_y = startgoal[:,1]

    success = True
    path_cells = {}
    path_cells[start_x * img_size + start_y] = False

    final_path = []
    while not (start_x == goal_x and start_y == goal_y):
        final_path += [(start_x, start_y)]
        neighbor = None
        min_prob = np.inf
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            x, y = start_x + dx, start_y + dy
            if x < 0 or y < 0 or x >= img_size or y >= img_size:
                continue
            if ppm[x, y] < min_prob and x * img_size + y not in path_cells and img[x, y] != 1:
                neighbor = (x, y)

        if neighbor:
            path_cells[neighbor[0] * img_size + neighbor[1]] = False
            start_x, start_y = neighbor
    else:
        success = False
            break
    
    return success, final_path

def get_true_length(path):
    length = 0
    for f, t in zip(path[:-1], path[1:]):
        length += np.sqrt((start_x - x) ** 2 + (start_y - y) ** 2)
    return length

img_size = 32
channels = 1
num_classes = 3
result_folder = './validation/ours_64_free_all_tests/'
dataset_dir = './dataset/ours_64_free_/'
device = torch.device("cuda:1")

os.makedirs(result_folder, exist_ok=True)

result_folders = ['./dataset/ours_64_free__results20_den/']

models = [
          define_G(channels, num_classes, img_size, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=True, unet=False, attn=True)
          ]

for model, path in zip(models, result_folders):
    model.load_state_dict(torch.load(path + 'generator.pt'))

batch_size = 6

val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='eval', img_size=img_size),
                             batch_size=1, shuffle=False, num_workers=1)


focal_astar = {1 : {'nodes': [], 'steps': [], 'length': []},
               1.5 : {'nodes': [], 'steps': [], 'length': []},
               2 : {'nodes': [], 'steps': [], 'length': []},
               5 : {'nodes': [], 'steps': [], 'length': []},
               10 : {'nodes': [], 'steps': [], 'length': []}}

success_rate = {'plain' : [], 'bresenham' : [], 'greedy' : []}
steps = {'plain' : [], 'bresenham' : [], 'greedy' : []}
lengths = {'plain' : [], 'bresenham' : [], 'greedy' : []}

for i, batch in enumerate(val_data_loader):
    if i > 5000: break
    input, target = batch[0].to(device), batch[1].to(device) # .to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        output = model(input)
        ppm = output[:, 0, ...]
        _, prediction = torch.max(output, 1, keepdim=True)
        success, succes_w_bresenham, final_path = check_connection(prediction.float().detach().cpu(),
                                                                   input.float().detach().cpu())
        success_rate['plain'] += [success]
        success_rate['bresenham'] += [succes_w_bresenham]
        
        success_greedy, final_path_algo = get_path_greedy(ppm.float().detach().cpu(),
                                                          input.float().detach().cpu())
        success_rate['greedy'] += [success_greedy]
        
        if success and succes_w_bresenham and success_greedy:
            steps['plain'] += [len(final_path)]
            steps['bresenham'] += [len(final_path)]
            steps['greedy'] += [len(final_path_algo)]
        
            lengths['plain'] += [get_true_length(final_path)]
            lengths['bresenham'] += [get_true_length(final_path)]
            lengths['greedy'] += [get_true_length(final_path_algo)]
        

        grid = np.round((ppm.mean(2) - ppm.mean(2).min()) / ppm.mean(2).max(), 2)
        grid[grid == 1] = 0.99
        if np.argwhere(input.mean(2) == 0).shape[0] < 2:
            continue
        (sx, sy), (fx, fy) = np.argwhere(input.mean(2) == 0)
        grid[input.mean(2) == 127] = 1.00
        inp_map = np.where(input.mean(2) == 127, 1, 0)
        fout = open(result_folder  + ('%d.xml' % i), 'w')
        fout.write('<?xml version="1.0" encoding="UTF-8" ?>\n<root>\n    <map>\n        <width>' + \
                   str(img_size) + '</width>\n        <height>' + \
                   str(img_size) + '</height>\n        <startx>' + \
                   str(sy) + '</startx>\n        <starty>' + \
                   str(sx) + '</starty>\n        <finishx>' + \
                   str(fy) + '</finishx>\n        <finishy>' + \
                   str(fx) + '</finishy>\n        <grid>\n')
                   for row in inp_map.astype('str'):
                       fout.write('            <row>' + ' '.join(row.tolist()) + '</row>\n')
                   fout.write('        </grid>\n')
                   fout.write('        <grid_pred>\n')
                       for row in grid:
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
        file = result_folder  + ('%d.xml' % i)
        for w in [1, 1.5, 2, 5, 10]:
            bashCommand = './AStar-JPS-ThetaStar ' + file + ' ' + str(w)
            out =  subprocess.call(bashCommand, shell=True)
            if not os.path.isfile(file[:-4] + '_log.xml'):
                print('not')
                continue

            f = open(file[:-4] + '_log.xml')
            s = f.read()
            steps = re.findall(r'numberofsteps=\"(.+)\" no', s)
            nodes = re.findall(r'nodescreated=\"(.+)\" length=', s)
            length = re.findall(r'length=\"(.+)\" l', s)


            focal_astar[w]['steps'] += [int(steps[0])]
            focal_astar[w]['nodes'] += [int(nodes[0])]
            focal_astar[w]['length'] += [float(length[0])]

            os.system('rm ' + file[:-4] + '_log.xml')


np.save('focal_astar_results.npy', focal_astar)
np.save('success_results.npy', success_rate)
np.save('steps_results.npy', steps)
np.save('lengths_results.npy', lengths)

with open('results.txt', 'w') as f:
    f.writelines(["steps focal = %s\n" % np.array(focal_astar[w]['steps']).mean() for w in [1, 1.5, 2, 5, 10]])
    f.writelines(["nodes focal = %s\n" % np.array(focal_astar[w]['nodes']).mean() for w in [1, 1.5, 2, 5, 10]])
    f.writelines(["length focal = %s\n" % np.array(focal_astar[w]['length']).mean() for w in [1, 1.5, 2, 5, 10]])

    f.writelines("success plain = %s\n" % np.array(success_rate['plain']).mean())
    f.writelines("success bres = %s\n" % np.array(success_rate['bresenham']).mean())
    f.writelines("success greedy = %s\n" % np.array(success_rate['greedy']).mean())

    f.writelines("steps plain = %s\n" % np.array(steps['plain']).mean())
    f.writelines("steps bres = %s\n" % np.array(steps['bresenham']).mean())
    f.writelines("steps greedy = %s\n" % np.array(steps['greedy']).mean())

    f.writelines("length plain = %s\n" % np.array(lengths['plain']).mean())
    f.writelines("length bres = %s\n" % np.array(lengths['bresenham']).mean())
    f.writelines("length greedy = %s\n" % np.array(lengths['greedy']).mean())

print("end")
