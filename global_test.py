
from __future__ import print_function
import argparse
import os
import re
import numpy as np
from math import log10
#from IPython.display import clear_output
import imageio
import matplotlib.pyplot as plt
import subprocess

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

def check_connection(out, img, img_size):
    grid = out.reshape(img_size, img_size)
    img = img.reshape(img_size, img_size)
    grid[np.array(img.tolist()).astype('uint8') == 0] = 0
    grid[np.array(img.tolist()).astype('uint8') == 1] = 1


    inds = np.where(np.array(grid.reshape(img_size, img_size).tolist()).astype('uint8') == 0)

    path_cells = {}
    for x, y in zip(inds[0], inds[1]):
        path_cells[x * img_size + y] = True

    startgoal = np.array(np.where(np.array(img.tolist()).astype('uint8') == 0))
    if startgoal.shape[1] < 2:
        return 0, 0, [], grid
    start_x, start_y = startgoal[:,-2]
    goal_x, goal_y = startgoal[:,-1]

    success = True
    succes_w_bresenham = True
    final_path = []

    path_cells[start_x * img_size + start_y] = False
    while not (start_x == goal_x and start_y == goal_y):
        final_path += [(start_x, start_y)]
        neighbor = False
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            x, y = start_x + dx, start_y + dy
            if x < 0 or y < 0 or x >= img_size or y >= img_size:
                continue
            if x * img_size + y in path_cells and path_cells[x * img_size + y]:
                path_cells[x * img_size + y] = False
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
            final_path += [cell]
            if grid[cell[0], cell[1]] == 1:
                succes_w_bresenham = False
            else:
                grid[cell[0], cell[1]] = 0
        if succes_w_bresenham:
            start_x, start_y = closest
            path_cells[closest[0] * img_size + closest[1]] = False
        else:
            break
    final_path += [(goal_x, goal_y)]
    return success, succes_w_bresenham, final_path, grid

def get_path_greedy(ppm, img, img_size):
    img = img.reshape(img_size, img_size)
    grid = img.copy()
    ppm = ppm.reshape(img_size, img_size)
    ppm = np.round((ppm - ppm.min()) / ppm.max(), 2)
    startgoal = np.array(np.where(np.array(img.tolist()).astype('uint8') == 0))
    if startgoal.shape[1] < 2:
        return 0, [], grid
    start_x, start_y = startgoal[:,-2]
    goal_x, goal_y = startgoal[:,-1]

    success = True
    path_cells = {}
    path_cells[start_x * img_size + start_y] = False

    final_path = []
    while not (start_x == goal_x and start_y == goal_y):
        final_path += [(start_x, start_y)]
        grid[start_x, start_y] = 0
        neighbor = None
        min_prob = -1
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            x, y = start_x + dx, start_y + dy
            if x < 0 or y < 0 or x >= img_size or y >= img_size:
                continue

            key = x * img_size + y
            #print(ppm[x, y], min_prob, img[x, y], key not in path_cells)
            if ppm[x, y] > min_prob and img[x, y] != 1 and (x,y) not in final_path:
                #print('chosen', ppm[x, y], min_prob, img[x, y])
                min_prob = ppm[x, y]
                neighbor = (x, y)

        if neighbor:
            path_cells[neighbor[0] * img_size + neighbor[1]] = False
            start_x, start_y = neighbor
        else:
            success = False
            break
    final_path += [(goal_x, goal_y)]
    return success, final_path, grid

def get_true_length(path):
    length = 0
    for f, t in zip(path[:-1], path[1:]):
        #print(f, t, np.sqrt((f[0] - t[0]) ** 2 + (f[1] - t[1]) ** 2))
        length += np.sqrt((f[0] - t[0]) ** 2 + (f[1] - t[1]) ** 2)
    return length

img_size = 32
channels = 1
num_classes = 3
result_folder = './ours_32_res/'
dataset_dir = './ours_32_test/'
device = torch.device("cuda:2")

os.makedirs(result_folder, exist_ok=True)

result_folders = ['./']

models = [
          define_G(channels, num_classes, img_size, 'batch', False, 'normal', 0.02, gpu_id=device, use_ce=True, unet=False)
          ]

for model, path in zip(models, result_folders):
    model.load_state_dict(torch.load(path + 'generator_ours_32.pt', map_location='cpu'))

batch_size = 6

val_data_loader = DataLoader(ImageDataset(dataset_dir, mode='eval', img_size=img_size),
                             batch_size=1, shuffle=False, num_workers=1)


focal_astar = {1 : {'nodes': [], 'steps': [], 'length': []},
               1.5 : {'nodes': [], 'steps': [], 'length': []},
               2 : {'nodes': [], 'steps': [], 'length': []},
               5 : {'nodes': [], 'steps': [], 'length': []},
               10 : {'nodes': [], 'steps': [], 'length': []}}

success_rate = {'plain' : [], 'bresenham' : [], 'greedy' : []}
steps_number= {'plain' : [], 'bresenham' : [], 'greedy' : []}
lengths = {'plain' : [], 'bresenham' : [], 'greedy' : []}

for i, batch in enumerate(val_data_loader):
    if i > 1000: break
    input, target = batch[0], batch[1]#.to(device), batch[1].to(device)
    predictions = []
    for num, model in enumerate(models):
        output = model(input)
        ppm = output[:, 0, ...]
        _, prediction = torch.max(output, 1, keepdim=True)

        save_image(ppm.float().data, result_folder  + ('%d_ppm.png' % i), nrow=1, normalize=True)
        ppm = imageio.imread(result_folder  + ('%d_ppm.png' % i)).mean(2)

        success, succes_w_bresenham, final_path, grid = check_connection(prediction.float().detach().cpu().numpy(),
                                                                         input.float().detach().cpu().numpy(),
                                                                         img_size)

        if not success and succes_w_bresenham:
            save_image(torch.from_numpy(grid), result_folder  + ('_%d_bres.png' % i), nrow=1, normalize=True)
            save_image(prediction.float().data, result_folder  + ('_%d_pred.png' % i), nrow=1, normalize=True)


        success_rate['plain'] += [success]
        success_rate['bresenham'] += [succes_w_bresenham]
        print(success, succes_w_bresenham, len(final_path), get_true_length(final_path))

        success_greedy, final_path_algo, grid_algo = get_path_greedy(ppm,
                                                          input.float().detach().cpu().numpy(),
                                                          img_size)
        #if i % 100 == 0:
        #    save_image(torch.from_numpy(grid_algo), result_folder  + ('%d_greedy.png' % i), nrow=1, normalize=True)
        success_rate['greedy'] += [success_greedy]

        steps_number['plain'] += [len(final_path)]
        steps_number['bresenham'] += [len(final_path)]
        steps_number['greedy'] += [len(final_path_algo)]

        print('succ done')
        input = input.float().detach().cpu().numpy().reshape(img_size, img_size)
        grid = np.round((ppm - ppm.min()) / ppm.max(), 2)
        grid[grid == 1] = 0.99
        startgoal = np.array(np.where(np.array(input.tolist()).astype('uint8') == 0))
        if startgoal.shape[1] < 2:
            continue

        sx, sy = startgoal[:,-2]
        fx, fy = startgoal[:,-1]
        grid[input == 1] = 1.00
        inp_map = np.where(input == 1, 1, 0)
        fout = open(result_folder  + ('%d.xml' % i), 'w')
        fout.write('<?xml version="1.0" encoding="UTF-8" ?>\n<root>\n    <map>\n        <width>' + \
                    str(img_size) + '</width>\n        <height>' + \
                    str(img_size) + '</height>\n        <startx>' + \
                    str(sy) + '</startx>\n        <starty>' + \
                    str(sx) + '</starty>\n        <finishx>' + \
                    str(fy) + '</finishx>\n        <finishy>' + \
                    str(fx) + '</finishy>\n        <grid>1\n')
        for row in inp_map.astype('str'):
            fout.write('            <row>' + ' '.join(row.tolist()) + '</row>\n')
        fout.write('        </grid>\n')
        fout.write('        <grid_pred>1\n')
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
        print(file)
        for w in [1, 1.5]: #, 2, 5, 10]:
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
            print(steps[0], nodes[0], length[0])

            focal_astar[w]['steps'] += [int(steps[0])]
            focal_astar[w]['nodes'] += [int(nodes[0])]
            focal_astar[w]['length'] += [float(length[0])]

        if success:
            #print(success, get_true_length(final_path), focal_astar[1]['length'][-1])
            lengths['plain'] += [100 * get_true_length(final_path) / focal_astar[1]['length'][-1]]
        if succes_w_bresenham:
            #print(succes_w_bresenham, get_true_length(final_path), focal_astar[1]['length'][-1])
            lengths['bresenham'] += [100 * get_true_length(final_path) / focal_astar[1]['length'][-1]]
        if success_greedy:
            #print(success_greedy, get_true_length(final_path_algo), focal_astar[1]['length'][-1])
            lengths['greedy'] += [100 * get_true_length(final_path_algo) / focal_astar[1]['length'][-1]]


np.save(result_folder + 'focal_astar_results.npy', focal_astar)
np.save(result_folder + 'success_results.npy', success_rate)
np.save(result_folder + 'steps_results.npy', steps)
np.save(result_folder + 'lengths_results.npy', lengths)

with open(result_folder + 'results.txt', 'w') as f:
    f.writelines(["steps focal = %s\n" % np.array(focal_astar[w]['steps']).mean() for w in [1, 1.5]])
    f.writelines(["nodes focal = %s\n" % np.array(focal_astar[w]['nodes']).mean() for w in [1, 1.5]])
    f.writelines(["length focal = %s\n" % np.array(focal_astar[w]['length']).mean() for w in [1, 1.5]])

    f.writelines("success plain = %s\n" % np.array(success_rate['plain']).mean())
    f.writelines("success bres = %s\n" % np.array(success_rate['bresenham']).mean())
    f.writelines("success greedy = %s\n" % np.array(success_rate['greedy']).mean())

    f.writelines("steps plain = %s\n" % np.array(steps_number['plain']).mean())
    f.writelines("steps bres = %s\n" % np.array(steps_number['bresenham']).mean())
    f.writelines("steps greedy = %s\n" % np.array(steps_number['greedy']).mean())

    f.writelines("length plain = %s\n" % np.array(lengths['plain']).mean())
    f.writelines("length bres = %s\n" % np.array(lengths['bresenham']).mean())
    f.writelines("length greedy = %s\n" % np.array(lengths['greedy']).mean())

print("end")
