import argparse

import numpy as np
import matplotlib.pyplot as plt
import re
import subprocess
import math
import glob
import os

import imageio


class Node:
    def __init__(self, i, j, g=math.inf, h=math.inf, f=None, parent=None):
        self.i = i
        self.j = j
        self.g = g
        if f is None:
            self.f = self.g + h
        else:
            self.f = f
        self.parent = parent

    def __eq__(self, other):
        return (self.i == other.i) and (self.j == other.j)


def get_neighbors(grid, i, j):
    neighbors = []
    height, width = np.array(grid).shape
    delta = [[0, 1], [1, 0], [0, -1], [-1, 0]]

    for d in delta:
        if 0 <= i + d[0] < height and 0 <= j + d[1] < width and not grid[i + d[0]][j + d[1]]:
            neighbors.append((i + d[0], j + d[1]))

    delta = [[1, 1], [-1, -1], [-1, 1], [1, -1]]
    for d in delta:
        if 0 <= i + d[0] < height and 0 <= j + d[1] < width and \
                not grid[i + d[0]][j + d[1]] and not grid[i + d[0]][j] and not grid[i][j + d[1]]:
            neighbors.append((i + d[0], j + d[1]))

    return neighbors


def get_min(open_list):
    best_f = math.inf
    best_ind = 0
    for i in range(len(open_list)):
        if (open_list[i].f < best_f) or (abs(open_list[i].f - best_f) < EPS):
            best_ind = i
            best_f = open_list[i].f

    best = open_list.pop(best_ind)
    return best


def add_node(open_list, item: Node):
    for coord in range(len(open_list)):
        if open_list[coord].i == item.i and open_list[coord].j == item.j:
            if (open_list[coord].g > item.g) or (abs(open_list[coord].g - item.g) < EPS):
                open_list[coord].f = item.f
                open_list[coord].g = item.g
                open_list[coord].parent = item.parent
                return open_list
            else:
                return open_list
    open_list.append(item)
    return open_list


def calculate_cost(i1, j1, i2, j2):
    return math.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)


def dijkstra(grid, start_i, start_j, goal_i, goal_j):
    OPEN = []
    start = Node(start_i, start_j, 0, 0)
    OPEN = add_node(OPEN, start)

    CLOSED = []
    length_values = grid.copy()

    goal_node = None

    while len(OPEN):
        current = get_min(OPEN)
        CLOSED += [current]
        length_values[current.i][current.j] = current.g

        if current.i == goal_i and current.j == goal_j:
            goal_node = current
            # return (True, current, CLOSED, OPEN, length_values)

        for (i, j) in get_neighbors(grid, current.i, current.j):
            new_node = Node(i, j)
            if new_node not in CLOSED:
                new_node.g = current.g + calculate_cost(current.i, current.j, i, j)
                new_node.f = new_node.g
                new_node.parent = current
                OPEN = add_node(OPEN, new_node)

    return True, goal_node, CLOSED, OPEN, length_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='./size_64/20_den/', help='Folder with generated image maps.')
    parser.add_argument('--theta', type=bool, default=False, help='If dataset is with theta.')

    parsed_args = parser.parse_args()

    EPS = 0.000001
    dataset_folder = parsed_args.dataset_folder
    theta = parsed_args.theta
    files = glob.glob(dataset_folder + '*_img.png')
    print(files)

    for file in files:
        image = plt.imread(file)
        path_p, obst_p, free_p = np.unique(image)
        image_map = (2 * np.ones_like(image)).astype(int)
        image_map[image == path_p] = 0
        image_map[image == obst_p] = 1

        (si, sj), (fi, fj) = np.argwhere(image_map == 0)
        start = Node(si, sj)
        goal = Node(fi, fj)

        cells = np.zeros_like(image_map)
        cells[image_map == 1] = 1

        result = dijkstra(cells, start.i, start.j, goal.i, goal.j)
        result_rev = dijkstra(cells, goal.i, goal.j, start.i, start.j)

        org_length = result[1].g
        lengths = []
        if theta:
            weights = [0.75, 0.6, 0.45, 0.3, 0.25, 0.15, 0.05]
            
        else:
            weights = [1, 0.6, 0.45, 0.3, 0.25, 0.15, 0.05]

        for percent in [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            lengths += [org_length + round(org_length * percent)]

        ppm = np.zeros_like(result_rev[-1]).astype(float)
        dists = result[-1].copy() + result_rev[-1].copy()
        prev_length = lengths[0] - 2
        for length, weight in zip(lengths, weights):
            ppm[(prev_length < dists) & (dists <= length)] += weight
            prev_length = length

        ppm[image_map == 1] = -1
        
        if theta:

            sx, sy = sj, si
            fx, fy = fj, fi
            field_size = ppm.shape[0]

            fout = open(file[:-8] + '.xml', 'w')
            fout.write('<?xml version="1.0" encoding="UTF-8" ?>\n<root>\n    <map>\n        <width>' +
                       str(field_size) + '</width>\n        <height>' +
                       str(field_size) + '</height>\n        <startx>' +
                       str(sx) + '</startx>\n        <starty>' +
                       str(sy) + '</starty>\n        <finishx>' +
                       str(fx) + '</finishx>\n        <finishy>' +
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
            alg = '    <algorithm>\n        <searchtype>theta</searchtype>\n        <metrictype>euclid</metrictype>\n' + \
                  '        <breakingties>g-max</breakingties>\n        <hweight>1</hweight>\n' + \
                  '        <allowdiagonal>true</allowdiagonal>\n        <cutcorners>false</cutcorners>\n' + \
                  '         <allowsqueeze>false</allowsqueeze>\n    </algorithm>\n    <options>\n' + \
                  '        <loglevel>1</loglevel>\n        <logpath />\n        <logfilename />\n    </options>\n'
            fout.write(alg)
            fout.write('</root>\n')
            fout.close()

            bashCommand = './AStar-JPS-ThetaStar ' + file[:-8] + '.xml'
            out = subprocess.call(bashCommand, shell=True)
            os.system('rm ' + file[:-8] + '.xml')
            print(file[:-8] + '_log.xml')
            if not os.path.isfile(file[:-8] + '_log.xml'):
                print('not')
                os.system('rm ' + file)
                continue

            f = open(file[:-8] + '_log.xml')
            s = f.read()

            path_x = np.array(re.findall(r'node x=\"(.+)\" y', s)).astype(int)
            path_y = np.array(re.findall(r'node x=\".+\" y=\"(.+)\" nu', s)).astype(int)
            ppm[path_y, path_x] = 1

        ppm[image_map == 1] = -0.5

        print('saving ' + file[:-8] + '_log.png')
        map_img = -ppm
        map_img = np.interp(map_img, (map_img.min(), map_img.max()), (0, 2))
        imageio.imwrite(file[:-8] + "_log.png", map_img)
        os.system('rm ' + file[:-8] + '_log.xml')
