import argparse

import numpy as np
import math
from PIL import Image
import os
import imageio

def save_map(current_map, map_img, file_num):
    start_x = np.random.randint(indent, size=tasks_num)
    start_y = np.random.randint(field_size - 1, size=tasks_num)
    finish_x = np.random.randint(field_size - indent - 1, field_size - 1, size=tasks_num)
    finish_y = np.random.randint(field_size - 1, size=tasks_num)
    for sx, sy, fx, fy in zip(start_x, start_y, finish_x, finish_y):
        fout = open(files_path + str(file_num) + '.xml', 'w')
        fout.write('<?xml version="1.0" encoding="UTF-8" ?>\n<root>\n    <map>\n        <width>' +
                   str(field_size) + '</width>\n        <height>' +
                   str(field_size) + '</height>\n        <startx>' +
                   str(sx) + '</startx>\n        <starty>' +
                   str(sy) + '</starty>\n        <finishx>' +
                   str(fx) + '</finishx>\n        <finishy>' +
                   str(fy) + '</finishy>\n        <grid>\n')
        map_img[sy][sx] = 0
        map_img[fy][fx] = 0
        for row in current_map.astype('str'):
            fout.write('            <row>' + ' '.join(row.tolist()) + '</row>\n')
        fout.write('        </grid>\n')
        fout.write('    </map>\n')
        alg = '    <algorithm>\n        <searchtype>astar</searchtype>\n        <metrictype>diagonal</metrictype>\n' +\
              '        <breakingties>g-max</breakingties>\n        <hweight>1</hweight>\n        <allowdiagonal>true</allowdiagonal>\n' +\
              '        <cutcorners>false</cutcorners>\n         <allowsqueeze>false</allowsqueeze>\n    </algorithm>\n' +\
              '    <options>\n        <loglevel>1</loglevel>\n        <logpath />\n        <logfilename />\n    </options>\n'
        fout.write(alg)
        fout.write('</root>\n')
        fout.close()
                                       
        imageio.imwrite(files_path + str(file_num) + "_img.png", map_img.astype('uint8'))
        map_img[sy][sx] = 2
        map_img[fy][fx] = 2
        file_num += 1
    
    return file_num


def add_obst(current_map, map_img, current_number, file_num, obsts, random=False):
    if current_number >= obsts:
        file_num += 1
        file_num = save_map(current_map, map_img, file_num)
        return file_num
    
    proportion = np.random.choice(np.arange(1, max_proportion + 1))
    obst_width = 2 * proportion
    obst_height = 3 * proportion
    x = np.random.choice(np.arange(indent, field_size - 1 - indent - obst_height))
    y = np.random.choice(np.arange(field_size - 1 - obst_height))

    type_ = 1
    if random:
        type_ = np.random.choice(np.arange(1, 4))

    if type_ == 1:
        if current_number % 2 == 0:
            for dx in range(obst_width):
                for dy in range(obst_height):
                    current_map[y + dy][x + dx] = 1
                    map_img[y + dy][x + dx] = 1
        else:
            for dx in range(obst_height):
                for dy in range(obst_width):
                    current_map[y + dy][x + dx] = 1
                    map_img[y + dy][x + dx] = 1

    elif type_ == 2:
        for dx in range(obst_width // 2):
            for dy in range(obst_height // 2):
                if dx ** 2 + dy ** 2 < (obst_width // 2) ** 2:
                    current_map[y + dy][x + dx] = 1
                    map_img[y + dy][x + dx] = 1
                    current_map[y - dy][x - dx] = 1
                    map_img[y - dy][x - dx] = 1

                    current_map[y - dy][x + dx] = 1
                    map_img[y - dy][x + dx] = 1
                    current_map[y + dy][x - dx] = 1
                    map_img[y + dy][x - dx] = 1

    elif type_ == 3:
        for dx in range(obst_width // 2):
            for dy in range(obst_height // 2):
                if dx + dy < obst_width // 2:
                    current_map[y + dy][x + dx] = 1
                    map_img[y + dy][x + dx] = 1
                    current_map[y - dy][x - dx] = 1
                    map_img[y - dy][x - dx] = 1

                    current_map[y - dy][x + dx] = 1
                    map_img[y - dy][x + dx] = 1
                    current_map[y + dy][x - dx] = 1
                    map_img[y + dy][x - dx] = 1

    file_num = add_obst(current_map, map_img, current_number + 1, file_num, obsts)
    return file_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--field_size', type=int, default=64, help='Size of the input/output grid.')
    parser.add_argument('--dencity', type=float, default=0.2, help='From 0 to 1, proportion of the obstacles.')
    parser.add_argument('--obstacles_num', type=int, default=6, help='Max number of obstacles.')
    parser.add_argument('--dataset_size', type=int, default=500, help='Number of unique grids.')
    parser.add_argument('--tasks_num', type=int, default=10, help='Number of tasks per grid.')
    parser.add_argument('--indent', type=int, default=3,
                        help='Free space width around the frame. ')
    parser.add_argument('--random_shapes', type=bool, default=False, help='Shapes of the obstacles.')
    parser.add_argument('--theta', type=bool, default=False, help='If theta.')

    parsed_args = parser.parse_args()

    field_size = parsed_args.field_size
    dencity = parsed_args.dencity
    obstacles_num = parsed_args.obstacles_num
    dataset = parsed_args.dataset_size
    tasks_num = parsed_args.tasks_num
    indent = parsed_args.indent
    random_shapes = parsed_args.random_shapes
    theta = parsed_args.theta
    
    if not theta:
        files_path = './size_' + str(field_size) + '/' + str(int(dencity * 100)) + '_den/'
    else: 
        files_path = './size_' + str(field_size) + '/' + str(int(dencity * 100)) + '_den_non_theta/'
    
    print('random_shapes:', random_shapes)

    os.makedirs(files_path)
    file_num = 0
    for i in range(dataset):
        obsts = np.random.choice(np.arange(1, obstacles_num + 1))
        #dencity = np.random.choice(np.arange(1, 3))
        max_proportion = math.floor(math.sqrt(field_size ** 2 * dencity // 6))

        # print(obsts)
        current_map = np.zeros((field_size, field_size)).astype('int')
        map_img = np.ones((field_size, field_size)) * 2
        file_num = add_obst(current_map, map_img, 0, file_num, obsts, random_shapes)

