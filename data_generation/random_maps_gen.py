import numpy as np
import math
from PIL import Image
import os
import imageio

def save_map(current_map, map_img, file_num):
    start_x = np.random.randint(indent, size=(tasks_num))
    start_y = np.random.randint(field_size - 1, size=(tasks_num))
    finish_x = np.random.randint(field_size - indent - 1, field_size - 1, size=(tasks_num))
    finish_y = np.random.randint(field_size - 1, size=(tasks_num))
    task = 1
    for sx, sy, fx, fy in zip(start_x, start_y, finish_x, finish_y):
        fout = open(files_path + str(file_num) + '.xml', 'w')
        fout.write('<?xml version="1.0" encoding="UTF-8" ?>\n<root>\n    <map>\n        <width>' + \
                   str(field_size) + '</width>\n        <height>' + \
                   str(field_size) + '</height>\n        <startx>' + \
                   str(sx) + '</startx>\n        <starty>' + \
                   str(sy) + '</starty>\n        <finishx>' + \
                   str(fx) + '</finishx>\n        <finishy>' + \
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
        task += 1



def add_obst(current_map, map_img, current_den, file_num, den):
    #Sprint(current_den, den)
    if current_den >= den:
        file_num += 1
        save_map(current_map, map_img, file_num)
        return file_num

    proportion = np.random.choice(np.arange(1, math.sqrt(den) // 6))
    obst_width = 2 * int(proportion)
    obst_height = 3 * int(proportion)
    x = np.random.choice(np.arange(indent + obst_height // 2, field_size - 1 - indent - obst_width))
    y = np.random.choice(np.arange(obst_height // 2, field_size - 1 - obst_height))
    type_ =  np.random.choice(np.arange(1, 4))
    if type_ == 1:
        for dx in range(obst_width):
            for dy in range(obst_height):
                current_map[y + dy][x + dx] = 1
                map_img[y + dy][x + dx] = 1
                current_den += 1
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

                    current_den += 4
    elif type_ == 3:
        for dx in range(obst_width // 2):
            for dy in range(obst_height // 2):
                if dx  + dy < obst_width // 2:
                    current_map[y + dy][x + dx] = 1
                    map_img[y + dy][x + dx] = 1
                    current_map[y - dy][x - dx] = 1
                    map_img[y - dy][x - dx] = 1

                    current_map[y - dy][x + dx] = 1
                    map_img[y - dy][x + dx] = 1
                    current_map[y + dy][x - dx] = 1
                    map_img[y + dy][x - dx] = 1

                    current_den += 4

    file_num = add_obst(current_map, map_img, current_den, file_num, den)
    return file_num



field_size = 64
dataset = 50000
tasks_num = 1
indent = 3

files_path = './size_' + str(field_size) + '/random/'

os.makedirs(files_path)
file_num = 0
for i in range(dataset):
    dencity = np.random.choice(np.arange(0.05, 0.8, 0.05))

   # print(obsts)
    current_map = np.zeros((field_size, field_size)).astype('int')
    map_img = np.ones((field_size, field_size)) * 2
    file_num = add_obst(current_map, map_img, 0, file_num, field_size ** 2 * dencity)

