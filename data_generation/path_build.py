import argparse
import numpy as np
import math
from PIL import Image
import os
import subprocess
import glob
import imageio
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--field_size', type=int, default=64, help='Size of the input/output grid.')
    parser.add_argument('--dencity', type=float, default=0.2, help='From 0 to 1, proportion of the obstacles.')
    parser.add_argument('--obstacles_num', type=int, default=6, help='Max number of obstacles.')
    parser.add_argument('--dataset_size', type=int, default=500, help='Number of unique grids.')
    parser.add_argument('--tasks_num', type=int, default=10, help='Number of tasks per grid.')
    parser.add_argument('--indent', type=int, default=3,
                        help='Free space width around the frame. ')

    parsed_args = parser.parse_args()

    field_size = parsed_args.field_size
    dencity = parsed_args.dencity
    obstacles_num = parsed_args.obstacles_num
    dataset = parsed_args.dataset_size
    tasks_num = parsed_args.tasks_num
    indent = parsed_args.indent

    files_path = './size_' + str(field_size) + '/' + str(int(dencity * 100)) + '_den/'
    files = glob.glob(files_path + '*.xml')

    for file in files:
        bashCommand = './AStar-JPS-ThetaStar ' + file
        out = subprocess.call(bashCommand, shell=True)
        if not os.path.isfile(file[:-4] + '_log.xml'):
            print('not')
            os.system('rm ' + file[:-4] + "_img.png")
        os.system('rm ' + file)

    files = glob.glob(files_path + '*log.xml')
    for file in files:
        f = open(file)
        s = f.read()
        if 'NOT' in s or 'not' in s:
            print('no path')
            os.system('rm ' + file)
            os.system('rm ' + file[:-7] + 'img.png')
            continue
        r = re.findall(r'<row number=\".+\">(.+) <', s)
        map_img = np.ones((field_size, field_size)) * 2
        for i in range(len(r)):
            row = r[i].split()
            for j in range(len(row)):
                if row[j] == '1':
                    map_img[i][j] = 1
                if row[j] == '*':
                    map_img[i][j] = 0

        imageio.imwrite(file[:-4] + ".png", map_img.astype('uint8'))
        os.system('rm ' + file)


