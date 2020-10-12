import numpy as np
import math
from PIL import Image
import os
import subprocess
import glob
import imageio
import re

field_size = 64
#dencity = 0.5
obstacles_num = 5
dataset = 1000
tasks_num = 10
indent = 3


files_path = './size_' + str(field_size) + '/20_den_val/'
files = glob.glob(files_path + '*.xml')

for file in files:
    bashCommand = './AStar-JPS-ThetaStar ' + file
    out =  subprocess.call(bashCommand, shell=True)
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

