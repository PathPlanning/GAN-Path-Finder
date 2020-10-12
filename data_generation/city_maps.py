import numpy as np
import math
from PIL import Image
import os
import subprocess
import glob
import imageio
import re

size_map = 1024
size_gan = 128

files_path = './MovingAI_CityMaps/xml/'
files = glob.glob(files_path + '*' + str(size_map) + '.xml')
for file in files:
    f = open(file)
    s = f.read()
    if 'NOT' in s or 'not' in s:
        print('no path')
        os.system('rm ' + file)
        os.system('rm ' + file[:-7] + 'img.png')
        continue
    r = re.findall(r'<row>(.+)<', s)
    map_img = np.zeros((size_map, size_map))
    for i in range(len(r)):
        row = r[i].split()
        for j in range(len(row)):
            if row[j] == '1':
                map_img[i][j] = 1
            if row[j] == '0':
                map_img[i][j] = 0

    imageio.imwrite(file[:-4] + "128.png", map_img.astype('uint8'))
