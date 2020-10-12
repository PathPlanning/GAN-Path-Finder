import numpy as np
import math
from PIL import Image
import os
import subprocess
import glob
import imageio
import re

files_path = './visualization_dense_64/'
files = glob.glob(files_path + '*.xml')

focal_astar = {1 : {'nodes': [], 'steps': [], 'length': []},
               1.5 : {'nodes': [], 'steps': [], 'length': []},
               2 : {'nodes': [], 'steps': [], 'length': []},
               5 : {'nodes': [], 'steps': [], 'length': []},
               10 : {'nodes': [], 'steps': [], 'length': []}}

for file in files:
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

        print(steps[0], nodes[0], length[0])

        focal_astar[w]['steps'] += [int(steps[0])]
        focal_astar[w]['nodes'] += [int(nodes[0])]
        focal_astar[w]['length'] += [float(length[0])]

        os.system('rm ' + file[:-4] + '_log.xml')

np.save('focal_astar_results.npy', focal_astar)

with open('results_astar_vs_focal.txt', 'w') as f:
    f.writelines(["steps focal = %s\n" % np.array(focal_astar[w]['steps']).mean() for w in [1, 1.5, 2, 5, 10]])
    f.writelines(["nodes focal = %s\n" % np.array(focal_astar[w]['nodes']).mean() for w in [1, 1.5, 2, 5, 10]])
    f.writelines(["length focal = %s\n" % np.array(focal_astar[w]['length']).mean() for w in [1, 1.5, 2, 5, 10]])
