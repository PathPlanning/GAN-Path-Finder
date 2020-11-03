## Customized Data Generation

It is possible to tweak the dataset generation (e.g. create maps of different size, with more tasks per map etc.). In this case the following scripts should be run manually.

**First**, one needs to create maps without the paths (input images). This is done by invoking either `maps_gen.py` script in the following fashion.  

```
python ./data_generation/maps_gen.py --field_size 64 --density 0.2 --obstacles_num 5 --indent 3 --dataset_size 5000 --tasks_num 10 --random_shape False
```
The following parameters are supported: 
+ field_size - the size of the square grid (_default_ 64x64).
+ density - value from 0 to 1, the proportion of the obstacles on the grid (0.2 means ~20% of the map is obstacles).
+ obstacles_num - maximal number of obstacles (the actual sizes of the obstacles are calculated according to maximal number of obstacles and density).
+ indent - number of grid cells near the left/rigth edges of the map, where obstacles are not generated. In these grid columns start/goal locations will be chosen (with random vertical offset).
+ dataset_size - number of unique grids/maps with unique obstacles.
+ tasks_num - number of tasks (start/goal positions for each grid, ex. 5000 maps and 10 tasks per map makes 50000 images in the dataset).
+ random_shape - False to generate obstacles that are only rectangular in shape (the default setting of our data) or for True maps with obstacles of the shapes: rectangle, square, circle (which often look like diamond on a grid)

Resultant images (with obstacles and with the start and goal pixels highlighted) will be saved to the folder `./size_{field_size}/{density * 100}_den/`, _default_ `./size_64/20_den/`. Some auxiliry files (xml files describing the path planning instances) will be generated as well. They are needed at the next step when paths should be generated for each path planning task encoded by the generated image.

**Alternative First** is not to generate maps with random obstacles, but to read real maps from `.xml` files. In our case we had xml versions of several cities 1024x1024, 512x512 and 256x256 in size, that can be sliced, according to the dataset format (ex. 64x64, 128x128) and converted into `.png` files (see `city_maps.py` file for details).

**Second**, one needs to generate the corresponding ground-truth images, i.e. the images that besides the obstacles and the start-goal pixels show the shortest path from start to goal. To generate those images `path_build.py` script should be run. It runs A* under the hood and should be parameterized similarly to `maps_gen.py` (`random_maps_gen.py`).

```
python ./data_generation/path_build.py --field_size 64 --density 0.2 --obstacles_num 5 --indent 3
```
Technically this scipt invokes A* [binary file](https://github.com/PathPlanning/GAN-Path-Finder/blob/master/data_generation/AStar-JPS-ThetaStar) built for Linux, which is the part of the repository. In case of any problems with binary file, due to using different platform, binary file could be build directly from the [Astar](https://github.com/PathPlanning/AStar-JPS-ThetaStar) repository. This binary takes generated xml-files as the input and ouputs the xml-files that encode the solution (=have the A* path from start to goal). The latter are further converted to images (while the xml files, both input and output, are deleted from the directory). As a result the directory will contain now pairs of images: one encoding the task (obstacles + start-goal) and the other encoding the solution (obstacles + start-goal + path from start to goal).

An examples of such a pair of images (map size 64, rectangular obstacles, density 20%) is shown here: [examples](https://github.com/PathPlanning/GAN-Path-Finder/tree/master/examples/size_64/20_den).