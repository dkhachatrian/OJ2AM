# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:34:43 2016

@author: David G. Khachatrian
"""

####################
##
##
##
#
#The OrientationJ-to-Anisotropy-Map (OJ2AM) to construct a 2- or 3-dimensional matrix representing the anisotropy values in an image (or z-stack of images) processed by OrientationJ in ImageJ.


#### To ensure the working directory starts at where the script is located...
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
dep = os.path.join(dname, 'dependencies')
os.chdir(dname)
####


import numpy as np
from PIL import Image
from lib import tools as t


np.set_printoptions(precision=2, suppress = True) #for easier-to-look-at numbbers when printing in pdb

outdir = os.path.join(dname, 'outputs') #directory for output files


#im, im_name = t.get_image(dep)

# We shall assume the image's HSV values came in the order: orientation, coherence, energy

im_name = 'swirl-analyzed-rgb.jpg'
im = Image.open(os.path.join(dep, im_name))

outmap_fname = im_name + ' aniso_map.dat'


#currently assuming the information is contained in the RGB layers
if im.mode != 'RGB':
    print("Mode isn't RGB!")
    

im_data = np.array(im)
im_data = im_data / 255 #normalized to [0,1] interval for all values

# bands = t.get_bands(im)
#cost_function = t.ori_cost


# build graph

aniso_map = t.Graph()

ind = np.ndindex(im_data.shape[:-1]) #allows loops over all but the last dimension (see below)

#add Nodes
j = 0

for i in ind:
    j += 1
    if j % 1000 == 0:
        print('Have gone through another 1000 data points. Iteration count: ' + str(int(j/1000)))
    coords = tuple(reversed(i)) # numpy arrays are indexed e_k, e_(k-1), ..., e_1
    #print('Working on coordinates ' + str(coords) + '...')
    node = t.Node(im_data[i], *coords) # (*coords) "unpacks" the iterable into individual values to pass in as arguments
    aniso_map.add_node(node)
    
# establish edges
aniso_map.make_connections()


#save maps made of particular images

outmap_path = os.path.join(outdir, outmap_fname)
if not os.path.lexists(outmap_path):
    with open(outmap_path) as outf:
        outf.write(aniso_map)


# Test points

start_coord = (2,3) #(x,y) coordinate
start_node = t.Node(im_data[start_coord], *reversed(start_coord))

end_coord = (10,39)
end_node = t.Node(im_data[end_coord], *reversed(end_coord))

# pathfinding algorithm 
pathway = t.optimize_path(graph = aniso_map, start = start_node, end = end_node)

print('Done!')
