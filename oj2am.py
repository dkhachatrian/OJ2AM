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


im, im_name = t.get_image(dep)
# We shall assume the image's HSV values came in the order: orientation, coherence, energy

im_data = np.array(im)

im_data = im_data / 255 #normalized to [0,1] interval for all values

# bands = t.get_bands(im)

cost_function = t.ori_cost


aniso_map = t.Graph()

ind = np.ndindex(im_data.shape[:-1]) #allows loops over all but the last dimension (see below)

# build graph

for i in ind:
    node = t.Node(im_data(i), *i) # (*i) "unpacks" the tuple into individual values to pass in as arguments
    t.aniso_map.add_node(node)
    


start_coord = (2,3) #(x,y) coordinate
end_coord = (10,39)







# pathfinding algorithm 
pathway = t.predict_path(graph = im_data, start = start_coord, end = end_coord, cost_f = cost_function)


