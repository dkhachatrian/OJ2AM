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

import pickle #save/cache Graphs made from images

from matplotlib import pyplot as plt # plot distributions



np.set_printoptions(precision=2, suppress = True) #for easier-to-look-at numbbers when printing in pdb

outdir = os.path.join(dname, 'outputs') #directory for output files


#im, im_name = t.get_image(dep)

# We shall assume the image's HSV values came in the order: orientation, coherence, energy

im_name = 'fft-swirl-analyzed-rgb-cropped.jpg'
im = Image.open(os.path.join(dep, im_name))

outmap_fname = im_name + ' aniso_map.p' # 'p' for pickle file

outmap_path = os.path.join(outdir, outmap_fname)


im_data = np.array(im)
im_data = im_data / 255 #normalized to [0,1] interval for all values


#used cached data if already processed
if os.path.lexists(outmap_path):
    with open(outmap_path, 'rb') as inf:
        aniso_map = pickle.load(inf)
        
#otherwise make Graph from scratch:

else:
    #currently assuming the information is contained in the RGB layers
    if im.mode != 'RGB':
        print("Mode isn't RGB!")
        
    

    
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
    
    
    if not os.path.lexists(outmap_path):
        with open(outmap_path, 'wb') as outf:
            pickle.dump(aniso_map, outf, pickle.HIGHEST_PROTOCOL)


# Test points

start_coord = (2,3) #(x,y) coordinate
start_node = t.Node(im_data[tuple(reversed(start_coord))], *start_coord)

end_coord = (20,14)
end_node = t.Node(im_data[tuple(reversed(end_coord))], *end_coord)

# pathfinding algorithm
path_generator = t.optimize_path(graph = aniso_map, start = start_node, end = end_node, orig_data = im_data)
path_list = []


while True:
    try:
        path_list.append(next(path_generator))
    except StopIteration:
        break

path_dict = path_list[-1] #last thing path_generator sends is the dict
path_list.pop() #pop_list now has, in order, the optimal coordinates to be traveled

#dump data
with open(os.path.join(outdir, 'path_dict.p'), 'wb') as outf:
    pickle.dump(path_dict, outf, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(outdir, 'path_list.p'), 'wb') as outf:
    pickle.dump(path_list, outf, pickle.HIGHEST_PROTOCOL)


black = [0,0,0]
white = [1,1,1] #RGB value for black pixels. Will be used to mark the optimal path on the original image

#ind = np.ndindex(im_data.shape[:-1]) #allows loops over all but the last dimension (see below)


#color in black the optimal path
for index in path_list:
    index = tuple(reversed(index[:-1])) #slicing due to this only being 2D
    index = tuple(np.subtract(index, np.ones(len(index))).astype(int))
    im_data[index] = black #slicing due to this only being 2D

#save as new image
im_data = im_data * 255
im_data = im_data.astype('uint8')
out_im = Image.fromarray(im_data)
out_imfname = im_name + ' with_optimized_path.jpg'
out_impath = os.path.join(outdir, out_imfname)
out_im.save(out_impath)


print('Done!')
