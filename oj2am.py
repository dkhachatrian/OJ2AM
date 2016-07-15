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
from lib import Graph as G
import sys

import pickle #save/cache Graphs made from images

from matplotlib import pyplot as plt # plot distributions



np.set_printoptions(precision=2, suppress = True) #for easier-to-look-at numbbers when printing in pdb

outdir = os.path.join(dname, 'outputs') #directory for output files



# ask for input data

graph_data = t.get_data(dep)


# ask for original image. TODO: move to tools

file_name = input("Please state the name of the file corresponding to the data in the previous input files, or enter nothing to quit: \n")        
while not os.path.isfile(os.path.join(dep, file_name)):
    if file_name == '':
        sys.exit()
    file_name = input("File not found! Please check the spelling of the filename input. Re-enter the name of the original image being analyzed (or enter nothing to quit): \n")

orig_im = Image.open(os.path.join(dep, file_name))
out_prefix = file_name


# Ask for start/end locations

start_coord = (2,3,0) #(x,y) coordinate

end_coord = (20,14,0)



#im_name = 'fft-swirl-analyzed-rgb-cropped.jpg'
#im = Image.open(os.path.join(dep, im_name))
#


# Paths for different possibly extant cached files

aniso_map_fname = '{0} start={1} end={2} aniso_map.p'.format(out_prefix, start_coord, end_coord)
aniso_map_path = os.path.join(outdir, aniso_map_fname)

#used cached data if already processed
if os.path.lexists(outmap_path): 
    with open(outmap_path, 'rb') as inf:
        aniso_map = pickle.load(inf)
        
#otherwise make Graph from scratch:

else:
    
    # build graph
    
    aniso_map = G.Graph()
    aniso_map.populate(graph_data)    
    

    #save maps made of particular images
    
    with open(outmap_path, 'wb') as outf:
        pickle.dump(aniso_map, outf, pickle.HIGHEST_PROTOCOL)
    
    
    
    # Below uses the "generic" implemetation of Graph. Runs slow -- O(n**2)
#    aniso_map = t.Graph()
#    
#    ind = np.ndindex(graph_data.shape[:-1]) #allows loops over all but the last dimension (see below)
#    
#    #add Nodes
#    j = 0
#    
#    for i in ind:
#        j += 1
#        if j % 1000 == 0:
#            print('Have gone through another 1000 data points. Iteration count: ' + str(int(j/1000)))
#        coords = tuple(reversed(i)) # numpy arrays are indexed e_k, e_(k-1), ..., e_1
#        #print('Working on coordinates ' + str(coords) + '...')
#        node = t.Node(graph_data[i], *coords) # (*coords) "unpacks" the iterable into individual values to pass in as arguments
#        aniso_map.add_node(node)
#        
#    # establish edges
#    aniso_map.make_connections()
#    
    

# now that Graph is made, load up Nodes in the Graph corresponding to the coordinates of interest

start_node = aniso_map.coord2node[start_coord]

if end_coord is None:
    end_node = None
else:
    end_node = aniso_map.coord2node[end_coord]




paths_info_fname = '{0} start={1} end={2} paths_info.p'.format(out_prefix, start_coord, end_coord)
paths_info_path = os.path.join(outdir, paths_info_fname)


preds_fname = '{0} start={1} end={2} preds.p'.format(out_prefix, start_coord, end_coord)
preds_path = os.path.join(outdir, preds_fname)



#
#
#graph_data = np.array(im)
#graph_data = graph_data / 255 #normalized to [0,1] interval for all values





# Test points


if os.path.lexists(paths_info_path) and os.path.lexists(preds_path):
    paths_info = pickle.load(paths_info_path)
    preds = pickle.load(preds_path)



else:
    paths_info, preds = t.Dijkstra(aniso_map, start_node, end_node)
    #dump data
    with open(paths_info_path, 'wb') as outf:
        pickle.dump(paths_info, outf, pickle.HIGHEST_PROTOCOL)
    
    with open(preds_path, 'wb') as outf:
        pickle.dump(preds, outf, pickle.HIGHEST_PROTOCOL)



if end_coord is not None:
    end_coord = input('Please specify where you would like to place the endpoint')

if end_coord is not None:
    path_list = t.optimal_path(preds, start_node, end_node)

    
    black = [0,0,0]
    white = [1,1,1] #RGB value for black pixels. Will be used to mark the optimal path on the original image
    
    #ind = np.ndindex(graph_data.shape[:-1]) #allows loops over all but the last dimension (see below)
    
    path_im_data = np.ones(graph_data.shape)
    
    #color in black the optimal path
    for index in path_list:
        index = index[:-1] #slicing due to this only being 2D, 
        index = tuple(reversed(index))
        index = tuple(np.subtract(index, np.ones(len(index))).astype(int))
        path_im_data[index] = black #slicing due to this only being 2D
    
    #save as new image, could be used as mask or for an overlay
    path_im_data = (path_im_data * 255).astype('uint8')
    #path_im_data = path_im_data.astype('uint8')
    path_im = Image.fromarray(path_im_data)
    
    should_overlay = input('Would you like the optimized path to be overlaid over the original image? (Y/N):\n')
    if should_overlay.lower() == 'y':
        overlaid = t.overlay(fg = path_im, bg = orig_im)
        path_im_fname = out_prefix + ' overlay.jpg'
        path_im_path = os.path.join(outdir, path_im_fname)
        overlaid.save(path_im_path)
    
    
    save_path_separately = input('Would you like the optimized path to be saved separetely as a grayscale image? (Y/N):\n')
    if save_path_separately.lower() == 'y':
        
        path_im_fname = out_prefix + ' optimized_path.jpg'
        path_im_path = os.path.join(outdir, path_im_fname)
        path_im.save(path_im_path)





print('Done!')
