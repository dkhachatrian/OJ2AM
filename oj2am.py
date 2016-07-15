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
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#dep = os.path.join(dname, 'dependencies')
#os.chdir(dname)
####

from lib import globe as g

import numpy as np
from PIL import Image
from lib import tools as t
from lib import Graph as G
import sys
import time

import pickle #save/cache Graphs made from images

from matplotlib import pyplot as plt # plot distributions



np.set_printoptions(precision=2, suppress = True) #for easier-to-look-at numbbers when printing in pdb






# TODO: Ask for image name. Coordinates. Then look at cache for possibly useful files






orig_im = Image.open(os.path.join(g.dep, g.file_name))





# if there's 'full' information regarding the start or end coordinates, load that instead of creating graph

cache_loaded = False

for fname in os.listdir(g.cache_dir):
    if g.out_prefix in fname and str(None) in fname:
        for coord in (g.start_coord, g.end_coord):
            if str(coord) in fname:
        #        if 'aniso_map' in fname:
        #            aniso_map = pickle.load(fname)
                if 'paths_info' in fname:
                    paths_info = pickle.load(fname)
                    cache_loaded = True
                if 'preds' in fname:
                    preds = pickle.load(fname)
                    cache_loaded = True
        


#im_name = 'fft-swirl-analyzed-rgb-cropped.jpg'
#im = Image.open(os.path.join(dep, im_name))



# Paths for different possibly extant cached files

#if there already exists a "full" map with



#used cached data if already processed

#if not cache_loaded:
for poss_path in g.aniso_map_paths:
    try:
        type(aniso_map)
        break
    except NameError:
        if os.path.lexists(poss_path): 
            with open(poss_path, 'rb') as inf:
                start = time.clock()
                aniso_map = pickle.load(inf)
                end = time.clock()
                print('Loading a map with {0} Nodes took {1} seconds.'.format(len(aniso_map.nodes), end-start))
        
#otherwise make Graph from scratch:

else:
    # ask for input data
    graph_data = t.get_data()
    # build graph
    
    aniso_map = G.Graph()
    aniso_map.populate(graph_data)    
    

    #save maps made of particular images
    
    with open(g.aniso_map_path, 'wb') as outf:
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

start_node = aniso_map.coord2node[g.start_coord]

if g.end_coord is None:
    end_node = None
else:
    end_node = aniso_map.coord2node[g.end_coord]





try:
    #path_list = G.optimal_path(preds, start_node, end_node)
    type(preds) # if already loaded, that means using a more "general" amp
except NameError: #it doesn't know what 'preds' is...
#    if os.path.lexists(g.paths_info_path) and os.path.lexists(g.preds_path):
#        with open(g.paths_info_path, 'rb') as pa_info:
#            with open(g.preds_path, 'rb') as pr_info:
#                paths_info = pickle.load(pa_info)
#                preds = pickle.load(pr_info)
#    else:
        start = time.clock()
        paths_info, preds = G.Dijkstra(aniso_map, start_node, end_node)
        end = time.clock()
        print('The pathfinding algorithm, working on a map with {0} Nodes, took {1} seconds.'.format(len(aniso_map.nodes), end-start))
        #dump data
        with open(g.paths_info_path, 'wb') as outf:
            pickle.dump(paths_info, outf, pickle.HIGHEST_PROTOCOL)
        with open(g.preds_path, 'wb') as outf:
            pickle.dump(preds, outf, pickle.HIGHEST_PROTOCOL)
finally: #can figure out the path to take
    path_list = G.optimal_path(preds, g.start_coord, g.end_coord)



if g.end_coord is not None:
    t.draw_path_onto_image(orig_im.size, path_list)





print('Done!')
