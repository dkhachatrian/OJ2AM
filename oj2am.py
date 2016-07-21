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
import pdb

import pickle #save/cache Graphs made from images

from matplotlib import pyplot as plt # plot distributions


def mat2path(image_name, start_coord, end_coord, gen_paths_info = None, gen_preds = None):
    """
    Determines the optimal path (as determined by minimizing the cost as defined in Graph.py) to get from start to end for the Graph corresponding to image_name.
    image_name = original image name (already checked for existence)
    start_coord = start coordinate on image (padded to (x,y,z))
    end_coord = end coordinate
    
    Returns:
    path_list = a list of coordinates, in order, from start_coord to end_coord
    """
    if start_coord is None and end_coord is None:
        print("No coordinates given to mat2path!")
        pdb.set_trace()
    
    #general predecessors loaded for specific start/endpoint. So skip the rest
    if gen_paths_info is not None and gen_preds is not None:
        if (end_coord is not None and start_coord is not None):
            return G.optimal_path(gen_preds, start_coord, end_coord)
        else:
            return #don't need to recreate general solution
    
    
    np.set_printoptions(precision=2, suppress = True) #for easier-to-look-at numbbers when printing in pdb
    
    
#    orig_im = Image.open(os.path.join(g.dep, image_name))
    
    
    
    # to load up cache of specific traversals along g.aniso_map

    
    
    # if there's 'full' information regarding the start or end coordinates, load that instead of creating graph
    
    paths_info_loaded = False
    preds_loaded = False
        
    
    # there was no general cache. But let's see if there's specific cache
    for fname in os.listdir(g.cache_dir):
        if paths_info_loaded and preds_loaded:
            #specific_cache_loaded = True
            break
        with open(os.path.join(g.cache_dir,fname), 'rb') as inf:
            if g.out_prefix in fname and str(g.start_coord) in fname and str(end_coord) in fname: #specific route has already been asked for
                if not paths_info_loaded and 'paths_info' in fname:
                    paths_info = pickle.load(inf)
                    paths_info_loaded = True
                if not preds_loaded and 'preds' in fname:
                    preds = pickle.load(inf)
                    preds_loaded = True
    

    
    
    
    
    try:
        #path_list = G.optimal_path(preds, start_node, end_node)
        type(preds) # if already loaded, that means the map has been solved for one of the endpoints -- no need to do it again
    except NameError: #it doesn't know what 'preds' is...
    #    if os.path.lexists(g.paths_info_path) and os.path.lexists(g.preds_path):
    #        with open(g.paths_info_path, 'rb') as pa_info:
    #            with open(g.preds_path, 'rb') as pr_info:
    #                paths_info = pickle.load(pa_info)
    #                preds = pickle.load(pr_info)
    #    else:
    
        # load up Nodes in the Graph corresponding to the coordinates of interest (or None if solving the "general" case first)
        start_node = g.aniso_map.coord2node[start_coord]
        
        if end_coord is None:
            end_node = None
        else:
            end_node = g.aniso_map.coord2node[end_coord]
            
        #run algorithm
        start = time.clock()
        paths_info, preds = G.Dijkstra(g.aniso_map, start_node, end_node)
        end = time.clock()
        print('The pathfinding algorithm, working on a map with {0} Nodes, took {1} seconds.'.format(len(g.aniso_map.nodes), end-start))
        
        # load in results into general solution, if ran the algorithm in full
        if end_coord is None:
            gen_paths_info = paths_info
            gen_preds = preds
        
        #dump data
        paths_info_fname = '{0} start={1} end={2} paths_info.p'.format(g.out_prefix, start_coord, end_coord)
        paths_info_path = os.path.join(g.cache_dir, paths_info_fname)
        preds_fname = '{0} start={1} end={2} preds.p'.format(g.out_prefix, start_coord, end_coord)
        preds_path = os.path.join(g.cache_dir, preds_fname)
            
        with open(paths_info_path, 'wb') as outf:
            pickle.dump(paths_info, outf, pickle.HIGHEST_PROTOCOL)
        with open(preds_path, 'wb') as outf:
            pickle.dump(preds, outf, pickle.HIGHEST_PROTOCOL)
#    finally: #can figure out the path to take
#        path_list = G.optimal_path(preds, g.start_coord, g.end_coord)

    
    
#    if (end_coord is not None and start_coord is not None): #TODO: move the handling of drawing neighbors to __main__
#        # testing for "stability" of optimal path to small startpoint perturbations
#        path_list = []
#        if g.should_draw_neighbors:
#            for start in G.generate_neighbor_coords(start_coord, *g.orig_im.size):
#                if start in preds:
#                    path_list.extend(G.optimal_path(preds,start,end_coord)) #TODO: maybe make a list of lists and color-code different optimal paths? Probably only worth it if we notice large deviations in paths.
#                else:
#                    print("Neighbor coordinate not settled!")
#                    print("Re-run script with the end coordinate unspecified, then repeat your current request to fix this.")
#        path_list.extend(G.optimal_path(preds, start_coord, end_coord))
#    t.draw_path_onto_image(orig_im.size, path_list)

    if (end_coord is not None and start_coord is not None):
        return G.optimal_path(preds, start_coord, end_coord)

#
#    print('Done!')
