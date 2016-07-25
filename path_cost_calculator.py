# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:46:02 2016

@author: David G. Khachatrian

Cost Calculator

Given a grayscale image with colored (yellow) path drawn over it, compute the cost as determined in Graph.cost(a,b)
"""

import numpy as np
from PIL import Image
from matplotlib import colors as c
from lib import Graph as G
import pdb


def get_path_from_image(im):
    """
    Uses colored path on otherwise grayscale image to return a list of coordinates.
    """
    marker = c.hex2color(c.cnames['yellow']) 
    
    im_data = np.array(im)
    im_data = im_data / 255 #'uint8' to 0-1 range
    
    coords = []
    
    index = np.ndindex(im_data.shape[:-1])
    #pdb.set_trace()
    
    for ind in index:
        if np.array_equal(im_data[ind], marker):
        #if tuple(im_data[ind]) == marker:
        #if tuple(im_data[ind]) != [1.0,1.0,1.0]:
            ind = tuple(reversed(ind)) #there's a flipping of dimension order between array and image...
            if len(ind) == 2:
                ind = (*ind, 0)
            coords.append(ind)

    return coords
    
def cost_through_path(graph, coords_list):
    """
    Given a graph and a list of coordinates/Nodes to traverse, returns the cost associated with traversing the Nodes in the order specified in the list, or 'None' if there is a traversal across unconnected Nodes is attempted.
    """
    # hardly the most efficient. But ideally, path lengths won't be >>1000 so won't take too long  
    
    result = 0
    accounted_for = []
    for coord in coords_list:
        node = graph.coord2node[coord]
        for connected_node in graph.edges[node]:
            pair = (node.coords, connected_node.coords)
            rev_pair = (connected_node.coords, node.coords) #avoiding double-counting...
            
            if connected_node.coords in coords_list and pair not in accounted_for:
                #order doesn't matter since bidirectional with equivalent costs
                result += G.cost(node, connected_node)
                accounted_for.append(pair)
                accounted_for.append(rev_pair) #avoiding double-counting...
                
    return result
        
    