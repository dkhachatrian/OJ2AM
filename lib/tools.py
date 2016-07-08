# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:41:09 2016

@author: David
"""

import os
import sys
from PIL import Image
from matplotlib import colors
import numpy as np
from collections import defaultdict

vals_dict = {'ori': 0, 'coh': 1, 'ener': 2} #maps labels to elements in array


#######################################
#### Node/Graph-Related Functions #####
#######################################


class Node: # Node will be 
    def __init__(self, data, x, y, z = 0):
        self.coords = (x,y,z) #unique ID for Node -- ensures each Node is different
        self.orientation = data[0]
        self.coherence = data[1]
        self.energy = data[2]

class Graph: #bidirectional
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list) #key = from_node, values = to_node(s)
        self.costs = {}

    def add_node(self, n):
        self.nodes.add(n)

    def add_edge(self, a, b, cost):
        self.edges[a].append(b)
        self.edges[b].append(a) #bidirectional
        self.costs[(a,b)] = cost
        self.costs[(b,a)] = cost
        


def cost(a,b):
    """ Given two adjacent Nodes, calculate the weight (determined by relative anisotropies, coherences, and energies). """
#    kappa = 1 #coherence parameter
#    epsilon = 1 #energy parameter
#    omega = 1 #orientation parameter
#    
    #dtheta = abs(b[vals_dict['ori']] - a[vals_dict['ori']]) #change in angle
    #thought process is that traversal along similar orientation is less costly
    
    return (a.energy*(1-a.coherence) + b.energy*(1-b.coherence)) / (a.energy + b.energy)
    




















def get_image(dep):
    """ Prompts user for name of image. (Pass in the location of the dependencies folder.) Returns open Image, and image name. """

    image_name = input("Please state the full filename for the image of interest (located in the dependencies directory of this script), or enter nothing to quit: \n")
    
#. To process all images in the dependencies direcctory as a z-stack, type 'all'
    
    while not os.path.isfile(os.path.join(dep, image_name)):
        if image_name == '':
            sys.exit()
        image_name = input("File not found! Please check the spelling of the filename input. Re-enter filename (or enter no characters to quit): \n")
    
    im_orig = Image.open(os.path.join(dep, image_name))

    return im_orig, image_name

def get_bands(image):
    """ Pull out bands from image.
    (We assume bands correspond to the order orientation, coherence, energy)"""
    
    data = np.array(image)

    data /= 255 #each of the values are on 8-bit scales. (They don't all necessarily reach 255)
    
    hsv = colors.rgb_to_hsv(data)
    #bands = np.dsplit(hsv)
    bands = []
    
    for band in np.dsplit(hsv, hsv.shape[-1]):
        #bands.append(np.squeeze(band)) #have the 1-element arrays be scalars
        bands.append(band)
        
    return bands










