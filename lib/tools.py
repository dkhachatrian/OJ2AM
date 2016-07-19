# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:41:09 2016

@author: David
"""

from lib import globe as g
import os
import sys
from PIL import Image
#from matplotlib import colors
import numpy as np
import itertools


#######################
#### UI Functions #####
#######################

def get_image():
    """
    Prompts user for name of image, looking in the dependencies directory.
    """
    file_name = input("Please state the name of the file corresponding to the data in the previous input files, or enter nothing to quit: \n")        
    while not os.path.isfile(os.path.join(g.dep, file_name)):
        if file_name == '':
            sys.exit()
        file_name = input("File not found! Please check the spelling of the filename input. Re-enter the name of the original image being analyzed (or enter nothing to quit): \n")
    
    return file_name, Image.open(os.path.join(g.dep, file_name))


def get_data():
    """
    Prompts user for names of files corresponding to outputs of OrientationJ's parameters: orientation, coherence, and energy.
    Input files are searched for in the script's dependencies folder.
    Input files must haev been saved as a Text Image using ImageJ.
    
    Returns a NumPy array of the data stacked such that the final axis has the data in order [orientation, coherence, energy].
    """

    print('Hello! Please place relevant files in the dependencies directory of this script. Please have the files saved as a "Text Image" in ImageJ.')
    
    data_list = []
    data_names = ['orientation', 'coherence', 'energy']    


    

    while len(data_list) < 3:
        file_name = input("Please state the name of the file corresponding to the " + str(data_names[len(data_list)]) + " for the image of interest, or enter nothing to quit: \n")        
        while not os.path.isfile(os.path.join(g.dep, file_name)):
            if file_name == '':
                sys.exit()
            file_name = input("File not found! Please check the spelling of the filename input. Re-enter the name of the file corresponding to the " + str(data_names[len(data_list)]) + " for the image of interest (or enter nothing to quit): \n")
        with open(os.path.join(g.dep, file_name), 'r') as inf:
            d_layer = np.loadtxt(inf, delimiter = '\t')
            d_layer = np.around(d_layer, decimals = 3) #rarely are more than 3 decimal places needed -- just takes more time and space when left unrounded...
            data_list.append(d_layer) #delimiter for Text Images is tab

    #stack arrays

    data = np.stack(data_list, axis = -1) #axis = -1 makes data the last dimension
    
    return data


def overlay(fg, bg, mask = None):
    """
    Overlay two images, using a mask. The mask defaults to path_im converted to 'L' mode.
    """
    if mask is None:
        mask = fg.convert('L')
    
    out_im = fg.copy()  
    
    out_im.paste(bg, (0, 0), mask) #this order because I elected to make the path black (==> 0 at path, 255 elsewhere)
    return out_im
    
    
def draw_path_onto_image(image_shape, path_list):
    """
    Draws path onto image data, rendering an overlay if desired.
    """
    
    black = [0,0,0]
    white = [1,1,1] #RGB value for black pixels. Will be used to mark the optimal path on the original image
    yellow = [1,1,0]
    
    #ind = np.ndindex(graph_data.shape[:-1]) #allows loops over all but the last dimension (see below)
    mask_data = np.ones(tuple(reversed(image_shape)))
    path_im_data = np.ones((*reversed(image_shape),3)) #the '3' is for normalized RGB values at each pixel. Reversed because array dimensions in opposite order of image.shape tuple
    
    #color in black the optimal path
    for index in path_list:
        index = index[:-1] #slicing due to this only being 2D; TODO: remove when using 3D
        index = tuple(reversed(index))
        index = tuple(np.subtract(index, np.ones(len(index))).astype(int))
        mask_data[index] = 0 #slicing due to this only being 2D
        path_im_data[index] = yellow
    
    #save as new image, could be used as mask or for an overlay
    path_im_data = (path_im_data * 255).astype('uint8')
    mask_data = (mask_data*255).astype('uint8')
    #path_im_data = path_im_data.astype('uint8')
    path_im = Image.fromarray(path_im_data)
    mask_im = Image.fromarray(mask_data)
    
    should_overlay = input('Would you like the optimized path to be overlaid over the original image? (Y/N):\n')
    if should_overlay.lower() == 'y':
        overlaid = overlay(fg = path_im, bg = g.orig_im, mask = mask_im)
        path_im_fname = '{0} start={1} end={2} multiple_paths={3} overlay.jpg'.format(g.out_prefix, g.start_coord, g.end_coord, g.should_draw_neighbors)
        path_im_path = os.path.join(g.outdir, path_im_fname)
        overlaid.save(path_im_path)
    
    
    save_path_separately = input('Would you like the optimized path to be saved separetely as a grayscale image? (Y/N):\n')
    if save_path_separately.lower() == 'y':
        
        path_im_fname = '{0} start={1} end={2} multiple_paths={3} optimized_path.jpg'.format(g.out_prefix, g.start_coord, g.end_coord, g.should_draw_neighbors)
        path_im_path = os.path.join(g.outdir, path_im_fname)
        path_im.save(path_im_path)

 








#############################
#### Iterating Functions ####
#############################

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable, n=2)
    next(b, None)
    return zip(a, b)












#
#vals_dict = {'ori': 0, 'coh': 1, 'ener': 2} #maps labels to elements in array
#
#discriminant_dist_sq = 3
#
########################################
##### Node/Graph-Related Functions #####
########################################
#
#
#class Node: # Node will be 
#    def __init__(self, data, x, y, z = 0):
#        self.coords = (x,y,z) #unique ID for Node -- ensures each Node is different
#        self.orientation = data[0]
#        self.coherence = data[1]
#        self.energy = data[2]
#    
#    def info(self):
#        print("Coords: " + str(self.coords))
#        print("Orientation: " + str(self.orientation))
#        print("Coherence: " + str(self.coherence))
#        print("Energy: " + str(self.energy))
#
#    #for comparisons of attributes
#    def __eq__(self, other):
#        if isinstance(other, self.__class__):
#            return self.__dict__ == other.__dict__
#        else:
#            return False
#
#    def __ne__(self, other):
#        return not self.__eq__(other)
#        
#    def __key(self):
#        return (self.coords, self.orientation, self.coherence, self.energy)
#
#    def __hash__(self):
#        return hash(self.__key())
#
#
#
#
#
#
#
#class Graph: #bidirectional
#    def __init__(self):
#        self.nodes = set()
#        self.edges = defaultdict(list) #key = from_node, values = to_node(s)
#        self.costs = {}
#
#    def add_node(self, n):
#        self.nodes.add(n)
#
#    def add_edge(self, a, b, cost):
#        self.edges[a].append(b)
#        self.edges[b].append(a) #bidirectional
#        self.costs[(a,b)] = cost
#        self.costs[(b,a)] = cost
#    
#    # establishing connections for our anisotropy graph is simple -- we use adjacent arrays of data
#    def make_connections(self): # O(n**2)...
#        for from_node in self.nodes:
#            print('Making connections for Node with coordinates ' + str(from_node.coords) + '...')
#            for to_node in self.nodes:
#                if should_be_connected(from_node, to_node) and not self.is_connected(from_node, to_node):
#                    self.add_edge(from_node, to_node, cost = cost(from_node, to_node))
#            
#    def is_connected(self,a,b):
#        try:
#            self.costs[(a,b)] #does an entry for its cost exist?
#            return True
#        except KeyError: #no associated cost --> not connected
#            return False
#            
#def should_be_connected(self, a, b):
#    if a == b:
#        return False #no need to be connected with oneself...
#    distance_sq = sum((t-f)**2 for t,f in zip(a.coords, b.coords))
#    return (distance_sq < discriminant_dist_sq)
#                    
#
#def make_node(im_data, x, y, z = 0):
#    if im_data.ndim == 3: #last dimension is data
#        return Node(im_data[y,x], x, y)
#    elif im_data.ndim == 4:
#        return Node(im_data[z,y,x],x,y,z)
#
#
#def access_coords(coords, data):
#    """ Access coordinates in (x,y,z,...) format from a NumPy array. """
#    return data[tuple(reversed(coords))] #dimensions are accessed 'backwards'
#
#
#def cost(a,b):
#    """ Given two adjacent Nodes, calculate the weight (determined by relative anisotropies, coherences, and energies). """
##    kappa = 1 #coherence parameter
##    epsilon = 1 #energy parameter
##    omega = 1 #orientation parameter
##    
#    #dtheta = abs(b[vals_dict['ori']] - a[vals_dict['ori']]) #change in angle
#    #thought process is that traversal along similar orientation is less costly
#    
#    if a.energy + b.energy > 0:
#        result = (a.energy*(1-a.coherence) + b.energy*(1-b.coherence)) / (a.energy + b.energy)
#    else:
#        result = ((1-a.coherence) + (1-b.coherence))/2 #limit as energies approach zero
#        
#    if math.isnan(result):
#        print('Got NaN as a cost! Welp...')
#    else:
#        return result
#    
    
    

    






