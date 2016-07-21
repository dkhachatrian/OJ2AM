# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:41:09 2016

@author: David
"""

from lib import globe as g
from lib import Graph as G
import os
import sys
from PIL import Image
#from matplotlib import colors
import numpy as np
import itertools
from matplotlib import colors as c
import pickle
import time


#######################
#### UI Functions #####
#######################

def get_image():
    """
    Prompts user for name of image, looking in the dependencies directory.
    """
    while True:
        try:
            file_name = input("Please state the name of the file corresponding to the data to be input, or enter nothing to quit: \n")
            if file_name == '':
                sys.exit()
            im = Image.open(os.path.join(g.dep, file_name))
            break
        except FileNotFoundError:
            print("File not found! Please check the spelling of the filename input, and ensure the filename extension is written as well.")
            continue
        except IOError: #file couldn't be read as an Image
            print("File could not be read as an image! Please ensure you are typing the filename of the original image..")
            continue
        
    
    return file_name, im


def choose_program_mode():
    """
    Chooses how this program will run. Details in print statements below.
    """
    
    print("Hello!")
    print("Please indicate your preference:")
    while True:
        print("To draw multiple paths with endpoint fixed, type 'm'. You will then be asked to list the other desired coordinates.")
        print("If you will indicate only one pair of coordinates, please type 's'.")
        resp = input("Please type your choice now:\n")
        
        if resp.lower() == 'm':
            return g.MULTI
        elif resp.lower() == 's':
            return g.SINGLE
        elif resp.lower() == 'q':
            sys.exit()
        else:
            print("Input not recognized! Please re-read the instructions and try again.")


def get_coords(mode):
    """ Ask for start and end coordinates. Ensure they're in the image size."""
    
#    # TODO: collapse things down -- too much repeated code between modes...
#    
#    start_coord = None
#    end_coords = [] #will either end up containing one or multiple elements depending on mode
##    if mode == g.SINGLE:
##        end_coord = None
##    elif mode == g.MULTI:
##        end_coords = []
##    
    coords = []
    
    while True:
        coord_ok = True
        #print("(Input 'q' to quit.)")
        print("Selected image's size is {0}.".format(g.orig_im.size))
        try:
            if len(coords) == 0:
                tup = input("Please input the desired start coordinate:\n")
            else:
                if mode == g.SINGLE:
                    if len(coords) == 2:
                        break
                    tup = input("Please input the desired end coordinate:\n")
                elif mode == g.MULTI:
                    print("You have currently input {0} end coordinate(s).".format(len(coords)-1))
                    tup = input("Please input a desired end coordinate. Type 'c' to finish input and continue.\n")
                    if tup == 'c':
                        break
            
            if tup == 'q':
                sys.exit()
            
            tup.strip('() ')
            nums = [int(x) for x in tup.split(',')]
        except ValueError:
            print('Error! Numbers not entered. Please try again.')
            continue
    
    
        if len(nums) != len(g.orig_im.size):
            print('Error! Input coordinates do not match image dimensions. Please try again.')
            continue
        for i,num in enumerate(nums):
            if num < 0 or num >= g.orig_im.size[i]:
                print('Error! Input values were out of image-size bounds! Image size bounds is {0}. Please try again.'.format(g.orig_im.size))
                coord_ok = False
                break
        if coord_ok:
            if len(nums) == 2:
                tup = (*nums, 0)
            elif len(nums) == 3:
                tup = tuple(nums)
            coords.append(tup)
    
    
    start_coord = coords.pop(0)
    end_coords = coords
    return start_coord, end_coords
    
    
    
def make_neighbor_ll(coord_list):
    """
    From a list of coordinates, return a list of a list of coordinates. Each list of coordinates contains a coordinate from coord_list and its valid neighbors (that remain within the bounds of the original image).
    """

    coord_ll = []

    for coord in coord_list:
        clist = G.generate_neighbor_coords(coord, *g.orig_im.size)
        clist.append(coord)
        coord_ll.append(clist)
    
    return coord_ll
    
    
    
    

def prompt_user_about_neighbors():
    """ Ask user whether to draw paths for neighbors as well as the indicated startpoint. """
    
    while True:
        resp = input("Would you like to draw paths for nearby startpoints as well? [Y/N]:\n")
        if resp.lower() == 'y':
            return True
        elif resp.lower() == 'n':
            return False
        else:
            print("Input not recognized! Please respond with either 'Y', 'y', 'N', or 'n'.")




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
            #d_layer = np.around(d_layer, decimals = 3) #rarely are more than 3 decimal places needed -- just takes more time and space when left unrounded...
            data_list.append(d_layer) #delimiter for Text Images is tab

    #stack arrays

    data = np.stack(data_list, axis = -1) #axis = -1 makes data the last dimension
    np.around(data, decimals = 3, out = data)
    
    return data


def prompt_saving_paths():
    
    while True:
        tup = input("Would you like the individual paths to be saved separately? [Y/N]: \n")
        if tup.lower() == 'y':
            return True
        elif tup.lower() == 'n':
            return False
        else:
            print("Input not recognized! Please try again.")


    
def draw_paths_onto_image(orig_im, paths_ll, save_paths, color = None):
    """
    Draws path onto image data.
    Returns Images of the image with path overlaid, and the path alone as an image.
    """
    
    black = [0,0,0]
    white = [1,1,1] #RGB value for black pixels. Will be used to mark the optimal path on the original image
    yellow = [1,1,0]
    
    if color is None:
        color = 'yellow'
    
    color = c.hex2color(c.cnames[color])
    
    #ind = np.ndindex(graph_data.shape[:-1]) #allows loops over all but the last dimension (see below)
    mask_data = np.ones(tuple(reversed(orig_im.size)))
    path_im_data = np.ones((*reversed(orig_im.size),3)) #the '3' is for normalized RGB values at each pixel. Reversed because array dimensions in opposite order of image.shape tuple
    
    #color in black the optimal path
    for path_list in paths_ll:
        for index in path_list:
            index = index[:-1] #slicing due to this only being 2D; TODO: remove when using 3D
            index = tuple(reversed(index))
            index = tuple(np.subtract(index, np.ones(len(index))).astype(int))
            mask_data[index] = 0
            path_im_data[index] = color
    
    #save as new image, could be used as mask or for an overlay
    path_im_data = (path_im_data * 255).astype('uint8')
    mask_data = (mask_data*255).astype('uint8')
    #path_im_data = path_im_data.astype('uint8')
    path_im = Image.fromarray(path_im_data)
    mask_im = Image.fromarray(mask_data)
    
    
    overlaid = overlay(fg = path_im, bg = orig_im, mask = mask_im)
    
    if save_paths:
        path_im_fname = '{0} start={1} end={2} draw_neighbors={3} optimized_path.jpg'.format(g.out_prefix, path_list[0], path_list[-1], g.should_draw_neighbors)
        path_im_path = os.path.join(g.outdir, path_im_fname)
        path_im.save(path_im_path)
    
    return overlaid
#    return overlaid, path_im    
    


 
def prompt_saving_overlay_to_file(overlay, start, end):
    """
    """
    
    should_overlay = input('Would you like the optimized path(s) to be overlaid over the original image? (Y/N):\n')
    if should_overlay.lower() == 'y':
        # TODO: should change behavior depending on mode? Don't think it's necessary...
        path_im_fname = '{0} start={1} end={2} draw_neighbors={3} overlay.jpg'.format(g.out_prefix, start, end, g.should_draw_neighbors)
        path_im_path = os.path.join(g.outdir, path_im_fname)
        overlay.save(path_im_path)
    
    



def overlay(fg, bg, mask = None):
    """
    Overlay two images, using a mask. The mask defaults to path_im converted to 'L' mode.
    """
    if mask is None:
        mask = fg.convert('L')
    
    out_im = fg.copy()  
    
    out_im.paste(bg, (0, 0), mask) #this order because I elected to make the path black (==> 0 at path, 255 elsewhere)
    return out_im
    





#############################
#### Iterating Functions ####
#############################

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable, n=2)
    next(b, None)
    return zip(a, b)




#########################
#### Cache Functions ####
#########################


def load_map():
    """
    Loads up the aniso_map corresponding to the chosen image, or creates it (and caches it) if it doesn't exist.
    """
    
    ## Map Cache filenames
    aniso_map_fname = '{0} aniso_map.p'.format(g.out_prefix)
    aniso_map_path = os.path.join(cache_dir, aniso_map_fname)


    if aniso_map_fname in os.listdir(g.cache_dir):
        with open(os.path.join(g.cache_dir, aniso_map_fname), 'rb') as inf:
            start = time.clock()
            aniso_map = pickle.load(inf)
            end = time.clock()
            print('Loading a map with {0} Nodes took {1} seconds.'.format(len(aniso_map.nodes), end-start))
#                
#        for poss_path in g.aniso_map_paths:
#            try:
#                type(aniso_map)
#                break
#            except NameError:
#                if os.path.lexists(poss_path): 
#                    with open(poss_path, 'rb') as inf:
#                        start = time.clock()
#                        aniso_map = pickle.load(inf)
#                        end = time.clock()
#                        print('Loading a map with {0} Nodes took {1} seconds.'.format(len(aniso_map.nodes), end-start))
#                
    
    else:
        # ask for input data
        graph_data = get_data()
        # build graph
        aniso_map = G.Graph()
        start = time.clock()
        aniso_map.populate(graph_data)
        end = time.clock()
        print('Creating a map with {0} Nodes took {1} seconds.'.format(len(aniso_map.nodes), end-start))
    
    
        #save maps made of particular images
        
        with open(aniso_map_path, 'wb') as outf:
            pickle.dump(aniso_map, outf, pickle.HIGHEST_PROTOCOL)

    return aniso_map

def load_general_solution(im_name, root_coord):
    """
    Loads general solutions to the Graph corresponding to im_name starting/ending at root_coord, if it exists.
    Returns loaded data if exists in the order (paths_info,preds). Else returns (None, None)
    """
    
    paths_info_loaded = False
    preds_loaded = False
    
    
    for fname in os.listdir(g.cache_dir):
        if paths_info_loaded and preds_loaded:
            #specific_cache_loaded = True
            break
        with open(os.path.join(g.cache_dir,fname), 'rb') as inf:
            if g.out_prefix in fname and str(None) in fname and str(root_coord) in fname:
                if not paths_info_loaded and 'paths_info' in fname:
                    start = time.clock()
                    paths_info = pickle.load(inf)
                    end = time.clock()
                    print("Loading gen_paths_info took {0} seconds.".format(end-start))
                    paths_info_loaded = True
                if not preds_loaded and 'preds' in fname:
                    start = time.clock()
                    preds = pickle.load(inf)
                    end = time.clock()
                    print("Loading gen_preds took {0} seconds.".format(end-start))
                    preds_loaded = True
    
    try:
        return paths_info, preds
    except NameError:
        return None, None


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
    
    

    






