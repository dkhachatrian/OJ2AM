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

discriminant_dist_sq = 3

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
    
    # establishing connections for our anisotropy graph is simple -- we use adjacent arrays of data
    def make_connections(self): # O(n**2)...
        for from_node in self.nodes:
            for to_node in self.nodes:
                if not self.is_connected(from_node, to_node) and self.should_be_connected(from_node, to_node):
                    self.add_edge(from_node, to_node, cost = cost(from_node, to_node))
            
    def is_connected(self,a,b):
        try:
            self.costs[(a,b)] #does an entry for its cost exist?
            return True
        except KeyError: #no associated cost --> not connected
            return False
            
    def should_be_connected(self, a, b):
        distance_sq = sum((t-f)**2 for t,f in zip(a.coords, b.coords))
        return (distance_sq < discriminant_dist_sq)
                    


def cost(a,b):
    """ Given two adjacent Nodes, calculate the weight (determined by relative anisotropies, coherences, and energies). """
#    kappa = 1 #coherence parameter
#    epsilon = 1 #energy parameter
#    omega = 1 #orientation parameter
#    
    #dtheta = abs(b[vals_dict['ori']] - a[vals_dict['ori']]) #change in angle
    #thought process is that traversal along similar orientation is less costly
    
    return (a.energy*(1-a.coherence) + b.energy*(1-b.coherence)) / (a.energy + b.energy)
    

def optimize_path(graph, start, end):
    """ Given a Graph with weighted edges (graph), determine the optimal path from the starting Node (start) to the target Node (end) (using Dijsktra's algorithm).
    Will yield the coordinates (one of Node's member variables, corresponding to its location in the original dataset) traversed from start to end.
    Will return a dictionary of all Nodes that were settled while getting from start to end, with keys being the Nodes and values being their costs.
    Note: this function does *not* explicitly determine the optimal costs to reach every Node! It stops once it reaches the 'end' Node."""
 
    epsilon = 0 # '0' may be changed to some epsilon, to be less sensitive to noise in data (or less-than-perfect cost functions)
    
 
    remaining_nodes = graph.nodes() #all unsettled nodes, visited or nonvisited
    unsettled = {} #visited and unsettled
    settled = {start: 0} #0 cost to get from start to start
    remaining_nodes.remove(start)
    
    yield start.coords #start-point
    
    current_node = start    
    
    while remaining_nodes:
        #update values for nodes adjacent to current node
        for adj_node in graph.edges[current_node]:
            if adj_node in settled:
                continue
            edge = graph.costs[(current_node, adj_node)]
            if adj_node not in unsettled:
                unsettled[adj_node] = edge
            else:
                new_cost = settled[current_node] + edge
                if new_cost < unsettled[adj_node]:
                    unsettled[adj_node] = new_cost
        #settle nodes
        m = min(unsettled.values())
        #TODO: what if multiple settle at once? How to decide which way to go a priori?
            # recursion that compares costs from bifurcating points to target?
        
        newly_settled = []
        for node in unsettled:
            if unsettled[node] <= m + epsilon: #if cost is the lowest among unsettled nodes, should be settled
                settled[node] = unsettled[node] #add to settled dict (with cost as value)
                remaining_nodes.remove(node) #remove node from remaining_nodes
                newly_settled.append(node) #checking to see if more than one Node settles at once...
                
        if len(newly_settled) > 1:
            graph_results = defaultdict(list)
            for tine in newly_settled:
                graph_results[optimize_path(graph, newly_settled, end)[end]].append(newly_settled) # key == the cost to get to the end from the newly settled node. value == the newly settled node. So we can find the min amongst the keys, and choose the corresponding node
            if len(graph_results[min(graph_results)]) > 1:
                pass #equal likelihood of choosing either path
            
            #find lowest-cost one
            
            #pass #TODO: figure out what we're doing
            # most likely have to perform recursion on each of the newly settled Nodes, to figure out which to yield for the path from start-to-end
            # but watch out for it bouncing back-and-forth between bifurcation fork and tine
            # simple cop-out is "just take the first Node and roll with it". But may not actually be true...
            
        elif len(newly_settled) >= 1: #just taking the first in the list
            current_node = newly_settled[0]
            yield current_node.coords
            
            if current_node == end: #reached the goal!
                return settled #returns dict of settled Nodes and their costs
        
            #otherwise we haven't. Continues to iterate
        









#####################################
#### UI.Image-Related Functions #####
#####################################




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










