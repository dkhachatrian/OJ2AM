# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:14:18 2016

@author: David G. Khachatrian
"""

# Graph clas


# General Graph class


import numpy as np
from collections import defaultdict
import math
#from lib import tools as t
#from lib import globe as g
import time
import pdb

import pickle

from collections import deque

discriminant_dist = 2


#######################################
#### Node/Graph-Related Functions #####
#######################################

    
    

class Node: # Node will be 
    def __init__(self, data, x, y, z = 0):
        self.coord = (x,y,z) #unique ID for Node -- ensures each Node is different
        #self.id = self.coords
        self.orientation = data[0]
        self.coherence = data[1]
        self.energy = data[2]
    
    def __str__(self):
#        info = []
#        info.append("Coords: {0}".format(self.coords))
#        info.append("Orientation: {}".format(self.orientation))
#        info.append("Coherence: {}".format(self.coherence))
#        info.append("Energy: {}".format(self.energy))
#        return '\n'.join(info)
        info = "Coord: {0}\nOrientation: {1}\nCoherence: {2}\nEnergy: {3}".format(self.coord, self.orientation, self.coherence, self.energy)
        return info
#        print("Coords: " + str(self.coords))
#        print("Orientation: " + str(self.orientation))
#        print("Coherence: " + str(self.coherence))
#        print("Energy: " + str(self.energy))
        
    def __repr__(self):
        return self.__str__()

    #for comparisons of attributes
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __key(self):
        return (self.coord, self.orientation, self.coherence, self.energy)

    def __hash__(self):
        return hash(self.__key())






 #bidirectional. Due to simplicity in determining adjacency, edges are "hardwired" and requires the original NumPy array of data to initialize (in O(n) time)
#
class Graph:
    """
    Bidirectional graph.
    self.populate(data) and generate_neighbor_coords(coord, *bounds) are made specifically for the simpler case where the Graph is only connected to coordinates within math.sqrt(discriminant_dist_sq) of each other.
    Also, given: (1) the nature of the Nodes' connections being largely static for its use here; (2) the cost function being subject to change as the anisotropic cost function needs to be fine-tuned; and (3) the desire to cache the Graph object (instead of starting from scratch every time). Given these things, the weights of the edges between Nodes is not saved within the Graph, and instead can be obtained by calling the cost(node_a,node_b) function outside of this class.
    (Honestly, given the input data and connections, pathfinding could probably be performed using the original input data without having to create a Graph form of it. But the use of Nodes makes the information in the array easier to read.)
    """
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list) #key = from_node, values = to_node(s)
        self.costs = {}
        self.coord2node = {}
    
    def populate(self, data):
        self.create_adjacency_matrix(data)

    def add_node(self, n):
        self.nodes.add(n)

    def add_edge(self, a, b):
        if b.coord not in self.edges[a.coord]:
            self.edges[a.coord].append(b.coord)
        if a.coord not in self.edges[b.coord]:
            self.edges[b.coord].append(a.coord) #bidirectional
            
        cur_cost = cost(a,b)
        self.costs[(a.coord,b.coord)] = cur_cost
        self.costs[(b.coord,a.coord)] = cur_cost
        
#        cur_cost = cost(a,b)
#        self.costs[(a,b)] = cur_cost
#        self.costs[(b,a)] = cur_cost
#    
    def create_adjacency_matrix(self, data):
        """ Create adjacency list for a Graph. Simple case for image. """
        bounds = tuple(reversed(data.shape[:-1]))
        ind = np.ndindex(bounds)
    
        #create lookup table for coords -> nodes. O(n)
        for i in ind:
            #i = tuple(reversed(i)) #flip to (x,y,z)
            node = make_node(data, *i)
            self.add_node(node)
            if len(i) == 2:
                self.coord2node[(*i,0)] = node
            elif len(i) == 3:
                self.coord2node[i] = node
            else:
                print('Unexpected length of index in create_adjacency_matrix!')
        
        #establish connections, looking up the nodes by their coords
        # Average case O(n) * O(1) = O(n)
        
        ## TODO: CHANGE THIS is the connections ever become more complicated
        for cur_k in self.coord2node:
            neighbor_coords = generate_neighbor_coords(cur_k, *bounds)
            cur_node = self.coord2node[cur_k]
            for coord in neighbor_coords:
                neighbor_node = self.coord2node[coord]
                self.add_edge(cur_node, neighbor_node)
        
        
    def is_connected(self,a,b):
        #return (a in self.edges[b])
        
        # Below implementation requires a baked-in costs dictionary
        try:
            self.costs[(a.coord,b.coord)] #does an entry for its cost exist?
            return True
        except KeyError: #no associated cost --> not connected
            return False
        
        
#        for i in ind:
#            if i in coord2node:
#                cur_node = coord2node[i]
#            else:
#                cur_node = t.make_node(data, *i)
#                coord2node[i] = cur_node #don't need to flip index for coord2node as it is a temporary lookup while making the adjacency matrix. The edges dictionary will have the expected index order
#                
#            if cur_node not in self.nodes:
#                self.add_node(cur_node)
#            
#            neighbor_coords = generate_neighbor_coords(i)
#            for neighbor_coord in neighbor_coords:
#                if neighbor_coord in coord2node:
#                    self.add_edge(cur_node, coord2node[neighbor_coord])
#                else:
#                    neighbor_node = t.make_node(data, *neighbor_coord)
#                    coord2node[i] = neighbor_node
#                    
#        
#        
        
        
    
            





def generate_neighbor_coords(t, x_max, y_max, z_max = 1):
    """From original tuple, generate nonnegative-integer tuples within discrminant_dist of the center. Returns a list of such tuples."""

    dt = discriminant_dist
    

    if len(t) == 3:
        x,y,z = t
#        neighbors = [(i,j,k) for i in range(x - dt, x + dt) if lambda i: i >= 0 and i < x_max\
#                            for j in range(y-dt, y+dt) if j >= 0 and j < y_max\
#                            for k in range(z - dt, z + dt) if k >= 0 and k < z_max\
#                            if (i,j,k) != t\
#                            if sum((t-f)**2 for t,f in zip(t, (i,j,k))) < discriminant_dist**2\
#                            ]
    
    
        # have to use several nested loops because parameters like x_max and y_max do not fall under the scope of list comprehensions when used in conditional statements
        neighbors = []
        for i in range(x-dt, x+dt):
            if i >=0 and i < x_max:
                for j in range(y-dt,y+dt):
                    if j>=0 and j<y_max:
                        for k in range(z-dt,z+dt):
                            if k>=0 and k<z_max:
                                cur = (i,j,k)
                                if cur != t:
                                    if (sum((to_-f)**2 for to_,f in zip(t,cur)))<discriminant_dist**2:
                                        neighbors.append(cur)
                                elif cur == t:
                                    pass
        return neighbors
#    elif len(t) == 2:
#        x,y = t
#        neighbors = [(i,j,0) for i in range(x - dt, x + dt) if i >= 0 and i < x_max\
#                            for j in range(y-dt, y+dt) if j >= 0 and j < y_max\
#                            if (i,j) != t\
#                            if sum((t-f)**2 for t,f in zip(t, (i,j))) <=\
#                            discriminant_dist**2\
#                            ]
    else:
        print('Unexpected length of tuple for generate_neighbor_coords!')
    
    return neighbors




def make_node(im_data, x, y, z = 0):
    """ Given a NumPy array and coordinates x,y,(and z), generate a Node for use in a Graph. """
    if im_data.ndim == 3: #last dimension is data
        return Node(im_data[y,x], x, y)
    elif im_data.ndim == 4:
        return Node(im_data[z,y,x],x,y,z)


def access_coords(coords, data):
    """ Access coordinates in (x,y,z,...) format from a NumPy array. """
    return data[tuple(reversed(coords))] #dimensions are accessed 'backwards'


EPSILON = 0.1
PENALTY_COST = 10000000

import random

def cost(a,b):
    """ Given two adjacent Nodes, calculate the weight (determined by relative anisotropies, coherences, and energies). """
#    kappa = 1 #coherence parameter
#    epsilon = 1 #energy parameter
#    omega = 1 #orientation parameter
#    
    #dtheta = abs(b[vals_dict['ori']] - a[vals_dict['ori']]) #change in angle
    #thought process is that traversal along similar orientation is less costly
    
    ax = a.coord[0]
    ay = a.coord[1]

    if (ax > 100 or ax < 700) and (ay > 110):
        return PENALTY_COST
    else:
        return random.randint(0,100)
    
#    return random.randint(0, 100000)
#    movement_angle = get_angle(a,b)
#    dtheta = abs(a.orientation - movement_angle)
#    if dtheta > math.pi/2:
#        dtheta = math.pi - dtheta
#    theta_factor = (dtheta / (math.pi/2))**2
#    # bounded between 0 and 1
#    # 1 ==> lines spanned by movement_angle vector and orientation vector are orthogonal
#    # 0 ==> lines spanned by movement_angle vector and orientation vector are the same
#    # the thought is: the more along the orientation vector the movement is, the lower the cost
#    
#    
#    if (a.energy + b.energy)/2 > EPSILON:
#        result = theta_factor * (a.energy*(1-a.coherence) + b.energy*(1-b.coherence)) / (a.energy + b.energy) #bounded within [0,1]
#    else:
#        result = PENALTY_COST # try to prevent movement across areas with low/no energy (and so essentially isotropic areas)
#        #result = ((1-a.coherence) + (1-b.coherence))/2 #limit as energies approach zero
#        
##    if math.isnan(result):
##        print('Got NaN as a cost! Welp...')
##    else:
##        return result
#    return result


def get_angle(a,b):
    """ Given two Nodes with coordinates, return the angle made by their adjoining line segment with the horizontal.
    Bounded in [-pi/2, pi/2] to match bounds of structure tensor orientation.
    Currently only working from 2D vectors. (3D support will cost time.)"""
    
    #2D implementation
    tup = np.subtract(a.coord, b.coord)
    return math.atan(tup[1]/tup[0]) #arctan(y/x)
    
    
#    
#    tup = np.subtract(a.coord, b.coord)
#    np.angle()
#
#
##rotation matrix
#
#from numpy import cross, eye, dot
#from scipy.linalg import expm3, norm
#
#def M(axis, theta):
#    """ Return the rotational matrix along axis by angle theta.
#    From http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector """
##    Let a be the unit vector along axis, i.e. a = axis/norm(axis)
##and A = I × a be the skew-symmetric matrix associated to a, i.e. the cross product of the identity matrix with a
##Then M = exp(θ A) is the rotation matrix.
#    return expm3(cross(eye(3), axis/norm(axis)*theta))


###############################################
### Dijkstra's Algorithm and Pathfinding ######
###############################################


def Dijkstra(graph, start, end = None):
    """ Given a Graph with weighted edges (graph), determine the optimal path from the starting Node (start) to the target Node (end) (using Dijsktra's algorithm).
    Will yield the coordinates (one of Node's member variables, corresponding to its location in the original dataset) traversed from start to end.
    Will return:
        settled: a dictionary such that settled[settled_node] = optimal_cost
        predecessors: a ditionary"""
 
 
    total_nodes = len(graph.nodes)
    settled = {start.coord: 0}
    v_unsettled = {} #visited but unsettled Nodes
    pred = {start.coord: None} #dictionary of predecessors
    processing_queue = deque([start.coord])
    start = time.clock()
    num_notices = 0
    

    # logging purposes
    lens_of_newly_settled = []    
    
    
    while len(settled) < total_nodes:

        
        dt = time.clock() - start
        if dt > (num_notices+1) * 10:
            print("{0} seconds have passed since starting Dijkstra's algorithm. Currently, {1} out of {2} Nodes have been settled.".format(int(dt), len(settled), total_nodes))
            num_notices += 1
        
        while len(v_unsettled) == 0:
#            try:
            cur_coord = processing_queue.popleft()
#            current_node = graph.coord2node[cur_coord]
            #current_node = processing_queue.popleft()
#            except IndexError:
#                print('Processing_queue was empty!')

#            if cur_coord == (36,312,0): #still looks OK
#                pdb.set_trace()
#            elif cur_coord == (37,317,0): #start of straight-line
#                pdb.set_trace()
#            elif cur_coord == (51,331,0): #middle of straight-line
#                pdb.set_trace()

#            if current_node.coord == (36,312,0): #still looks OK
#                pdb.set_trace()
#            if current_node.coord == (37,317,0): #start of straight-line
#                pdb.set_trace()
#            if current_node.coord == (51,331,0): #middle of straight-line
#                pdb.set_trace()
            
            current_min_cost = settled[cur_coord]
            #current_min_cost = settled[current_node.coord]
            
            #update Nodes with potentially new lower min costs
            for coord in graph.edges[cur_coord]:
            #for coord in graph.edges[current_node.coord]:
                #node = graph.coord2node[coord]
                #additional_cost = cost(current_node, node)      
                additional_cost = graph.costs[(cur_coord, coord)]
                #additional_cost = graph.costs[(current_node.coord, node.coord)]
                new_cost = current_min_cost + additional_cost
                if coord not in settled:
                    if coord not in v_unsettled\
                    or new_cost < v_unsettled[coord]: #first visit of Node corresponding to coord; or new cost is lower
                        v_unsettled[coord] = new_cost
                        pred[coord] = cur_coord
#                if node not in settled:
#                    if node not in v_unsettled: #first visit of Node
#                        v_unsettled[node.coord] = new_cost
#                        pred[node.coord] = current_node.coord
#                    else: #update cost if lower
#                        if new_cost < v_unsettled[node]:
#                            v_unsettled[node.coord] = new_cost.coord
#                            pred[node.coord] = current_node.coord
            
#        if len(v_unsettled) == 0:
#            break #nothing left visited but unsettled!
        
        #find any newly settled Nodes
        min_v = min(v_unsettled.values())

#        #even if more than one Node has reached a minimum, will only settle one...
#        for c in v_unsettled:
#            if v_unsettled[c] == min_v:
#                settled[c] = v_unsettled[c]
#                pred[c] = cur_coord
#                v_unsettled.pop(c)
#                processing_queue.append(c)
#                break
        
        newly_settled = [n for n in v_unsettled if v_unsettled[n] == min_v]
        lens_of_newly_settled.append(len(newly_settled)) #logging/debugging

#        for x in random.randrange(len(newly_settled)): # so there isn't a preference for the upper-left corner (?)
#            c = newly_settled[x]
#        random.shuffle(newly_settled) #so there isn't a preference for the upper-left corner (?)
        for c in newly_settled:
        
#        for c in newly_settled:
#            if len(newly_settled) > 1:
#                pdb.set_trace()
#                print('Bifurcation in optimal path.')
            settled[c] = v_unsettled[c] #v_settled[n] == min_v (at least, should be)

            
            #pred[n] = current_node
            v_unsettled.pop(c) #remove from dictionary whose costs are compared to find new mins
            processing_queue.append(c) #add to queue of settled nodes whose neighbors should be updated
        
        #get out if we reached a specified end Node.
        # Note: if done this way, cannot arbitrarily choose any end Node afterward
        if end in newly_settled:
#        if end in settled:
            break
        
#    pdb.set_trace()
    return settled, pred    




def optimal_path(results, start, end):
    """ Using the results of Dijkstra's algorithm, construct the optimal path from the start and end coordinates. """
    
    #results = Dijkstra(graph,start,end)
    
    cur = end
    path = []
    
    while cur is not None: # in algorithm, the predecessor of start is None
        path.append(cur)
        cur = results[cur]
    
    return list(reversed(path)) #to go from S->E instead of E->S
    
    
    

    
#    # TODO: probably need to fix the logic here...
#    result = 0
#    for (before,after) in t.pairwise(coords_list):
#        if after is None:
#            return result
#        else:
#            n1 = graph.coord2node[before]
#            n2 = graph.coord2node[after]
#            try:
#                result += cost(n1,n2)
##                result += graph.costs[(before,after)]
#            except KeyError:
#                print("coords_list was invalid!")
#                return None
#    
#    
#    





