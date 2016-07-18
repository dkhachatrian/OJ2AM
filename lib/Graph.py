# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:14:18 2016

@author: David
"""

# Graph clas


# General Graph class


import numpy as np
from collections import defaultdict
import math


from collections import deque

discriminant_dist = 2


#######################################
#### Node/Graph-Related Functions #####
#######################################

    
    

class Node: # Node will be 
    def __init__(self, data, x, y, z = 0):
        self.coords = (x,y,z) #unique ID for Node -- ensures each Node is different
        self.orientation = data[0]
        self.coherence = data[1]
        self.energy = data[2]
    
    def info(self):
        print("Coords: " + str(self.coords))
        print("Orientation: " + str(self.orientation))
        print("Coherence: " + str(self.coherence))
        print("Energy: " + str(self.energy))

    #for comparisons of attributes
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __key(self):
        return (self.coords, self.orientation, self.coherence, self.energy)

    def __hash__(self):
        return hash(self.__key())






 #bidirectional. Due to simplicity in determining adjacency, edges are "hardwired" and requires the original NumPy array of data to initialize (in O(n) time)
class Graph:
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
        self.edges[a].append(b)
        self.edges[b].append(a) #bidirectional
        cur_cost = cost(a,b)
        self.costs[(a,b)] = cur_cost
        self.costs[(b,a)] = cur_cost
    
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
        for cur_k in self.coord2node:
            neighbor_coords = generate_neighbor_coords(cur_k, *bounds)
            cur_node = self.coord2node[cur_k]
            for coord in neighbor_coords:
                neighbor_node = self.coord2node[coord]
                self.add_edge(cur_node, neighbor_node)
        
        
    def is_connected(self,a,b):
        try:
            self.costs[(a,b)] #does an entry for its cost exist?
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



def cost(a,b):
    """ Given two adjacent Nodes, calculate the weight (determined by relative anisotropies, coherences, and energies). """
#    kappa = 1 #coherence parameter
#    epsilon = 1 #energy parameter
#    omega = 1 #orientation parameter
#    
    #dtheta = abs(b[vals_dict['ori']] - a[vals_dict['ori']]) #change in angle
    #thought process is that traversal along similar orientation is less costly
    
    if a.energy + b.energy > 0:
        result = (a.energy*(1-a.coherence) + b.energy*(1-b.coherence)) / (a.energy + b.energy)
    else:
        result = ((1-a.coherence) + (1-b.coherence))/2 #limit as energies approach zero
        
    if math.isnan(result):
        print('Got NaN as a cost! Welp...')
    else:
        return result





###############################################
### Dijkstra's Algorithm and Pathfinding ######
###############################################


def Dijkstra(graph, start, end = None):
    """ Given a Graph with weighted edges (graph), determine the optimal path from the starting Node (start) to the target Node (end) (using Dijsktra's algorithm).
    Will yield the coordinates (one of Node's member variables, corresponding to its location in the original dataset) traversed from start to end.
    Will return:
        settled: a dictionary such that settled[settled_node] = (optimal_cost, node_traversed_just_prior_to_reaching_settled_node)
        predecessors: a ditionary"""
 
 
    total_nodes = len(graph.nodes)
    settled = {start: 0}
    v_unsettled = {} #visited but unsettled Nodes
    pred = {start.coords: None} #dictionary of predecessors
    processing_queue = deque([start])
    
    while len(settled) < total_nodes:
        while len(v_unsettled) == 0:
            try:
                current_node = processing_queue.popleft()
            except IndexError:
                print('Processing_queue was empty!')
                
            current_min_cost = settled[current_node]
            
            #update Nodes with potentially new lower min costs
            for node in graph.edges[current_node]:
                additional_cost = graph.costs[(current_node, node)]
                new_cost = current_min_cost + additional_cost
                if node not in settled:
                    if node not in v_unsettled: #first visit of Node
                        v_unsettled[node] = new_cost
                        pred[node.coords] = current_node.coords
                    else: #update cost if lower
                        if new_cost < v_unsettled[node]:
                            v_unsettled[node] = new_cost
                            pred[node.coords] = current_node.coords
            
#        if len(v_unsettled) == 0:
#            break #nothing left visited but unsettled!
        
        #find any newly settled Nodes
        min_v = min(v_unsettled.values())
        newly_settled = [n for n in v_unsettled if v_unsettled[n] == min_v]
        
        for n in newly_settled:
#            if len(newly_settled) > 1:
#                print('Bifurcation in optimal path.')
            settled[n] = v_unsettled[n] #v_settled[n] == min_v (at least, should be)

            
            #pred[n] = current_node
            v_unsettled.pop(n) #remove from dictionary whose costs are compared to find new mins
            processing_queue.append(n) #add to queue of settled nodes whose neighbors should be updated
        
        #get out if we reached a specified end Node.
        # Note: if done this way, cannot arbitrarily choose any end Node afterward
        if end in newly_settled:
            break
        
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
    
    
    





