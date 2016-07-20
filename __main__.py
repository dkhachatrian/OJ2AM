# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:13:34 2016

@author: David G. Khachatrian
"""



#### To ensure the working directory starts at where the script is located...
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
dep = os.path.join(dname, 'dependencies')
os.chdir(dname)
####




from lib import globe as g
from lib import tools as t
import oj2am as o
from PIL import Image


# create directories if necessary

if not os.path.isdir(g.outdir):
    os.mkdir(g.outdir)

if not os.path.isdir(g.cache_dir):
    os.mkdir(g.cache_dir)



mode = t.choose_program_mode()
im_name, orig_im = t.get_image()

end_coords = t.get_coords(mode)
start_coord = end_coords.pop(0)


paths_ll = [] #will be a list of (path_lists)

if len(end_coords) > 1:
    o.mat2path(im_name, start_coord, None) #run Dijkstra's algorithm for all Nodes so we don't have to redo for each endpoint

for end_coord in end_coords:
    paths_ll.append(o.mat2path(im_name,start_coord,end_coord)) #collect paths

overlay = orig_im.copy()

should_save_paths = t.prompt_saving_paths()

for path_list in paths_ll:
    overlay = t.draw_path_onto_image(orig_im.shape, path_list, color = None, save_paths = should_save_paths)

t.prompt_saving_overlay_to_file(overlay, start = start_coord, end = end_coords)


print('Done!')