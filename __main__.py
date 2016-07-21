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



#mode = t.choose_program_mode()
#im_name, orig_im = t.get_image()
#
#end_coords = t.get_coords(mode)
#start_coord = end_coords.pop(0)


paths_ll = [] #will be a list of (path_lists)

if len(g.end_coords) > 1:
    o.mat2path(g.im_name, g.start_coord, None) #run Dijkstra's algorithm on entire image and cache results so we don't have to redo for each endpoint

for end_coord in g.end_coords:
    paths_ll.append(o.mat2path(g.im_name, g.start_coord, end_coord)) #collect paths

overlay = g.orig_im.copy()

should_save_paths = t.prompt_saving_paths()

for i,path_list in enumerate(paths_ll): #paths limited by length of path_colors. If ever necessary, can expand selection.
    try:
        overlay = t.draw_path_onto_image(overlay, path_list, save_paths = should_save_paths, color = g.path_colors[i])
    except IndexError: #g.path_colors ran out. Default to yellow
        overlay = t.draw_path_onto_image(overlay, path_list, save_paths = should_save_paths, color = 'yellow')

t.prompt_saving_overlay_to_file(overlay, start = g.start_coord, end = g.end_coords)


print('Done!')