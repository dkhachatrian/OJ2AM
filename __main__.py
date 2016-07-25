# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:13:34 2016

@author: David G. Khachatrian
"""

# python -WError -m pdb <__main__.py>   ## to stop at Warnings


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
import pdb
#from PIL import Image


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

# TODO: Deal with neighbors in a clean way...

paths_ll = [] #will be a list of (path_lists) for related sets of paths
paths_lll = [] #will contain a list of paths_ll's

gen_paths_info, gen_preds = t.load_general_solution(g.im_name, root_coord = g.start_coord)

if len(g.end_coords) > 1 and (gen_paths_info is None and gen_preds is None):
    o.mat2path(g.im_name, g.start_coord, None) #run Dijkstra's algorithm on entire image and cache results so we don't have to redo for each endpoint
    gen_paths_info, gen_preds = t.load_general_solution(g.im_name, root_coord = g.start_coord)

for end_coord_list in g.end_coords_ll:
    paths_ll = []
    for end_coord in end_coord_list:
        paths_ll.append(o.mat2path(g.im_name, g.start_coord, end_coord, gen_paths_info, gen_preds)) #collect paths
    paths_lll.append(paths_ll)

overlay = g.orig_im.copy()

#should_save_paths = t.prompt_saving_paths()
should_save_paths = True

# TODO: fix colors...

for i,paths_ll in enumerate(paths_lll):
    try:
        overlay = t.draw_paths_onto_image(overlay, paths_ll, save_paths = should_save_paths, color = g.path_colors[i])
    except IndexError: #g.path_colors ran out. Default to yellow
        overlay = t.draw_paths_onto_image(overlay, paths_ll, save_paths = should_save_paths, color = 'yellow')


t.prompt_saving_overlay_to_file(overlay, start = g.start_coord, end = g.end_coords)


print('Done!')