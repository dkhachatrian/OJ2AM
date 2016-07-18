# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:03:29 2016

@author: David G. Khachatrian
"""

### "Global" variables for the duration of a single execution of this script


#### To ensure the working directory starts at where the script is located...
import os
abspath = os.path.abspath(__file__) #in lib directory
dname = os.path.dirname(os.path.dirname(abspath)) #twice to get path to main directory
dep = os.path.join(dname, 'dependencies')
os.chdir(dname)
####

from lib import tools as t
from PIL import Image

outdir = os.path.join(dname, 'outputs') #directory for output files
cache_dir = os.path.join(dname, 'cache')




# ask for original image. TODO: move to tools

#file_name, orig_im = t.get_image()


# quick test
#file_name = 'fft-swirl-analyzed-rgb-cropped.jpg' #test-cases
#start_coord = (2,3,0) #(x,y) coordinate
#end_coord = (20,14,0)
# harder test
file_name = 'Pair2_NSC008_M6_DiI_aligned_cropped_falsecolor.jpg'
start_coord = (114,180,0)
end_coord = None
#end_coord = (330,440,0)

orig_im = Image.open(os.path.join(dep, file_name))

out_prefix = file_name


# Ask for start/end locations
# t.prompt_coords








# Cache filenames
aniso_map_paths = []

for coord in (start_coord, end_coord):
    aniso_map_fname = '{0} aniso_map.p'.format(out_prefix)    
    #aniso_map_fname = '{0} initial={1} aniso_map.p'.format(out_prefix, coord)
    aniso_map_path = os.path.join(cache_dir, aniso_map_fname)
    aniso_map_paths.append(aniso_map_path)



paths_info_fname = '{0} start={1} end={2} paths_info.p'.format(out_prefix, start_coord, end_coord)
paths_info_path = os.path.join(cache_dir, paths_info_fname)


preds_fname = '{0} start={1} end={2} preds.p'.format(out_prefix, start_coord, end_coord)
preds_path = os.path.join(cache_dir, preds_fname)
