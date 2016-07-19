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
import sys


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







orig_im = Image.open(os.path.join(dep, file_name))

out_prefix = file_name


# Ask for start/end locations
# t.prompt_coords


def get_coords():
    """ Ask for start and end coordinates. Ensure they're in the image size."""
    
    coords = []
    n_dict = {0: 'start', 1: 'end'}
    
    while len(coords) < 2:
        coord_ok = True
        print("(Input 'q' to quit.)")
        print("Selected image's size is {0}.".format(orig_im.size))
        try:
            if len(coords) == 0:
                tup = input("Please input the desired {0} coordinates:\n".format(n_dict[len(coords)]))
            else:
                print("Please input the desired {0} coordinates.".format(n_dict[len(coords)]))
                print("Input nothing to create a cached version of pathfinding to (or from) the previously designated start coordinate.")
                tup = input("Enter input now:\n")
        
            if tup == 'q':
                sys.exit()
                
            if len(coords) == 1 and tup is '':
                coords.append(None)
                break
                
            tup = tup.strip('() ')
            nums = [int(x) for x in tup.split(',')]
        except ValueError:
            print("Error! Numbers not entered. Please try again.")
            continue
        
        if len(nums) != len(orig_im.size):
            print('Error! Input coordinates do not match image dimensions. Please try again.')
            continue
        for i,num in enumerate(nums):
            if num < 0 or num >= orig_im.size[i]:
                print('Error! Input values were out of image-size bounds! Image size bounds is {0}. Please try again.'.format(orig_im.size))
                coord_ok = False
                break
        if coord_ok:
            if len(nums) == 2:
                tup = (*nums, 0)
            elif len(nums) == 3:
                tup = tuple(nums)
            coords.append(tup)

    return coords


#Hardcoding
#
#start_coord = (24,333,0)
##start_coord = (114,180,0)
##end_coord = None
##end_coord = (330,440,0)
#end_coord = (574,134,0)




endpoints = get_coords()
start_coord = endpoints[0]
end_coord = endpoints[1]




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

should_draw_neighbors = prompt_user_about_neighbors()


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
