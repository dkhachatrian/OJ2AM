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

outdir = os.path.join(dname, 'outputs') #directory for output files
cache_dir = os.path.join(dname, 'cache')