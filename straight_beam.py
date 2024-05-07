"""
Using a straight beam.
"""

# Import libraries
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot
from utils import save_load
from inertial_forces import inertial_loads_fun

# Model input json file name
f_model_json = "straight_beam_no_offset_static.json"

# Input files folder
inputfolder = os.path.join(os.getcwd(),'straight_beam')
mainfile = os.path.join(inputfolder,f_model_json)

# Extra geometry info
hub_di = 160 # Hub diameter [m]
# The original blade section is around 80 m away (160 diameter).

# - Instantiate beam
save_load([0], inputfolder, onlyy=True) # Creates a force file
beam = ComplBeam(mainfile)
original_nodel_loc = beam.nodeLocations

# - Get the radius location of nodes
r = beam.nodeLocations[:,2] # z-axis position

# - Get the mass matrix
# For non dynamic calculations the mass matrix is not calculated
f_model_json = "straight_beam_no_offset_dynamic.json"
mainfile_dyn = os.path.join(inputfolder,f_model_json)

# Calculate mass matrix
beam_dynamic = ComplBeam(mainfile_dyn)
M_mat_full = beam_dynamic.M_mat_full