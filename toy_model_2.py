"""
Toy model with the structural solver and inertial loads.
"""

# Import libraries
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot
from utils import save_load
from inertial_forces_v2 import inertial_loads_fun

# Model input json file name
f_model_json = "iea15mw_toy_model.json"

# Input files folder
inputfolder = os.path.join(os.getcwd(),'iea15mw')
mainfile = os.path.join(inputfolder,f_model_json)

# Extra geometry info
root_di = 5.2 # Root diameter [m]

# Operations info
rpm = 7
omega = rpm*np.pi/30
pitch_deg = 0
pitch_rad = np.deg2rad(pitch_deg)

# Main loop options
epsilon = 1e-3
delta_u = epsilon + 1
iter_max = 20

# Initialize loop

# - Instantiate beam
save_load([0], inputfolder, onlyy=True) # Creates a force file
beam = ComplBeam(mainfile)

# - Get the radius location of nodes
r = beam.nodeLocations[:,2] # z-axis position

# - Get the mass matrix
# For non dynamic calculations the mass matrix is not calculated
f_model_json = "iea15mw_dynamic.json"
mainfile_dyn = os.path.join(inputfolder,f_model_json)

# Calculate mass matrix
beam_dynamic = ComplBeam(mainfile_dyn)
M_mat_full = beam_dynamic.M_mat_full

# - Init position
force = r/r[-1]*100
save_load(force,inputfolder,onlyy=True)
beam = ComplBeam(mainfile)
corotobj = CoRot(beam,numForceInc=4,max_iter=20)

# - Init variables
tip_init_pos = np.reshape(corotobj.final_pos, (-1,6))[-1,0:3]
old_tip_def = 0 
new_tip_def =  np.linalg.norm(tip_init_pos)
tip_def_history = []
iter = 0

while abs(delta_u) > epsilon and iter < iter_max:
    # Update variables
    old_tip_def = new_tip_def
    tip_def_history.append(old_tip_def)
    iter += 1

    # Calculate inertial loads
    load = -inertial_loads_fun(corotobj.final_pos,root_di,M_mat_full,omega,pitch_rad)
    save_load(load,inputfolder)

    # Calculate deflections
    beam = ComplBeam(mainfile)
    corotobj = CoRot(beam,numForceInc=10,max_iter=20)
    final_pos = np.reshape(corotobj.final_pos, (-1,6))
    tip_pos = final_pos[-1,0:3]
    new_tip_def = np.linalg.norm(tip_pos - tip_init_pos)

    # Calculate delta
    delta_u = new_tip_def - old_tip_def

    print("--- Iteration finished ---")
    print(f"Iteration {iter}")
    print(f"Delta = {delta_u:.5f} m")
    print(f"Old tip def = {old_tip_def:.2f}")
    print(f"Tip def: {new_tip_def:.2f} m")

tip_def_history.append(new_tip_def)
print("---- Coupled ----")
print(f"Num. iterations: {iter}")
print(f"Tip def: {new_tip_def:.2f} m")
print(f"History tip def:")
print(f"\t{tip_def_history}")
print("Operation conditions:")
print(f"\tPitch angle: {pitch_deg} degrees")
print(f"\tRotor speed: {rpm} RPM")