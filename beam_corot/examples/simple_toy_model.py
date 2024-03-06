"""This couple model used CoRot and a toy aerodynamic model.

The aerodynamic model applies a lineal distributed load: F(r,L)= L*r/R

The lift, L, parameter is calculate as function of the tip deflection. It is a simple
concave parabola: L(tip_def) = -A*(tip_def - tip_def_opt)**2 + L_max

"""
# Import libraries
import os
import sys
sys.path.append(r"C:\Users\Public\OneDrive - Danmarks Tekniske Universitet\Dokumenter\ICAI\DTU\_Thesis\repos\beam-corotational")
import numpy as np
from matplotlib import pyplot as plt
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot

def aeroload (tip_def, r):
    """
    The aerodynamic model applies a lineal distributed load: F(r,L)= L*r/R

    The lift, L, parameter is calculate as function of the tip deflection. It is a simple
    concave parabola: L(tip_def) = -A*(tip_def - tip_def_opt)**2 + L_max

    Input:
        tip_def float: deflection of tip of the blade
        r np.array: the radial positions of the blade where to apply the load
    return:
        laod np.array: load for each node
    """
    tip_def_opt = 1 # [m]
    lift_max = 1e4
    A = 100

    lift = -A*(tip_def -tip_def_opt)**2 + lift_max

    load = lift*r/r[-1]

    return load

def save_load (load, folder):

    force_line = []
    force_line += "# Node_num    Fx      Fy      Fz      Mx      My      Mz \n"
    for node_i,f_node in enumerate(load):
        force_line +=  f"{node_i+1:4d}  0.0  {f_node:>15.6e}  0.0  0.0  0.0  0.0 \n"

    with open(os.path.join(folder,"force.dat"), 'w') as out_file:
        for line in force_line:
            out_file.write(line)

# Model input json file name
f_model_json = "iea15mw_toy_model.json"

# Input files folder
inputfolder = os.path.join(os.getcwd(),'iea15mw')
mainfile = os.path.join(inputfolder,f_model_json)

epsilon = 1e-3
delta_u = epsilon + 1
iter_max = 20

# Initialize loop

# - Instantiate beam
save_load([0], inputfolder) # Creates a force file
beam = ComplBeam(mainfile)

# - Get the radius location of nodes
r = beam.nodeLocations[:,2] # z-axis position

# - Init variables
tip_init_pos = beam.nodeLocations[-1,1]
old_tip_def = 0.5 # to give initial load (avoid convergance issues)
new_tip_def = 0.5
tip_def_history = []
iter = 0

while abs(delta_u) > epsilon and iter < iter_max:
    # Update variables
    old_tip_def = new_tip_def
    tip_def_history.append(old_tip_def)
    iter += 1

    # Calculate aerodynamic loads
    load = aeroload(old_tip_def,r)
    save_load(load,inputfolder)

    # Calculate deflections
    beam = ComplBeam(mainfile)
    corotobj = CoRot(beam,numForceInc=10,max_iter=20)
    tip_pos = corotobj.final_pos[-2]
    new_tip_def = tip_pos - tip_init_pos

    # Calculate delta
    delta_u = new_tip_def - old_tip_def

    print("--- Iteration finsihed ---")
    print(f"Iteration {iter} Delta = {delta_u:.5f}")
    print(f"Tip def: {new_tip_def:.2f} m")
    print(f"Tip load: {load[-1]:.2f} N/m")

print(tip_def_history)


