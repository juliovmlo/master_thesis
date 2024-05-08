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
from inertial_forces import inertial_loads_fun, inertial_loads_fun_v04
from cg_offset import get_cg_offset

# Model input json file name
f_model_json = "iea15mw_toy_model.json"

# Input files folder
inputfolder = os.path.join(os.getcwd(),'iea15mw')
mainfile = os.path.join(inputfolder,f_model_json)

# Extra geometry info
hub_di = 7.94 # Hub diameter [m]

# Operations info
rpm = 7
omega = rpm*np.pi/30
pitch_deg = 0
pitch_rad = np.deg2rad(pitch_deg)

# Main loop options
epsilon = 1e-3
delta_u_rel = epsilon + 1
iter_max = 20

# Initialize loop

# - Instantiate beam
save_load([0], inputfolder, onlyy=True) # Creates a force file
beam = ComplBeam(mainfile)
original_nodel_loc = beam.nodeLocations
cg_offset = get_cg_offset(beam)

# - Get the radius location of nodes
r = beam.nodeLocations[:,2] # z-axis position

# - Get the mass matrix
# For non dynamic calculations the mass matrix is not calculated
f_model_json = "iea15mw_dynamic.json"
mainfile_dyn = os.path.join(inputfolder,f_model_json)

# Calculate mass matrix
beam_dynamic = ComplBeam(mainfile_dyn)
M_mat_full = beam_dynamic.M_mat_full

# - Init variables
pos_new = np.zeros(M_mat_full.shape[0]) # The initial pos are 0. This gives no inertial forces
tip_init_pos=beam.nodeLocations[-1]
old_tip_def = np.zeros(3)
new_tip_def = np.zeros(3)
tip_def_history = []
delta_history = []
iter = 0

while abs(delta_u_rel) > epsilon and iter < iter_max:
    # Update variables
    pos_old = pos_new
    old_tip_def = new_tip_def
    tip_def_history.append(np.linalg.norm(old_tip_def))
    iter += 1

    # Calculate inertial loads
    load = -inertial_loads_fun_v04(pos_old,cg_offset,M_mat_full,hub_di,omega,pitch_rad)
    save_load(load,inputfolder)

    # Calculate deflections
    beam = ComplBeam(mainfile)
    corotobj = CoRot(beam,numForceInc=10,max_iter=20)
    pos_new = np.reshape(corotobj.final_pos, (-1,6))
    tip_pos = pos_new[-1,0:3]
    new_tip_def = tip_pos - tip_init_pos

    # Calculate delta
    delta_u_rel = np.linalg.norm(new_tip_def - old_tip_def)/r[-1]
    delta_history.append(delta_u_rel)

    print("--- Iteration finished ---")
    print(f"Iteration {iter}")
    print(f"Delta = {delta_u_rel:.5f} m")
    print(f"Old tip def = {old_tip_def}")
    print(f"Tip def: {new_tip_def} m")

tip_def_history.append(np.linalg.norm(new_tip_def))
print("---- Coupled ----")
print(f"Num. iterations: {iter}")
print("Operation conditions:")
print(f"\tPitch angle: {pitch_deg} degrees")
print(f"\tRotor speed: {rpm} RPM")
print(f"Final tip def: {new_tip_def} m")
print(f"History tip def. module:")
print(f"\t{tip_def_history}")
print(f"History delta ({epsilon = }):")
print(f"\t{delta_history}")


# Plot final result versus original state

fig, axs = plt.subplots(2,1,figsize=(14,10),sharex=True, layout="tight")
plot_title = ["r_x", "r_y"]
ylabel = ["x", "y"]

for i, ax in enumerate(axs):
    ax.plot(original_nodel_loc[:,2], original_nodel_loc[:,i],marker='o', label=f"Original")
    ax.plot(pos_new[:,2], pos_new[:,i],marker='o', label=f"Final")
    ax.set_ylabel(ylabel[i])

plt.grid()
plt.xlabel("Span, z [m]")
plt.legend()
plt.show()


