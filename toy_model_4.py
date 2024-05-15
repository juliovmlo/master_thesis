# Import libraries
import os
import numpy as np
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot
from pybevc import PyBEVC
from utils import save_load, save_distributed_load, c2_to_node
from inertial_forces import inertial_loads_fun_v04
from cg_offset import get_cg_offset
from input import createProjectFolder

#%%
# Operations info
omega_rpm = 7 # [RPM]
omega = omega_rpm*np.pi/30 # [rad/s]
pitch_deg = 0 # [deg]
pitch_rad = np.deg2rad(pitch_deg) # [rad]

# Climate conditions
U0 = 8 # [m/s]

# Coupling parameters
epsilon = 1e-3
delta_u_rel = epsilon + 1
iter_max = 20

#%%
# Folders and file names
inputfolder = 'examples/input_iea15mw'
projectfolder = 'examples/project_folder'
c2_file = 'c2_pos.dat'
st_file = 'IEA_15MW_RWT_Blade_st_FPM.st'
ae_file = 'IEA_15MW_RWT_ae.dat'
pc_file = 'IEA_15MW_RWT_pc.dat'

createProjectFolder(inputfolder,projectfolder,c2_file,st_file,ae_file,pc_file)

#%% Initializing the structural model
# Model input json file  name
f_model_json = "config.json"

# Input files folder
inputfolder_stru = os.path.join(os.getcwd(),'examples/project_folder/stru')
mainfile_beam = os.path.join(inputfolder_stru,f_model_json)

# Initialize beam model
save_load([0], inputfolder_stru, onlyy=True) # Creates a force file
beam = ComplBeam(mainfile_beam)
cg_offset = get_cg_offset(beam)

#%% Initializing the aerodynamic model
bevc = PyBEVC()

inputfolder_aero = 'examples/project_folder/aero'
# I use the trick of using the HTC file
bevc.from_htc_file(os.path.join(inputfolder_aero,'htc_files/IEA-15-240-RWT-Onshore/htc/IEA_15MW_RWT_Onshore.htc'))
# bevc.from_ae_file(os.path.join(inputfolder_aero,'ae_file.dat'))
# bevc.from_pc_file(os.path.join(inputfolder_aero,'pc_file.dat'))
# bevc.from_c2_file(os.path.join(inputfolder_aero,'c2_pos.dat'))

# Setting inputs manually 
bevc.U0 = U0
# bevc.TSR = omega*bevc.R/U0
bevc.omega_rpm = omega_rpm
bevc.pitch_deg = pitch_deg
bevc.flag_a_CT = 2
hub_di = bevc.r_hub*2 # For inertial model

# Display all data as an Xarray object
xr_inp = bevc.as_xarray()

# Load matrix
aero_distr_forces = np.zeros((len(bevc.s),3))
aero_moments = np.zeros((len(bevc.s),3))
aero_loads = np.zeros((len(bevc.s),6))

#%% Initialize coupling
pos_new = np.zeros(beam.M_mat_full.shape[0]) # The initial pos are 0. This gives no inertial forces
tip_init_pos = beam.nodeLocations[-1]
old_tip_def = np.zeros(3)
new_tip_def = np.zeros(3)
tip_def_history = []
delta_history = []
iter = 0

#%% Coupling loop

while abs(delta_u_rel) > epsilon and iter < iter_max:
    # Update variables
    pos_old = pos_new
    old_tip_def = new_tip_def
    tip_def_history.append(np.linalg.norm(old_tip_def))
    iter += 1

    # Calculate aero loads
    res_bevc = bevc.run()
    
    load_names = ['fx_b', 'fy_b', 'fz_b', 'sec_mx_b', 'sec_my_b', 'sec_mz_b']
    for i, name in enumerate(load_names):
        aero_loads[:,i] = getattr(res_bevc, name)
    aero_loads = c2_to_node(beam,aero_loads) # Change location

    # Calculate inertial loads
    inert_load = -inertial_loads_fun_v04(pos_old,cg_offset,beam.M_mat_full,hub_di,omega,pitch_rad)
    inert_load = np.reshape(inert_load,(-1,6))

    # Add up loads and save
    load = inert_load.copy()
    load[:,3:] += aero_loads[:,3:]
    load_distr = np.zeros_like(load)
    load_distr[:,:3] = aero_loads[:,:3]

    save_load(load,inputfolder_stru)
    save_distributed_load(load_distr,inputfolder_stru)

    # Calculate deflections
    beam = ComplBeam(mainfile_beam)
    corotobj = CoRot(beam,numForceInc=10,max_iter=20)
    pos_new = np.reshape(corotobj.final_pos, (-1,6))
    tip_pos = pos_new[-1,0:3]
    new_tip_def = tip_pos - tip_init_pos

    # Save deflections

    # Calculate delta
    delta_u_rel = np.linalg.norm(new_tip_def - old_tip_def)/bevc.R
    delta_history.append(delta_u_rel)

    print("--- Iteration finished ---")
    print(f"Iteration {iter}")
    print(f"Delta = {delta_u_rel:.5f} m")
    print(f"Old tip def = {old_tip_def} m")
    print(f"Tip def: {new_tip_def} m")


