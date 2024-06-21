# Import libraries
import os
import numpy as np
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot
from pybevc import PyBEVC
from inertial_forces import inertial_loads_fun_v04
from input import createProjectFolder
from utils import (
    save_load,save_distributed_load,
    c2_to_node,
    save_deflections,
    get_cg_offset,
    save_results,
)

#%%
# Operations info
# omega_rpm = 7.46 # [RPM]
# omega = omega_rpm*np.pi/30 # [rad/s]
omega = 0.596902
omega_rpm = omega*30/np.pi
print(f'{omega_rpm = }')
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
inputfolder = 'examples/input_iea15mw_ang_50'
projectfolder = 'examples/project_folder_ang_50'

createProjectFolder(inputfolder,projectfolder)

#%% Initializing the structural model
# Model input json file  name
f_model_json = "config.json"

# Input files folder
inputfolder_stru = os.path.join(os.getcwd(),projectfolder+'/stru')
mainfile_beam = os.path.join(inputfolder_stru,f_model_json)

# Initialize beam model
beam = ComplBeam(mainfile_beam)
cg_offset = get_cg_offset(beam) # TODO: update it in the loop

#%% Initializing the aerodynamic model
bevc = PyBEVC()

inputfolder_aero = projectfolder + '/aero'
# I use the trick of using the HTC file
# bevc.from_htc_file('examples/input_iea15mw_ang_stiff/IEA_15MW_RWT_ae_nsec_50_stiff.htc',model_path='./')
bevc.from_htc_file(os.path.join(inputfolder, 'IEA_15MW_RWT_ae_nsec_50.htc'),model_path='./')
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

# Load matrix
aero_distr_forces = np.zeros((len(bevc.s),3))
aero_moments = np.zeros((len(bevc.s),3))
aero_loads = np.zeros((len(bevc.s),6))
aero_loads_34c = np.zeros((len(bevc.s),6))
pos_34c =  np.zeros((len(bevc.s),3))

#%% Initialize coupling
pos_new = np.zeros((beam.numNode,6)) # The initial node positions are given, but not the rotations
pos_new[:,:3] = beam.nodeLocations
c2_pos_new = beam.c2Input[:,1:4]
c2_pos_ini = beam.c2Input[:,1:4]
twist_ini = beam.c2Input[:,-1] # [deg]
tip_init_pos = beam.nodeLocations[-1]
old_tip_def = np.zeros(3)
new_tip_def = np.zeros(3)
defl_hist = np.zeros((beam.numNode,6,0))
pos_hist = np.zeros((beam.numNode,3,0)) # Position of the C2 nodes.
pos_34c_hist = np.zeros((beam.numNode,3,0)) # Position of the 3/4 chord nodes
inertial_loads_hist = np.zeros((beam.numNode,6,0)) # Inertial loads in 
aero_force_hist = np.zeros((beam.numNode,6,0)) # Aerodynamic forces in C2. Distributed forces and punctual moments
aero_force_34c_hist = np.zeros((beam.numNode,6,0)) # The same but the moments in the 3/4 chord
alpha_deg_hist = np.zeros((beam.numNode,0))
power_hist = np.zeros((0))
thrust_hist = np.zeros((0))
tip_def_history = []
delta_history = []
iter = 0

#%% Coupling loop

while abs(delta_u_rel) > epsilon and iter < iter_max:
    # Update variables
    pos_old = pos_new
    c2_pos_old = c2_pos_new
    old_tip_def = new_tip_def
    tip_def_history.append(np.linalg.norm(old_tip_def))
    iter += 1

    # Calculate aero loads
    res_bevc = bevc.run()
    
    load_names = ['fx_b', 'fy_b', 'fz_b', 'sec_mx_b', 'sec_my_b', 'sec_mz_b']
    for i, name in enumerate(load_names):
        aero_loads[:,i] = getattr(res_bevc, name)

    load_names = ['fx_b', 'fy_b', 'fz_b', 'sec_mx_34c', 'sec_my_34c', 'sec_mz_34c']
    for i, name in enumerate(load_names):
        aero_loads_34c[:,i] = getattr(res_bevc, name)

    power = res_bevc.power*3
    thrust = res_bevc.thrust*3
    
    names = ['x_34c_r','y_34c_r','z_34c_r']
    for i, name in enumerate(names):
        pos_34c[:,i] = getattr(res_bevc, name)

    # Move aero loads to elastic centre
    node_pos = pos_old[:,:3]
    aero_loads_n = c2_to_node(c2_pos_old,node_pos,aero_loads) # Change location
    aero_loads_n = aero_loads # Ignoring 'c2_to_node'

    # Calculate inertial loads

    inert_load = -inertial_loads_fun_v04(pos_old,cg_offset,beam.M_mat_full,hub_di,omega,pitch_rad)
    inert_load = np.reshape(inert_load,(-1,6))

    # Add up loads and save
    load = inert_load.copy()
    load[:,3:] += aero_loads_n[:,3:]
    load_distr = np.zeros_like(load)
    load_distr[:,:3] = aero_loads_n[:,:3]

    save_load(load,inputfolder_stru)
    save_distributed_load(load_distr,inputfolder_stru)

    # Calculate deflections
    beam = ComplBeam(mainfile_beam)
    corotobj = CoRot(beam,numForceInc=10,max_iter=20)
    pos_new = np.reshape(corotobj.final_pos, (-1,6))
    tip_pos = pos_new[-1,0:3]
    new_tip_def = tip_pos - tip_init_pos

    # Save deflections
    # Find new position of c2 and twist. Then create the c2_pos file
    defl =  np.reshape(corotobj.total_disp,(-1,6))
    # defl_nodes,_,defl_twist = np.split(defl,[3,5],axis=1)
    c2_pos_new = c2_pos_ini + defl[:,:3]
    twist_new = twist_ini + np.rad2deg(defl[:,-1]) 

    c2_file_dict = {'x': c2_pos_new[:,0], 'y': c2_pos_new[:,1], 'z': c2_pos_new[:,2], 'twist_deg': twist_new}

    c2_file_new = np.column_stack((c2_pos_new,twist_new))

    # Update aero model
    bevc.from_dict(c2_file_dict)

    # Calculate delta
    delta_u_rel = np.linalg.norm(new_tip_def - old_tip_def)/bevc.R
    delta_history.append(delta_u_rel)

    pos_hist =  np.concatenate((pos_hist,np.expand_dims(c2_pos_new, axis=2)),axis=2)
    pos_34c_hist = np.concatenate((pos_34c_hist,np.expand_dims(pos_34c, axis=2)),axis=2)
    defl_hist = np.concatenate((defl_hist,np.expand_dims(defl, axis=2)),axis=2)
    aero_force_hist = np.concatenate((aero_force_hist,np.expand_dims(aero_loads, axis=2)),axis=2)
    aero_force_34c_hist = np.concatenate((aero_force_hist,np.expand_dims(aero_loads_34c, axis=2)),axis=2)
    alpha_deg_hist = np.concatenate((alpha_deg_hist,np.expand_dims(res_bevc.alpha_deg, axis=1)),axis=1)
    power_hist = np.concatenate((power_hist,np.expand_dims(power, axis=0)),axis=0)
    thrust_hist = np.concatenate((thrust_hist,np.expand_dims(thrust, axis=0)),axis=0)


    print("--- Iteration finished ---")
    print(f"Iteration {iter}")
    print(f"Delta = {delta_u_rel:.5f} m")
    print(f"Old tip def = {old_tip_def} m")
    print(f"Tip def: {new_tip_def} m")


results_dict = {
    'span': beam.scurve*beam.scurveLength,
    'pos_hist': pos_hist,
    'pos_34c_hist': pos_34c_hist,
    'defl_hist': defl_hist,
    'aero_force_hist': aero_force_hist,
    'aero_force_34c_hist': aero_force_34c_hist,
    'alpha_deg_hist': alpha_deg_hist,
    'power_hist': power_hist,
    'thrust_hist': thrust_hist,
}

save_results(results_dict,os.path.join(projectfolder,'res/results.json'))

