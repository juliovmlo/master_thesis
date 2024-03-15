"""Here the inertial forces are implemented.
"""
import numpy as np
#%%
pitch_mat = lambda pitch_rad: np.array([
                                        [np.cos(pitch_rad), -np.sin(pitch_rad), 0],
                                        [np.sin(pitch_rad), np.cos(pitch_rad),  0],
                                        [0,                 0,                  1],
                                        ])

vec = np.array([1,1,1])

ang_rad = np.pi/2 # 90 degrees

print(vec)
print(pitch_mat(ang_rad)@vec)

# %%
# Import libraries
import os
import sys
# sys.path.append(r"C:\Users\Public\OneDrive - Danmarks Tekniske Universitet\Dokumenter\ICAI\DTU\_Thesis\repos\beam-corotational")
import numpy as np
from matplotlib import pyplot as plt
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot

def aeroload (tip_def, r):
    """
    The aerodynamic model applies a lineal distributed load: F(r,L)= L*r/R

    The lift, L, parameter is calculate as function of the tip deflection. It is a simple
    concave linear: L(tip_def) = -A*abs(tip_def) + L_max

    Input:
        tip_def float: deflection of tip of the blade
        r np.array: the radial positions of the blade where to apply the load
    return:
        laod np.array: load for each node
    """
    tip_def_opt = 1 # [m]
    lift_max = 1e4
    A = 1000

    # Linear
    lift = -A*abs(tip_def) + lift_max

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

# Initialize beam model
save_load([0], inputfolder) # Creates a force file
beam = ComplBeam(mainfile)

# Get some loads
r = beam.nodeLocations[:,2] # z-axis position
load = aeroload(0.5,r)
save_load(load,inputfolder)

# Calculate deflections
beam = ComplBeam(mainfile)
corotobj = CoRot(beam,numForceInc=10,max_iter=20)
 
# For non dynamic calculations the mass matrix is not calculated
f_model_json = "iea15mw_dynamic.json"
inputfolder = os.path.join(os.getcwd(),'iea15mw')
mainfile = os.path.join(inputfolder,f_model_json)

save_load(load,inputfolder)

# Calculate mass matrix
beam_dynamic = ComplBeam(mainfile)
beam.M_mat_full = beam_dynamic.M_mat_full

# %%
print(beam_dynamic.M_mat_full)

#%%
# Get the accelerations

omega = 1 # rad/s

pos_mat = np.reshape(corotobj.final_pos, (-1,6))

mask = np.array([-omega**2, 0, -omega**2, 0, 0, 0])

acc_mat = pos_mat * mask

acc_vec = acc_mat.flatten()

print(acc_vec)

# %%
# Get the node loads: f = M @ a

inertial_loads = beam.M_mat_full @ acc_vec

# If there is any pitch: f = M @ R @ a

pitch = np.deg2rad(90)

def pitch_rot (pitch_rad, vec):
    """Apply the pitch rotation to all the nodes.
    
    Input:
        pitch_rad: pitch angle in radians

        vec: 1D vector of size 2*3*N. N is the number of nodes

    Result:
        vec_rot: `vec`rotated
    """
    
    # Evaluate the rotational matrix
    pitch_mat_eval = pitch_mat(pitch_rad)

    # Reshape the vector into an array (2*N, 3)
    vec = np.reshape(vec,(-1,3))

    vec_rot = np.zeros_like(vec)
    for i in range(vec.shape[0]):
        vec_rot_i = pitch_mat_eval @ vec[i,:]
        vec_rot[i,:] = vec_rot_i

    # Reshape to the original 1D shape 
    vec_rot = vec_rot.flatten()

    return vec_rot

inertial_loads_pitch = beam.M_mat_full @ pitch_rot(pitch, acc_vec)

def inertial_loads_fun (pos_vec, m_mat, omega, pitch_rad):
    """Function with all the needed steps to get the inertial loads for each of the nodes.
    """
    pos_mat = np.reshape(pos_vec, (-1,6))

    pos2acc = np.array([-omega**2, 0, -omega**2, 0, 0, 0])

    acc_mat = pos_mat * pos2acc # Element wise multiplication

    acc_vec = acc_mat.flatten()

    inertial_loads = m_mat @ pitch_rot(pitch_rad, acc_vec)

    return inertial_loads

def obtain_M_mat ():

    return M_mat_full


