import os
import numpy as np

def save_load (load, folder, onlyy = False):
    """Function to create a force.dat file for a given load array. There is the option
    to only give the y-axis load.

    Input:
        load: flat array with the loads for each node in the 6 DoF (Fx, Fy, Fz, Mx, My, Mz)
    """
    force_line = []
    force_line += "# Node_num    Fx      Fy      Fz      Mx      My      Mz \n"

    if not onlyy:
        load = np.reshape(load,(-1,6))
        for node_i in range(load.shape[0]):
            load_str = [f"{f_node:>15.6e}" for f_node in load[node_i]]
            force_line += f"{node_i+1:4d} {' '.join(load_str)}\n"
    else:
        for node_i,f_node in enumerate(load):
            force_line +=  f"{node_i+1:4d}  0.0  {f_node:>15.6e}  0.0  0.0  0.0  0.0 \n"

    with open(os.path.join(folder,"load.dat"), 'w') as out_file:
        for line in force_line:
            out_file.write(line)

def save_distributed_load (load, folder):
    """Save a distributed load
    Function to create a force_distr.dat file for a given distributed load array.
    
    Input:
        load: flat array with the loads for each node in the 6 DoF (Fx, Fy, Fz, Mx, My, Mz)
    """
    # Element loads in their first and second nodes
    force_line = []
    force_line += "# Element_num    Fx_n    Fy_n    Fz_n    Mx_n    My_n    Mz_n    Fx_n+1    Fy_n+1    Fz_n+1    Mx_n+1    My_n+1    Mz_n+1 \n"

    load = np.reshape(load,(-1,6))
    for ele_i in range(load.shape[0]-1):
        node_i = ele_i
        load_str = [f"{f_node:>15.6e}" for f_node in load[node_i]]
        load_str += [f"{f_node:>15.6e}" for f_node in load[node_i+1]]
        force_line += f"{ele_i+1:4d} {' '.join(load_str)}\n"

    with open(os.path.join(folder,"load_distr.dat"), 'w') as out_file:
        for line in force_line:
            out_file.write(line)

def save_deflections(c2_pos, folder: str):

    header = ";1/2 chord locations of the cross-sections\n; X_coordinate[m]	Y_coordinate[m]	Z_coordinate[m]	Twist[deg.]"
    
    with open(os.path.join(folder,"c2_pos.dat"), 'w') as out_file:
        np.savetxt(out_file,c2_pos,header=header,comments=';')

from beam_corot.ComplBeam import ComplBeam

def c2_to_node(beam: ComplBeam, loads_c2):
    """
    Moves the loads from the half chord centre to the node centre (elastic centre) position.

    Input:
        beam: instance of the class ComplBeam
        loads_c2: array of loads Nx6 in C2 in the blade root axis. Forces and moments
    
    Output:
        loads_n: array of loads Nx6 in the nodes in the blade root axis. Forces and moments
    """
    # Nodes are located in the elastic centre
    c2_pos = beam.c2Input[:, 1:4] # Half chord position in blade root FR
    node_pos = beam.nodeLocations # Elastic centre in blade root FR
    r_node2c2 = c2_pos - node_pos

    f_c2, m_c2 = np.split(loads_c2, [3], axis = 1)
    f_n = f_c2
    m_n = m_c2.copy()
    for node_i in range(loads_c2.shape[0]):
        m_n += np.cross(r_node2c2[node_i,:],f_c2[node_i,:])

    loads_n = np.concatenate((f_n,m_n),axis=1)

    return loads_n

from beam_corot.ComplBeam import ComplBeam
from beam_corot.utils import subfunctions as subf

def get_cg_offset(beam: ComplBeam)->np.ndarray:
    """Gives back the centre of gravity offset from the elastic centre (node) in the blade root coordinate system.

    Input:
        beam: instance of the class ComplBeam

    Output:
        cg_offset: Nx3 matrix with the centre of gravity offset from the elastic centre (node) in the blade root coordinate system.
    """
    # Getting centre of gravity and elastic centre in the half chord nodes. The spline and
    # variables used in ComplBeam are used.
    cg_s = np.zeros((beam.numNode,3))
    cg_s[:,0],cg_s[:,1] = beam.struturalPropertyInterploator['x_cg'](beam.scurve),beam.struturalPropertyInterploator['y_cg'](beam.scurve)
    ec_s = np.zeros_like(cg_s)
    ec_s[:,0],ec_s[:,1] = beam.struturalPropertyInterploator['x_e'](beam.scurve),beam.struturalPropertyInterploator['y_e'](beam.scurve)
    twist_deg = beam.c2Input[:, 4]
    # Getting the offset in the nodes coordinate system
    ec_to_cg = cg_s - ec_s

    # Move the offset from element coordinates to blade root coordinates. Just rotate them
    # The tools used in ComplBeam are used. The tanget of the spline and the twist are used.
    cg_offset = np.zeros_like(cg_s)
    for i in range(beam.numNode):
        cg_offset[i,:] = subf.get_tsb(beam.v1, beam.scurveTangent[i, :], np.deg2rad(twist_deg[i])) @ ec_to_cg[i, :]

    return cg_offset

import json

def save_results(results_dict, path):
    """
    Saves a dictionary with the results in the desired path.

    Input:
        results_dict: dict Dictionary of numpy arrays with the desired data to save.
        path: str Path where to save the data
    """
    # Makes sure the extension is always .json
    base, _ = os.path.splitext(path)
    path = base + '.json'

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Convert numpy arrays to lists
    json_ready_dict = {key: value.tolist() for key, value in results_dict.items()}
    
    # Save the dictionary as a JSON file
    with open(path, 'w') as f:
        json.dump(json_ready_dict, f, indent=4)