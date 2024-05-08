import os
import numpy as np

def save_load (load, folder, onlyy = False):
    """Function to create a force.data file for a given load array. There is the option
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

    with open(os.path.join(folder,"force.dat"), 'w') as out_file:
        for line in force_line:
            out_file.write(line)

def save_distributed_load (load, folder):
    """Save a distributed load"""
    force_line = []
    force_line += "# Node_num    Fx      Fy      Fz      Mx      My      Mz \n"

    load = np.reshape(load,(-1,6))
    for node_i in range(load.shape[0]):
        load_str = [f"{f_node:>15.6e}" for f_node in load[node_i]]
        force_line += f"{node_i+1:4d} {' '.join(load_str)}\n"

    with open(os.path.join(folder,"distributed_loads.dat"), 'w') as out_file:
        for line in force_line:
            out_file.write(line)


def input_data (folder):
    """Function where the input data is handled."""

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

