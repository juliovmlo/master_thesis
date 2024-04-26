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
