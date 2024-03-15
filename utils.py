import os

def save_load (load, folder):

    force_line = []
    force_line += "# Node_num    Fx      Fy      Fz      Mx      My      Mz \n"
    for node_i,f_node in enumerate(load):
        force_line +=  f"{node_i+1:4d}  0.0  {f_node:>15.6e}  0.0  0.0  0.0  0.0 \n"

    with open(os.path.join(folder,"force.dat"), 'w') as out_file:
        for line in force_line:
            out_file.write(line)