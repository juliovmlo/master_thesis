import os
import json
import numpy as np
import shutil
from wetb.hawc2 import AEFile, PCFile, StFile

def createProjectFolder(inputfolder: str, projectfolder: str):
    """
    Creates the project folder for beamCompl.CoRot and BEAVC from the c2, st, ae and pc
    files.

    JSON file named 'file_names.json' in 'inputfolder' directory:
    {
    "c2_file": "c2_pos.dat",
    "st_file": "IEA_15MW_RWT_Blade_st_FPM.st",
    "ae_file": "IEA_15MW_RWT_ae.dat",
    "pc_file": "IEA_15MW_RWT_pc.dat"
    }

    createProjectFolder(inputfolder, projectfolder)

    """

    # Read the file names
    with open(os.path.join(inputfolder, 'file_names.json'), 'r') as f:
        file_names = json.load(f)

    # TODO: raise an exception when a name is missing
    c2_file = file_names.get("c2_file", "")
    st_file = file_names.get("st_file", "")
    ae_file = file_names.get("ae_file", "")
    pc_file = file_names.get("pc_file", "")

    # The default names of files
    c2_filename = 'c2_pos.dat'
    st_filename = 'st_file.st'
    ae_filename = 'ae_file.dat'
    pc_filename = 'pc_file.dat'
    boundary_filename = 'boundary.dat'
    load_filename = 'load.dat'
    load_distr_filename = 'load_distr.dat'
    config_filename = 'config.json'

    # Check and create directories
    if not os.path.exists(projectfolder):
        os.makedirs(projectfolder)
    stru_folder = os.path.join(projectfolder, 'stru')
    if not os.path.exists(stru_folder):
        os.makedirs(stru_folder)
    aero_folder = os.path.join(projectfolder, 'aero')
    if not os.path.exists(aero_folder):
        os.makedirs(aero_folder)
    
    # C2 file
    shutil.copy(
        os.path.join(inputfolder, c2_file),
        os.path.join(stru_folder, c2_filename),
        )
    # Change comments
    with open(os.path.join(stru_folder, c2_filename), 'r') as file:
        content = file.read()
    new_content = content.replace(';', '#')
    with open(os.path.join(stru_folder, c2_filename), 'w') as new_file:
        new_file.write(new_content)

    shutil.copy(
        os.path.join(inputfolder, c2_file),
        os.path.join(aero_folder, c2_filename),
        )
    
    # Here the first row is removed
    with open(os.path.join(aero_folder, c2_filename), 'r') as file:
        lines = file.readlines()
    with open(os.path.join(aero_folder, c2_filename), 'r') as file:
        content = np.loadtxt(file,comments=';')

    nodes_num = content.shape[0]
        
    lines[1] = lines[1].replace('node ','')
    header = (lines[0].replace('; ','')+lines[1].replace('; ',''))[:-2]
    # Remove node row
    content = content[:,1:] # Remove first row
    with open(os.path.join(aero_folder, c2_filename), 'w') as new_file:
        np.savetxt(new_file, content,comments=';', delimiter='\t', header=header)

    # St file
    st_data = StFile(os.path.join(inputfolder, st_file)).main_data_sets[1][1]
    headline = (
        '=====================================================================\n'
        'r [0]                   m [1]                 x_cg [2]                y_cg [3]                ri_x [4]                ri_y [5]                pitch [6]                x_e [7]                 y_e [8]                	K_11 [9]                K_12 [10]                K_13 [11]                K_14 [12]                K_15 [13]                K_16 [14]                K_22 [15]                K_23 [16]                K_24 [17]                K_25 [18]                K_26 [19]                K_33 [20]                K_34 [21]                K_35 [22]                K_36 [23]                K_44 [24]                K_45 [25]                K_46 [26]                K_55 [27]                K_56 [28]                K_66 [29] \n'       
        '====================================================================='
    )
    np.savetxt(
        os.path.join(stru_folder, st_filename),
        st_data,delimiter='\t',
        header=headline,
        )

    # AE file
    shutil.copy(
        os.path.join(inputfolder, ae_file),
        os.path.join(aero_folder, ae_filename),
        )

    # PC file
    shutil.copy(
        os.path.join(inputfolder, pc_file),
        os.path.join(aero_folder, pc_filename),
        )

    # Create boundary file
    boundary_file_content = "# Node# x_t_dof y_t_dof z_t_dof x_r_dof y_r_dof z_r_dof\n1 0 0 0 0 0 0"
    with open(os.path.join(stru_folder, boundary_filename), 'w') as new_file:
        new_file.write(boundary_file_content)

    # Create loads files
    headline = 'Node_num    Fx      Fy      Fz      Mx      My      Mz'
    nodes_vec = np.arange(1, nodes_num + 1)
    loads = np.zeros((len(nodes_vec),6))
    data = np.column_stack((nodes_vec,loads))
    formats = ['%d'] + ['%.1f'] * (data.shape[1] - 1)
    np.savetxt(
        os.path.join(stru_folder, load_filename),
        data,fmt=formats,delimiter='\t',header=headline
        ) 
    np.savetxt(
        os.path.join(stru_folder, load_distr_filename),
        data,fmt=formats,delimiter='\t',header=headline
        )

    # Create ComplBeam config file
    config_data = {
        "Properties": st_filename,
        "c2_pos":c2_filename,
        "Boundary": boundary_filename,
        "Analysistype": "static",
        "static_load": load_filename,
        # This is not used yet
        # "static_load_distributed": load_distr_filename,
        "static_load_distributed_segment": load_distr_filename,
        "int_type": "gauss",
        "Npmax": 5,
        "Nip": 6
    }
    
    with open( os.path.join(stru_folder, config_filename), 'w') as json_file:
        json.dump(config_data, json_file, indent=4) # indent=4 for pretty formatting


if __name__=="__main__":
    inputfolder = r"examples/input_iea15mw"
    projectfolder =  os.path.dirname(inputfolder) + "/project_" + os.path.basename(inputfolder)

    createProjectFolder(inputfolder,projectfolder)