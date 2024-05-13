"""
The input files are read, and the input files for both models are writen.

We need a json file that tells the location of files or a python script.
"""
import os
import json
import numpy as np
import shutil
from wetb.hawc2 import AEFile, PCFile, StFile

def readInput(inputfolder: str, c2_file: str, st_file: str, ae_file: str, pc_file: str):
    try:
        with open(os.path.join(inputfolder, c2_file)) as file:
            c2_data = np.loadtxt(file, comments=";")
            c2_file_content = file.read()
        # with open(os.path.join(inputfolder, st_file)) as file:
        #     st_data = np.loadtxt(file, comments=";")
        st_data = StFile(os.path.join(inputfolder, st_file))
        ae_data = AEFile(os.path.join(inputfolder, ae_file))
        pc_data = PCFile(os.path.join(inputfolder, pc_file))
        # with open(os.path.join(inputfolder, ae_file)) as file:
        #     ae_data = np.loadtxt(file, comments=";")
        
        # with open(os.path.join(inputfolder, pc_file)) as file:
        #     pc_data = np.loadtxt(file, comments=";")
    except:
        print("Error loading data")

    return c2_data, st_data, ae_data, pc_data

def witeInput(projectfolder: str, c2_content, st_data, ae_data, pc_data):
    # C2: the version saved in the structural folder has different comment simbol
    c2_content_stru = c2_content.replace(';', '#')

    with open(os.path.join(projectfolder, 'stru/c2_pos.dat'), 'w') as new_file:
        new_file.write(c2_content_stru)
    
    with open(os.path.join(projectfolder, 'aero/c2_pos.dat'), 'w') as new_file:
        new_file.write(c2_content)

def input2(inputfolder, projectforlder, c2_file: str, st_file: str, ae_file: str, pc_file: str):
    # The default names of files
    c2_filename = 'c2_pos.dat'
    st_filename = 'st_file.st'
    ae_filename = 'ae_file.dat'
    pc_filename = 'pc_file.dat'
    boundary_filename = 'boundary.dat'
    load_filename = 'load.dat'
    load_distr_filename = 'load_distr.dat'
    config_filename = 'config'


    # Check and create directories
    if not os.path.exists(projectfolder):
        os.makedirs(projectfolder)
    stru_folder = os.path.join(projectforlder, 'stru')
    if not os.path.exists(stru_folder):
        os.makedirs(stru_folder)
    aero_folder = os.path.join(projectforlder, 'aero')
    if not os.path.exists(aero_folder):
        os.makedirs(aero_folder)
    
    # C2 file
    shutil.copy(os.path.join(inputfolder, c2_file),os.path.join(stru_folder, c2_filename))
    # Change comments
    with open(os.path.join(stru_folder, 'c2_pos.dat'), 'r') as file:
        content = file.read()
    modified_content = content.replace(';', '#')
    with open(os.path.join(stru_folder, 'c2_pos.dat'), 'w') as new_file:
        new_file.write(modified_content)

    shutil.copy(os.path.join(inputfolder, c2_file),os.path.join(aero_folder, c2_filename))

    # St file
    shutil.copy(os.path.join(inputfolder, st_file),os.path.join(stru_folder, st_filename))

    # AE file
    shutil.copy(os.path.join(inputfolder, ae_file),os.path.join(aero_folder, ae_filename))

    # PC file
    shutil.copy(os.path.join(inputfolder, pc_file),os.path.join(aero_folder, pc_filename))

    # Create boundary file
    boundary_file_content = "# Node# x_t_dof y_t_dof z_t_dof x_r_dof y_r_dof z_r_dof\n1 0 0 0 0 0 0"
    with open(os.path.join(stru_folder, boundary_filename), 'w') as new_file:
        new_file.write(boundary_file_content)

    # Create loads files
    headline = 'Node_num    Fx      Fy      Fz      Mx      My      Mz'
    data = np.array([[1,0,0,0,0,0,0]])
    np.savetxt(os.path.join(stru_folder, load_filename),data,fmt='%d',delimiter='\t',header=headline) 
    np.savetxt(os.path.join(stru_folder, load_distr_filename),data,fmt='%d',delimiter='\t',header=headline)

    # Create ComplBeam config file
    config_data = {
        "Properties": st_filename,
        "c2_pos":c2_filename,
        "Boundary": boundary_filename,
        "Analysistype": "static",
        "static_load": load_filename,
        "static_load_distributed": load_distr_filename,
        "int_type": "gauss",
        "Npmax": 5,
        "Nip": 6
    }
    
    with open( os.path.join(stru_folder, config_filename), 'w') as json_file:
        json.dump(config_data, json_file, indent=4) # indent=4 for pretty formatting


if __name__=="__main__":
    inputfolder = r"input_iea15mw"
    projectfolder = "project_"+inputfolder
    c2_file = 'c2_pos.dat'
    st_file = 'IEA_15MW_RWT_Blade_st_FPM.st'
    ae_file = 'IEA_15MW_RWT_ae.dat'
    pc_file = 'IEA_15MW_RWT_pc.dat'

    # c2_data, st_data, ae_data, pc_data = readInput(inputfolder,c2_file,st_file,ae_file,pc_file)
    input2(inputfolder,projectfolder,c2_file,st_file,ae_file,pc_file)
