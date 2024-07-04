import os
import json
import numpy as np
import shutil
from wetb.hawc2 import AEFile, PCFile, StFile, HTCFile

def createProjectFolder(inputfolder: str, projectfolder: str):
    """
    Creates the project folder for beamCompl.CoRot and BEAVC from the c2, st, ae and pc
    files.

    JSON file named 'file_names.json' in 'inputfolder' directory:
    {
    "c2_file": "c2_pos.dat",
    "st_file": "IEA_15MW_RWT_Blade_st_FPM.st",
    "st_subset": 1,
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
    st_subset = file_names.get("st_subset","")
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
    st_data = StFile(os.path.join(inputfolder, st_file)).main_data_sets[1][st_subset]
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
    headline_distr = "# Element_num    Fx_n    Fy_n    Fz_n    Mx_n    My_n    Mz_n    Fx_n+1    Fy_n+1    Fz_n+1    Mx_n+1    My_n+1    Mz_n+1 \n"
    ele_vec = nodes_vec[:-1]
    loads_distr = np.zeros((len(ele_vec),12))
    data_distr = np.column_stack((ele_vec,loads_distr))
    formats_distr = ['%d'] + ['%.1f'] * (data_distr.shape[1] - 1) 
    np.savetxt(
        os.path.join(stru_folder, load_distr_filename),
        data_distr,fmt=formats_distr,delimiter='\t',header=headline_distr
        )

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

def createProjectFolderV02(htc_filename: str, model_path: str, projectfolder: str):
    # Read a HTC file
    dat_out = dict()
    model_path = './'
    htc = HTCFile(htc_filename, ".." if (model_path is None) else model_path)
    htc_stru = htc.new_htc_structure
    for body in htc_stru.contents.values():
        if ("name" in body.contents):
                # Blade center line
                if (body.name.values[0] == "blade1"):
                    nsec = body.c2_def.nsec.values[0]
                    dat_out["nsec"] = nsec # Number of sections
                    dat_out["x"] = np.zeros(nsec)
                    dat_out["y"] = np.zeros(nsec)
                    dat_out["z"] = np.zeros(nsec)
                    dat_out["twist_deg"] = np.zeros(nsec)
                    for isec in range(nsec):
                        (dat_out["x"][isec], dat_out["y"][isec],
                        dat_out["z"][isec], dat_out["twist_deg"][isec]) = getattr(body.c2_def, "sec__%d"%(isec+1)).values[1:]
                    st_filedir_original = os.path.join(htc.modelpath, body.timoschenko_input.filename[0])
                    st_subset = body.timoschenko_input.set[1]
                # hub radius
                if (body.name.values[0] == "hub1"):
                    nsec = body.c2_def.nsec.values[0]
                    dat_out["r_hub"] = 0.0
                    for isec in range(nsec):
                        r_hub = np.linalg.norm(getattr(body.c2_def, "sec__%d"%(isec+1)).values[1:-1])
                        if r_hub > dat_out["r_hub"]:
                            dat_out["r_hub"] = r_hub

    # Check and create directories
    # projectfolder/
    # ├─ aero/
    # ├─ stru/
    if not os.path.exists(projectfolder):
        os.makedirs(projectfolder)
    stru_folder = os.path.join(projectfolder, 'stru')
    if not os.path.exists(stru_folder):
        os.makedirs(stru_folder)
    aero_folder = os.path.join(projectfolder, 'aero')
    if not os.path.exists(aero_folder):
        os.makedirs(aero_folder)

    # HTC file to aerodynamics folder
    # shutil.copy(
    #     htc_filename,
    #     os.path.join(aero_folder, "htc_file.htc"),
    #     )
    
    # C2 file
    c2_filename = "c2_pos.dat"
    nodes_num = dat_out["x"].shape[0]
    c2_data = np.zeros((nodes_num, 5))
    c2_data[:,0] = list(range(1,nodes_num+1))
    for i, name in enumerate(["x", "y", "z", "twist_deg"]):
        c2_data[:,i+1] = dat_out[name]

    headline = (
        "; 1/2 chord locations of the cross-sections\n"
        "; node  X_coordinate[m]	Y_coordinate[m]	Z_coordinate[m]	Twist[deg.]"
    )
    formats = ['%d'] + ['%.6e'] * (c2_data.shape[1] - 1)
    np.savetxt(
        os.path.join(stru_folder, c2_filename),
        c2_data,fmt=formats,delimiter='\t',header=headline
        )

    # St file 
    st_filename = "st_file.st"
    st_data = StFile(st_filedir_original).main_data_sets[1][st_subset]
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
    
    # Create boundary file
    boundary_filename = 'boundary.dat' # Default name
    boundary_file_content = "# Node# x_t_dof y_t_dof z_t_dof x_r_dof y_r_dof z_r_dof\n1 0 0 0 0 0 0"
    with open(os.path.join(stru_folder, boundary_filename), 'w') as new_file:
        new_file.write(boundary_file_content)

    # Create loads files
    load_filename = 'load.dat'
    load_distr_filename = 'load_distr.dat'
    config_filename = 'config.json'
    headline = 'Node_num    Fx      Fy      Fz      Mx      My      Mz'
    nodes_vec = np.arange(1, nodes_num + 1)
    loads = np.zeros((len(nodes_vec),6))
    data = np.column_stack((nodes_vec,loads))
    formats = ['%d'] + ['%.6e'] * (data.shape[1] - 1)
    np.savetxt(
        os.path.join(stru_folder, load_filename),
        data,fmt=formats,delimiter='\t',header=headline
        )
    headline_distr = "# Element_num    Fx_n    Fy_n    Fz_n    Mx_n    My_n    Mz_n    Fx_n+1    Fy_n+1    Fz_n+1    Mx_n+1    My_n+1    Mz_n+1 \n"
    ele_vec = nodes_vec[:-1]
    loads_distr = np.zeros((len(ele_vec),12))
    data_distr = np.column_stack((ele_vec,loads_distr))
    formats_distr = ['%d'] + ['%.6e'] * (data_distr.shape[1] - 1) 
    np.savetxt(
        os.path.join(stru_folder, load_distr_filename),
        data_distr,fmt=formats_distr,delimiter='\t',header=headline_distr
        )
    
    # Create ComplBeam config file
    config_data = {
        "Properties": st_filename,
        "c2_pos": c2_filename,
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

    # Create the BEVC config file
    config_data = {
        "htc_filename": htc_filename,
        "model_path": model_path,
    }

    with open( os.path.join(aero_folder, config_filename), 'w') as json_file:
        json.dump(config_data, json_file, indent=4) # indent=4 for pretty formatting


if __name__=="__main__":
    # inputfolder = r"examples/input_iea15mw"
    # projectfolder =  os.path.dirname(inputfolder) + "/project_" + os.path.basename(inputfolder)

    # createProjectFolder(inputfolder,projectfolder)

    htc_filename = r"examples/input_only_hacw2/IEA_15MW_RWT_ae_nsec_50_stiff.htc"
    projectfolder = "examples/project_" + os.path.basename(htc_filename)
    createProjectFolderV02(htc_filename,projectfolder)