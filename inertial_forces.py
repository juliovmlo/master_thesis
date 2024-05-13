"""Here the inertial forces are implemented.
"""
import numpy as np

## Pitch rotation

# Definition:
# Rotation matrix to convert from frame of reference with no pitch (BNP) to one with pitch (BP).
# Pitch is defined as the possitive rotation of the AXIS in the z-axis.
# vec_BP = pitch_mat @ vec_BNP
# The invers is the transpose: vec_BNP = np.transpose(pitch_mat) @ vec_BP
pitch_mat = lambda pitch_rad: np.array([
                                        [np.cos(pitch_rad), np.sin(pitch_rad), 0],
                                        [-np.sin(pitch_rad), np.cos(pitch_rad),  0],
                                        [0,                 0,                  1],
                                        ])

def b1_to_b2 (vec, pitch_rad, inv=False):
    """Apply the pitch rotation to all the nodes. Thus going from B1 to B2.
    
    Input:
        vec: 1D vector of size 2*3*N. N is the number of nodes
        pitch_rad: pitch angle in radians
        inv: Apply the inverse, which is `b2_to_b1`. By default False.
    Result:
        vec_rot: `vec`rotated
    """
    
    # Evaluate the rotational matrix
    pitch_mat_eval = pitch_mat(pitch_rad)

    # Reshape the vector into an array (2*N, 3)
    vec = np.reshape(vec,(-1,3))

    n_node = vec.shape[0] # In reality is double the num. of nodes
    vec_rot = np.zeros_like(vec)
    for node_i in range(n_node):
        if not inv:
            vec_rot[node_i,:] = pitch_mat_eval @ vec[node_i,:]
        else:
            vec_rot[node_i,:] = np.transpose(pitch_mat_eval) @ vec[node_i,:]

    # Reshape to the original 1D shape 
    vec_rot = vec_rot.flatten()

    return vec_rot

# Transformation matrix

def pos_b1_to_b2 (vec: np.ndarray, pitch_rad: float, r_hub: float, inv: bool=False)->np.ndarray:
    """Apply transformation to go from the rotor frame of reference (B1) to the blade root
    FR. Applies pitch rotation and balde root translation. Used for position vector transformations.

    Input:
        pitch_rad: pitch angle in radians
        r_root: root distance to the rotor axis.
        vec: 1D vector of size 2*3*N. N is the number of nodes
        inv: Apply the inverse, which is `b2_to_b1`. By default False.

    Result:
        vec_rot: `vec`rotated

    """

    # Evaluate the rotational matrix
    pitch_mat_eval = pitch_mat(pitch_rad)

    # Reshape the vector into an array (2*N, 3)
    vec = np.reshape(vec,(-1,3))

    n_node = int(vec.shape[0]/2)
    vec_rot = np.zeros_like(vec)
    for node_i in range(n_node):
        if not inv: # B1 to B2
            vec_rot[node_i*2,:] = pitch_mat_eval @ vec[node_i*2,:] - np.array([0, 0, r_hub])
            vec_rot[node_i*2+1,:] = pitch_mat_eval @ vec[node_i*2+1,:]
        else: # B2 to B1
            vec_rot[node_i*2,:] = np.transpose(pitch_mat_eval) @ vec[node_i*2,:] + np.array([0, 0, r_hub])
            vec_rot[node_i*2+1,:] = np.transpose(pitch_mat_eval) @ vec[node_i*2+1,:]

    # Reshape to the original 1D shape 
    vec_rot = vec_rot.flatten()

    return vec_rot

def pos_b2_to_b1 (vec: np.ndarray, pitch_rad: float, r_hub: float)->np.ndarray:
    """Applies the inverse of `pos_b1_to_b2`.
    """
    return pos_b1_to_b2 (vec, pitch_rad, r_hub, inv=True)

## Inertial loads

def inertial_loads_fun_v04 (pos_vec_B2, cg_offset_mat, m_mat, r_hub, omega, pitch_rad):
    """
    Function with all the needed steps to get the inertial loads for each of the nodes.

    Input:
        pos_vec_B2: 1D vector with all the 6 DoF positions of all nodes in B2. Length N*6
        cg_offset_vec: Nx3 matrix with the position of the CoG with respect to the node position in B2
        m_mat: mass matrix of the beam. Square matrix of the same size as pos_vec
        hub_di: hub diameter in meters
        omega: angular velocity of the rotor
        pitch_rad: pitch angle of the blades

    Return:
        inertial_loads: 1D array of the same size as pos_vec_B2 with the applyed load for
        each node and DoF
    
    """

    ## Obtaining the CoG positions in rotor base (B1)
    # First in blade root base (B2)
    pos_mat_B2 = np.reshape(pos_vec_B2,(-1,6))
    pos_cg_B2 = pos_mat_B2.copy()
    pos_cg_B2[:,:3] += cg_offset_mat

    # Put the pos_vec in the rotor frame of reference, B1
    pos_cg_B1 = pos_b1_to_b2(pos_cg_B2.flatten(), pitch_rad, r_hub, inv=True) # The inverse is used!!

    ## Obataining CoG accelerations
    pos_cg_mat = np.reshape(pos_cg_B1, (-1,6))

    pos2acc = np.array([-omega**2, 0, -omega**2, 0, 0, 0])

    acc_mat = pos_cg_mat * pos2acc

    acc_vec = acc_mat.flatten()

    ## Obataining inertial loads in nodes

    # The acc_vec is used in the blade rood frame of reference, B2
    inertial_loads = m_mat @ b1_to_b2(acc_vec, pitch_rad)

    return inertial_loads

def loads_cg2node (load_vec, cg_offset):
    """Calculates the equivalent loads in the nodes positions from the CoG.

    Input:
        load_vec: 1D vector of length N*6
        cg_offset: 1D vector of length N*3 with the offset of the CoG for each node
    """
    loads = np.reshape(load_vec,(-1,6))
    cg_offset = np.reshape(cg_offset,(-1,3))

    force, moment_cg = np.split(loads,[3],axis=1)
    moment = moment_cg.copy()
    for node_i in range(loads.shape[0]):
        moment[node_i] += np.cross(cg_offset[node_i,:],force[node_i,:])

    loads = np.concatenate((force,moment),axis=1)

    return loads.flatten()

def loads_in_global_cg(load_vec, node_pos,cg_pos, verbose=False):
    """
    Calculates the loads in the global CoG given its position.
    """
    load_vec = np.reshape(load_vec,(-1,6))
    node_pos = np.reshape(node_pos,(-1,6))
    cg_pos = np.reshape(cg_pos,(-1,3))

    force, moment = np.split(load_vec,[3],axis=1)
    force_cg = np.sum(force, axis=0)

    r_globcg2node = node_pos[:,:3] - cg_pos
    if verbose: print(f'{r_globcg2node =}')

    moment_cg = np.sum(moment,axis=0)
    for node_i in range(node_pos.shape[0]):
        moment_cg += np.cross(r_globcg2node[node_i],force[node_i])

    if verbose:
        print(f'{force_cg =}')
        print(f'{r_globcg2node =}')
        print(f'{moment_cg =}')

    return force_cg, moment_cg




if __name__=="__main__":
    ## Example inertial forces

    # Import libraries
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from beam_corot.ComplBeam import ComplBeam
    from beam_corot.CoRot import CoRot
    from utils import save_load

    # Model input json file  name
    f_model_json = "iea15mw_toy_model.json"

    # Input files folder
    inputfolder = os.path.join(os.getcwd(),'iea15mw')
    mainfile = os.path.join(inputfolder,f_model_json)

    # Initialize beam model
    save_load([0], inputfolder, onlyy=True) # Creates a force file
    beam = ComplBeam(mainfile)

    # Get initial loads
    r = beam.nodeLocations[:,2] # z-axis position
    load = r/r[-1]*100
    save_load(load,inputfolder, onlyy=True)
    beam = ComplBeam(mainfile) # The beam has to be reinstanciated to load the loads

    # For non dynamic calculations, the mass matrix is not calculated
    f_model_json = "iea15mw_dynamic.json"
    inputfolder = os.path.join(os.getcwd(),'iea15mw')
    mainfile = os.path.join(inputfolder,f_model_json)

    # Calculate mass matrix and apply to static beam
    beam_dynamic = ComplBeam(mainfile)
    beam.M_mat_full = beam_dynamic.M_mat_full

    # Calculate deflections
    corotobj = CoRot(beam,numForceInc=10,max_iter=20)

    # Conditions
    pitch_rad = np.deg2rad(0)
    rpm = 7
    omega = rpm*np.pi/30

    # Root diameter
    root_di = 5.2 # [m]

    pitch_lst = np.deg2rad([0,45,90])


    ## Plot different F for different pitch angles

    fig, axs = plt.subplots(6,1,figsize=(14,10),sharex=True, layout="tight")
    plot_title = ["Fx load", "Fy load", "Fz load", "Mx load", "My load", "Mz load"]

    for pitch_rad in pitch_lst:

        inertial_loads = inertial_loads_fun(corotobj.final_pos,beam.M_mat_full,root_di,omega,pitch_rad)

        inertial_loads = np.reshape(inertial_loads, (-1, 6))

        for i, ax in enumerate(axs):
            ax.plot(r, inertial_loads[:,i],marker='o', label=f"{np.rad2deg(pitch_rad)} deg")
            ax.set_title(plot_title[i])
            ax.set_ylabel("Load") 
            ax.grid(True)

    fig.suptitle(f"Inertial loads with \n{rpm} RPM and {np.rad2deg(pitch_lst)} degrees pitch")
    plt.xlabel("Span [m]")
    plt.legend()
    deg_str = "-".join([f"{np.rad2deg(pitch_rad):02.0f}" for pitch_rad in pitch_lst])
    pdf_name = f"figures\inertial_loads_{rpm:02d}rpm_{deg_str}deg.pdf"
    # plt.savefig(pdf_name, bbox_inches="tight")
    plt.show()

    # plt.savefig(f"figures\inertial_loads_{rpm:02d}_{np.rad2deg(pitch_rad):02.0f}.pdf", bbox_inches="tight")

    ## First test

    fig1, axs1 = plt.subplots(6,1,figsize=(14,10),sharex=True, layout="tight")
    plot_title1 = ["Fx load", "Fy load", "Fz load", "Mx load", "My load", "Mz load"]

    acc_lst = [[1,0,0,0,0,0],
               [0,1,0,0,0,0],
               [0,0,1,0,0,0],
    ]

    for acc in acc_lst:

        inertial_loads = inertial_loads_test(beam.M_mat_full,acc)

        inertial_loads = np.reshape(inertial_loads, (-1, 6))

        for i, ax in enumerate(axs1):
            ax.plot(r, inertial_loads[:,i],marker='o', label=f"{acc}")
            ax.set_title(plot_title1[i])
            ax.set_ylabel("Load") 
            ax.grid(True)

    fig1.suptitle(f"Inertial loads for unitary acceleration in x, y and z-axis in all nodes")
    plt.xlabel("Span [m]")
    plt.legend()
    plt.show()


    ## Acc. in x-axis

    fig, axs = plt.subplots(3,1,figsize=(14,10),sharex=True, layout="tight")
    plot_title = ["Fx load", "Fy load", "Fz load"]
    
    acc = [1,0,0,0,0,0]

    inertial_loads = inertial_loads_test(beam.M_mat_full,acc)

    inertial_loads = np.reshape(inertial_loads, (-1, 6))

    for i, ax in enumerate(axs):
            ax.plot(r, inertial_loads[:,i], marker='o', label=f"{acc}")
            ax.set_title(plot_title[i])
            ax.set_ylabel("Load") 
            ax.grid(True)

    fig.suptitle(f"Inertial loads for unitary acceleration in x-axis in all nodes")
    plt.xlabel("Span [m]")
    plt.legend()
    plt.show()