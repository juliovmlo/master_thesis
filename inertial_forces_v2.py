"""Here the inertial forces are implemented.
"""
import numpy as np

## Pitch rotation

pitch_mat = lambda pitch_rad: np.array([
                                        [np.cos(pitch_rad), -np.sin(pitch_rad), 0],
                                        [np.sin(pitch_rad), np.cos(pitch_rad),  0],
                                        [0,                 0,                  1],
                                        ])

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

## Inertial loads

def inertial_loads_fun (pos_vec, root_di, m_mat, omega, pitch_rad):
    """
    Function with all the needed steps to get the inertial loads for each of the nodes.

    Input:
        pos_vec: 1D vector with all the 6 DoF positions of all nodes. 
        root_di: root diameter, used to center the blade axis in the rotor centre
        m_mat: mass matrix of the beam. Square matrix of the same size as pos_vec
        omega: angular velocity of the rotor
        pitch_rad: pitch angle of the blades

    Return:
        inertial_loads: 1D array of the same size as pos_vec with the applyed load for
        each node and DoF
    
    """
    pos_mat = np.reshape(pos_vec, (-1,6))

    # The z-axis is translated to the rotor center
    pos_mat = pos_mat + np.array([0,0,root_di/2,0,0,0])

    pos2acc = np.array([-omega**2, 0, -omega**2, 0, 0, 0])

    acc_mat = pos_mat * pos2acc # Element wise multiplication

    acc_vec = acc_mat.flatten()

    inertial_loads = m_mat @ pitch_rot(pitch_rad, acc_vec)

    return inertial_loads


if __name__=="__main__":
    ## Example

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
    save_load([0], inputfolder) # Creates a force file
    beam = ComplBeam(mainfile)

    # Get initial loads
    r = beam.nodeLocations[:,2] # z-axis position
    load = r/r[-1]*100
    save_load(load,inputfolder)
    beam = ComplBeam(mainfile) # The beam has to be reinstanciated to load the loads

    # For non dynamic calculations the mass matrix is not calculated
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

    # Root
    root_di = 5.2 #[m]

    inertial_loads = inertial_loads_fun(corotobj.final_pos,root_di,beam.M_mat_full,omega,pitch_rad)

    inertial_loads = np.reshape(inertial_loads, (-1, 6))

    fig, axs = plt.subplots(6,1,figsize=(14,10),sharex=True, layout="tight")

    plot_title = ["Fx load", "Fy load", "Fz load", "Mx load", "My load", "Mz load"]

    fig.suptitle(f"Inertial loads with \n{rpm} RPM and {np.rad2deg(pitch_rad)} degrees pitch")
    for i, ax in enumerate(axs):
        ax.plot(r+root_di/2, inertial_loads[:,i],marker='o')
        ax.set_title(plot_title[i])
        ax.set_ylabel("Load") 
    plt.xlabel("Span [m]")

    plt.savefig(f"figures\inertial_loads_{rpm:02d}_{np.rad2deg(pitch_rad):02.0f}.pdf", bbox_inches="tight")
    plt.show()









