import os
import numpy as np
from beam_corot.ComplBeam import ComplBeam
from beam_corot.utils import subfunctions as subf
from utils import save_load

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

if __name__=="__main__":
    
    # Model input json file name
    f_model_json = "straight_beam_cg_offset_test_static.json"

    # Input files folder
    inputfolder = os.path.join(os.getcwd(),'straight_beam')
    mainfile = os.path.join(inputfolder,f_model_json)

    # - Instantiate beam
    save_load([0], inputfolder, onlyy=True) # Creates a force file
    beam = ComplBeam(mainfile)

    cg_offset = get_cg_offset(beam)

    print(cg_offset)

    # Model input json file  name
    f_model_json = "iea15mw_toy_model.json"

    # Input files folder
    inputfolder = os.path.join(os.getcwd(),'iea15mw')
    mainfile = os.path.join(inputfolder,f_model_json)

    # Initialize beam model
    save_load([0], inputfolder, onlyy=True) # Creates a force file
    beam = ComplBeam(mainfile)

    # - Get the radius location of nodes
    r = beam.nodeLocations[:,2] # z-axis position

    cg_offset = get_cg_offset(beam)

    print(cg_offset)

    import matplotlib.pyplot as plt

    plt.figure()
    for axis_i in range(3):
        plt.plot(r, cg_offset[:,axis_i]*100)
    plt.legend(['x','y','z'])
    plt.xlabel("Span [m]")
    plt.ylabel("CoG offset [cm]")
    plt.savefig("figures/cg_offset_fig1.png")




