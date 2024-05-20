import os
import numpy as np
from beam_corot.ComplBeam import ComplBeam
from beam_corot.utils import subfunctions as subf
from utils import save_load

"""
The fucntion `get_cg_offset` was moved to `utils`. Here was kept some of the test that were carried out.
"""

if __name__=="__main__":

    # The function was moved to utils
    from utils import get_cg_offset
    
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
    plt.grid()
    plt.savefig("figures/cg_offset_fig1.png")




