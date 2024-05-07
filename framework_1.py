"""
The framework will structure 
"""

import numpy as np
import os

class coupling:
    def __init__(self, epsilon):
        # Main loop options
        self.epsilon = epsilon
        self.delta_u_rel = epsilon + 1
        self.iter_max = 20

        # Simulation conditions
        self.U0 = 10
        self.TSR = 7
        self.pitch_deg = 0

        self.initCoRot()
        self.initBEVC()

    def initCoRot (self):
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
        self.beam = ComplBeam(mainfile)

    def initBEVC(self):
        from pybevc import PyBEVC 

        # Instanciating a PyBEVC object
        self.blade_bevc = PyBEVC()

        # Setting inputs
        self.blade_bevc.flag_a_CT = 2
        htc_filename = 'iea15mw_bevc/IEA-15-240-RWT-Onshore/htc/IEA_15MW_RWT_Onshore.htc'
        self.blade_bevc.from_htc_file(htc_filename) # HAWC2 HTC file

        self.blade_bevc.U0 = self.U0
        self.blade_bevc.TSR = self.TSR
        self.blade_bevc.pitch_deg = self.pitch_deg


    def run(self):
        while abs(self.delta_u_rel) > self.epsilon and iter < self.iter_max:
            pass
    
    
