import os
import numpy as np
from pybevc import PyBEVC
from utils import save_deflections

class Aero:
    def __init__(self, aerofolder,U0,omega_rpm,pitch_deg):
        """
        Loads the aerodynamic data and instanciates the needed variables.
        """
        self.bevc = PyBEVC()

        # I use the trick of using the HTC file
        self.bevc.from_htc_file(os.path.join(aerofolder,'htc_files/IEA-15-240-RWT-Onshore/htc/IEA_15MW_RWT_Onshore.htc'))
        # bevc.from_ae_file(os.path.join(inputfolder_aero,'ae_file.dat'))
        # bevc.from_pc_file(os.path.join(inputfolder_aero,'pc_file.dat'))
        # bevc.from_c2_file(os.path.join(inputfolder_aero,'c2_pos.dat'))

        # Setting inputs manually 
        self.bevc.U0 = U0
        # bevc.TSR = omega*bevc.R/U0
        self.bevc.omega_rpm = omega_rpm
        self.bevc.pitch_deg = pitch_deg
        self.bevc.flag_a_CT = 2
        hub_di = self.bevc.r_hub*2 # For inertial model

        # Load matrix initialization
        self.aero_distr_forces = np.zeros((len(self.bevc.s),3))
        self.aero_moments = np.zeros((len(self.bevc.s),3))
        self.aero_loads = np.zeros((len(self.bevc.s),6))
        # They are not used

    def compute (self,c2_pos):
        """
        Takes the input, processes it, runs the model and gives an ouput.

        Input: the C1/2 positions and twist of the blade in a matrix (N,4).

        Output: the load in the C1/2 of distributed forces 'fX_b' and the punctual moments
        'sec_mX_b' in a matrix (N,6)
        """
        
        # Save in folder and update model
        save_deflections(c2_pos,self.inputfolder_aero)
        self.bevc.from_c2_file(os.path.join(self.inputfolder_aero,'c2_pos.dat'))

        # Run model
        res_bevc = self.bevc.run()

        # Take desired output
        load_names = ['fx_b', 'fy_b', 'fz_b', 'sec_mx_b', 'sec_my_b', 'sec_mz_b']
        for i, name in enumerate(load_names):
            self.aero_loads[:,i] = getattr(res_bevc, name)
        
        return self.aero_loads
    
from beam_corot.ComplBeam import ComplBeam
from inertial_forces import inertial_loads_fun_v04
from utils import (
    c2_to_node,
    get_cg_offset,
    save_load,
    save_distributed_load,
)

class Stru:
    def __init__(self, strufolder):
        # Model input json file  name
        f_model_json = "config.json"

        # Input files folder
        self.mainfile_beam = os.path.join(strufolder,f_model_json)

        # Initialize beam model
        self.beam = ComplBeam(self.mainfile_beam)
        self.cg_offset = get_cg_offset(self.beam) # TODO: update it in the loop

    def compute (self,aero_loads):


        self.aero_loads = c2_to_node(self.beam,self.aero_loads) # Change location

        # Calculate inertial loads
        inert_load = -inertial_loads_fun_v04(pos_old,self.cg_offset,self.beam.M_mat_full,self.hub_di,self.omega,self.pitch_rad)
        inert_load = np.reshape(inert_load,(-1,6))

        # Add up loads and save
        load = inert_load.copy()
        load[:,3:] += self.aero_loads[:,3:]
        load_distr = np.zeros_like(load)
        load_distr[:,:3] = self.aero_loads[:,:3]

        save_load(load,self.inputfolder_stru)
        save_distributed_load(load_distr,self.inputfolder_stru)