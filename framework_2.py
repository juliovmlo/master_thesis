import os
import numpy as np
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot
from pybevc import PyBEVC
from utils import save_load, save_distributed_load, c2_to_node, save_deflections
from inertial_forces import inertial_loads_fun_v04
from cg_offset import get_cg_offset
from input import createProjectFolder

class coupling:
    def __init__(self, inputforder, epsilon=1e-6, ):
        # Main loop options
        self.epsilon = epsilon
        self.delta_u_rel = epsilon + 1
        self.iter_max = 20

        # Simulation conditions
        self.U0 = 10
        self.TSR = 7
        self.pitch_deg = 0

        inputfolder = 'examples/input_iea15mw'
        projectfolder =  os.path.dirname(inputfolder) + "/project_" + os.path.basename(inputfolder)

        createProjectFolder(inputfolder,projectfolder)

        self.initCoRot(inputfolder)
        self.initBEVC(inputfolder,self.U0,self.omega_rpm,self.pitch_deg)

    def initCoRot (self,inputfolder):
        """Initializing the structural model"""
        # Model input json file  name
        f_model_json = "config.json"

        # Input files folder
        self.inputfolder_stru = os.path.join(inputfolder,'stru')
        self.mainfile_beam = os.path.join(self.inputfolder_stru,f_model_json)

        # Initialize beam model
        self.beam = ComplBeam(self.mainfile_beam)
        self.cg_offset = get_cg_offset(self.beam) # TODO: update it in the loop

    def initBEVC(self,inputfolder,U0,omega_rpm,pitch_deg):
        """Initializing the aerodynamic model"""
        self.bevc = PyBEVC()

        self.inputfolder_aero = os.path.join(inputfolder,'aero')
        # I use the trick of using the HTC file
        self.bevc.from_htc_file(os.path.join(self.inputfolder_aero,'htc_files/IEA-15-240-RWT-Onshore/htc/IEA_15MW_RWT_Onshore.htc'))
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

    def run(self):
        #%% Initialize coupling
        pos_new = np.zeros(self.beam.M_mat_full.shape[0]) # The initial pos are 0. This gives no inertial forces
        tip_init_pos = self.beam.nodeLocations[-1]
        old_tip_def = np.zeros(3)
        new_tip_def = np.zeros(3)
        tip_def_history = []
        delta_history = []
        iter = 0

        #%% Coupling loop

        while abs(delta_u_rel) > self.epsilon and iter < self.iter_max:
            # Update variables
            pos_old = pos_new
            old_tip_def = new_tip_def
            tip_def_history.append(np.linalg.norm(old_tip_def))
            iter += 1

            # Calculate aero loads
            res_bevc = self.bevc.run()
            
            load_names = ['fx_b', 'fy_b', 'fz_b', 'sec_mx_b', 'sec_my_b', 'sec_mz_b']
            for i, name in enumerate(load_names):
                self.aero_loads[:,i] = getattr(res_bevc, name)
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

            # Calculate deflections
            beam = ComplBeam(self.mainfile_beam)
            corotobj = CoRot(beam,numForceInc=10,max_iter=20)
            pos_new = np.reshape(corotobj.final_pos, (-1,6))
            tip_pos = pos_new[-1,0:3]
            new_tip_def = tip_pos - tip_init_pos

            # Save deflections
            # Find new position of c2 and twist. Then create the c2_pos file
            defl = np.reshape(self.beam.defl_full,(-1,6)) #
            c2_pos_old = self.beam.c2Input[:,1:4]
            twist_old = self.beam.c2Input[:,-1]
            c2_pos_new = defl[:,:3] + c2_pos_old
            twist_new = twist_old + defl[:,-1] # The rotation around z-axis changes the twist

            c2_file_new = np.column_stack((c2_pos_new,twist_new))

            save_deflections(c2_file_new,self.inputfolder_aero)
            self.bevc.from_c2_file(os.path.join(self.inputfolder_aero,'c2_pos.dat'))

            # Calculate delta
            delta_u_rel = np.linalg.norm(new_tip_def - old_tip_def)/self.bevc.R
            delta_history.append(delta_u_rel)

            print("--- Iteration finished ---")
            print(f"Iteration {iter}")
            print(f"Delta = {delta_u_rel:.5f} m")
            print(f"Old tip def = {old_tip_def} m")
            print(f"Tip def: {new_tip_def} m")