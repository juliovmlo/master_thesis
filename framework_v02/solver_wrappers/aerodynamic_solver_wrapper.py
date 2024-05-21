# solver_wrappers/aerodynamic_solver_wrapper.py
import os
import numpy as np
from coupling_framework.solver_wrapper_interface import SolverWrapper
from pybevc import PyBEVC
from utils import save_deflections

class AerodynamicSolverWrapper(SolverWrapper):
    def __init__(self, input_folder, U0, omega_rpm, pitch_deg):
        self.solver = PyBEVC()
        self.input_folder = input_folder
        self.U0 = U0
        self.omega_rpm = omega_rpm
        self.pitch_deg = pitch_deg

    def initialize(self, initial_conditions=None):
        # Initialize aerodynamic model using the HTC file
        self.solver.from_htc_file(os.path.join(self.input_folder, 'htc_files/IEA-15-240-RWT-Onshore/htc/IEA_15MW_RWT_Onshore.htc'))
        self.solver.U0 = self.U0
        self.solver.omega_rpm = self.omega_rpm
        self.solver.pitch_deg = self.pitch_deg
        self.solver.flag_a_CT = 2
        self.hub_di = self.solver.r_hub * 2

    def update(self, structural_data):
        # Save new deflections
        c2_file_new = np.column_stack((structural_data['c2_pos_new'], structural_data['twist_new']))
        save_deflections(c2_file_new, self.input_folder)
        self.solver.from_c2_file(os.path.join(self.input_folder, 'c2_pos.dat'))

    def get_results(self):
        # Calculate aerodynamic loads
        res_bevc = self.solver.run()
        aero_loads = self._extract_aero_loads(res_bevc)

        return {
            'aero_loads': aero_loads,
            'hub_di': self.hub_di,
            'omega': self.solver.omega_rpm * np.pi / 30,
            'pitch_rad': np.deg2rad(self.solver.pitch_deg),
            'R': self.solver.R
        }

    def _extract_aero_loads(self, res_bevc):
        aero_loads = np.zeros((len(self.solver.s), 6))
        load_names = ['fx_b', 'fy_b', 'fz_b', 'sec_mx_b', 'sec_my_b', 'sec_mz_b']
        for i, name in enumerate(load_names):
            aero_loads[:, i] = getattr(res_bevc, name)
        return aero_loads
