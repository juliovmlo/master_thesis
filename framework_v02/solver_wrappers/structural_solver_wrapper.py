# solver_wrappers/structural_solver_wrapper.py
import os
import numpy as np
from coupling_framework.solver_wrapper_interface import SolverWrapper
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot
from inertial_forces import inertial_loads_fun_v04
from utils import get_cg_offset, save_load, save_distributed_load, c2_to_node

class StructuralSolverWrapper(SolverWrapper):
    def __init__(self, config_filename, input_folder):
        self.config_filename = config_filename
        self.input_folder = input_folder
        self.beam = None
        self.cg_offset = None
        self.tip_init_pos = None

    def initialize(self, initial_conditions=None):
        # Model input json file name
        mainfile_beam = os.path.join(self.input_folder, self.config_filename)
        self.beam = ComplBeam(mainfile_beam)
        self.cg_offset = get_cg_offset(self.beam)
        self.tip_init_pos = self.beam.nodeLocations[-1]

    def update(self, aerodynamic_data):
        # Calculate inertial loads
        pos_old = aerodynamic_data.get('pos_old', np.zeros(self.beam.M_mat_full.shape[0]))
        inert_load = self._calculate_inertial_loads(pos_old, aerodynamic_data)
        aero_loads = c2_to_node(self.beam, aerodynamic_data['aero_loads'])
        
        # Add up loads and save
        load = self._add_loads(inert_load, aero_loads)
        load_distr = self._distribute_loads(aero_loads)
        
        save_load(load, self.input_folder)
        save_distributed_load(load_distr, self.input_folder)

    def get_results(self):
        # Calculate deflections
        mainfile_beam = os.path.join(self.input_folder, self.config_filename)
        self.beam = ComplBeam(mainfile_beam)
        corotobj = CoRot(self.beam, numForceInc=10, max_iter=20)
        pos_new = np.reshape(corotobj.final_pos, (-1, 6))
        tip_pos = pos_new[-1, 0:3]
        new_tip_def = tip_pos - self.tip_init_pos

        defl = np.reshape(self.beam.defl_full, (-1, 6))
        c2_pos_old = self.beam.c2Input[:, 1:4]
        twist_old = self.beam.c2Input[:, -1]
        c2_pos_new = defl[:, :3] + c2_pos_old
        twist_new = twist_old + defl[:, -1]

        return {
            'pos_new': pos_new,
            'new_tip_def': new_tip_def,
            'c2_pos_new': c2_pos_new,
            'twist_new': twist_new,
            # 'R': self.beam.R
        }

    def _calculate_inertial_loads(self, pos_old, aerodynamic_data):
        inertial_loads = -inertial_loads_fun_v04(
            pos_old, self.cg_offset, self.beam.M_mat_full,
            aerodynamic_data['hub_di'], aerodynamic_data['omega'],
            aerodynamic_data['pitch_rad']
        )
        return np.reshape(inertial_loads,(-1,6))

    def _add_loads(self, inert_load, aero_loads):
        load = inert_load.copy()
        load[:, 3:] += aero_loads[:, 3:]
        return load

    def _distribute_loads(self, aero_loads):
        load_distr = np.zeros_like(aero_loads)
        load_distr[:, :3] = aero_loads[:, :3]
        return load_distr
