import os
import numpy as np
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot
from pybevc import PyBEVC
from couplingFramework import SolverWrapper
from utils import (
    save_load,save_distributed_load,
    c2_to_node,
    save_deflections,
    get_cg_offset,
)

class StructuralSolverWrapper(SolverWrapper):
    def __init__(self, input_folder):
        self.beam_class = ComplBeam
        self.solver_class = CoRot
        self.input_folder = input_folder
        self.solver = None

    def initialize(self, initial_conditions):
        # Construct the path to the main file
        self.file_beam = os.path.join(self.input_folder, 'config.json')

        # Initialize beam model
        self.beam = self.beam_class(self.file_beam)
        self.solver = None # CoRot doesn't have a 'run' method

    def update(self, aerodynamic_data, inertial_data):

        # Get the puntual and distributed loads
        load, load_distr = self.convert_data(aerodynamic_data, inertial_data)

        # Update the beam files
        save_load(load,self.input_folder)
        save_distributed_load(load_distr,self.input_folder)

        # Reinstanciate the beam with new loads
        self.beam = self.beam_class(self.file_beam)

    def get_results(self):
        # Solve the beam with CoRot
        self.solver = CoRot(self.beam,numForceInc=10,max_iter=20)
        pos_new = np.reshape(self.solver.final_pos, (-1,6))
        return pos_new

    def convert_data(self, aerodynamic_data, inertial_data):
        # Perform necessary data conversion
        # Combine aerodynamic and inertial data into a format suitable for the structural solver
        aero_loads = aerodynamic_data
        inert_load = inertial_data

        load = inert_load.copy()
        load[:,3:] += aero_loads[:,3:]
        load_distr = np.zeros_like(load)
        load_distr[:,:3] = aero_loads[:,:3]

        save_load(load,self.input_folder)
        save_distributed_load(load_distr,self.input_folder)
        
        return load, load_distr

class AerodynamicSolverWrapper(SolverWrapper):
    def __init__(self, input_folder):
        self.solver_class = PyBEVC
        self.input_folder = input_folder

    def initialize(self, op_conditions):
        
        self.solver = self.solver_class()

        # TODO: generalize the HTC file
        self.solver.from_htc_file(os.path.join(self.input_folder,'htc_files/IEA-15-240-RWT-Onshore/htc/IEA_15MW_RWT_Onshore.htc'))

        # Setting inputs manually TODO: make this more general
        self.solver.U0 = op_conditions['U0']
        # bevc.TSR = omega*bevc.R/U0
        self.solver.omega_rpm = op_conditions['omega_rpm']
        self.solver.pitch_deg = op_conditions['pitch_deg']
        self.solver.flag_a_CT = 2 # I don't understand this setting

        # Variables
        

    def update(self, structural_data):
        # Convert structural data to the format required by the aerodynamic solver
        c2_file = self.get_c2_file(structural_data)

        # Update the aerodynamic solver with the new data
        save_deflections(c2_file,self.input_folder)
        # Updates BEVC
        self.solver.from_c2_file(os.path.join(self.input_folder,'c2_pos.dat'))

    def get_results(self):
        # Retrieve results from the aerodynamic solver
        res_bevc = self.solver.run()

        aero_loads = np.zeros((len(self.solver.s),6))
        load_names = ['fx_b', 'fy_b', 'fz_b', 'sec_mx_b', 'sec_my_b', 'sec_mz_b']
        for i, name in enumerate(load_names):
            aero_loads[:,i] = getattr(res_bevc, name)

        return aero_loads

    def get_c2_file(self, structural_data):
        """Finds the new position of c2 and twist. Then create the c2_pos file.
        The c2_pos file has c2 positions and twist."""
        defl = structural_data
        c2_pos_old = self.solver.c2Input[:,1:4]
        twist_old = self.solver.c2Input[:,-1]
        c2_pos_new = defl[:,:3] + c2_pos_old
        twist_new = twist_old + defl[:,-1] # The rotation around z-axis changes the twist
        c2_file_new = np.column_stack((c2_pos_new,twist_new))
        return c2_file_new



## Examples

# class StructuralSolverWrapper:
#     def __init__(self, solver):
#         self.solver = solver

#     def initialize(self, initial_conditions):
#         # Initialize the structural solver with initial conditions
#         self.solver.initialize(initial_conditions)

#     def update(self, aerodynamic_data, inertial_data):
#         # Convert aerodynamic data to the format required by the structural solver
#         structural_input = self.convert_data(aerodynamic_data, inertial_data)
#         # Update the structural solver with the new data
#         self.solver.solve(structural_input)

#     def get_results(self):
#         # Retrieve results from the structural solver
#         return self.solver.get_results()

#     def convert_data(self, aerodynamic_data, inertial_data):
#         # Perform necessary data conversion
#         # Combine aerodynamic and inertial data into a format suitable for the structural solver
#         return combined_data

# class AerodynamicSolverWrapper:
#     def __init__(self, solver):
#         self.solver = solver

#     def initialize(self, initial_conditions):
#         # Initialize the aerodynamic solver with initial conditions
#         self.solver.initialize(initial_conditions)

#     def update(self, structural_data):
#         # Convert structural data to the format required by the aerodynamic solver
#         aerodynamic_input = self.convert_data(structural_data)
#         # Update the aerodynamic solver with the new data
#         self.solver.solve(aerodynamic_input)

#     def get_results(self):
#         # Retrieve results from the aerodynamic solver
#         return self.solver.get_results()

#     def convert_data(self, structural_data):
#         # Perform necessary data conversion
#         return aerodynamic_input

