import sys
import os

# Add the master_thesis directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from solver_wrappers.structural_solver_wrapper import StructuralSolverWrapper
from solver_wrappers.aerodynamic_solver_wrapper import AerodynamicSolverWrapper
from coupling_framework.coupling_framework import CouplingFramework
from input import createProjectFolder

# Operations info
omega_rpm = 7.46  # [RPM]
U0 = 10.5  # [m/s]
pitch_deg = 0  # [deg]

# Coupling parameters
epsilon = 1e-3
delta_u_rel = epsilon + 1
iter_max = 20

# Folders and file names
input_folder = 'examples/input_iea15mw'
project_folder = 'examples/project_folder'
createProjectFolder(input_folder, project_folder)

# Structural Solver Initialization
f_model_json = "config.json"
input_folder_stru = os.path.join(os.getcwd(), 'examples/project_folder/stru')
structural_solver_wrapper = StructuralSolverWrapper(f_model_json, input_folder_stru)
structural_solver_wrapper.initialize()

# Aerodynamic Solver Initialization
input_folder_aero = 'examples/project_folder/aero'
aerodynamic_solver_wrapper = AerodynamicSolverWrapper(input_folder_aero, U0, omega_rpm, pitch_deg)
aerodynamic_solver_wrapper.initialize()

# Coupling Framework Initialization
coupling_framework = CouplingFramework(structural_solver_wrapper, aerodynamic_solver_wrapper)

# Run Coupling Loop
coupling_framework.run()
