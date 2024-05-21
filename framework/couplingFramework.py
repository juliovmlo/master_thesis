from abc import ABC, abstractmethod
from solverWrapper import (
    StructuralSolverWrapper,
    AerodynamicSolverWrapper,
)

class SolverWrapper(ABC):
    """
    Interface for the solver wrappers. The wrappers consist of this minimum building blocks.
    """

    @abstractmethod
    def initialize(self):
        """Instantiates the solver and loads the necesary data."""
        pass

    @abstractmethod
    def update(self, data):
        """Update the state of the solver with the input data. It can run the solver."""
        pass

    @abstractmethod
    def get_results(self):
        """Executes the solver (optional) and returns the resulting data.
        
        Output: dictionary of the following data:
            Structural wrapper:
                's' : nodes positions along the span of the blade [m]
                'defl': deflections of the elastic centres in the blade root FR, [m]
                'final_pos': final position of the elastic centre, [m]
            Aerodynamic wrapper:
                's': nodes positions along the span of the blade [m]
                'force': force distribution in C1/2 [N/m]
                'moments': moments in C1/2 nodes [Nm]
        """
        pass


class CouplingFramework:
    def __init__(self, structural_solver, aerodynamic_solver):
        self.structural_solver = StructuralSolverWrapper(structural_solver)
        self.aerodynamic_solver = AerodynamicSolverWrapper(aerodynamic_solver)
        self.data_manager = DataManager()

    def run(self):
        converged = False
        while not converged:
            # Step 1: Structural solver update
            self.structural_solver.update(self.data_manager.get_aerodynamic_data(),
                                          self.data_manager.get_inertial_data())
            structural_data = self.structural_solver.get_results()
            self.data_manager.set_structural_data(structural_data)

            # Step 2: Aerodynamic solver update
            self.aerodynamic_solver.update(self.data_manager.get_structural_data())
            aerodynamic_data = self.aerodynamic_solver.get_results()
            self.data_manager.set_aerodynamic_data(aerodynamic_data)

            # Step 3: Check for convergence
            converged = self.check_convergence(structural_data, aerodynamic_data)

    def check_convergence(self, structural_data, aerodynamic_data):
        # Implement convergence criteria
        pass

class DataManager:
    def __init__(self):
        self.structural_data = {}
        self.aerodynamic_data = {}
        self.inertial_data = {}

    def get_structural_data(self):
        return self.structural_data

    def set_structural_data(self, data):
        self.structural_data = data

    def get_aerodynamic_data(self):
        return self.aerodynamic_data

    def set_aerodynamic_data(self, data):
        self.aerodynamic_data = data

    def get_inertial_data(self):
        # Calculate and return inertial loads based on current state
        pass