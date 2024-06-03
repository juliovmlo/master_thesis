# coupling_framework.py
from coupling_framework.data_manager import DataManager
from coupling_framework.solver_wrapper_interface import SolverWrapper
import numpy as np


class CouplingFramework:
    def __init__(self, structural_solver: SolverWrapper, aerodynamic_solver: SolverWrapper, epsilon=1e-3, iter_max=20,save_path='results'):
        self.structural_solver = structural_solver
        self.aerodynamic_solver = aerodynamic_solver
        self.epsilon = epsilon
        self.iter_max = iter_max
        self.data_manager = DataManager()
        self.save_path = save_path

    def run(self):

        self.structural_solver.initialize()
        self.aerodynamic_solver.initialize()

        iter_count = 0
        delta_u_rel = self.epsilon + 1
        tip_def_history = []
        delta_history = []

        while delta_u_rel > self.epsilon and iter_count < self.iter_max:
            iter_count += 1

            # Get aerodynamic results and update data manager
            aero_results = self.aerodynamic_solver.get_results()
            self.data_manager.set_aerodynamic_data(aero_results)

            # Update structural solver with aerodynamic data
            self.structural_solver.update(self.data_manager.get_aerodynamic_data())

            # Get structural results and update data manager
            structural_results = self.structural_solver.get_results()
            self.data_manager.set_structural_data(structural_results)

            # Update aerodynamic solver with new structural data
            self.aerodynamic_solver.update(self.data_manager.get_structural_data())

            # Calculate convergence criteria
            old_tip_def = self.data_manager.get_structural_data().get('old_tip_def', [0, 0, 0])
            new_tip_def = structural_results['new_tip_def']
            delta_u_rel = self._calculate_delta_u_rel(new_tip_def, old_tip_def, self.data_manager.get_aerodynamic_data()['R'])
            delta_history.append(delta_u_rel)

            # Update old tip deflection
            self.data_manager.get_structural_data()['old_tip_def'] = new_tip_def

            # Logging iteration info
            print(f"--- Iteration {iter_count} finished ---")
            print(f"Delta = {delta_u_rel:.5f} m")
            print(f"Old tip def = {old_tip_def} m")
            print(f"Tip def: {new_tip_def} m")

        print("Coupling converged" if delta_u_rel <= self.epsilon else "Coupling did not converge")

        self.data_manager.save_data(self.save_path)

    def _calculate_delta_u_rel(self, new_tip_def, old_tip_def, radius):
        return np.linalg.norm(np.array(new_tip_def) - np.array(old_tip_def)) / radius
