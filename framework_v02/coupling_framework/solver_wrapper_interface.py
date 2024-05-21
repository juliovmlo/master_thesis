# solver_wrapper_interface.py
from abc import ABC, abstractmethod

class SolverWrapper(ABC):
    @abstractmethod
    def initialize(self, initial_conditions=None):
        pass

    @abstractmethod
    def update(self, data):
        pass

    @abstractmethod
    def get_results(self):
        pass
