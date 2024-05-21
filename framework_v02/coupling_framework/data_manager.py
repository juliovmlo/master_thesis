# data_manager.py

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
        return self.inertial_data

    def set_inertial_data(self, data):
        self.inertial_data = data
