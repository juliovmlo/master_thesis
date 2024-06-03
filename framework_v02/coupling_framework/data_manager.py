# data_manager.py
import os
import json

class DataManager:
    def __init__(self):
        self.structural_data = {}
        self.aerodynamic_data = {}
        self.inertial_data = {}

    def get_structural_data(self):
        return self.structural_data

    def set_structural_data(self, data):
        self.structural_data.update(data)

    def get_aerodynamic_data(self):
        return self.aerodynamic_data

    def set_aerodynamic_data(self, data):
        self.aerodynamic_data.update(data)

    def get_inertial_data(self):
        return self.inertial_data

    def set_inertial_data(self, data):
        self.inertial_data.update(data)

    def save_data(self, file_path):
        # Combine all data into a single dictionary
        combined_data = {
            'structural_data': self.structural_data,
            'aerodynamic_data': self.aerodynamic_data,
            'inertial_data': self.inertial_data
        }

        # Write the combined data to a JSON file
        with open(file_path, 'w') as file:
            json.dump(combined_data, file, indent=4)

        print(f"Data saved to {file_path}")