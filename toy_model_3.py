# Importing modules
from os import path
from pybevc import PyBEVC
from pybevc.test.data import test_data_path
from pybevc.file_io import read_operation
import matplotlib.pyplot as plt
# import hvplot.xarray

# Instanciating a PyBEVC object
bevc = PyBEVC()
# Printing all solver inputs
print(bevc)

# Setting inputs manually 
bevc.U0 = 8.0
bevc.flag_a_CT = 2

# Setting inputs from files
#bevc.from_dict(bevc_inp_dict) # Setting input from a dict
bevc.from_windIO(path.join(test_data_path, "IEA-3.4-130-RWT.yaml"));  # windIO file
#bevc.from_htc_file(htc_filename) # HAWC2 HTC file
#bevc.from_ae_file(ae_filename) # HAWC2 AE file
#bevc.from_pc_file(pc_filename) # HAWC2 PC file
#bevc.from_c2_file(c2_filename) # HAWC2 C2 file