# Importing modules
from os import path
from pybevc import PyBEVC
from pybevc.test.data import test_data_path
from pybevc.file_io import read_operation
import matplotlib.pyplot as plt
import hvplot, xarray


# Instanciating a PyBEVC object
bevc = PyBEVC()

# Setting inputs from htc file
htc_filename = 'iea15mw_bevc/IEA-15-240-RWT-Onshore/htc/IEA_15MW_RWT_Onshore.htc'
bevc.from_htc_file(htc_filename, model_path='data') # HAWC2 HTC file

# Setting inputs manually 
bevc.U0 = 8.0 # Hub wind speed
bevc.TSR = 7 # Tip speed ratio
bevc.pitch_deg = 2 # Pitch angle

bevc.flag_a_CT = 2



# Running the BEVC solver and returning an Xarray DataSet
res = bevc.run()

print(res['fx_b'])



