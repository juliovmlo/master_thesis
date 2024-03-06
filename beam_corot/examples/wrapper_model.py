# Julio debug
import sys
sys.path.append(r"C:\Users\Public\OneDrive - Danmarks Tekniske Universitet\Dokumenter\ICAI\DTU\_Thesis\repos\beam-corotational")
import os
print(os.getcwd())

# Import libraries
import os
import numpy as np
from beam_corot.TimoBeamV2 import TimoBeamV2

# Model input json file name
f_model_json = "iea15mw_dynamic.json"

# Input files folder
inputfolder = os.path.join(os.getcwd(),'iea15mw')

mainfile = os.path.join(inputfolder,f_model_json)

beamT2 = TimoBeamV2(mainfile)