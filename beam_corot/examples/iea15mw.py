# Julio debug
import sys
sys.path.append(r"C:\Users\Public\OneDrive - Danmarks Tekniske Universitet\Dokumenter\ICAI\DTU\_Thesis\repos\beam-corotational")
import os
print(os.getcwd())

# Plot of verification of case 1-3
# - Location of gauss point on compl beam with gauss integration
# Load models
from beam_corot.TimoBeam import TimoBeam 
from beam_corot.TimoBeamV2 import TimoBeamV2
from beam_corot.ComplBeam import ComplBeam
from beam_corot.CoRot import CoRot

# Import libraries
import os
from matplotlib import pyplot as plt

# Model input json file name
f_model_json = "iea15mw_dynamic.json"


# Input files folder
inputfolder = os.path.join(os.getcwd(),'iea15mw')

mainfile = os.path.join(inputfolder,f_model_json)

# Beam models
beamT = TimoBeam(mainfile)
beamT2 = TimoBeamV2(mainfile)
beamC = ComplBeam(mainfile)

#corotobj = CoRot(beamC,numForceInc=2,max_iter=20)

freqT = beamT.freqs[:5]
freqT2 = beamT2.freqs[:5]
freqC = beamC.freqs[:5]

#
fmt1 = "{:^10d}"
fmt2 = "{:<15.6e}"*3
print("------------------------------------------------------")
print("   Mode   Compliance     Timoshenko-2   Timoshenko    ")
print("------------------------------------------------------")
for i in range(freqT.shape[0]):
    print(fmt1.format(i),fmt2.format(freqC[i],freqT2[i],freqT[i]))
print("------------------------------------------------------")