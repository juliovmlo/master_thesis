# For Julio's computer
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
import numpy as np
from matplotlib import pyplot as plt

# Model input json file name
f_model_json = "iea15mw_static_my_code.json"

# Input files folder
inputfolder = os.path.join(os.getcwd(),'iea15mw')
mainfile = os.path.join(inputfolder,f_model_json)

#
n_el = 33
tip_f = 1e5

force_line = []
force_line += "# Node_num    Fx      Fy      Fz      Mx      My      Mz \n"
force_line +=  f"{n_el+1:4d}  0.0  {tip_f:>15.6e}  0.0  0.0  0.0  0.0 \n"

with open(os.path.join(inputfolder,"force.dat"), 'w') as out_file:
    for line in force_line:
        out_file.write(line)

# Beam models
beamT = TimoBeam(mainfile)
beamT2 = TimoBeamV2(mainfile)
beamC = ComplBeam(mainfile)
corotobj = CoRot(beamC,numForceInc=10,max_iter=20)

mask = np.full(((n_el+1)*6,),False)
mask[-6:-4] = True
mask[-1] = True

deffT = beamT.defl_full[mask]
deffT2 = beamT2.defl_full[mask]
deffC = beamC.defl_full[mask]
deffCorot = corotobj.final_pos[mask]

py = np.zeros((4,3))
py[0,:] = deffT
py[1,:] = deffT2
py[2,:] = deffC
py[3,:] = deffCorot

#%%
## Plot

#
p_lab = ['TimoBeam','TimoBeamV2','ComplBeam','Co-rot'] 
xlab = ['Method used']
ylab = ['Deflection [ ]','Deflection [ ]','Twist [ ]', ]
sub_title = ['Tip x-deflection','Tip y-deflection', 'Tip z-twist']

fig, axs = plt.subplots(1,3,figsize=(8,6),layout="tight")

axs = axs.ravel()

for j in range(3):
    axs[j].bar(p_lab, py[:,j])
    
    axs[j].set_xlabel(xlab[0])
    axs[j].set_ylabel(ylab[j])
    axs[j].set_title(sub_title[j])

for ax in fig.get_axes():
    ax.tick_params(axis='both')
    ax.grid(which='major',axis='both', linestyle=':', linewidth=1)
    ax.set_xticklabels(p_lab, rotation=45)

plt.suptitle(f"Force on the tip of {tip_f/1e3} kN in the y direction")

plt.show()
# %%
