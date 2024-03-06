"""The models used here are linear except the co-rotational model. The Abaqus benchmark
is also linear, that is why the co-rotational model results don't converge to the same
result.
"""

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
f_model_json = "twisted_beam_main_static.json"
model_folder = "twisted_beam"

# Input files folder
inputfolder = os.path.join(os.getcwd(),model_folder)
mainfile = os.path.join(inputfolder,f_model_json)

#
tip_f = 1e6
deffAbaq = np.array([0.546617,1.018174]) # Abaqus 3D results
#n_el_case = [1,2,3,4,5,10,20]
n_el_case = [5,10,15,20]

deffT = np.zeros((len(n_el_case),2))
deffT2 = np.zeros_like(deffT)
deffC = np.zeros_like(deffT)
deffCorot = np.zeros_like(deffT)
# solution
for i_el,n_el in enumerate(n_el_case):
    c2_lines = []
    c2_lines += "# 1/2 chord locations of the cross-sections \n"
    c2_lines += "node  X_coordinate[m]	Y_coordinate[m]	Z_coordinate[m]	Twist[deg.]\n"

    p_n = np.linspace(0,10,n_el+1)

    fmt="{:>9.3f}"

    for i in range(n_el+1):
        c2_lines += "{:4d}  0.0  0.0  {:>15.6f}  0.0 \n".format(i+1,p_n[i])

    with open(os.path.join(inputfolder,"c2_pos.dat"), 'w') as out_file:
        for line in c2_lines:
            out_file.write(line)

    force_line = []
    force_line += "# Node_num    Fx      Fy      Fz      Mx      My      Mz \n"
    force_line +=  "{:4d}  0.0  {:>15.6e}  0.0  0.0  0.0  0.0 \n".format(n_el+1,tip_f)

    with open(os.path.join(inputfolder,"force.dat"), 'w') as out_file:
        for line in force_line:
            out_file.write(line)

    #### Case 1 ####

    # Beam models
    beamT = TimoBeam(mainfile)
    beamT2 = TimoBeamV2(mainfile)
    beamC = ComplBeam(mainfile)
    corotobj = CoRot(beamC,numForceInc=2,max_iter=20,eps=1e-8)

    deffT[i_el,:] = beamT.defl_full[-6:-4]
    deffT2[i_el,:] = beamT2.defl_full[-6:-4]
    deffC[i_el,:] = beamC.defl_full[-6:-4]
    deffCorot[i_el,:] = corotobj.total_disp[-6:-4]

resT  = np.divide(deffT - deffAbaq , deffAbaq)
resT2 = np.divide(deffT2 - deffAbaq, deffAbaq)
resC  = np.divide(deffC - deffAbaq , deffAbaq)
resCorot = np.divide(deffCorot - deffAbaq , deffAbaq)

px = np.array(n_el_case)
py = np.zeros((len(n_el_case),4,2))
py[:,0,:] = resC[:,:2] * 100
py[:,1,:] = resT[:,:2] * 100
py[:,2,:] = resT2[:,:2] * 100
py[:,3,:] = resCorot[:,:2] * 100

## Plot

#
p_lab = ['ComplBeam','TimoBeam','TimoBeamV2','Co-rot'] 
xlab = ['Number of elements']
ylab = ['Relative error [%]' ]
sub_title = ['Tip x-deflection','Tip y-deflection']

fig, axs = plt.subplots(1,2)

axs = axs.ravel()

for j in range(2):
    for i in range(4):
        axs[j].plot(px,py[:,i,j],label=p_lab[i])

    axs[j].set_xlabel(xlab[0])
    axs[j].set_ylabel(ylab[0])
    axs[j].set_title(sub_title[j])

for ax in fig.get_axes():
    ax.tick_params(axis='both')
    ax.grid(which='major',axis='both', linestyle=':', linewidth=1)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',ncol=3)

plt.show()