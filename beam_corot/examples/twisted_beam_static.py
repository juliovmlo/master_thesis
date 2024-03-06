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
import numpy as np
from matplotlib import pyplot as plt

# Model input json file name
f_model_json = "twisted_beam_main_static.json"
model_folder = "twisted_beam"

# Input files folder
inputfolder = os.path.join(os.getcwd(),model_folder)

#
tip_f = 1e6
deffAbaq = np.array([0.546617,1.018174]) # Abaqus 3D results
#n_el_case = [1,2,3,4,5,10,20]
n_el_case = [5,10,15,20]

deffT = np.zeros((len(n_el_case),2))
deffT2 = np.zeros_like(deffT)
deffC = np.zeros_like(deffT)
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
    mainfile = os.path.join(inputfolder,f_model_json)

    # Beam models
    beamT = TimoBeam(mainfile)
    beamT2 = TimoBeamV2(mainfile)
    beamC = ComplBeam(mainfile)

    deffT[i_el,:] = beamT.defl_full[-6:-4]
    deffT2[i_el,:] = beamT2.defl_full[-6:-4]
    deffC[i_el,:] = beamC.defl_full[-6:-4]


resT  = np.divide(deffT - deffAbaq , deffAbaq)
resT2 = np.divide(deffT2 - deffAbaq, deffAbaq)
resC  = np.divide(deffC - deffAbaq , deffAbaq)
corotobj = CoRot(beamC,numForceInc=2,max_iter=20,eps=1e-8)
deffCorot = corotobj.total_disp[-6:]

px = np.array(n_el_case)
py = np.zeros((len(n_el_case),3,2))
py[:,0,:] = resC[:,:2] * 100
py[:,1,:] = resT[:,:2] * 100
py[:,2,:] = resT2[:,:2] * 100
#
## PLOT
p_clr = ['#000000','#FF0000','#0000FF','#000000']
p_mrk   = ['','','','s','','']
p_mrk_f = ['none','none','none','none']
p_line = ['solid','solid',(0, (2.5, 2.5)) ]
p_mrk_s = [4.0,8.0,8.0,8.0]
p_mrk_e = [1.5,1.5,1.5,1.5]
p_line_t = [3,3,3,3]
mark_rate = [1,1,1,1]
#
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
font_size = 32 

p_size = [16,9]
p_adjust=[0.15,0.1,0.99,0.95]
#
p_lab = ['ComplBeam','TimoBeam','TimoBeamV2','Co-rot'] 
xlab = ['Number of elements']
ylab = ['Relative error [\%]' ]
sub_title = ['Tip x-deflection','Tip y-deflection']


fig, axs = plt.subplots(1,2, facecolor='w', edgecolor='k')
fig.set_size_inches(p_size[0],p_size[1])
fig.subplots_adjust(left=p_adjust[0], bottom=p_adjust[1],
                    right=p_adjust[2], top=p_adjust[3],wspace = 0.1,hspace = 0.5)

axs = axs.ravel()

for j in range(2):
    for i in range(3):
        axs[j].plot(px,py[:,i,j], marker=p_mrk[i],label=p_lab[i],
                    markersize=p_mrk_s[i],linestyle=p_line[i],
                    fillstyle=p_mrk_f[i],color=p_clr[i],
                    markeredgewidth=p_mrk_e[i],linewidth=p_line_t[i],markevery=mark_rate[i])

    axs[j].set_xlabel(xlab[0], fontsize=font_size)
    axs[j].set_ylabel(ylab[0], fontsize=font_size)
    axs[j].set_title(sub_title[j] , fontsize=font_size)

#axs[0].legend(fontsize=font_size,bbox_to_anchor=(0, 1.1 ),loc="upper left",ncol=3)

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=font_size)
    ax.grid(which='major',axis='both', linestyle=':', linewidth=1)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',fontsize=font_size,bbox_to_anchor=(0.5, 1.03 ),ncol=3)

# fig.tight_layout(rect=(0,0,1,0.93))

plt.show()

# plt.savefig('twist_beam.pdf')
# plt.close()

# print('----------- End of analysis ------------')
