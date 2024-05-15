"""Make the HAWC2 tower files from the yaml

Requirements: numpy, matplotlib, pyyaml
"""
from datetime import date
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import yaml


save_twr = 1  # save the HAWC2 tower st file?
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

# paths
yaml_path = os.path.join(mydir,'../../../WT_Ontology/IEA-15-240-RWT.yaml')  # yaml file with data
ed_path = os.path.join(mydir,'../../../OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn_tower.dat')  # elastodyn file
h2_st_path = os.path.join(mydir,'../data/IEA_15MW_RWT_Tower_st.dat')  # file to write for HAWC2 model


# load the yaml file as nested dictionaries
with open(yaml_path, 'r', encoding='utf-8') as stream:
    try:
        res = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

body = 'tower'

# get tower dimensions
twr_stn = np.array(res['components'][body]['outer_shape_bem']['reference_axis']['z']['values'])
twr_stn -= twr_stn[0]
outfitting_factor = res['components'][body]['internal_structure_2d_fem']['outfitting_factor']
out_diam = np.array(res['components'][body]['outer_shape_bem']['outer_diameter']['values'])
twr_wall = res['components'][body]['internal_structure_2d_fem']['layers'][0]
assert twr_wall['name'] == body + '_wall'
t = np.array(twr_wall['thickness']['values'])

# get material properties
material = twr_wall['material']
mat_props = [d for d in res['materials'] if d['name'] == material][0]
E, G, rho = mat_props['E'], mat_props['G'], mat_props['rho']

print('s [m]  OD [m]    t [mm]')
for i in range(t.size):
    print(f'{twr_stn[i]:5.1f}   {out_diam[i]:4.1f}m   {t[i]*1000:5.1f}mm')

# create the figures for the tower report
plt.figure(1, figsize=(7, 3))
plt.clf()
plt.subplot(1, 2, 1)
plt.plot(out_diam, twr_stn)
plt.xlabel('Outer diameter [m]'); plt.ylabel('Tower height [m]'); plt.grid('on')
plt.subplot(1, 2, 2)
plt.plot(t*1000, twr_stn)
plt.xlabel('Wall thickness [mm]'); plt.ylabel('Tower height [m]'); plt.grid('on')
plt.tight_layout()
plt.show()
# plt.savefig(cont_dir + r'\diam_thick.png', dpi=150)
#
# intermediates
r_out = out_diam / 2  # outer diameter [m]
r_in = r_out - t  # inner diameter [m]

area = pi * (r_out**2 - r_in**2)  # cross-sectional area [m^2]
mom_iner = pi/4 * (r_out**4 - r_in**4)  # area moment of inertia [m^4]
tors_const = pi/2 * (r_out**4 - r_in**4)  # torsional constant [m^4]
shear_red = 0.5 + 0.75 * t / r_out  # shear reduction factor [-]

# create the array of values to write
nstn = 19
out_arr = np.full((twr_stn.size, nstn), np.nan)
out_arr[:, 0] = twr_stn  # tower station
out_arr[:, 1] = rho*area*outfitting_factor  # mass/length
out_arr[:, 2] = 0  # center of mass, x
out_arr[:, 3] = 0  # center of mass, y
out_arr[:, 4] = np.sqrt(mom_iner/area)  # radius gyration, x
out_arr[:, 5] = np.sqrt(mom_iner/area)  # radius gyration, y
out_arr[:, 6] = 0  # shear center, x
out_arr[:, 7] = 0  # shear center, y
out_arr[:, 8] = E  # young's modulus
out_arr[:, 9] = G  # shear modulus
out_arr[:, 10] = mom_iner  # area moment of inertia, x
out_arr[:, 11] = mom_iner  # area moment of inertia, y
out_arr[:, 12] = tors_const  # torsional stiffness constant
out_arr[:, 13] = shear_red  # shear reduction, x
out_arr[:, 14] = shear_red  # shear reduction, y
out_arr[:, 15] = area  # cross-sectional area
out_arr[:, 16] = 0  # structural pitch
out_arr[:, 17] = 0  # elastic center, x
out_arr[:, 18] = 0  # elastic center, y

# compare results with elastodyn

ed_st = np.loadtxt(ed_path, skiprows=19, max_rows=10)
h2_stn = (out_arr[:, 0] - out_arr[0, 0])/(out_arr[-1, 0] - out_arr[0, 0])

# visualize the difference
plt.figure(2, figsize=(7, 3))
plt.clf()
plt.subplot(1, 3, 1)  # mass density
plt.plot(ed_st[:, 1], ed_st[:, 0], label='OpenFAST')
plt.plot(out_arr[:, 1], h2_stn, '--', label='HAWC2')
plt.xlabel('Mass density [m]'); plt.ylabel('Tower height [m]'); plt.grid('on')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(ed_st[:, 2], ed_st[:, 0])
plt.plot(out_arr[:, 8]*out_arr[:, 10], h2_stn, label='HAWC2')  # EIxx
plt.xlabel('TwFAStif [mm]'); plt.grid('on')
plt.subplot(1, 3, 3)
plt.plot(ed_st[:, 3], ed_st[:, 0])
plt.plot(out_arr[:, 8]*out_arr[:, 11], h2_stn, label='HAWC2')  # EIyy
plt.xlabel('TwSSStif [mm]'); plt.grid('on')
plt.tight_layout()
plt.show()
if save_twr:
    # flexible
    header1 = f'#1 Tower made by automatic script on {date.today().strftime("%d-%b-%Y")}'
    header2 = '\t'.join(['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'x_sh', 'y_sh', 'E', 'G',
                     'I_x', 'I_y', 'I_p', 'k_x', 'k_y', 'A', 'pitch', 'x_e', 'y_e'])
    header = '\n'.join([header1, header2, f'$1 {out_arr.shape[0]} Flexible'])
    fmt = ['%.3f'] + ['%.4E'] * 18
    np.savetxt(h2_st_path, out_arr, delimiter='\t', fmt=fmt, header=header, comments='')
    # flexible (no torsion)
    out_arr[:, 9] *= 1e7  # increase G
    header = '\n'.join([header2, f'$2 {out_arr.shape[0]} Flexible (no torsion)'])
    with open(h2_st_path, 'a') as f:
        np.savetxt(f, out_arr, delimiter='\t', fmt=fmt, header=header, comments='')
    # rigid
    out_arr[:, 8] *= 1e7  # increase E
    header = '\n'.join([header2, f'$3 {out_arr.shape[0]} Rigid'])
    with open(h2_st_path, 'a') as f:
        np.savetxt(f, out_arr, delimiter='\t', fmt=fmt, header=header, comments='')

plt.show()