from coupled_model_v04_wrapper import coupled_model


fidelities_lst = [20, 40, 50, 80]



projectfolder = r'examples/prj_fidelity_study_v02'

for fidelity in fidelities_lst:
    filename = f'examples/input_hawc2_no_e_offset/IEA_15MW_RWT_ae_nsec_{fidelity}.htc'
    results_dict = coupled_model(U0=8,omega=0.59,pitch_deg=0,htc_filename=filename,projectfolder=projectfolder,epsilon=None)

    print('_________________')
    print(f'Fidelity {fidelity} DONE.')

print('_________________')
print('Fidelity study finished.')
print(f'Cases studied: {fidelities_lst}')