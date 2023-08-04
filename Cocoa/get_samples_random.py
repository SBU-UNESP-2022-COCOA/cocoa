import sys,os
import numpy as np
import pynolh

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config, get_lhs_params_list, get_params_list


from cocoa_emu.sampling import EmuSampler
import emcee


#configfile   = "./projects/lsst_y1/generate_dv_lhs.yaml"
configfile   = "./projects/lsst_y1/generate_dv_lhs_3x2.yaml"
#savedir      = './projects/lsst_y1/emulator_output/random/dvs_for_training_4M_new '
#savedir      = './projects/lsst_y1/emulator_output_3x2/random/dvs_for_training_4M'
savedir      = './projects/lsst_y1/emulator_output_3x2/random/dvs_for_validation_5k'
config       = Config(configfile)
n_samples    = 5000 #4M
param_minmax = config.lhs_minmax
# ============= LHS samples =================

def get_random_samples(N_dim, N_samples, param_minmax):
    unit_random_samples = np.random.rand(N_samples, N_dim)
    rdm_params = get_lhs_params_list(unit_random_samples, param_minmax) # use the existing code to convert to list; not really lhs
    return rdm_params

n_split = 1

#get full list
rdm_params = get_random_samples(config.n_dim, n_samples, param_minmax)
print("dim of parameter space for training:", config.n_dim)
print("checking train sample shape: ", np.shape(rdm_params))

try:
    os.makedirs(savedir)
except FileExistsError:
    pass


N = np.size(rdm_params) // n_split 

for i in range(n_split):
    start_idx = i * N
    if i != n_split-1:
        end_idx = start_idx + N - 1
    else:
        end_idx = np.size(rdm_params) - 1
    print("start_idx, end_idx = ", start_idx, end_idx)
    np.save(savedir + '/train_' + str(i+1) + '_samples.npy', rdm_params[start_idx:end_idx])
    print("saved to: ", savedir + '/train_' + str(i+1) + '_samples.npy')
    print("testing", np.shape(rdm_params[start_idx:end_idx]))

print("DONE") 

