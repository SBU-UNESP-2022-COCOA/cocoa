import sys,os
import numpy as np
import pynolh
from pyDOE2 import lhs

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config, get_lhs_params_list, get_params_list


from cocoa_emu.sampling import EmuSampler
import emcee


configfile = "./projects/lsst_y1/generate_dv_lhs_3x2.yaml"
config = Config(configfile)

# ============= LHS samples =================

def get_lhs_samples(N_dim, N_lhs, lhs_minmax):
    unit_lhs_samples = lhs(N_dim, N_lhs, criterion='center')
    lhs_params = get_lhs_params_list(unit_lhs_samples, lhs_minmax)
    return lhs_params

n_split = 2


#get full list
lhs_params = get_lhs_samples(config.n_dim, config.n_lhs, config.lhs_minmax)
print("dim of parameter space for training:", config.n_dim)
print("checking train sample shape: ", np.shape(lhs_params))

print(lhs_params[0:2], "\n")

print(type(lhs_params), type(lhs_params[0]), type(config.lhs_minmax))
quit()


try:
    os.makedirs(config.savedir)
except FileExistsError:
    pass


N = np.size(lhs_params) // n_split 

for i in range(n_split):
    start_idx = i * N
    if i != n_split-1:
        end_idx = start_idx + N - 1
    else:
        end_idx = np.size(lhs_params) - 1
    print("start_idx, end_idx = ", start_idx, end_idx)
    np.save(config.savedir + '/train_' + str(i+1) + '_samples.npy', lhs_params[start_idx:end_idx])
    print("saved to: ", config.savedir + '/train_' + str(i+1) + '_samples.npy')
    print("testing", np.shape(lhs_params[start_idx:end_idx]))

print("DONE") 

