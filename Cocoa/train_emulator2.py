import sys,os
from mpi4py import MPI
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel
#from cocoa_emu.emulator import NNEmulator, GPEmulator  #KZ: not working, no idea why
from cocoa_emu import NNEmulator, GPEmulator  #KZ: not working, no idea why


from cocoa_emu.sampling import EmuSampler
import emcee

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

configfile = sys.argv[1]
config = Config(configfile)

# this script do the same thing for as train_emulator.py, but instead calculate the data_vactors for training, a set of
# calculated data_vectors(using cocoa) is being provided.  
train_samples      = np.load(sys.argv[2])
train_data_vectors = np.load(sys.argv[3])



def get_chi_sq_cut(train_data_vectors):
    chi_sq_list = []
    for dv in train_data_vectors:
        delta_dv = (dv - config.dv_obs)[config.mask]
        chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
        chi_sq_list.append(chi_sq)
    chi_sq_arr = np.array(chi_sq_list)
    select_chi_sq = (chi_sq_arr < config.chi_sq_cut)
    return select_chi_sq
        # ===============================================
select_chi_sq = get_chi_sq_cut(train_data_vectors)
selected_obj = np.sum(select_chi_sq)
total_obj    = len(select_chi_sq)
        # ===============================================
        
train_data_vectors = train_data_vectors[select_chi_sq]
train_samples      = train_samples[select_chi_sq]


print("Training emulator...")
if(config.emu_type=='nn'):
    emu = NNEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std)
    emu.train(torch.Tensor(train_samples), torch.Tensor(train_data_vectors),\
                      batch_size=config.batch_size, n_epochs=config.n_epochs)

    emu.save(config.savedir + '/model_direct')
elif(config.emu_type=='gp'):
    emu = GPEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std)
    emu.train(train_samples, train_data_vectors)

    emu.save(config.savedir + '/model_direct') #KZ: save here for safety


print("DONE!!")    
