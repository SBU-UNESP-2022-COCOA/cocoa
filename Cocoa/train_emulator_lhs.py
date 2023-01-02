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

    
# ============= LHS samples =================
from pyDOE import lhs

def get_lhs_samples(N_dim, N_lhs, lhs_minmax):
    unit_lhs_samples = lhs(N_dim, N_lhs)
    lhs_params = get_lhs_params_list(unit_lhs_samples, lhs_minmax)
    return lhs_params

# ================== Calculate data vectors ==========================



cocoa_model = CocoaModel(configfile, config.likelihood)

def get_local_data_vector_list(params_list, rank):
    train_params_list      = []
    train_data_vector_list = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        params_arr  = np.array(list(params_list[i].values()))
        #KZ: make EE2 return 0 if out of boundary
        try:
            data_vector = cocoa_model.calculate_data_vector(params_list[i])
        except:
            print("param out of bound of EE2")
            data_vectpr = np.zeros(len(config.dv_fid))
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
    return train_params_list, train_data_vector_list

def get_data_vectors(params_list, comm, rank):
    local_params_list, local_data_vector_list = get_local_data_vector_list(params_list, rank)
    if rank!=0:
        comm.send([local_params_list, local_data_vector_list], dest=0)
        train_params       = None
        train_data_vectors = None
    else:
        data_vector_list = local_data_vector_list
        params_list      = local_params_list
        for source in range(1,size):
            new_params_list, new_data_vector_list = comm.recv(source=source)
            data_vector_list = data_vector_list + new_data_vector_list
            params_list      = params_list + new_params_list
        train_params       = np.vstack(params_list)    
        train_data_vectors = np.vstack(data_vector_list)        
    return train_params, train_data_vectors

for n in range(config.n_train_iter):
    print("Iteration: %d"%(n))
    if(n<=1):
        train_samples_list = []
        train_data_vectors_list = []
    if(n==0):
        if(rank==0):
            lhs_params = get_lhs_samples(config.n_dim, config.n_lhs, config.lhs_minmax)
        else:
            lhs_params = None
        lhs_params = comm.bcast(lhs_params, root=0)
        params_list = lhs_params
    else:
        params_list = get_params_list(next_training_samples, config.param_labels)
            
    current_iter_samples, current_iter_data_vectors = get_data_vectors(params_list, comm, rank)    
    
    if(n>1):
        train_samples_list.append(current_iter_samples)
        train_data_vectors_list.append(current_iter_data_vectors)
        if(n >= 3):
            del train_samples_list[0]
            del train_data_vectors_list[0]            
        train_samples      = np.vstack(train_samples_list)
        train_data_vectors = np.vstack(train_data_vectors_list)
    else:
        train_samples      = current_iter_samples
        train_data_vectors = current_iter_data_vectors
    # ================== Train emulator ==========================
    if(rank==0):
    # ========================================================
        #KZ: create the directory if not present
        try:
            os.makedirs(config.savedir)
        except FileExistsError:
            pass
        if(config.save_train_data):
            np.save(config.savedir + '/train_data_vectors_%d.npy'%(n), current_iter_data_vectors)
            np.save(config.savedir + '/train_samples_%d.npy'%(n), current_iter_samples)
        print("Training emulator...")
        if(config.emu_type=='nn'):
            emu = NNEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std)
            emu.train(torch.Tensor(train_samples), torch.Tensor(train_data_vectors),\
                      batch_size=config.batch_size, n_epochs=config.n_epochs)
            if(config.save_intermediate_model):
                emu.save(config.savedir + '/model_%d'%(n))
        elif(config.emu_type=='gp'):
            emu = GPEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std)
            emu.train(train_samples, train_data_vectors)
            emu.save(config.savedir + '/model_%d'%(n)) #KZ: save here for safety
    else:
        next_training_samples = None
    next_training_samples = comm.bcast(next_training_samples, root=0)

if(rank==0):
    emu.save(config.savedir + '/model_%d'%(n))
    print("DONE!!")    
MPI.Finalize