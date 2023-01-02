import sys,os
from mpi4py import MPI
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel


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
    count = 0    
    for i in range(rank * N_local, (rank + 1) * N_local):
        params_arr  = np.array(list(params_list[i].values()))
        data_vector = cocoa_model.calculate_data_vector(params_list[i])
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
        if rank==0:
            count +=1
        if rank==0 and count % 50 == 0:
            print("calculation progress, count = ", count)
    return train_params_list, train_data_vector_list

def get_data_vectors(params_list, comm, rank):
    local_params_list, local_data_vector_list = get_local_data_vector_list(params_list, rank)

    ## for some weird reason, this Barrier() makes it SUPER slow, maybe some node is off
    # if rank==0:
    #     print("rank 0 done, waiting other tasks")
    # comm.Barrier()
    # if rank==0:
    #     print("Every task is done, gathering")
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


##### START dv calculation###

print("generating datavectors from LHS")

if(rank==0):
    lhs_params = get_lhs_samples(config.n_dim, config.n_lhs, config.lhs_minmax)
else:
    lhs_params = None

lhs_params = comm.bcast(lhs_params, root=0)
params_list = lhs_params

print("number of dvs to be calculated: ", len(params_list))

train_samples, train_data_vectors = get_data_vectors(params_list, comm, rank) 

if(rank==0):
    print("checking train sample shape: ", np.shape(train_samples))
    print("checking dv set shape: ", np.shape(train_data_vectors))
    # ================== Chi_sq cut ==========================
    print("not applying chi2 cut")
    try:
        os.makedirs(config.savedir)
    except FileExistsError:
        pass
    if(config.save_train_data):
        np.save(config.savedir + '/train_post_data_vectors.npy', train_data_vectors)
        np.save(config.savedir + '/train_post_samples.npy', train_samples)

print("DONE") 
MPI.Finalize

