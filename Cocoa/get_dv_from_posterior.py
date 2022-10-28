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
samplefile = sys.argv[2]


config = Config(configfile)


#KZ:  debugging
# print("testing......\n")
# print(config.n_train_iter)
# print(config.config_args_lkl)
# print(config.debug)
# print(config.likelihood)
# try:
#     os.makedirs(config.savedir)
# except FileExistsError:
#     pass
# quit()
    
# ============= LHS samples =================
from pyDOE import lhs

def get_lhs_samples(N_dim, N_lhs, lhs_minmax):
    unit_lhs_samples = lhs(N_dim, N_lhs)
    lhs_params = get_lhs_params_list(unit_lhs_samples, lhs_minmax)
    return lhs_params


# ============= samples from posterior =========
def get_samples_from_posterior(file_name):
    posterior_params = np.load(file_name)
    return posterior_params

# ================== Calculate data vectors ==========================



cocoa_model = CocoaModel(configfile, config.likelihood)

def get_local_data_vector_list(params_list, rank):
    train_params_list      = []
    train_data_vector_list = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        params_arr  = np.array(list(params_list[i].values()))
        data_vector = cocoa_model.calculate_data_vector(params_list[i])
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
    return train_params_list, train_data_vector_list


##should implement a BLOCK here for MPI safety
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


print("Calculating data vectors from posterior")

if(rank==0):
    posterior_params = get_samples_from_posterior(samplefile)
    print("testing:", np.shape(posterior_params))
else:
    posterior_params = None

posterior_params = comm.bcast(posterior_params, root=0)
try:
    params_list = get_params_list(posterior_params, config.param_labels)
except:
    print("something wrong with input, it should be a .npy file generated with 'get_samples_from_posterior.py', where you do burn-in and thinning ")

            
train_samples, train_data_vectors = get_data_vectors(params_list, comm, rank)    
    
print("checking train sample shape: ", np.shape(train_samples))
print("checking dv set shape: ", np.shape(train_data_vectors))
    # ================== Train emulator ==========================
if(rank==0):
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