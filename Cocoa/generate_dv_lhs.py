import sys, os, signal, time
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

# ============= signal handler =============
# ==Not receiving sigTerm, not sure why===use timer for now===
interrupted = False
converged = False
def signal_handler(signum, frame):
    global interrupted
    interrupted = True

start_minutes = time.time() / 60
end_minutes = 60*7.9 ## 8 hours are usually maximum on seawulf, end the program at 7.9 hours to avoid losing everthing
end_minutes = 60*47.9 ## 48 hours

# ============= signal handler =============

# ============= LHS samples =================
from pyDOE import lhs

def get_lhs_samples(N_dim, N_lhs, lhs_minmax):
    unit_lhs_samples = lhs(N_dim, N_lhs, criterion='center')
    print("lhs samples generated with CRITERION = CENTER")

    # unit_lhs_samples = lhs(N_dim, N_lhs, criterion='corr')
    # print("lhs samples generated with CRITERION = CORR")
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
        signal.signal(signal.SIGTERM, signal_handler)
        if interrupted or converged:
            print("interrupted or convered, trying to save what we have for now. interrupted = "\
                , interrupted, "convered = ", convered)
            print("!!!!NOTE!!!!, this is not working on seawulf, not sure why")
            break
        if (time.time()/60 - start_minutes) > end_minutes:
            print("about timeout, try to save what we have for now.")
            break
            
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
    
    ###for some weird reason, this Barrier() makes it SUPER slow, maybe some node is off
    if rank==0:
        print("rank 0 done, waiting other tasks")
    comm.Barrier()
    if rank==0:
        print("Every task is done, gathering")
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


# ================== Start calculating data vectors (main) ==========================
if(rank==0):
    print("generating datavectors from LHS")
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
    print("saving to ", config.savedir)
    # ================== Chi_sq cut ==========================
    print("not applying chi2 cut")
    try:
        os.makedirs(config.savedir)
    except FileExistsError:
        pass
    if(config.save_train_data):
        np.save(config.savedir + '/train_data_vectors.npy', train_data_vectors)
        np.save(config.savedir + '/train_samples.npy', train_samples)

    print("DONE") 
MPI.Finalize

