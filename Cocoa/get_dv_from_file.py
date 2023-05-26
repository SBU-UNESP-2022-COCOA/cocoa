import sys,os, signal, time
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

# ============= signal handler =============
# ==Not receiving sigTerm, not sure why===use timer for now===
interrupted = False
converged = False
def signal_handler(signum, frame):
    global interrupted
    interrupted = True

start_minutes = time.time() / 60
end_minutes = 60*7.95 ## 8 hours are usually maximum on seawulf, end the program at 7.9 hours to avoid losing everthing


# ============= samples from posterior =========
def get_samples_from_posterior(file_name):
    posterior_params = np.load(file_name, allow_pickle=True)
    return posterior_params

start_minutes = time.time() / 60
end_minutes = 60*7.9 ## 8 hours are usually maximum on seawulf, end the program at 7.9 hours to avoid losing everthing
# end_minutes = 60*47.9 ## 48 hours

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
            print("interrupted or converged, trying to save what we have for now. interrupted = "\
                , interrupted, "convered = ", converged)
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

##should implement a BLOCK here for MPI safety
def get_data_vectors(params_list, comm, rank):
    local_params_list, local_data_vector_list = get_local_data_vector_list(params_list, rank)
    comm.Barrier()
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




if(rank==0):
    posterior_params = get_samples_from_posterior(samplefile)
    # print("testing0000",posterior_params[0])
    print("testing:", np.shape(posterior_params))
else:
    posterior_params = None

posterior_params = comm.bcast(posterior_params, root=0)


# # Can format better in the future. The following is needed when input is pure array (get_samples_from_posterior). 
# # Not needed when input is list (get_samples_lhs.py). For example {"ns": 0.9 ....}
# # The later is easier, but the former is needed for now.
# # TODO: edit get_samples_from_posterior.py to make it a list directly
# try:
#     params_list = get_params_list(posterior_params, config.param_labels)
# except:
#     print("something wrong with input, it should be a .npy file generated with 'get_samples_from_posterior.py' ")
#     print("what we get is: ", np.shape(posterior_params))
#     quit()

try:
    assert len(config.param_labels) == len(posterior_params[0])
    params_list = posterior_params
except:
    print("something wrong with input. It should be a list. It should be a .npy file with 'get_samples_lhs.py' ")
    print("what we get is: ", np.shape(posterior_params))
    quit()

print("Calculating data vectors from posterior")            
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
        print("saving to: ", config.savedir)
        np.save(config.savedir + '/train_data_vectors.npy', train_data_vectors)
        np.save(config.savedir + '/train_samples.npy', train_samples)

print("DONE") 
MPI.Finalize