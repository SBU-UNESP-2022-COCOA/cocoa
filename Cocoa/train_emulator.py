import sys,os
from mpi4py import MPI
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel
#from cocoa_emu.emulator import NNEmulator, GPEmulator  #KZ: not working, no idea why
from cocoa_emu import GPEmulator, NNEmulator  #KZ: not working, no idea why


from cocoa_emu.sampling import EmuSampler
import emcee

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

configfile = sys.argv[1]
config = Config(configfile)

# this script do the same thing for as train_emulator.py, but instead calculate the data_vactors for training, a set of
# calculated data_vectors(using cocoa) is being provided.  
train_samples_file1      = np.load(sys.argv[2]+'_samples.npy')
train_data_vectors_file1 = np.load(sys.argv[2]+'_data_vectors.npy')

train_samples = train_samples_file1
train_data_vectors = train_data_vectors_file1

print("length of samples from LHS: ", len(train_samples))

if config.probe=='cosmic_shear':
    print("training for cosmic shear only")
    OUTPUT_DIM = 780
    train_data_vectors = train_data_vectors[:,:OUTPUT_DIM]
    cov_inv = np.linalg.inv(config.cov)[0:OUTPUT_DIM, 0:OUTPUT_DIM] #NO mask here for cov_inv enters training
    mask_cs = config.mask[0:OUTPUT_DIM]
    
    dv_fid =config.dv_fid[0:OUTPUT_DIM]
    dv_std = config.dv_std[0:OUTPUT_DIM]
elif config.probe=='3x2pt':
    print("trianing for 3x2pt")
    train_data_vectors = train_data_vectors
    cov_inv = np.linalg.inv(config.cov) #NO mask here for cov_inv enters training
    OUTPUT_DIM = config.output_dim #config will do it automatically, check config.py
    dv_fid =config.dv_fid
    dv_std = config.dv_std
else:
    print('probe not defnied')
    quit()

def get_chi_sq_cut(train_data_vectors, chi2_cut):
    chi_sq_list = []
    for dv in train_data_vectors:
        if config.probe=='cosmic_shear':
            delta_dv = (dv - config.dv_obs[0:OUTPUT_DIM])[mask_cs] #technically this should be masked(on a fiducial scale cut), but the difference is small
            chi_sq = delta_dv @ cov_inv[mask_cs][:,mask_cs] @ delta_dv
        elif config.probe=='3x2pt':
            delta_dv = (dv - config.dv_obs)[config.mask]
            chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv


        chi_sq_list.append(chi_sq)
    chi_sq_arr = np.array(chi_sq_list)
    select_chi_sq = (chi_sq_arr < chi2_cut)
    return select_chi_sq
# ====================chi2 cut for train dvs===========================
# select_chi_sq = get_chi_sq_cut(train_data_vectors, config.chi_sq_cut)
select_chi_sq = get_chi_sq_cut(train_data_vectors, 1e6)
selected_obj = np.sum(select_chi_sq)
total_obj    = len(select_chi_sq)
        
train_data_vectors = train_data_vectors[select_chi_sq]
train_samples      = train_samples[select_chi_sq]

print("training LHC samples after chi2 cut: ", len(train_samples))

#adding points from chains here to avoid chi2 cut
if len(sys.argv) > 3:
    print("training with posterior samples")
    train_samples_file2      = np.load(sys.argv[3]+'_samples.npy')
    train_data_vectors_file2 = np.load(sys.argv[3]+'_data_vectors.npy')[:,:OUTPUT_DIM]
    
    train_samples = np.vstack((train_samples, train_samples_file2))
    train_data_vectors = np.vstack((train_data_vectors, train_data_vectors_file2))
    print("posterior samples contains: ", len(train_samples_file2))

print("Total samples enter the training: ", len(train_samples))

##Normalize the data vectors for training based on the maximum##
dv_max = np.abs(train_data_vectors).max(axis=0)
train_data_vectors = train_data_vectors / dv_max

# np.savetxt('input_norm.txt',train_data_vectors, fmt='%s' )




###============= Setting up validation set ============
validation_samples = np.load('./projects/lsst_y1/emulator_output/emu_test/test_samples.npy')
validation_data_vectors = np.load('./projects/lsst_y1/emulator_output/emu_test/test_data_vectors.npy')[:,:OUTPUT_DIM]
#        ====================chi2 cut for test dvs===========================
select_chi_sq = get_chi_sq_cut(validation_data_vectors, 7000)
selected_obj = np.sum(select_chi_sq)
total_obj    = len(select_chi_sq)
        
validation_data_vectors = validation_data_vectors[select_chi_sq]
validation_samples      = validation_samples[select_chi_sq]

print("validation samples after chi2 cut: ", len(validation_samples))


##### shuffeling #####
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

train_samples, train_data_vectors = unison_shuffled_copies(train_samples, train_data_vectors)
validation_samples, validation_data_vectors = unison_shuffled_copies(validation_samples, validation_data_vectors)


print("Training emulator...")
if(config.emu_type=='nn'):
    emu = NNEmulator(config.n_dim, OUTPUT_DIM, dv_fid, dv_std, cov_inv, dv_max)
    emu.train(torch.Tensor(train_samples), torch.Tensor(train_data_vectors), torch.Tensor(validation_samples), torch.Tensor(validation_data_vectors),\
                      batch_size=config.batch_size, n_epochs=config.n_epochs)
    print("out put to model")
    emu.save(config.savedir + '/model')
elif(config.emu_type=='gp'):
    print("Gaussian Progression NOT implemented yet")
    quit()


print("DONE!!")    
