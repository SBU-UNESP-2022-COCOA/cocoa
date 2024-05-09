import sys,os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config
from cocoa_emu import NNEmulator

debug=False

configfile = "./projects/lsst_y1/train_emulator.yaml"
config = Config(configfile)
savedir = config.savedir

### Training set
files = ["./projects/lsst_y1/emulator_output/lhs/dvs_for_training_1M/train", #LHS
         # "./projects/lsst_y1/emulator_output/random/train_1",  #Random-1 (~800k)
         "./projects/lsst_y1/emulator_output/random/train_2", #Random-2 (1M)
        #  "./projects/lsst_y1/emulator_output/random/train_3", #Random-3 (1M)
        #  "./projects/lsst_y1/emulator_output/random/train_4", #Random-4 (1M)
        #  "./projects/lsst_y1/emulator_output/random/train_5", #Random-5 (1M)
        #  "./projects/lsst_y1/emulator_output/random/train_6", #Random-6 (1M)
        #  "./projects/lsst_y1/emulator_output/random/train_7", #Random-7 (1M)
        ]
train_samples      = []
train_data_vectors = []
for file in files:
    train_samples.append(np.load(file+'_samples.npy').astype(np.float32))
    train_data_vectors.append(np.load(file+'_data_vectors.npy').astype(np.float32))
train_samples = np.vstack(train_samples)
train_data_vectors = np.vstack(train_data_vectors)

### Validation set
# LHS
# validation_samples =      np.load('./projects/lsst_y1/emulator_output/lhs/dvs_for_validation_5k/validation_samples.npy').astype(np.float32)
# validation_data_vectors = np.load('./projects/lsst_y1/emulator_output/lhs/dvs_for_validation_5k/validation_data_vectors.npy').astype(np.float32)
# Random
validation_samples =      np.load('./projects/lsst_y1/emulator_output/random/dvs_for_validation_5k/validation_samples.npy').astype(np.float32)
validation_data_vectors = np.load('./projects/lsst_y1/emulator_output/random/dvs_for_validation_5k/validation_data_vectors.npy').astype(np.float32)


### Select NN model
# nn_model = "Transformer"
# savedir = savedir+"Transformer/8M"
# nn_model = "resnet"
# savedir = savedir+"ResNet/8M"
nn_model = "MLP_v2"
savedir = savedir+"MLP/2M"
# nn_model = "Simple_1D_CNN"
# savedir = savedir+"Simple_1D_CNN/2M"
###

print("length of samples from LHS: ", train_samples.shape)
print("saving the results to: ", savedir)

if config.probe=='cosmic_shear':
    print("training for cosmic shear only")
    OUTPUT_DIM = 780
    train_data_vectors = train_data_vectors[:,:OUTPUT_DIM]
    validation_data_vectors = validation_data_vectors[:,:OUTPUT_DIM]
    cov     = config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM]
    cov_inv = np.linalg.inv(cov)#NO mask here for cov_inv enters training
    mask_cs = config.mask[0:OUTPUT_DIM]
    
    dv_fid =config.dv_fid[0:OUTPUT_DIM]
    dv_std = config.dv_std[0:OUTPUT_DIM]
elif config.probe=='3x2pt':
    print("trianing for 3x2pt")
    OUTPUT_DIM = 1560
    train_data_vectors = train_data_vectors[:,:OUTPUT_DIM]
    validation_data_vectors = validation_data_vectors[:,:OUTPUT_DIM]
    cov     = config.cov
    cov_inv = np.linalg.inv(config.cov) #NO mask here for cov_inv enters training
    OUTPUT_DIM = config.output_dims #config will do it automatically, check config.py
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
print("not applying chi2 cut to lhs")
# select_chi_sq = get_chi_sq_cut(train_data_vectors, 1e6)
# selected_obj = np.sum(select_chi_sq)
# total_obj    = len(select_chi_sq)
# train_data_vectors = train_data_vectors[select_chi_sq]
# train_samples      = train_samples[select_chi_sq]


print("Total samples enter the training: ", len(train_samples))

###============= Normalize the data vectors for training; 
###============= used to be based on dv_max; but change to eigen-basis is better##
dv_max = np.abs(train_data_vectors).max(axis=0)
dv_mean = np.mean(train_data_vectors, axis=0)

cov = config.cov[0:OUTPUT_DIM,0:OUTPUT_DIM] #np.loadtxt('lsst_y1_cov.txt')
# do diagonalization C = QLQ^(T); Q is now change of basis matrix
eigensys = np.linalg.eigh(cov)
evals = eigensys[0].astype(np.float32)
evecs = eigensys[1].astype(np.float32)
#change of basis
tmp = np.array([dv_mean for _ in range(len(train_data_vectors))])
train_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(train_data_vectors - tmp)))#[pc_idxs])
tmp = np.array([dv_mean for _ in range(len(validation_data_vectors))])
validation_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(validation_data_vectors - tmp)))#[pc_idxs])

#====================chi2 cut for test dvs===========================

print("not doing chi2 cut")

#print("Training emulator...")
# cuda or cpu
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    torch.set_num_interop_threads(35) # Inter-op parallelism
    torch.set_num_threads(35) # Intra-op parallelism

print('Using device: ',device)
    

TS = torch.Tensor(train_samples)
#TS.to(device)
TDV = torch.Tensor(train_data_vectors)
#TDV.to(device)
VS = torch.Tensor(validation_samples)
#VS.to(device)
VDV = torch.Tensor(validation_data_vectors)
#VDV.to(device)

print("training with the following hyper paraters: batch_size = ", config.batch_size, 'n_epochs = ', config.n_epochs)
print("emulator info. INPUT_DIM = ", config.n_dim, "OUTPUT_DIM  = ", OUTPUT_DIM )
emu = NNEmulator(config.n_dim, OUTPUT_DIM, 
                        dv_fid, dv_std, cov, dv_max, dv_mean, config.lhs_minmax,
                        device, model=nn_model)
emu.train(TS, TDV, VS, VDV, batch_size=config.batch_size, n_epochs=config.n_epochs)
print("model saved to ",str(savedir))
try:
    os.makedirs(savedir)
except FileExistsError:
    pass
emu.save(savedir + '/model_CS')

print("DONE!!")   
