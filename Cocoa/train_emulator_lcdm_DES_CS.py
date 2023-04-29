import sys,os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config,NNEmulator, boundary_check

debug=False

configfile =  "./projects/des_y3/lhs_cs_NLA.yaml"
config = Config(configfile)

# Training set
file ="./projects/des_y3/emulator_output_cs_lcdm_NLA/lhs/dvs_for_training_2000k/train"

resume     = False
set_range  = False
log_norm   = False
model_path = ''

### CONCATENATE TRAINING DATA
#train_samples = []
#train_data_vectors = []

#for file in train_samples_files:
train_samples=np.load(file+'_samples.npy')
train_data_vectors=np.load(file+'_data_vectors.npy')
if debug:
    print('(debug)')
    print('lhs')
    #print(train_samples[0])
    #print(train_data_vectors[0])
    print('(end debug)')



if set_range:
    print("should not use this now")
    quit()
    print("setting ranges of samples")
    IA1 = train_samples[:,12]
    IA2 = train_samples[:,13]
    IA1_max  = +6  
    IA1_min  = -6 
    IA2_max  = +6  
    IA2_min  = -6
    rows_to_delete = []
    for i in range(len(train_samples)):
        if IA1[i] > IA1_max or  IA1[i] < IA1_min:
            rows_to_delete.append(int(i))
            continue
        elif IA2[i] > IA2_max or  IA2[i] < IA2_min:
            rows_to_delete.append(int(i))
            continue
    train_samples           = np.delete(train_samples,      rows_to_delete , 0)
    train_data_vectors      = np.delete(train_data_vectors, rows_to_delete , 0)


# this script do the same thing for as train_emulator.py, but instead calculate the data_vactors for training, a set of

print("length of samples from LHS: ", train_samples.shape)

if config.probe=='cosmic_shear':
    print("training for cosmic shear only======DES")
    OUTPUT_DIM = 400
    train_data_vectors = train_data_vectors[:,:OUTPUT_DIM]
    cov     = config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM]
    cov_inv = np.linalg.inv(cov) #NO mask here for cov_inv enters training
    mask_cs = config.mask[0:OUTPUT_DIM]
    
    dv_fid =config.dv_fid[0:OUTPUT_DIM]
    dv_std = config.dv_std[0:OUTPUT_DIM]
elif config.probe=='3x2pt':
    print("trianing for 3x2pt======DES")
    train_data_vectors = train_data_vectors
    cov     = config.cov
    cov_inv = np.linalg.inv(config.cov) #NO mask here for cov_inv enters training
    OUTPUT_DIM = config.output_dims #config will do it automatically, check config.py
    dv_fid =config.dv_fid
    dv_std = config.dv_std
elif config.probe=='2x2pt':
    print('probe not defnied')
    quit()
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

print("training LHC samples after chi2 cut: ", len(train_samples))

#adding points from chains here to avoid chi2 cut
if len(sys.argv) > 3:
    print("training with posterior samples")
    train_samples_file2      = np.load(sys.argv[3]+'_samples.npy')
    train_data_vectors_file2 = np.load(sys.argv[3]+'_data_vectors.npy')[:,:OUTPUT_DIM]
    if debug:
        print('(debug)')
        print('posterior')
        #print(train_samples_file2[0])
        #print(train_data_vectors_file2[0])
        print('(end debug)')
    
    train_samples = np.vstack((train_samples, train_samples_file2))
    train_data_vectors = np.vstack((train_data_vectors, train_data_vectors_file2))
    print("posterior samples contains: ", len(train_samples_file2))

print("Total samples enter the training: ", len(train_samples))

###============= Setting up validation set ============

print("loading DES-LCDM validation set")
validation_samples =      np.load('./projects/des_y3/emulator_output_cs_lcdm_NLA/lhs/dvs_for_validation_10k/validation_samples.npy')
validation_data_vectors = np.load('./projects/des_y3/emulator_output_cs_lcdm_NLA/lhs/dvs_for_validation_10k/validation_data_vectors.npy')[:,:OUTPUT_DIM]

print("removing boundaries for validation set")
rows_to_delete = []
for k in range(len(validation_samples)):
    if boundary_check(validation_samples[k], config.lhs_minmax, rg=0.1)==True:
        rows_to_delete.append(int(k))
validation_samples           = np.delete(validation_samples,      rows_to_delete , 0)
validation_data_vectors      = np.delete(validation_data_vectors, rows_to_delete , 0)
print("Total samples for Validation: ", len(validation_samples))


###============= Normalize the data vectors for training; 
###============= Get dv_max and dv_mean for training; will be saved along with the model to avoid confusion##
dv_max  = np.abs(train_data_vectors).max(axis=0)
dv_mean = np.mean(train_data_vectors, axis=0)

cov = config.cov[0:OUTPUT_DIM,0:OUTPUT_DIM] #np.loadtxt('lsst_y1_cov.txt')
# do diagonalization C = QLQ^(T); Q is now change of basis matrix
eigensys = np.linalg.eigh(cov)
evals = eigensys[0]
evecs = eigensys[1]
#change of basis
tmp = np.array([dv_mean for _ in range(len(train_data_vectors))])
train_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(train_data_vectors - tmp)))#[pc_idxs])
tmp = np.array([dv_mean for _ in range(len(validation_data_vectors))])
validation_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(validation_data_vectors - tmp)))#[pc_idxs])


# #======Special normalization: this is probably very bad======
# train_samples[:,12]      = train_samples[:,12] / 4
# train_samples[:,13]      = train_samples[:,13] / 4
# validation_samples[:,12] = validation_samples[:,12] / 4
# validation_samples[:,13] = validation_samples[:,13] / 4


#====================chi2 cut for test dvs===========================

print("not doing chi2 cut")

#print("Training emulator...")
# cuda or cpu
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    torch.set_num_interop_threads(30) # Inter-op parallelism
    torch.set_num_threads(30) # Intra-op parallelism

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
emu = NNEmulator(config.n_dim, OUTPUT_DIM, 
                    dv_fid, dv_std, cov, dv_max, dv_mean,config.lhs_minmax,
                    device, model='DES')
if resume:
    print("resume training, loading from previous emulator", model_path)
    emu.load(model_path, map_location=torch.device(device))
emu.train(TS, TDV, VS, VDV, batch_size=config.batch_size, n_epochs=config.n_epochs)
print("model saved to ",str(config.savedir))
try:
    os.makedirs(config.savedir)
except FileExistsError:
    pass
emu.save(config.savedir + '/model_1')
##KZ: 12/31/2022: The bin-by-bin version of this file was accidentally deleted. should be able to get back with the convention in plot_validation


print("DONE!!")   
