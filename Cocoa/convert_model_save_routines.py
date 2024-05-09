##NOTE: PLEASE DON'T USE THIS SCRIPT UNLESS YOU KNOW WHAT YOUR'RE DOING


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from cocoa_emu import Config, NNEmulator
from cocoa_emu.sampling import EmuSampler

torch.set_default_dtype(torch.double)

# OUTPUT_DIM = 780
# BIN_SIZE   = 26 # number of angular bins in each z-bin
# BIN_NUMBER = 30 # number of z-bins

##3x2 setting: separate cosmic shear and 2x2pt
BIN_SIZE     = 780 # number of angular bins in each z-bin
BIN_NUMBER   = 2   # number of z-bins
INPUT_DIM_CS = 13  # input dim of cosmic shear, excluding for example dz_lens and point-mass parameters

cut_boundary = False

configfile              = './projects/lsst_y1/train_emulator_3x2.yaml'
samples_validation_file = './projects/lsst_y1/emulator_output_3x2/random/validation_samples.npy'
dv_validation_file      = './projects/lsst_y1/emulator_output_3x2/random/validation_data_vectors.npy'

emu_model_cs  = 'projects/lsst_y1/emulator_output/modelsTransformer/4M/model_CS'

emu_model_2x2 = 'projects/lsst_y1/emulator_output_3x2_limber/modelsTransformer/2M/param/model_2x2'; model_prefix = "Transformer"; nn_model="Transformer_2x2pt"
# emu_model_2x2 = 'projects/lsst_y1/emulator_output_3x2_limber/modelsResNet/2M//param/model_2x2'; model_prefix = "ResNet"; nn_model="resnet" 
# emu_model_2x2 = 'projects/lsst_y1/emulator_output_3x2_limber/modelsMLP/2M/param/model_2x2'; model_prefix = "MLP"; nn_model="simply_connected"
#emu_model_2x2 = 'projects/lsst_y1/emulator_output_3x2/modelsTransformer/2MSimple_1D_CNN/model_2x2'; model_prefix = "Simple_1D_CNN"
#emu_model_2x2 = 'projects/lsst_y1/emulator_output_3x2/modelsTransformer/2M/model_2x2'; model_prefix = "Transformer"

#emu_model_3x2 = 'projects/lsst_y1/emulator_output_3x2/models/Transformer/model_3x2'


def get_chi2(dv_predict, dv_exact, mask, cov_inv):

    ## GPU emulators works well with float32
    delta_dv = (dv_predict - np.float32(dv_exact) )[mask]
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv)) , delta_dv  )   
    return chi2


os.environ["OMP_NUM_THREADS"] = "1"
config = Config(configfile)

samples_validation = np.load(samples_validation_file)
dv_validation      = np.load(dv_validation_file)

if config.probe =='cosmic_shear':
    dv_validation = dv_validation[:,:OUTPUT_DIM]
    mask = config.mask[0:OUTPUT_DIM]
elif config.probe =='3x2pt':
    dv_validation = dv_validation
    mask = config.mask
else:
    print('probe not tested')
    quit()

full_dim       = 1560    
cov            = config.cov[0:full_dim, 0:full_dim]
cov_inv        = np.linalg.inv(config.cov[0:full_dim, 0:full_dim])
cov_inv_masked = np.linalg.inv(config.cov[0:full_dim, 0:full_dim][mask][:,mask])


bin_count = 0
start_idx = 0
end_idx   = 0

print("KZ testing", config.n_dim)


device='cpu'
emu = NNEmulator(config.n_dim, BIN_SIZE, config.dv_fid, config.dv_std, cov, config.dv_fid,config.dv_fid, config.lhs_minmax ,device, nn_model) #should privde dv_max instead of dv_fid, but emu.load will make it correct

print("using emulator of 2x2 part")
# emu.load(emu_model_2x2, map_location=torch.device(device))


#Transformer
emu.convert_parallel_saved_models(emu_model_2x2, map_location=torch.device(device), save_dir = "projects/lsst_y1/emulator_output_3x2_limber/modelsTransformer/2M/param/test/model_2x2")
#RESNET
# emu.convert_parallel_saved_models(emu_model_2x2, map_location=torch.device(device), save_dir = "projects/lsst_y1/emulator_output_3x2_limber/modelsResNet/2M/param/test/model_2x2")
#MLP
# emu.convert_parallel_saved_models(emu_model_2x2, map_location=torch.device(device), save_dir = "projects/lsst_y1/emulator_output_3x2_limber/modelsTransformer/2M/param/test/model_2x2")







