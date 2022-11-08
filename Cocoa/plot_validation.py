##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from cocoa_emu import Config, CocoaModel, NNEmulator, GPEmulator
from cocoa_emu.sampling import EmuSampler

def get_chi2(theta, dv_exact, mask):
    if(config.emu_type=='nn'):
        theta = torch.Tensor(theta)
    elif(config.emu_type=='gp'):
        theta = theta[np.newaxis]
    dv_emu = emu.predict(theta)[0]   
    delta_dv = (dv_emu - dv_exact)[mask]
    chi2 = delta_dv @ config.masked_inv_cov @ delta_dv  
    
    return chi2


os.environ["OMP_NUM_THREADS"] = "1"

configfile = './projects/lsst_y1/train_emulator.yaml'
config = Config(configfile)


samples_validation = np.load('projects/lsst_y1/emulator_output/emu_validation/train_validation_samples.npy')
dv_validation = np.load('projects/lsst_y1/emulator_output/emu_validation/train_validation_data_vectors.npy')

if config.probe =='cosmic_shear':
    dv_validation = dv_validation[:,:780]
    mask = config.mask[0:780]
else:
    print("3x2 not tested")
    quit()

print('number of points to plot',len(samples_validation))

emu = NNEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std, config.masked_inv_cov, config.dv_fid) #should privde dv_max instead of dv_fid, but emu.load will make it correct
emu.load('projects/lsst_y1/emulator_output/models/model')
print('emulator loaded')
emu_sampler = EmuSampler(emu, config)

chi2_list = []
for i in range(len(samples_validation)):
    chi2 = get_chi2(samples_validation[i], dv_validation[i], mask)
    chi2_list.append(chi2)

chi2_list = np.array(chi2_list)

print(chi2_list)

logA = samples_validation[:,0]
Omegam = samples_validation[:,4]

cmap = plt.cm.get_cmap('coolwarm')

plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
plt.colorbar()

plt.xlabel(r'$\log A$')
plt.ylabel(r'$\Omega_m$')


plt.legend()

plt.savefig("validation.pdf")