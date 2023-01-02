##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from cocoa_emu import Config, nn_pca_emulator
from cocoa_emu.sampling import EmuSampler

torch.set_default_dtype(torch.double)

OUTPUT_DIM = 780
BIN_SIZE   = 26 # number of angular bins in each z-bin
BIN_NUMBER = 30 # number of z-bins

BIN_SIZE   = 780 # number of angular bins in each z-bin
BIN_NUMBER = 1 # number of z-bins

# def get_chi2(theta, dv_exact, mask, cov_inv_masked):
#     if(config.emu_type=='nn'):
#         theta = torch.Tensor(theta)
#     elif(config.emu_type=='gp'):
#         theta = theta[np.newaxis]

#     dv_emu = emu.predict(theta)[0]
#     delta_dv = (dv_emu - np.float32(dv_exact) )[mask]
#     ## GPU emulators works well with float32
#     chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv_masked)) , delta_dv  )   
#     #print(dv_emu.dtype, np.float32(dv_exact).dtype, config.masked_inv_cov.dtype) 
#     #print(np.float32(dv_exact))
#     return chi2

def get_chi2(dv_predict, dv_exact, mask, cov_inv):

    ## GPU emulators works well with float32
    delta_dv = (dv_predict - np.float32(dv_exact) )[mask]
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv)) , delta_dv  )   
    return chi2


os.environ["OMP_NUM_THREADS"] = "1"

configfile = './projects/lsst_y1/train_emulator.yaml'
config = Config(configfile)

# samples_validation = np.load('projects/lsst_y1/emulator_output/emu_validation/noshift' + '/validation_samples.npy')
# dv_validation      = np.load('projects/lsst_y1/emulator_output/emu_validation/noshift' + '/validation_data_vectors.npy')

# samples_validation = np.load('projects/lsst_y1/emulator_output/emu_validation/noshift_withCMB' + '/validation_samples.npy')
# dv_validation      = np.load('projects/lsst_y1/emulator_output/emu_validation/noshift_withCMB' + '/validation_data_vectors.npy')


samples_validation = np.load('projects/lsst_y1/emulator_output/lhs/dvs_for_training_30k/train_30k_samples.npy')
dv_validation      = np.load('projects/lsst_y1/emulator_output/lhs/dvs_for_training_30k/train_30k_data_vectors.npy')

if config.probe =='cosmic_shear':
    dv_validation = dv_validation[:,:OUTPUT_DIM]
    mask = config.mask[0:OUTPUT_DIM]

    dv_fid =config.dv_fid[0:OUTPUT_DIM]
    dv_std = config.dv_std[0:OUTPUT_DIM]
else:
    print("3x2 not tested")
    quit()

cov_inv        = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM])
cov_inv_masked = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM][mask][:,mask])

#print(samples_validation[1])

logA = samples_validation[:,0]
ns = samples_validation[:,1]
H0 = samples_validation[:,2]
Omegab = samples_validation[:,3]
Omegam = samples_validation[:,4]
Omegam_growth = samples_validation[:,5]
dz1 = samples_validation[:,6]
dz2 = samples_validation[:,7]
dz3 = samples_validation[:,8]
dz4 = samples_validation[:,9]
dz5 = samples_validation[:,10]

#####Set Range####
print("setting ranges of validation plot")
rows_to_delete = []

## rg = range to be taken from the boundary, eg. rg=0.1 means 80% of EE box
rg = 0.1
logA_max = np.log(25) - rg*(np.log(25)-np.log(17))
logA_min = np.log(17) + rg*(np.log(25)-np.log(17))
omm_max  = 0.4  - rg*(0.4-0.24)
omm_min  = 0.24 + rg*(0.4-0.24)
ommg_max = 0.4  - rg*(0.4-0.24)
ommg_min = 0.24 + rg*(0.4-0.24)
ns_max   = 1.0  - rg*(1.0-0.92)
ns_min   = 0.92 + rg*(1.0-0.92)
omb_max  = 0.06 - rg*(0.06-0.04)
omb_min  = 0.04 + rg*(0.06-0.04)

dz1_max  = 0.008  - rg*(0.008+0.008) 
dz1_min  = -0.008 + rg*(0.008+0.008) 
dz2_max  = 0.008  - rg*(0.008+0.008) 
dz2_min  = -0.008 + rg*(0.008+0.008) 
dz3_max  = 0.008  - rg*(0.008+0.008) 
dz3_min  = -0.008 + rg*(0.008+0.008) 
dz4_max  = 0.008  - rg*(0.008+0.008) 
dz4_min  = -0.008 + rg*(0.008+0.008) 
dz5_max  = 0.008  - rg*(0.008+0.008) 
dz5_min  = -0.008 + rg*(0.008+0.008) 

IA1_max  = 5  + rg*(10) 
IA1_min  = -5 + rg*(10)
IA2_max  = 5  + rg*(10) 
IA2_min  = -5 + rg*(10) 


for i in range(len(samples_validation)):
    if logA[i]>logA_max or logA[i]<logA_min:
        rows_to_delete.append(int(i))
        continue
    elif Omegam[i] > omm_max or  Omegam[i] < omm_min:
        rows_to_delete.append(int(i))
        continue
    elif Omegam_growth[i] > ommg_max or  Omegam_growth[i] < ommg_min:
        rows_to_delete.append(int(i))
        continue
    elif ns[i] > ns_max or  ns[i] < ns_min:
        rows_to_delete.append(int(i))
        continue
    elif Omegab[i] > omb_max or  Omegab[i] < omb_min:
        rows_to_delete.append(int(i))
        continue
    elif dz1[i] > dz1_max or  dz1[i] < dz1_min:
        rows_to_delete.append(int(i))
        continue
    elif dz2[i] > dz2_max or  dz2[i] < dz2_min:
        rows_to_delete.append(int(i))
        continue
    elif dz3[i] > dz3_max or  dz3[i] < dz3_min:
        rows_to_delete.append(int(i))
        continue
    elif dz4[i] > dz4_max or  dz4[i] < dz4_min:
        rows_to_delete.append(int(i))
        continue
    elif dz5[i] > dz5_max or  dz5[i] < dz5_min:
        rows_to_delete.append(int(i))
        continue

samples_validation = np.delete(samples_validation, rows_to_delete , 0)
dv_validation      = np.delete(dv_validation, rows_to_delete , 0)

logA = samples_validation[:,0]
ns = samples_validation[:,1]
Omegam = samples_validation[:,4]
Omegam_growth = samples_validation[:,5]

######

print('number of points to plot',len(samples_validation))




bin_count = 0
start_idx = 0
end_idx   = 0


#Loop over the models glue them together
#It's more intuitive to take one sample at a time, but that would require too many loading of the emulator
#The loop below is to get dv_predict of ALL samples, bin by bin.
for i in range(BIN_NUMBER):
    device='cuda'

    n_PCA = 780
    lsst_cov = config.cov[0:OUTPUT_DIM,0:OUTPUT_DIM] #np.loadtxt('lsst_y1_cov.txt')
    lsst_fid = config.dv_fid[0:OUTPUT_DIM] #np.loadtxt('lsst_y1_fid.txt')
    # do diagonalization C = QLQ^(T)
    eigensys = np.linalg.eig(lsst_cov)
    evals = eigensys[0]
    evecs = eigensys[1]

    # truncate PCAs
    # we need to keep ALL indices, cant forget unmodelled dimensions add to loss.
    sorted_idxs = np.argsort(1/evals)
    pc_idxs = sorted_idxs[:n_PCA]
    non_pc_idxs = sorted_idxs[n_PCA:] 
    cov_inv_pc = np.diag(1/evals[pc_idxs])
    cov_inv_npc = np.diag(1/evals[non_pc_idxs])
    emu = nn_pca_emulator(config.n_dim, OUTPUT_DIM, 
                        dv_fid, dv_std, cov_inv_pc,cov_inv_npc, 
                        pc_idxs, non_pc_idxs, lsst_cov,
                        device, reduce_lr=False)#, PCA_vecs=pca_vecs)
    emu.load('projects/lsst_y1/emulator_output/models/model_' + str(i+1))
    print('emulator loaded', i+1)
    tmp = []
    for j in range(len(samples_validation)):

        theta = torch.Tensor(samples_validation[j])

        dv_emu = emu.predict(theta)[0]
        tmp.append(dv_emu)
    tmp = np.array(tmp)

    if i==0:
        dv_predict = tmp
    else:
        dv_predict = np.append(dv_predict, tmp, axis = 1)



print("testing", np.shape(dv_predict))

chi2_list = []
count=0
count2=0
for i in range(len(dv_predict)):
    chi2 = get_chi2(dv_predict[i], dv_validation[i], mask, cov_inv_masked)
    chi2_list.append(chi2)
    if chi2>1:
        count +=1



chi2_list = np.array(chi2_list)

#print("testing",chi2_list)
print("average chi2 is: ", np.average(chi2_list))
print("Warning: This can be different from the training-validation loss. It depends on the mask file you use.")
print("points with chi2 > 1: ", count)


cmap = plt.cm.get_cmap('coolwarm')

#####PLOT 2d start######
#plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
#plt.scatter(logA, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(H0, Omegab, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())

plt.colorbar()

plt.xlabel(r'$\log A$')
plt.ylabel(r'$\Omega_m$')

#plt.xlabel(r'$\Omega_m$')
#plt.ylabel(r'$\Omega_m^{\rm growth}$')

#####PLOT 2d end######


##### PLOT 3d start###

# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")

# # Creating plot
# ax.scatter3D(logA, Omegam, Omegam_growth, c = chi2_list, s = 2, cmap=cmap, norm=matplotlib.colors.LogNorm())
# plt.title("simple 3D scatter plot")

# ax.azim = 150
# ax.elev = 15

##### PLOT 3d end###

plt.legend()
plt.savefig("validation.pdf")