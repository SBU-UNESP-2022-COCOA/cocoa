##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from cocoa_emu import Config, NNEmulator
from cocoa_emu.sampling import EmuSampler

torch.set_default_dtype(torch.double)

def get_chi2(theta, dv_exact, mask):
    if(config.emu_type=='nn'):
        theta = torch.Tensor(theta)
    elif(config.emu_type=='gp'):
        theta = theta[np.newaxis]

    dv_emu = emu.predict(theta)[0]
    delta_dv = (dv_emu - np.float32(dv_exact) )[mask]
    ## GPU emulators works well with float32
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(config.masked_inv_cov)) , delta_dv  )   
    #print(dv_emu.dtype, np.float32(dv_exact).dtype, config.masked_inv_cov.dtype) 
    #print(np.float32(dv_exact))
    return chi2


os.environ["OMP_NUM_THREADS"] = "1"

configfile = './projects/lsst_y1/train_emulator.yaml'
config = Config(configfile)


samples_validation = np.load('projects/lsst_y1/emulator_output/emu_validation/validation_samples.npy')
dv_validation = np.load('projects/lsst_y1/emulator_output/emu_validation/validation_data_vectors.npy')

#print(samples_validation[1])

logA = samples_validation[:,0]
ns = samples_validation[:,1]
Omegam = samples_validation[:,4]
Omegam_growth = samples_validation[:,5]

#####Set Range####
print("setting ranges of validation plot")
rows_to_delete = []

logA_max = 3.078
logA_min = 2.98
omm_max  = 0.32
omm_min  = 0.308
ommg_max = 0.39
ommg_min = 0.25
ns_min   = 0.94
ns_max   = 0.97

# for i in range(len(samples_validation)):
#     if logA[i]>logA_max or logA[i]<logA_min:
#         rows_to_delete.append(int(i))
#         continue
#     elif Omegam[i] > omm_max or  Omegam[i] < omm_min:
#         rows_to_delete.append(int(i))
#         continue
#     elif Omegam_growth[i] > ommg_max or  Omegam_growth[i] < ommg_min:
#         rows_to_delete.append(int(i))
#         continue
#     elif ns[i] > ns_max or  ns[i] < ns_min:
#         rows_to_delete.append(int(i))
#         continue

# samples_validation = np.delete(samples_validation, rows_to_delete , 0)
# dv_validation      = np.delete(dv_validation, rows_to_delete , 0)

# logA = samples_validation[:,0]
# ns = samples_validation[:,1]
# Omegam = samples_validation[:,4]
# Omegam_growth = samples_validation[:,5]

######



if config.probe =='cosmic_shear':
    dv_validation = dv_validation[:,:780]
    mask = config.mask[0:780]
else:
    print("3x2 not tested")
    quit()

print('number of points to plot',len(samples_validation))

device='cuda'
emu = NNEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std, config.masked_inv_cov, config.dv_fid, device) #should privde dv_max instead of dv_fid, but emu.load will make it correct
emu.load('projects/lsst_y1/emulator_output/models/model')
print('emulator loaded')

chi2_list = []
count=0
count2=0
for i in range(len(samples_validation)):
    chi2 = get_chi2(samples_validation[i], dv_validation[i], mask)
    chi2_list.append(chi2)
    if chi2>100:
        count +=1



chi2_list = np.array(chi2_list)

#print("testing",chi2_list)
print("average chi2 is: ", np.average(chi2_list))
print("Warning: This can be different from the training-validation loss. It depends on the mask file you use.")
print("points with chi2 > 100: ", count)


cmap = plt.cm.get_cmap('coolwarm')

#####PLOT 2d start######
#plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
#plt.scatter(logA, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())


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