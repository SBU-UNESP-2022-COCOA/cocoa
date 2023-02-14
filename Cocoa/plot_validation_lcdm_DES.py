##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from cocoa_emu import Config, NNEmulator, boundary_check

#torch.set_default_dtype(torch.double)

model_path = 'projects/des_y3/emulator_output_cs_lcdm_NLA/lhs/dvs_for_training_2000k/model_'

OUTPUT_DIM = 400
BIN_SIZE   = 26 # number of angular bins in each z-bin
BIN_NUMBER = 30 # number of z-bins

##3x2 setting: separate cosmic shear and 2x2pt
BIN_SIZE   = 400 # number of angular bins in each z-bin
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
    #print("testing", (dv_predict - dv_exact) / dv_exact)
    delta_dv = (dv_predict - np.float32(dv_exact) )[mask]
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv)) , delta_dv  )   
    return chi2


os.environ["OMP_NUM_THREADS"] = "1"

configfile = './projects/des_y3/lhs_cs_NLA.yaml'
config = Config(configfile)


#samples_validation = np.load('./projects/des_y3/emulator_output_cs_wcdm_NLA/lhs/dvs_for_validation_10k/validation_samples.npy')
#dv_validation      = np.load('./projects/des_y3/emulator_output_cs_wcdm_NLA/lhs/dvs_for_validation_10k/validation_data_vectors.npy')

samples_validation = np.load('./projects/des_y3/emulator_output_cs_lcdm_NLA/lhs/dvs_for_validation_10k/validation_samples.npy')
dv_validation      = np.load('./projects/des_y3/emulator_output_cs_lcdm_NLA/lhs/dvs_for_validation_10k/validation_data_vectors.npy')

if config.probe =='cosmic_shear':
    dv_validation = dv_validation[:,:OUTPUT_DIM]
    mask = config.mask[0:OUTPUT_DIM]
else:
    print("3x2 not tested")
    quit()
    
cov            = config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM]
cov_inv        = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM])
cov_inv_masked = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM][mask][:,mask])

#print(samples_validation[1])

#Note: the following order is not the same in LCDM and wCDM
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

IA1 = samples_validation[:,10]
IA2 = samples_validation[:,11]

#####Set Range####
print("setting ranges of validation plot")
rows_to_delete = []

## rg = range to be taken from the boundary, eg. rg=0.1 means 80% of EE box
rg = 0.05
logA_max = 3.21 - rg*(3.21-2.84)
logA_min = 2.84 + rg*(3.21-2.84)
omm_max  = 0.4  - rg*(0.4-0.24)
omm_min  = 0.24 + rg*(0.4-0.24)
ommg_max = 0.4  - rg*(0.4-0.24)
ommg_min = 0.24 + rg*(0.4-0.24)
ns_max   = 1.0  - rg*(1.0-0.92)
ns_min   = 0.92 + rg*(1.0-0.92)
omb_max  = 0.06 - rg*(0.06-0.04)
omb_min  = 0.04 + rg*(0.06-0.04)
H0_min   = 61   + rg*(73-61)
H0_max   = 73   - rg*(73-61)

# #4sigma version
# dz1_max  = 0.072  - rg*(0.072+0.072) 
# dz1_min  = -0.072 + rg*(0.072+0.072) 
# dz2_max  = 0.060  - rg*(0.060+0.060) 
# dz2_min  = -0.060 + rg*(0.060+0.060) 
# dz3_max  = 0.044  - rg*(0.044+0.044) 
# dz3_min  = -0.044 + rg*(0.044+0.044) 
# dz4_max  = 0.068  - rg*(0.068+0.068) 
# dz4_min  = -0.068 + rg*(0.068+0.068) 

rg = 0.0
# 3sigma version
dz1_max  = 0.054  - rg*(0.054+0.054) 
dz1_min  = -0.054 + rg*(0.054+0.054) 
dz2_max  = 0.045  - rg*(0.045+0.045) 
dz2_min  = -0.045 + rg*(0.045+0.045) 
dz3_max  = 0.033  - rg*(0.033+0.033) 
dz3_min  = -0.033 + rg*(0.033+0.033) 
dz4_max  = 0.051  - rg*(0.051+0.051) 
dz4_min  = -0.051 + rg*(0.051+0.051) 

#The exact number should be 5.0000040; can print from reduced range
# IA1_max  = 5  - rg*(10) 
# IA1_min  = -5 + rg*(10)
# IA2_max  = 5  - rg*(10) 
# IA2_min  = -5 + rg*(10)

IA1_max  = +2  - rg*(4) 
IA1_min  = -2 + rg*(4)
IA2_max  = +2  - rg*(2) 
IA2_min  = -2  + rg*(2) 


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
    elif H0[i] > H0_max or  H0[i] < H0_min:
        rows_to_delete.append(int(i))
        continue
    if dz1[i] > dz1_max or dz1[i] < dz1_min:
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
    elif IA1[i] > IA1_max or  IA1[i] < IA1_min:
        rows_to_delete.append(int(i))
        continue
    elif IA2[i] > IA2_max or  IA2[i] < IA2_min:
        rows_to_delete.append(int(i))
        continue

# ##=== another way start; they are almost equal, of course the first way has more flexibility
# rows_to_delete = []
# for k in range(len(samples_validation)):
#     if boundary_check(samples_validation[k], config.lhs_minmax, rg=0.1)==True:
#         rows_to_delete.append(int(k))
# ##=== another way end

samples_validation = np.delete(samples_validation, rows_to_delete , 0)
dv_validation      = np.delete(dv_validation, rows_to_delete , 0)

print("dv shape after boundary removal", np.shape(dv_validation))

# # #======Special normalization: this is probably not necessary======
# samples_validation[:,12] = samples_validation[:,12] / 4
# samples_validation[:,13] = samples_validation[:,13] / 4


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

IA1 = samples_validation[:,10]
IA2 = samples_validation[:,11]

######

print('number of points to plot',len(samples_validation))




bin_count = 0
start_idx = 0
end_idx   = 0


#Loop over the models glue them together
#It's more intuitive to take one sample at a time, but that would require too many loading of the emulator
#The loop below is to get dv_predict of ALL samples, bin by bin.
for i in range(BIN_NUMBER):
    device='cpu'
    emu = NNEmulator(config.n_dim, BIN_SIZE, config.dv_fid, config.dv_std, cov, config.dv_fid,config.dv_fid, config.lhs_minmax ,device) #should privde dv_max instead of dv_fid, but emu.load will make it correct; same for dv_mean
    emu.load(model_path + str(i+1) , map_location=torch.device('cpu'))
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

###PLOT chi2 start

num_bins = 100
plt.xlabel(r'$\chi^2$')
plt.ylabel('distribution')
plt.xscale('log')

plt.hist(chi2_list, num_bins, 
                            density = 1, 
                            color ='green',
                            alpha = 0.7)


plt.savefig("validation_chi2_lcdm_DES.pdf")

####PLOT chi2 end

#####PLOT 2d start######
plt.figure().clear()


# plt.scatter(Omegam, logA, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
# plt.xlabel(r'$\Omega_m^{\rm geo}$')
# plt.ylabel(r'$\log A$')

plt.scatter(IA1, IA2, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
plt.xlabel(r'$IA_1$')
plt.ylabel(r'$IA_2$')

# plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
# plt.xlabel(r'$omm^{\rm geo}$')
# plt.ylabel(r'$omm^{\rm growth}$')



cb = plt.colorbar()

plt.legend()
plt.savefig("validation_lcdm_DES.pdf")

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




