##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


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

#emu_model_2x2 = 'projects/lsst_y1/emulator_output_3x2/modelsTransformer/model_2x2'; model_prefix = "Transformer"
#emu_model_2x2 = 'projects/lsst_y1/emulator_output_3x2/modelsResNet/model_2x2'; model_prefix = "ResNet"
#emu_model_2x2 = 'projects/lsst_y1/emulator_output_3x2/modelsMLP/model_2x2'; model_prefix = "MLP"
emu_model_2x2 = 'projects/lsst_y1/emulator_output_3x2/modelsTransformer/2MSimple_1D_CNN/model_2x2'; model_prefix = "Simple_1D_CNN"
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
if cut_boundary:
    print("setting ranges of validation plot")
    rows_to_delete = []

    ## rg = range to be taken from the boundary, eg. rg=0.1 means 80% of EE box
    rg = 0.00
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

########################################
print("validating 3x2pt with seperating cosmic shear and 2x2")
for i in range(2):
    device='cpu'
    emu = NNEmulator(config.n_dim, BIN_SIZE, config.dv_fid, config.dv_std, cov, config.dv_fid,config.dv_fid, config.lhs_minmax ,device) #should privde dv_max instead of dv_fid, but emu.load will make it correct
    if i ==0:
        print("using emulator of cosmic shear part")
        emu.load(emu_model_cs, map_location=torch.device('cpu'))
        print('emulator loaded cosmic shear')
        tmp = []
        for j in range(len(samples_validation)):
            theta = torch.Tensor(samples_validation[j][0:INPUT_DIM_CS])
            dv_emu = emu.predict(theta)[0]
            tmp.append(dv_emu)
        tmp = np.array(tmp)

        dv_predict = tmp
    elif i==1:
        print("using emulator of 2x2 part")
        emu.load(emu_model_2x2, map_location=torch.device('cpu'))
        print('emulator loaded 2x2pt')
        tmp = []
        for j in range(len(samples_validation)):
            theta = torch.Tensor(samples_validation[j])
            dv_emu = emu.predict(theta)[0]
            tmp.append(dv_emu)
        tmp = np.array(tmp)

        dv_predict = np.append(dv_predict, tmp, axis = 1) # add cs and 2x2 part
########################################

########################################
# print("validating 3x2pt DIRECTLY")
# device='cpu'
# emu = NNEmulator(config.n_dim, BIN_SIZE, config.dv_fid, config.dv_std, cov, config.dv_fid,config.dv_fid, config.lhs_minmax ,device) #should privde dv_max instead of dv_fid, but emu.load will make it correct
# print("using emulator of 3x2")
# emu.load(emu_model_3x2, map_location=torch.device(device))
# print('emulator loaded 3x2')
# tmp = []
# for j in range(len(samples_validation)):
#     theta = torch.Tensor(samples_validation[j])
#     dv_emu = emu.predict(theta)[0]
#     tmp.append(dv_emu)
# tmp = np.array(tmp)

# dv_predict = tmp

########################################
        
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

np.savetxt('chi2_list_3x2'+model_prefix+'.txt',chi2_list)

#print("testing",chi2_list)
print("average chi2 is: ", np.average(chi2_list))
print("Warning: This can be different from the training-validation loss. It depends on the mask file you use.")
print("points with chi2 > 1: ", count)


###PLOT chi2 start

np.savetxt('chi2_list.txt',chi2_list)
num_bins = 100
plt.xlabel(r'$\chi^2$')
plt.ylabel('distribution')
plt.xscale('log')

plt.hist(chi2_list, num_bins, 
                            density = 1, 
                            color ='green',
                            alpha = 0.7)


plt.savefig("validation_chi2.pdf")

####PLOT chi2 end

#####PLOT 2d start######
plt.figure().clear()
cmap = plt.cm.get_cmap('coolwarm')

plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
plt.xlabel(r'$\log A$')
plt.ylabel(r'$\Omega_m$')

np.savetxt("chi2_distribution_3x2"+model_prefix+'.txt',[logA, Omegam, chi2_list],header="logA,Omegam, chi2_list")

cb = plt.colorbar()
plt.savefig("validation_lcdm_LSST_3x2_"+model_prefix+".pdf")

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




