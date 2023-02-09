import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

from mpl_toolkits.mplot3d import axes3d   


#samples_1 = np.load('projects/lsst_y1/emulator_output/post/noshift_200k/train_post_samples.npy',allow_pickle=True)
#samples_2 = np.load('projects/lsst_y1/emulator_validation/train_validation_samples.npy',allow_pickle=True)

samples_1 = np.load('projects/des_y3/emulator_output_cs_wcdm_NLA/lhs/dvs_for_validation_2000k/train_samples.npy',allow_pickle=True)
samples_2 = np.load('projects/des_y3/emulator_output_cs_wcdm_NLA/lhs/dvs_for_validation_10k/validation_samples.npy',allow_pickle=True)

#samples_1 = np.load('projects/lsst_y1/emulator_output/lhs/dvs_for_training_800k/lhs_800k_samples.npy',allow_pickle=True) #LHS
#samples_2 = np.load('projects/lsst_y1/emulator_output/lhs/dvs_for_training_30k/train_30k_samples.npy')

# logA_0 = samples_0[:,0]
# Omegam_0 = samples_0[:,4]
# Omegam_growth_0 = samples_0[:,5]

print(samples_1[0])
print(samples_2[0])

logA_1 = samples_1[:,0]
H0_1 = samples_1[:,2]
Omegab_1 = samples_1[:,3]
Omegam_1 = samples_1[:,4]
Omegam_growth_1 = samples_1[:,5]
testp1_1 = samples_1[:,12]
testp2_1 = samples_1[:,13]

logA_2 = samples_2[:,0]
H0_2 = samples_2[:,2]
Omegab_2 = samples_2[:,3]
Omegam_2 = samples_2[:,4]
Omegam_growth_2 = samples_2[:,5]
testp1_2 = samples_2[:,12]
testp2_2 = samples_2[:,13]


# x0 = logA_0
# y0 = Omegam_0
# z0 = Omegam_growth_0

# x1 = logA_1
# y1 = Omegam_1
# z1 = Omegam_growth_1

# x2 = logA_2
# y2 = Omegam_2
# z2 = Omegam_growth_2

x1 = testp1_1
y1 = testp2_1
z1 = Omegam_growth_1

x2 = testp1_2
y2 = testp2_2
z2 = Omegam_growth_2

# print("number of points from LHC: ", len(x0))
print("number of points from chains: ", len(x1))


#plt.scatter(x0, y0, c ="pink", label='Latin Hyper Cube', s = 2)
plt.scatter(x1, y1, c ='blue', label='Chains', s = 0.01, linewidths=0.5)
plt.scatter(x2, y2, c ='red', label='validation', s = 2, linewidths=0.5)
plt.xlabel(r'$P1$')
plt.ylabel(r'$P2$')

# plt.scatter(z0, y0, c ="pink", label='Latin Hyper Cube', s = 2)
# plt.scatter(z1, y1, c ='blue', label='Chain 1 + 2', s = 0.001, linewidths=0.5)
# plt.xlabel(r'$\Omega_m^{\rm growth}$')
# plt.ylabel(r'$\Omega_m$')



###########3D plot#######


# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
 
# # Creating plot
# ax.scatter3D(x1, y1, z1, color = "pink", s = 2)
# ax.scatter3D(x1, y1, z1, color = "blue", s = 0.001, linewidths=0.5)
# plt.title("simple 3D scatter plot")

# ax.azim = 200
# ax.elev = 45


plt.legend()

plt.savefig("samples.pdf")