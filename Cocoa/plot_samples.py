import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

from mpl_toolkits.mplot3d import axes3d   


samples_0 = np.load('projects/lsst_y1/emulator_output/dvs_for_training_100k/train_100k_samples.npy')
samples_1 = np.load('projects/lsst_y1/emulator_output/post/samples_from_posterior.npy',allow_pickle=True)
samples_1 = np.load('projects/lsst_y1/emulator_output/post/3sigmaonly/train_post_samples.npy',allow_pickle=True)


logA_0 = samples_0[:,0]
Omegam_0 = samples_0[:,4]
Omegam_growth_0 = samples_0[:,5]

logA_1 = samples_1[:,0]
Omegam_1 = samples_1[:,4]
Omegam_growth_1 = samples_1[:,5]


x0 = logA_0
y0 = Omegam_0
z0 = Omegam_growth_0

x1 = logA_1
y1 = Omegam_1
z1 = Omegam_growth_1

print("number of points from LHC: ", len(x0))
print("number of points from chains: ", len(x1))


plt.scatter(x0, y0, c ="pink", label='Latin Hyper Cube', s = 2)
plt.scatter(x1, y1, c ='blue', label='Chain 1 + 2', s = 0.001, linewidths=0.5)
plt.xlabel(r'$\log A$')
plt.ylabel(r'$\Omega_m$')

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