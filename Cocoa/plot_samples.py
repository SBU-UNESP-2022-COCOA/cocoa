import numpy as np
import matplotlib.pyplot as plt
import os



samples_0 = np.load('projects/lsst_y1/emulator_output/dvs_for_training_100k/train_100k_samples.npy')
samples_1 = np.load('projects/lsst_y1/emulator_output/post/samples_from_posterior.npy',allow_pickle=True)


logA_0 = samples_0[:,0]
Omegam_0 = samples_0[:,4]
Omegam_growth_0 = samples_0[:,5]

logA_1 = samples_1[:,0]
Omegam_1 = samples_1[:,4]
Omegam_growth_1 = samples_1[:,5]

# plt.scatter(logA_0, Omegam_0, c ="pink", label='Latin Hyper Cube', s = 2)
# plt.scatter(logA_1, Omegam_1, c ="blue", label='Chain 1 + 2', s = 2)
# plt.xlabel(r'$\log A$')
# plt.ylabel(r'$\Omega_m$')

plt.scatter(Omegam_0, Omegam_growth_0, c ="pink", label='Latin Hyper Cube', s = 2)
plt.scatter(Omegam_1, Omegam_growth_1, c ="blue", label='Chain 1', s = 2)
plt.xlabel(r'$\Omega_m$')
plt.ylabel(r'$\Omega_m^{\rm growth}$')

# print("testing")
# plt.scatter(samples_0[:,8], samples_0[:,9], c ="pink", label='Latin Hyper Cube', s = 2)
# plt.scatter(samples_1[:,8], samples_1[:,9], c ="blue", label='Chain 1 + 2', s = 2)
# plt.xlabel(r'$test$')
# plt.ylabel(r'$test$')

#print("testing", np.shape(logA_0))



plt.legend()

plt.savefig("samples.pdf")