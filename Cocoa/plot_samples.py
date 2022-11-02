import numpy as np
import matplotlib.pyplot as plt
import os



samples_0 = np.load('projects/lsst_y1/emulator_output/dvs_for_training_100k/train_100k_samples.npy')
samples_1 = np.load('projects/lsst_y1/emulator_output/post/samples_from_posterior.npy')

logA_0 = samples_0[:,0]
Omegam_0 = samples_0[:,4]

logA_1 = samples_1[:,0]
Omegam_1 = samples_1[:,4]

plt.scatter(logA_0, Omegam_0, c ="pink", label='Latin Hyper Cube', s = 2)

plt.scatter(logA_1, Omegam_1, c ="blue", label='Chain 1', s = 2)

print("LCS points: ", np.shape(logA_0))
print("chain1 points", np.shape(logA_1))

plt.xlabel(r'$\log A$')
plt.ylabel(r'$\Omega_m$')

plt.legend()

plt.savefig("samples.pdf")