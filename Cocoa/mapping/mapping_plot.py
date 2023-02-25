import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import axes3d


samples = np.loadtxt("mapping_w0wa.txt")

w0         = samples[:,0]
wa         = samples[:,1]
omm_geo    = samples[:,2]
omm_growth = samples[:,3]
delta_omm  = omm_growth - omm_geo



### ========= First Plot ========= ###
plt.figure().clear()
cmap = plt.cm.get_cmap('coolwarm')
plt.scatter(w0, wa, c=delta_omm, label=r'test', s = 5, cmap=cmap)

cb = plt.colorbar()
plt.xlabel(r'$w_0$')
plt.ylabel(r'$w_a$')

plt.legend()
plt.savefig("mapping_w0wa_v1.pdf")


### ========= Second Plot ========= ###
plt.figure().clear()
plt.scatter(omm_geo, omm_growth,label=r'test2', s = 5)

plt.xlabel(r'$\Omega^{\rm geo}$')
plt.ylabel(r'$\Omega^{\rm growth}$')

plt.legend()
plt.savefig("mapping_w0wa_v2.pdf")