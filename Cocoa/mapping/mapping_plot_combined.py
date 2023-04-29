import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import axes3d
from getdist import plots, MCSamples
import getdist
import seaborn as sns
import matplotlib.pylab as pylab
params = {'legend.fontsize': 28,
         'axes.labelsize': 28,
         'axes.titlesize':24,
         'xtick.labelsize':24,
         'ytick.labelsize':24}
pylab.rcParams.update(params)


samples_lcdm = np.loadtxt("./mapping/data/lcdm_planck/mapping_lcdm_0_19.txt")
samples_w0wa = np.loadtxt("./mapping/data/w0wa_planck_BAO_SN/mapping_w0wa_0_35.txt")
samples_w2bin = np.loadtxt("./mapping/data/2wbin_casarini/mapping_w2bin_0_37.txt")

omm_geo_1        = samples_lcdm[:,2]
omm_growth_1     = samples_lcdm[:,3]

omm_geo_2        = samples_w0wa[:,2]
omm_growth_2     = samples_w0wa[:,3]

omm_geo_3        = samples_w2bin[:,4]
omm_growth_3     = samples_w2bin[:,5]



# ### ========= Second Plot ========= ###
plt.figure().clear()

fig, ax = plt.subplots(1,3, figsize=(35,10))

ax[0].scatter(omm_geo_1, omm_growth_1,label=r'Best fit of $\Lambda$CDM', s = 3)
ax[1].scatter(omm_geo_2, omm_growth_2,label=r'Best fit of $w_0 w_a$', s = 3)
ax[2].scatter(omm_geo_3, omm_growth_3,label=r'Best fit of 2 w-bin', s = 3)


ax[0].set_xlabel(r'$\Omega^{\rm geo}$')
ax[1].set_xlabel(r'$\Omega^{\rm geo}$')
ax[2].set_xlabel(r'$\Omega^{\rm geo}$')
ax[0].set_ylabel(r'$\Omega^{\rm growth}$')
ax[1].set_yticks([])
ax[2].set_yticks([])

for i in range(0,len(ax)):
    ax[i].axline((0.29,0.29),(0.34,0.34), color='gray', ls='-',alpha=0.6)
    ax[i].axline((0.29,0.29+0.0639225),(0.34,0.34+0.0639225), color='gray', ls='-.',alpha=0.6)
    ax[i].axline((0.29,0.29-0.0639225),(0.34,0.34-0.0639225), color='gray', ls='-.',alpha=0.36)
    ax[i].set_xlim([0.271,0.355])
    ax[i].set_ylim([0.223,0.41])



ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.subplots_adjust(wspace=0.1)

#fig.tight_layout()
plt.savefig("./mapping/mapping_combined.pdf",bbox_inches='tight')


