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

end_idx = 35
models = ["lcdm", "w0wa", "w2bin", "w2bin_RT"]
for model in models:
    with open("./mapping/data/"+model+"/mapping_"+model+"_0_"+str(end_idx)+".txt",'w') as output:
        for i in range(end_idx+1):
            with open("./mapping/data/"+model+"/mapping_"+model+"_"+str(i)+".txt",'r') as input:
                for line in input:
                    output.write(line)
    samples = np.loadtxt("./mapping/data/"+model+"/mapping_"+model+"_0_"+str(end_idx)+".txt")

samples_lcdm = np.loadtxt("./mapping/data/lcdm/mapping_lcdm_0_35.txt")
samples_w0wa = np.loadtxt("./mapping/data/w0wa/mapping_w0wa_0_35.txt")
samples_w2bin = np.loadtxt("./mapping/data/w2bin/mapping_w2bin_0_35.txt")
samples_w2bin_RT = np.loadtxt("./mapping/data/w2bin_RT/mapping_w2bin_RT_0_35.txt")

omm_geo_1        = samples_lcdm[:,2]
omm_growth_1     = samples_lcdm[:,3]

omm_geo_2        = samples_w0wa[:,2]
omm_growth_2     = samples_w0wa[:,3]

omm_geo_3        = samples_w2bin[:,4]
omm_growth_3     = samples_w2bin[:,5]

omm_geo_4        = samples_w2bin_RT[:,4]
omm_growth_4     = samples_w2bin_RT[:,5]



### ========= First Plot ========= ###
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


### ========= Second Plot ========= ###
plt.figure().clear()

fig, ax = plt.subplots(1,4, figsize=(42,10))

ax[0].scatter(omm_geo_1, omm_growth_1,label=r'$\Lambda$CDM', s = 3)
ax[1].scatter(omm_geo_2, omm_growth_2,label=r'$w_0 w_a$', s = 3)
ax[2].scatter(omm_geo_3, omm_growth_3,label=r'2 w-bin with Pantheon', s = 3)
ax[3].scatter(omm_geo_4, omm_growth_4,label=r'2 w-bin with Roman', s = 3)


ax[0].set_xlabel(r'$\Omega^{\rm geo}$')
ax[1].set_xlabel(r'$\Omega^{\rm geo}$')
ax[2].set_xlabel(r'$\Omega^{\rm geo}$')
ax[3].set_xlabel(r'$\Omega^{\rm geo}$')
ax[0].set_ylabel(r'$\Omega^{\rm growth}$')
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[3].set_yticks([])

for i in range(0,len(ax)):
    ax[i].axline((0.29,0.29),(0.34,0.34), color='gray', ls='-',alpha=0.6)
    ax[i].axline((0.24,0.24),(0.40,0.24), color='gray', ls='--',alpha=0.6)
    ax[i].axline((0.24,0.40),(0.40,0.40), color='gray', ls='--',alpha=0.6)
    # ax[i].axline((0.29,0.29+0.0639225),(0.34,0.34+0.0639225), color='gray', ls='-.',alpha=0.6)
    # ax[i].axline((0.29,0.29-0.0639225),(0.34,0.34-0.0639225), color='gray', ls='-.',alpha=0.36)
    ax[i].set_xlim([0.271,0.355])
    ax[i].set_ylim([0.223,0.399])



ax[0].legend(fontsize=31, loc='upper left')
ax[1].legend(fontsize=31, loc='upper left')
ax[2].legend(fontsize=31, loc='upper left')
ax[3].legend(fontsize=31, loc='upper left')

plt.subplots_adjust(wspace=0.1)

#fig.tight_layout()
plt.savefig("./mapping/mapping_combined_v2.pdf",bbox_inches='tight')

