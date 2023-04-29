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
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


end_idx = 21

file_prefix = "./mapping/data/2wbin_RT/mapping_w2bin_RT_"

with open(file_prefix + "0_"+str(end_idx)+".txt",'w') as output:
    for i in range(end_idx+1):
        with open(file_prefix+str(i)+".txt",'r') as input:
            for line in input:
                output.write(line)

samples = np.loadtxt(file_prefix + "0_"+str(end_idx)+".txt")
print("checking shape: ",np.shape(samples))


w0             = samples[:,0]
w1             = samples[:,1]
w2             = samples[:,2]
w3             = samples[:,3]
omm_geo        = samples[:,4]
omm_growth     = samples[:,5]
delta_omm      = omm_growth - omm_geo
delta_omm_abs  = np.absolute(omm_growth - omm_geo)



# ### ========= Second Plot ========= ###
plt.figure().clear()
plt.scatter(omm_geo, omm_growth,label=r'Best fit of 2 w-bin (Roman)', s = 2)

plt.xlabel(r'$\Omega^{\rm geo}$')
plt.ylabel(r'$\Omega^{\rm growth}$')
plt.axline((0.29,0.29),(0.34,0.34), color='gray', ls='-',alpha=0.6)
plt.axline((0.29,0.29+0.0639225),(0.34,0.34+0.0639225), color='gray', ls='-.',alpha=0.6)
plt.axline((0.29,0.29-0.0639225),(0.34,0.34-0.0639225), color='gray', ls='-.',alpha=0.36)

plt.legend()
plt.savefig("./mapping/mapping_w2bin_RT_v2.pdf",bbox_inches='tight')

### ========= Third Plot ========= ###

