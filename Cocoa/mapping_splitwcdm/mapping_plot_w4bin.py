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


end_idx = 34

with open("./mapping_splitwcdm/mapping_w4bin_0_"+str(end_idx)+".txt",'w') as output:
    for i in range(end_idx+1):
        with open("./mapping_splitwcdm/mapping_w4bin_"+str(i)+".txt",'r') as input:
            for line in input:
                output.write(line)

samples = np.loadtxt("./mapping_splitwcdm/mapping_w4bin_0_"+str(end_idx)+".txt")
print("checking shape: ",np.shape(samples))


w0             = samples[:,0]
w1             = samples[:,1]
w2             = samples[:,2]
w3             = samples[:,3]
omm_geo        = samples[:,4]
omm_growth     = samples[:,5]
w_geo          = samples[:,6]
w_growth       = samples[:,7]
delta_omm      = omm_growth - omm_geo
delta_omm_abs  = np.absolute(omm_growth - omm_geo)

pc_geo         = -(w_geo)    + (omm_geo)
pc_growth      = -(w_growth) + (omm_growth)

# ### ========= Second Plot ========= ###
plt.figure().clear()
plt.scatter(pc_geo, pc_growth,label=r'Best fit of 4 w-bin', s = 2)

plt.xlabel(r'$\Omega^{\rm geo} - w^{\rm geo}$')
plt.ylabel(r'$\Omega^{\rm growth} - w^{\rm growth}$')
plt.axline((0.94,0.94),(1.7,1.7), color='gray', ls='-',alpha=0.6)
# plt.axline((0.29,0.29+0.0639225),(0.34,0.34+0.0639225), color='gray', ls='-.',alpha=0.6)
# plt.axline((0.29,0.29-0.0639225),(0.34,0.34-0.0639225), color='gray', ls='-.',alpha=0.36)

plt.legend()
plt.savefig("./mapping_splitwcdm/mapping_w4bin_v2.pdf")

# ### ========= Third Plot ========= ###
# plt.figure().clear()

# ##### sns pairplot #####
# # omm_geo_growth = samples[:, [2, 3]]
# # sns.pairplot(pd.DataFrame(omm_geo_growth), kind='kde')
# ##### sns pairplot

# ##### GetDist #####
# names  = ['w0', 'w1', 'w2', 'w3', 'omm_geo', 'omm_growth','logA','ns'] 
# labels = [r'$w_0$', r'$w_1$',r'$w_2$',r'$w_3$',r'\Omega^{\rm geo}',r'\Omega^{\rm growth}',r'logA', r'ns']
# getdist_samples = MCSamples(samples=samples,names = names, labels = labels)
# g = plots.get_single_plotter()
# g.plot_2d([getdist_samples], 'omm_geo', 'omm_growth', filled=True,contour_colors='#377eb8')

# plt.axline((0.29,0.29),(0.365,0.365), color='gray', ls='-',alpha=0.3)
# plt.axline((0.29,0.29+0.0639225),(0.365,0.365+0.0639225), color='gray', ls='-.',alpha=0.3)
# plt.axline((0.29,0.29-0.0639225),(0.365,0.365-0.0639225), color='gray', ls='-.',alpha=0.3)
# ##### GetDist #####

# plt.savefig("./mapping_splitwcdm/mapping_w2bin_v3.pdf")
