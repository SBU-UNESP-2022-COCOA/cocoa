import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import axes3d
from getdist import plots, MCSamples
import getdist
import seaborn as sns


end_idx = 35

with open("./mapping/mapping_w0wa_0_"+str(end_idx)+".txt",'w') as output:
    for i in range(end_idx+1):
        with open("./mapping/mapping_w0wa_"+str(i)+".txt",'r') as input:
            for line in input:
                output.write(line)

samples = np.loadtxt("./mapping/mapping_w0wa_0_"+str(end_idx)+".txt")
print("checking shape: ",np.shape(samples))


w0             = samples[:,0]
wa             = samples[:,1]
omm_geo        = samples[:,2]
omm_growth     = samples[:,3]
delta_omm      = omm_growth - omm_geo
delta_omm_abs  = np.absolute(omm_growth - omm_geo)



# ### ========= First Plot ========= ###
plt.figure().clear()
cmap = plt.cm.get_cmap('coolwarm')
plt.scatter(w0, wa, c=delta_omm_abs, label=r'$|\Omega_m^{\rm growth} - \Omega_m^{\rm geo}|$', s = 2, cmap=cmap)

cb = plt.colorbar()
plt.xlabel(r'$w_0$')
plt.ylabel(r'$w_a$')
plt.axhline(0.0, color="gray", ls='--',alpha=0.3)
plt.axvline(-1.0, color="gray", ls='--',alpha=0.3)

plt.legend()
plt.savefig("./mapping/mapping_w0wa_v1.pdf")


# ### ========= Second Plot ========= ###
plt.figure().clear()
plt.scatter(omm_geo, omm_growth,label=r'Best fit of $w_0 w_a$', s = 2)

plt.xlabel(r'$\Omega^{\rm geo}$')
plt.ylabel(r'$\Omega^{\rm growth}$')
plt.axline((0.29,0.29),(0.34,0.34), color='gray', ls='-',alpha=0.6)
plt.axline((0.29,0.29+0.0639225),(0.34,0.34+0.0639225), color='gray', ls='-.',alpha=0.6)
plt.axline((0.29,0.29-0.0639225),(0.34,0.34-0.0639225), color='gray', ls='-.',alpha=0.36)

plt.legend()
plt.savefig("./mapping/mapping_w0wa_v2.pdf")

### ========= Third Plot ========= ###
plt.figure().clear()

##### sns pairplot #####
# omm_geo_growth = samples[:, [2, 3]]
# sns.pairplot(pd.DataFrame(omm_geo_growth), kind='kde')
##### sns pairplot

##### GetDist #####
names  = ['w0', 'wa', 'omm_geo', 'omm_growth'] 
labels = [r'$w_0$', r'$w_a$',r'\Omega^{\rm geo}',r'\Omega^{\rm growth}']
getdist_samples = MCSamples(samples=samples,names = names, labels = labels)
g = plots.get_single_plotter()
g.plot_2d([getdist_samples], 'omm_geo', 'omm_growth', filled=True,contour_colors='#377eb8')

plt.axline((0.29,0.29),(0.365,0.365), color='gray', ls='-',alpha=0.3)
plt.axline((0.29,0.29+0.0639225),(0.365,0.365+0.0639225), color='gray', ls='-.',alpha=0.3)
plt.axline((0.29,0.29-0.0639225),(0.365,0.365-0.0639225), color='gray', ls='-.',alpha=0.3)
##### GetDist #####

plt.savefig("./mapping/mapping_w0wa_v3.pdf")
