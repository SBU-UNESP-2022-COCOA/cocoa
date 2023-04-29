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


end_idx = 19

with open("./mapping/data/lcdm_planck/mapping_lcdm_0_"+str(end_idx)+".txt",'w') as output:
    for i in range(end_idx+1):
        with open("./mapping/data/lcdm_planck/mapping_lcdm_"+str(i)+".txt",'r') as input:
            for line in input:
                output.write(line)

samples = np.loadtxt("./mapping/data/lcdm_planck/mapping_lcdm_0_"+str(end_idx)+".txt")
print("checking shape: ",np.shape(samples))


logA           = samples[:,0]
ns             = samples[:,1]
omm_geo        = samples[:,2]
omm_growth     = samples[:,3]
delta_omm      = omm_growth - omm_geo
delta_omm_abs  = np.absolute(omm_growth - omm_geo)

# ### ========= Second Plot ========= ###
plt.figure().clear()
plt.scatter(omm_geo, omm_growth,label=r'Best fit of LCDM', s = 2)

plt.xlabel(r'$\Omega^{\rm geo}$')
plt.ylabel(r'$\Omega^{\rm growth}$')
plt.axline((0.29,0.29),(0.34,0.34), color='gray', ls='-',alpha=0.6)
plt.axline((0.29,0.29+0.0639225),(0.34,0.34+0.0639225), color='gray', ls='-.',alpha=0.6)
plt.axline((0.29,0.29-0.0639225),(0.34,0.34-0.0639225), color='gray', ls='-.',alpha=0.36)

plt.legend()
plt.savefig("./mapping/mapping_lcdm_v2.pdf",bbox_inches='tight')

### ========= Third Plot ========= ###
plt.figure().clear()

##### sns pairplot #####
# omm_geo_growth = samples[:, [2, 3]]
# sns.pairplot(pd.DataFrame(omm_geo_growth), kind='kde')
##### sns pairplot

##### GetDist #####
names  = ['logA', 'ns', 'omm_geo', 'omm_growth'] 
labels = [r'$\mathrm{logA}$', r'$n_s$',r'\Omega^{\rm geo}',r'\Omega^{\rm growth}']
getdist_samples = MCSamples(samples=samples,names = names, labels = labels)
analysissettings={'smooth_scale_1D':0.0,'smooth_scale_2D':0.0,'ignore_rows': u'0.0', #read thinned chains
'range_confidence' : u'0.005'}
g = plots.get_single_plotter()
g.plot_2d([getdist_samples], 'omm_geo', 'omm_growth', filled=True,contour_colors='#377eb8',analysis_settings=analysissettings)

plt.axline((0.29,0.29),(0.365,0.365), color='gray', ls='-',alpha=0.3)
plt.axline((0.29,0.29+0.0639225),(0.365,0.365+0.0639225), color='gray', ls='-.',alpha=0.3)
plt.axline((0.29,0.29-0.0639225),(0.365,0.365-0.0639225), color='gray', ls='-.',alpha=0.3)
plt.axline((0.29,0.29+0.0319612),(0.365,0.365+0.0319612), color='gray', ls='--',alpha=0.3)
plt.axline((0.29,0.29-0.0319612),(0.365,0.365-0.0319612), color='gray', ls='-.',alpha=0.3)
##### GetDist #####

plt.savefig("./mapping/mapping_lcdm_v3.pdf",bbox_inches='tight')
