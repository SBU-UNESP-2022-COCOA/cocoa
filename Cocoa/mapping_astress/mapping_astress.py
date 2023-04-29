import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys, time
from io import StringIO 
import torch
from cocoa_emu import Config, NNEmulator
from cobaya.yaml import yaml_load_file
from cobaya.run import run
from cobaya.post import post
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt
from getdist import MCSamples
from getdist import loadMCSamples
#from mpi4py import MPI



chain_range = np.arange(1,10)
astress_chains_root    = "./mapping_astress/astress_dv/astress_"
g0 = [0, -1.0, -0.5, 0.5, 1.0,      -1.0, -0.5, 0.5, 1.0,]
cg = [0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03]
ggsplit_yaml_file   = "./projects/lsst_y1/EXAMPLE_MCMC514.yaml"
ggsplit_chains_root = "./projects/lsst_y1/chains/EXAMPLE_MCMC514"


omm_geo    = []
omm_growth = []
chi2       = []

start_time = time.time()
for i in chain_range:
	### No dv generation needed
	### Minimize the likelihood with GG-split
	info_emu = yaml_load_file(ggsplit_yaml_file)
	# need more accuracy for anistropic stress injection (mainly due to linear scale cut)
	info_emu["sampler"] = {'minimize': {'ignore_prior': False, 'best_of': 10, 'override_bobyqa': {'rhoend': 0.005}}}
	info_emu["force"]   = True
	# Add a prior on H0 and ns that has 4 times larger sigma. Values get from Planck 2018 Table 2.
	# The temperature boost should make this not model related; but still provide good additional information for LSST
	info_emu["params"]['ns']['prior']  = {'dist': 'norm', 'loc':  0.9649, 'scale': 0.0044*4}
	info_emu["params"]['H0']['prior']  = {'dist': 'norm', 'loc':   67.27, 'scale':  0.60*4} 
	info_emu["likelihood"]['lsst_y1.lsst_emu_cs_lcdm']['data_vector_file']   = astress_chains_root+str(i)+".modelvector" # use the data vector generated above
	info_emu["output"] = ggsplit_chains_root + "TMP" + str(i)

	# for anisotropic stress only
	info_emu["likelihood"]['lsst_y1.lsst_emu_cs_lcdm']['mask_file'] =  "./projects/lsst_y1/data/lsst_3x2_cs_only_linear.mask"
	#info_emu["likelihood"]['lsst_y1.lsst_emu_cs_lcdm']['mask_file'] =  "./projects/lsst_y1/data/lsst_3x2_cs_only.mask"
	#info_emu["likelihood"]['lsst_y1.lsst_emu_cs_lcdm']['mask_file'] =  "./projects/lsst_y1/data/lsst_3x2_linear.mask"
	# should probably add extra omm_geo prior since a linear scale cut is too noisy;
	# with and without this prior doesn't make a huge difference in chi2; but do make omm_geo around fiducial value
	info_emu["params"]['omegam']['prior']  = {'dist': 'norm', 'loc':   0.316, 'scale':  0.01*4} 


	updated_info_minimizer, minimizer = run(info_emu)
	omm_geo_minimized    = minimizer.products()["minimum"]["omegam"]
	omm_growth_minimized = minimizer.products()["minimum"]["omegam_growth"]
	chi2_minimized       = minimizer.products()["minimum"]["chi2"]
	omm_geo.append(omm_geo_minimized)
	omm_growth.append(omm_growth_minimized)
	chi2.append(chi2_minimized)

	sys.stdout = sys.__stdout__
	print("Progress: ", i)

### Finalize
end_time = time.time()
print("Minutes used: ",(end_time - start_time)/60 )
print([g0, cg, omm_geo, omm_growth, chi2])
np.savetxt('./mapping_astress/mapping_astress.txt', np.transpose([g0, cg, omm_geo, omm_growth, chi2]), fmt='%f')