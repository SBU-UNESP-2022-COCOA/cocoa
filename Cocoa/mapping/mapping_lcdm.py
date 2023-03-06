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




lcdm_chains_root    = "./projects/lsst_y1/chains/base_plikHM_TTTEEE_lowl"
ggsplit_yaml_file   = "./projects/lsst_y1/EXAMPLE_MCMC514.yaml"
ggsplit_chains_root = "./projects/lsst_y1/chains/EXAMPLE_MCMC514"

num_points_thin = 5000
analysissettings={'smooth_scale_1D':0.35,'smooth_scale_2D':0.35,'ignore_rows': u'0.5',
'range_confidence' : u'0.005'}

no_print = False
BASH_parallel = True



### Read w-bin parameters and generate a temporary data vector

samples = loadMCSamples(lcdm_chains_root,settings=analysissettings)
samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
p = samples.getParams()

omm_geo    = []
omm_growth = []

logA       = p.logA
ns   	     = p.ns
H0         = p.H0
omegabh2   = p.omegabh2
omegach2   = p.omegach2

if BASH_parallel:
	# job_num in bash paralell programs; similar idea to mpi.size()
	job_num = int(sys.argv[1])
	job_idx = int(sys.argv[2])
	if job_idx+1 > job_num:
		print("job index and number doesn't match")
		quit()
else:
	job_idx = int(1)

params   = np.transpose([logA, ns, H0, omegabh2, omegach2]) # Mind oder
# count: the size of each sub-task
ave, res = divmod(len(params), job_num)
count 	= [ave + 1 if p < res else ave for p in range(job_num)]
count	 	= np.array(count)
# displacement: the starting index of each sub-task
displ 	= [sum(count[:p]) for p in range(job_num)]
displ 	= np.array(displ)
params 	= params[displ[job_idx]: displ[job_idx]+count[job_idx]]

logA     = params[:, 0]
ns       = params[:, 1]
H0       = params[:, 2]
omegabh2 = params[:, 3]
omegach2 = params[:, 4]

class NullIO(StringIO):
    def write(self, txt):
       pass


start_time = time.time()
for i in range(len(logA)):
	info_lcdm = {'likelihood': {'lsst_y1.lsst_cosmic_shear':
								{'path': './external_modules/data/lsst_y1', 
								'data_file': 'LSST_Y1.dataset', 
								'print_datavector': True, 
								'print_datavector_file': "./projects/lsst_y1/data/tmp_"+str(job_idx)+".modelvector", # print to a tmp data vector
								'non_linear_emul': 1, #Use Euclid Emulator for LCDM-like data vector generation; hopefully no parameter out of bounds...
								'kmax_boltzmann': 5.0}},

				 'theory': {'camb': 
				 				{'path': './external_modules/code/CAMB', 
				 				'stop_at_error': False, 
				 				'use_renames': True, 
				 				'extra_args': {'halofit_version': 'takahashi', 'AccuracyBoost': 1.15, 'lens_potential_accuracy': 1.0, 'num_massive_neutrinos': 1, 'nnu': 3.046, 'dark_energy_model': 'ppf', 'accurate_massive_neutrino_transfers': False, 'k_per_logint': 20}}},
				 'sampler': {'evaluate': None}
	}
	### Adding param info
	info_lcdm["params"] = { 'logA':   {'value': logA[i], 'drop': True},
							'As':			    {'value': 'lambda logA: 1e-10*np.exp(logA)', 'latex': 'A_\\mathrm{s}'},
							'H0':    		 {'value': H0[i]},
							'ns':     		 {'value': ns[i]},
							'omegabh2':     {'value': omegabh2[i]},
							'omegach2': 	 {'value': omegach2[i]},
							'omegam_growth':{'value': -100}, # no gg-split
							'w':				 {'value': -1},   # no gg-split 
							'w_growth':		 {'value': -1},   # no gg-split 
							'mnu':		    {'value': 0.06},
							'LSST_DZ_S1': 	 {'value': 0.0},
							'LSST_DZ_S2': 	 {'value': 0.0},
							'LSST_DZ_S3': 	 {'value': 0.0},
							'LSST_DZ_S4': 	 {'value': 0.0},
							'LSST_DZ_S5': 	 {'value': 0.0},
							'LSST_A1_1': 	 {'value': 0.5}, # note this is non-zero
							'LSST_A1_2': 	 {'value': 0.0},
							'LSST_M1': 	 	 {'value': 0.0},
							'LSST_M2': 	 	 {'value': 0.0},
							'LSST_M3': 	 	 {'value': 0.0},
							'LSST_M4': 	 	 {'value': 0.0},
							'LSST_M5': 	 	 {'value': 0.0},
							}

	### Prevent printing, but still cosmolike log output
	if no_print:
		sys.stdout = NullIO()
	
	updated_info, evaluate = run(info_lcdm)

	### Minimize the likelihood with GG-split
	info_emu = yaml_load_file(ggsplit_yaml_file)
	if BASH_parallel:
		info_emu["sampler"] = {'minimize': {'ignore_prior': False, 'best_of': 2, 'override_bobyqa': {'rhoend': 0.01}}}
	else:
		info_emu["sampler"] = {'minimize': {'ignore_prior': False, 'best_of': 2, 'override_bobyqa': {'rhoend': 0.01}}}
	info_emu["force"]   = True
	# Add a prior on H0 and ns that has 4 times larger sigma. Values get from Planck 2018 Table 2.
	# The temperature boost should make this not model related; but still provide good additional information for LSST
	info_emu["params"]['ns']['prior']  = {'dist': 'norm', 'loc':  0.9649, 'scale': 0.0044*4}
	info_emu["params"]['H0']['prior']  = {'dist': 'norm', 'loc':   67.27, 'scale':  0.60*4} 
	info_emu["likelihood"]['lsst_y1.lsst_emu_cs_lcdm']['data_vector_file']   = "./projects/lsst_y1/data/tmp_"+str(job_idx)+".modelvector" # use the data vector generated above
	info_emu["output"] = ggsplit_chains_root + "TMP" + str(job_idx)

	updated_info_minimizer, minimizer = run(info_emu)
	omm_geo_minimized    = minimizer.products()["minimum"]["omegam"]
	omm_growth_minimized = minimizer.products()["minimum"]["omegam_growth"]
	omm_geo.append(omm_geo_minimized)
	omm_growth.append(omm_growth_minimized)

	sys.stdout = sys.__stdout__
	print("Progress: ", i)

### Finalize
end_time = time.time()
print("Minutes used: ",(end_time - start_time)/60 )
np.savetxt('./mapping/mapping_lcdm_'+str(job_idx)+'.txt', np.transpose([logA, ns, omm_geo, omm_growth]), fmt='%f')