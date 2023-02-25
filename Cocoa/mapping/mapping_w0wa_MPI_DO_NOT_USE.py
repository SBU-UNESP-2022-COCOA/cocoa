 #TODO: The MPI part is triky because minimizer has its own MPI.....
 #KZ Feb24 2023: should not try to hack cobaya to make mpi work. Just use Bash script...
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
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

w0wa_chains_root    = "./projects/lsst_y1/chains/base_w_wa_plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18"
ggsplit_yaml_file   = "./projects/lsst_y1/EXAMPLE_MCMC514.yaml"
ggsplit_chains_root = "./projects/lsst_y1/chains/EXAMPLE_MCMC514"

num_points_thin = 10
analysissettings={'smooth_scale_1D':0.35,'smooth_scale_2D':0.35,'ignore_rows': u'0.5',
'range_confidence' : u'0.005'}

no_print = False

class NullIO(StringIO):
    def write(self, txt):
       pass


if rank == 0:
	samples = loadMCSamples(w0wa_chains_root,settings=analysissettings)
	samples.thin(factor = int(np.sum(samples.weights)/num_points_thin))
	p = samples.getParams()
	logA       = p.logA
	ns   	   = p.ns
	H0         = p.H0
	omegabh2   = p.omegabh2
	omegach2   = p.omegach2
	# w0         = p.w
	# wa         = p.wa
	print("TESTING with LCDM")
	w0         = np.ones(len(H0)) * (-1.0) #TEST LCDM
	wa         = np.zeros(len(H0)) #TEST LCDM
	print("total points to be minimized: ", len(p.logA))

	params     = np.transpose([logA, ns, H0, omegabh2, omegach2, w0, wa]) # Mind oder
	# count: the size of each sub-task
	ave, res = divmod(len(params), size)
	count = [ave + 1 if p < res else ave for p in range(size)]
	count = np.array(count)
	# displacement: the starting index of each sub-task
	displ = [sum(count[:p]) for p in range(size)]
	displ = np.array(displ)

else:
    params = None
    # initialize count on worker processes
    count = np.zeros(size, dtype=np.int64)
    displ = np.zeros(size, dtype=np.int64)

params      = comm.bcast(params, root=0)
comm.Bcast(count, root=0)
comm.Bcast(displ, root=0)

local_params = params[displ[rank]: displ[rank]+count[rank]]

if rank==0:
	print("checking shape", np.shape(local_params), local_params[0])
	start_time = time.time()

### Minimzer Function for each local mpi worker
def get_minimized_local(cosmo_params):
	local_omm_geo    = []
	local_omm_growth = []
	# Mind the order
	local_logA     = cosmo_params[:, 0]
	local_ns       = cosmo_params[:, 1]
	local_H0       = cosmo_params[:, 2]
	local_omegabh2 = cosmo_params[:, 3]
	local_omegach2 = cosmo_params[:, 4]
	local_w0       = cosmo_params[:, 5]
	local_wa       = cosmo_params[:, 6]
	for i in range(len(cosmo_params)):
		print("DEBUG", local_w0[i], "i = ", i, "rank = ", rank)
		info_w0wa = {'likelihood': {'lsst_y1.lsst_cosmic_shear':
								{'path': './external_modules/data/lsst_y1', 
								'data_file': 'LSST_Y1_no_scale_cut.dataset', 
								'print_datavector': True, 
								'print_datavector_file': "./projects/lsst_y1/data/tmp_"+str(rank)+str(i)+".modelvector", # print to a tmp data vector
								'non_linear_emul': 2, #avoid using emulator, or gg-split
								'kmax_boltzmann': 5.0}},

				 'theory': {'camb': 
				 				{'path': './external_modules/code/CAMB', 
				 				'stop_at_error': False, 
				 				'use_renames': True, 
				 				'extra_args': {'halofit_version': 'takahashi', 'AccuracyBoost': 1.15, 'lens_potential_accuracy': 1.0, 'num_massive_neutrinos': 1, 'nnu': 3.046, 'dark_energy_model': 'ppf', 'accurate_massive_neutrino_transfers': False, 'k_per_logint': 20}}},
				 'sampler': {'evaluate': None}
		}
		### Adding param info
		info_w0wa["params"] = { 'logA':      {'value': local_logA[i], 'drop': True},
							'As':			 {'value': 'lambda logA: 1e-10*np.exp(logA)', 'latex': 'A_\\mathrm{s}'},
							'H0':    		 {'value': local_H0[i]},
							'ns':     		 {'value': local_ns[i]},
							'omegabh2':      {'value': local_omegabh2[i]},
							'omegach2': 	 {'value': local_omegach2[i]},
							'omegam_growth': {'value': -100}, # no gg-split
							'w':      		 {'value': local_w0[i]},
							'wa': 			 {'value': local_wa[i]},
							'mnu':		     {'value': 0.06},
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
	
		updated_info, evaluate = run(info_w0wa)

		### Minimize the likelihood with GG-split
		print("Minimizing the likelihood with GG-split")
		info_emu = yaml_load_file(ggsplit_yaml_file)
		info_emu["sampler"] = {'minimize_no_mpi.Minimize': {'ignore_prior': True, 'best_of': 4, 'override_bobyqa': {'rhoend': 0.01}}} #relax convergence criterion a little bit
		info_emu["force"]   = True
		info_emu["likelihood"]['lsst_y1.lsst_emu_cs_lcdm']['data_vector_file']   = "./projects/lsst_y1/data/tmp_"+str(rank)+str(i)+".modelvector" # use the data vector generated above

		updated_info_minimizer, minimizer = run(info_emu)
		omm_geo_minimized    = minimizer.products()["minimum"]["omegam"]
		omm_growth_minimized = minimizer.products()["minimum"]["omegam_growth"]
		comm.Barrier()
		# try:
		# 	updated_info_minimizer, minimizer = run(info_emu)
		# 	omm_geo_minimized    = minimizer.products()["minimum"]["omegam"]
		# 	omm_growth_minimized = minimizer.products()["minimum"]["omegam_growth"]
		# except:
		# 	print("passed points can't find minimum", "rank = ", rank)
		# 	omm_geo_minimized    = -100.
		# 	omm_growth_minimized = -100.
		# 	pass

		local_omm_geo.append(omm_geo_minimized)
		local_omm_growth.append(omm_growth_minimized)

		sys.stdout = sys.__stdout__
		if rank==0 and i % 5 == 0:
			print("Local Progress at rank0:", i)

	return np.transpose([local_w0, local_wa, np.asarray(local_omm_geo), np.asarray(local_omm_growth)])

### MPI RUN ###
local_result = get_minimized_local(local_params)
# #SOMETHING wrong? never stop; seems only when each worker doesn't have equal tasks
comm.Barrier()

# Finilize
if rank!=0:
    comm.send(local_result, dest=0)
    train_params       = None
    train_data_vectors = None
elif rank==0:
    result = local_result
    for source in range(1,size):
        new_result = comm.recv(source=source)
        result = np.append(result, new_result, axis=0)  
    end_time = time.time()
    print("Total number of points minimized: ", len(result))
    print("Minutes used: ",(start_time - end_time)/60 )
    np.savetxt('mapping_w0wa.txt', result, fmt='%f')
