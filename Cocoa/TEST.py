##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from cocoa_emu import Config, NNEmulator, get_lhs_params_list, get_params_list, get_lhs_box_reduced, boundary_check
from cocoa_emu.sampling import EmuSampler
from pyDOE import lhs


configfile = './projects/des_y3/lhs_cs_NLA.yaml'
config = Config(configfile)

def get_lhs_samples(N_dim, N_lhs, lhs_minmax):
    unit_lhs_samples = lhs(N_dim, N_lhs, criterion='center')
    print("lhs samples generated with CRITERION = CENTER")
    lhs_params = get_lhs_params_list(unit_lhs_samples, lhs_minmax)
    return lhs_params

validation_samples = np.load('./projects/des_y3/emulator_output_cs_lcdm_NLA/lhs/dvs_for_validation_10k/validation_samples.npy')

count = 0
for i in range(len(validation_samples)):
    if boundary_check(validation_samples[i], config.lhs_minmax, rg=0.1):
        count+=1
print(count)