"""
.. module:: samplers.nautilus

Kunhao Zhong: A low-level implementation for the nautilus cobaya wrapper
TODO:   1. Does NOT support derived parameters; NOT blocking (fast/slow hierachy)
        2. @classmethod about installation
        3. _correct_unphysical_fraction() 
        4. Parallel: Not working it seems///schwimmbad (across nodes); ipyparallel missing; using dynesty's own pool only
"""
# Global
import os
import sys
import numpy as np
import logging
import inspect
from itertools import chain
from typing import Any, Callable, Optional
from tempfile import gettempdir
import re
#nautlilus
from .nautilus_src.nautilus import Sampler as nautilus_Sampler
from .nautilus_src.nautilus import Prior as nautilus_Prior
from scipy.stats import norm, multivariate_normal
import yaml
import time
import pandas as pd

# Local
from cobaya.tools import read_dnumber, get_external_function, \
    find_with_regexp, NumberWithUnits, load_module, VersionCheckError
from cobaya.sampler import Sampler
from cobaya.mpi import is_main_process, share_mpi, sync_processes, get_mpi_rank
from cobaya.collection import SampleCollection
from cobaya.log import LoggedError, get_logger
from cobaya.yaml import yaml_dump_file
from cobaya.conventions import derived_par_name_separator, packages_path_arg, Extension


class nautilus(Sampler):
    r"""
    TODO
    """

    _base_dir_suffix = ""

    # variables from yaml
    nlive: NumberWithUnits
    is_dynamial: bool

    def initialize(self):
        n_liven_sampled = len(self.model.parameterization.sampled_params())
        self.n_derived = len(self.model.parameterization.derived_params())
        self.n_priors = len(self.model.prior)
        self.n_likes = len(self.model.likelihood)
        self.nDims = self.model.prior.d()
        self.sampled_params_names = list(self.model.parameterization.sampled_params().keys())
        # KZ 2024.5.7: let me test this: seems I can still call emulators to run these chains
        # if self.n_derived>0:
        #     raise LoggedError(
        #         self.log, "Does Not support derived parameters YET")

        # Import additional modules for parallel computing if requested
        self.pool = None

        # Prepare output folders and prefixes
        if self.output:
            self.file_root = self.output.prefix
            self.read_resume = self.output.is_resuming()
        else:
            output_prefix = share_mpi(hex(int(self._rng.random() * 16 ** 6))[2:]
                                      if is_main_process() else None)
            self.file_root = output_prefix
            # dummy output -- no resume!
            self.read_resume = False
        self.base_dir = self.get_base_dir(self.output)
        # self.output.create_folder(self.base_dir)
        self.mpi_info("Storing nautilus output to '%s'.", self.base_dir)


    def run(self):
        """
        Prepares the likelihood and prior_transform function and calls ``nautilus``'s ``run`` function.
        """
        self.mpi_info("Calling nautilus...")

        # get prior
        params_info = self.model.parameterization.sampled_params_info()
        nautilus_prior = nautilus_Prior()
        for name in self.sampled_params_names:
            prior = params_info[name]['prior']
            if ('dist' in prior and prior['dist']=='norm'):
                nautilus_prior.add_parameter(name, dist=norm(loc=prior['loc'], scale=prior['scale']))
            else:
                nautilus_prior.add_parameter(name, dist=(prior['min'], prior['max']))

        # loglikelihood fumction take dictionary as input
        def loglikelihood(param_dict):
            params_values = [param_dict[name] for name in self.sampled_params_names]
            result = self.model.logposterior(params_values)
            loglikes = result.loglikes
            return np.squeeze(loglikes).sum()


        # KZ: MPI run start: # MPI in cobaya style
        if self.parallel.get("kind") == "cobaya_mpi":
            print("Running Nautilus with cobaya-MPI")
            n_batch = min(self.n_live, self.parallel.get("n_batch"))
            sampler = nautilus_Sampler(nautilus_prior, loglikelihood, n_live=self.n_live, pool=None, split_threshold=100, n_batch=n_batch, cobaya_mpi=True)
            start =time.time()
            sampler.run(verbose=True)
            end   =time.time()
            print("run time = ", end-start)
        # KZ: MPI run end
        else:
            # KZ: you can still run nautilus without any parallization, but would be slow
            sampler = nautilus_Sampler(nautilus_prior, loglikelihood, n_live=self.n_live, pool=None)
            start =time.time()
            sampler.run(verbose=True)
            end   =time.time()
            print("run time = ", end-start)

        # same as cobaya manner
        # KZ NOTE: The sampler doesn't save logL seperately. Let me just use cobaya's collection class to make a template and save manually
        if is_main_process():
            self.save_raw(np.array([sampler.evidence()]))
            points, log_w, log_l = sampler.posterior(equal_weight=self.is_equal_weights)
            self.mpi_info("Saving posteriors, number of points is {}".format(len(points)))
            
            # save some points and output .1.txt as a example
            self.save_sample(points[0:1], log_w[0:1], log_l[0:1], "1tmp")
            
            # manual save from here:
            outfile_tmp = self.base_dir + '.1tmp.txt'
            outfile_cleaned = self.base_dir + '.1.txt'

            f = open(outfile_tmp, 'r')
            header = f.readline().split()
            tmp1 = f.readline().split() # read one line, remove nan
            
            header_cleaned = []

            # clear up nans
            for i in range(len(tmp1)):
                if tmp1[i] !='nan':
                    header_cleaned.append(header[1+i]) # there's a # in front

            # clear up last chi2 and minuslogprior__0
            header_cleaned.remove('minuslogprior__0')
            for i in range(1, len(header_cleaned)):
                if header_cleaned[-i] !='chi2':
                    header_cleaned.pop(-i)
                    break

            delimiter = '         '  # Single space as delimiter
            header_cleaned = delimiter.join(header_cleaned)

            data_to_output = np.zeros([len(points), len(points[0]) + 4]) # weight, logPosterior, logPrior, chi2
            if self.is_equal_weights:
                data_to_output[:, 0] = 1.0 # weight
            else:
                data_to_output[:, 0] = np.exp(log_w) # weight

            # get logPior
            logpriors=np.zeros(len(log_l))
            logpriors=np.array([self.model.logposterior(points[i]).logpriors[0] for i in range(len(log_l))])

            data_to_output[:, 1]  = - log_l - logpriors# minuslogpost
            data_to_output[:, 2: 2+len(points[0])] = points # sample points
            data_to_output[:, -1] = -2. * log_l # chi2
            data_to_output[:, -2] = - logpriors # minus log prior

            # assert len(header_cleaned.split())== len(data_to_output[0]), 'something wrong'
            
            np.savetxt(outfile_cleaned,data_to_output, header = header_cleaned, fmt='%f')
            
            
            self.mpi_info("nautilus finished")
        return
        # KZ: MPI RUN end

    def save_raw(self, logz):
        if is_main_process():
            # How to get directly output directory?
            np.savetxt(self.base_dir + '.logz.txt',logz)
        return

    def save_sample(self, samples, logweights, loglikes, name):
        if is_main_process():
            collection = SampleCollection(self.model, self.output, name=str(name))
            if self.is_equal_weights:
                for i in range(len(samples)):
                    collection.add(
                        samples[i],
                        # derived=row[2 + self.n_sampled:2 + self.n_sampled + self.n_derived], # can't handel derived now
                        weight = 1,
                        logpriors=[self.model.logposterior(samples[i]).logpriors[0]],
                        loglikes=[loglikes[i]]
                        )
            else:   
                for i in range(len(samples)):
                    # skip weight=0 samples
                    if np.exp(logweights[i])==0:
                        continue
                    collection.add(
                        samples[i],
                        # derived=row[2 + self.n_sampled:2 + self.n_sampled + self.n_derived], # can't handel derived now
                        weight = np.exp(logweights[i]),
                        logpriors=[self.model.logposterior(samples[i]).logpriors[0]],
                        loglikes=[loglikes[i]]
                        )
            # make sure that the points are written
            collection.out_update()
            return

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``SampleCollection`` containing the sequentially
           discarded live points.
        """
        if is_main_process():
            products = {
                "sample": self.collection, "logZ": self.logZ, "logZstd": self.logZstd}
            return products
        else:
            return {}
    @property
    def raw_prefix(self):
        return os.path.join(
            self.pc_settings.base_dir, self.pc_settings.file_root)
    @classmethod
    def get_base_dir(cls, output):
        if output:
            return output.add_suffix(cls._base_dir_suffix, separator="")
        return os.path.join(gettempdir(), cls._base_dir_suffix)