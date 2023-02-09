import numpy as np
def get_params_from_sample(sample, labels):
    """
    Format arrays into cocoa params
    """
    assert len(sample)==len(labels), "Length of the labels not equal to the length of samples"
    params = {}
    for i, label in enumerate(labels):
        param_i = sample[i]
        params[label] = param_i
    return params

def get_params_list(samples, labels):
    params_list = []
    for i in range(len(samples)):
        params = get_params_from_sample(samples[i], labels)
        params_list.append(params)
    return params_list

def get_params_from_lhs_sample(unit_sample, lhs_prior):
    """
    Format unit LHS arrays into cocoa params
    """
    assert len(unit_sample)==len(lhs_prior), "Length of the labels not equal to the length of samples"
    params = {}
    for i, label in enumerate(lhs_prior):
        lhs_min = lhs_prior[label]['min']
        lhs_max = lhs_prior[label]['max']
        param_i = lhs_min + (lhs_max - lhs_min) * unit_sample[i]
        params[label] = param_i
    return params

def get_lhs_params_list(samples, lhs_prior):
    params_list = []
    for i in range(len(samples)):
        params = get_params_from_lhs_sample(samples[i], lhs_prior)
        params_list.append(params)
    return params_list

# This function returns boolean type of whether the sample is in boundary or not. 
# return yes if it in a range of *percent (ex.10%) near the boundary
def boundary_check(params, lhs_prior, rg=0.1):
    reduced_range = get_lhs_box_reduced(lhs_prior,rg)
    assert len(params)==len(reduced_range), "Length of LHS parameters not match"
    for i in range(len(params)):
        if params[i]<reduced_range[i][0] or params[i]>reduced_range[i][1]:
            return True
        else:
            return False

#returns a simple 2D array representing the prior after boundary removal; 0.1 means 5% on each side
def get_lhs_box_reduced(lhs_prior, rg=0.1):
    reduced_range = np.zeros((len(lhs_prior),2))
    for i, label in enumerate(lhs_prior):
        lhs_min = lhs_prior[label]['min']
        lhs_max = lhs_prior[label]['max']
        lhs_min_reduced = lhs_min + (lhs_max - lhs_min) * rg / 2
        lhs_max_reduced = lhs_max - (lhs_max - lhs_min) * rg / 2
        reduced_range[i,0] = lhs_min_reduced
        reduced_range[i,1] = lhs_max_reduced
    return reduced_range



