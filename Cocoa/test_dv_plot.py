import numpy as np
import matplotlib.pyplot as plt

dv_true = np.loadtxt('test_dv.txt')[0]
dv_emu = np.loadtxt('test_dv.txt')[1]
cov_file = ('./projects/lsst_y1/data/cov_lsst_y1')

residue = (dv_emu - dv_true) / dv_true

def get_full_cov(cov_file):
    print("Getting covariance...")
    full_cov = np.loadtxt(cov_file)
    cov = np.zeros((1560, 1560))
    cov_scenario = full_cov.shape[1]
        
    for line in full_cov:
        i = int(line[0])
        j = int(line[1])
        if(cov_scenario==3):
            cov_ij = line[2]
        elif(cov_scenario==10):
            cov_g_block  = line[8]
            cov_ng_block = line[9]
            cov_ij = cov_g_block + cov_ng_block
        cov[i,j] = cov_ij
        cov[j,i] = cov_ij
    return cov
cov = get_full_cov(cov_file)[0:780, 0:780] #cut for cosmic shear
error = np.sqrt(np.diag(cov)) / dv_true


nxip = 390
nxim = 390
ind = np.arange(0, nxip + nxim)
zeros = np.zeros(nxip + nxim)

fig, axs = plt.subplots(2, 1)
# axs[0].set_yscale('log')
# #axs[0].set_ylim(5.e-8,0.99e-4)
# axs[0].set_xlim(0, nxip - 1)
# axs[0].set_ylabel(r'$\xi_+ $', fontsize = 18)
# axs[0].errorbar(ind, dv_true, error, marker='o', color='k',linestyle = '', markersize = 1.00, alpha = 0.25)
# axs[0].plot(ind, dv_true, marker='o', color='r',linestyle = '',markersize = 1.5)
# axs[0].plot(ind, dv_emu, marker='x', color='b',linestyle = '',markersize = 1.5)
# axs[0].set_xticks(np.arange(min(ind[0:nxip]), max(ind[0:nxip]), 20.0))
# for i in np.arange(min(ind[0:nxip]), max(ind[0:nxip]), 20.0):
#     axs[0].axvline(x=i, color ='grey', alpha = 0.15, ls='--', lw=1)

# axs[1].set_yscale('log')
# #axs[1].set_ylim(2.e-7,0.8e-5)
# axs[1].set_xlim(nxip, nxip+nxim-1)
# axs[1].set_ylabel(r'$\xi_- $', fontsize = 18)
# axs[1].errorbar(ind, dv_true, error, marker='o', color='k', linestyle = '', markersize = 1.00, alpha = 0.25)
# axs[1].plot(ind,dv_emu,marker='o', color='r',linestyle = '',markersize = 1.5)
# axs[1].plot(ind, dv_emu, marker='x', color='b',linestyle = '',markersize = 1.5)
# axs[1].set_xticks(np.arange(min(ind[nxip:nxip+nxim]), max(ind[nxip:nxip+nxim]), 20.0))
# for i in np.arange(min(ind[nxip:nxip+nxim]), max(ind[nxip:nxip+nxim]), 20.0):
#     axs[1].axvline(x=i, color ='grey', alpha = 0.15, ls='--', lw=1)


axs[0].set_yscale('linear')
axs[0].set_ylim(-0.3,0.3)
axs[0].set_xlim(0, nxip - 1)
axs[0].set_ylabel(r'$\xi_+ $ residue', fontsize = 18)
axs[0].errorbar(ind, zeros, error, marker='o', color='k',linestyle = '', markersize = 1.00, alpha = 0.25)
axs[0].plot(ind, residue, marker='o', color='r',linestyle = '',markersize = 1.5)
axs[0].set_xticks(np.arange(min(ind[0:nxip]), max(ind[0:nxip]), 20.0))
for i in np.arange(min(ind[0:nxip]), max(ind[0:nxip]), 20.0):
    axs[0].axvline(x=i, color ='grey', alpha = 0.15, ls='--', lw=1)

axs[1].set_yscale('linear')
axs[1].set_ylim(-0.3,0.3)
axs[1].set_xlim(nxip, nxip+nxim-1)
axs[1].set_ylabel(r'$\xi_- $ residue', fontsize = 18)
axs[1].errorbar(ind, zeros, error, marker='o', color='k', linestyle = '', markersize = 1.00, alpha = 0.25)
axs[1].plot(ind,residue,marker='o', color='r',linestyle = '',markersize = 1.5)
axs[1].set_xticks(np.arange(min(ind[nxip:nxip+nxim]), max(ind[nxip:nxip+nxim]), 20.0))
for i in np.arange(min(ind[nxip:nxip+nxim]), max(ind[nxip:nxip+nxim]), 20.0):
    axs[1].axvline(x=i, color ='grey', alpha = 0.15, ls='--', lw=1)

print("testing: ", residue[550])
print(error)

plt.tight_layout()
plt.savefig("plotdv1.pdf")
