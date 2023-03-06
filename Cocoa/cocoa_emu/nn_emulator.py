import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5
from torchvision import models
from torchinfo import summary
from .utils import *


class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size:
            self.skip = nn.Linear(in_size, out_size, bias=False)
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        # self.act1 = nn.PReLU()
        # self.act2 = nn.PReLU()
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2

class ResBlock2(nn.Module):
    def __init__(self, size, mid_size):
        super(ResBlock2, self).__init__()
        

        self.skip = nn.Identity()

        self.layer1 = nn.Linear(size, mid_size)
        self.layer2 = nn.Linear(mid_size, mid_size)
        self.layer3 = nn.Linear(mid_size, size)

        # self.layer1 = nn.Linear(size, mid_size)
        # self.layer2 = nn.Linear(mid_size, size)

        self.norm1 = Affine()
        self.norm2 = Affine()
        self.norm3 = Affine()

        # self.act1 = nn.PReLU()
        # self.act2 = nn.PReLU()
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        self.act3 = nn.Tanh()

    def forward(self, x):
        xskip = self.skip(x)
        o1 = self.layer1(self.act1(self.norm1(x)))  / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) 
        o3 = self.layer3(self.act3(self.norm3(o2))) / np.sqrt(10)+ xskip
        return o3

    # def forward(self, x):
    #     xskip = self.skip(x)
    #     o1 = self.layer1(self.act1(self.norm1(x)))  / np.sqrt(10)
    #     o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip
    #     return o2
    
class ResBottle(nn.Module):
    def __init__(self, size, N):
        super(ResBottle, self).__init__()

        self.size = size
        self.N = N
        encoded_size = size // N

        self.norm1  = torch.nn.BatchNorm1d(encoded_size)
        self.layer1 = nn.Linear(size,encoded_size)
        self.act1   = nn.Tanh()

        self.norm2  = torch.nn.BatchNorm1d(encoded_size)
        self.layer2 = nn.Linear(encoded_size,encoded_size)
        self.act2   = nn.Tanh()

        self.norm3  = torch.nn.BatchNorm1d(size)
        self.layer3 = nn.Linear(encoded_size,size)
        self.act3   = nn.PReLU() #nn.Tanh()

    def forward(self, x):
        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.act2(self.norm2(self.layer2(o1)))
        o3 = self.norm3(self.layer3(o2))
        o  = self.act3(o3+x)

        return o

class DenseBlock(nn.Module):
    def __init__(self, size, N):
        super(DenseBlock, self).__init__()

        self.size = size
        self.N = N
        encoded_size = size // N

        self.norm1  = torch.nn.BatchNorm1d(encoded_size)
        self.layer1 = nn.Linear(size,encoded_size)
        self.act1   = nn.Tanh()

        self.norm2  = torch.nn.BatchNorm1d(encoded_size)
        self.layer2 = nn.Linear(encoded_size,encoded_size)
        self.act2   = nn.Tanh()

        self.norm3  = torch.nn.BatchNorm1d(size)
        self.layer3 = nn.Linear(encoded_size,size)
        self.act3   = nn.PReLU() #nn.Tanh()

    def forward(self, x):
        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.act2(self.norm2(self.layer2(o1)))
        o3 = self.norm3(self.layer3(o2))
        o  = torch.cat((o3, x),axis=1)

        return self.act3(o)
#torch.nn.Conv1d(in_channels=7, out_channels=20, kernel_size=5, stride=2)

class NNEmulator:
    def __init__(self, N_DIM, OUTPUT_DIM, dv_fid, dv_std, cov, dv_max, dv_mean, lhs_minmax, device, model='resnet_small_LSST', optim=None):

        torch.set_default_dtype(torch.float64)
        self.N_DIM = N_DIM
        self.model = model
        self.optim = optim
        self.device = device
        self.trained = False
        self.model_type = model
        
        self.dv_fid     = torch.Tensor(dv_fid)
        self.dv_std     = torch.Tensor(dv_std)
        self.cov        = torch.Tensor(cov)
        self.cov_inv    = torch.Tensor(np.linalg.inv(cov))        
        self.dv_max     = torch.Tensor(dv_max)
        self.dv_mean    = torch.Tensor(dv_mean)
        self.lhs_minmax = lhs_minmax


        #Need to use np.linalg.eigh instead of .eig for numerical stability
        self.L_inv      = torch.Tensor(np.diag(1/np.linalg.eigh(self.cov)[0]))  # in the new basis where Cov = QLQ^(-1)
        self.evecs      = torch.Tensor(np.linalg.eigh(self.cov)[1]) #this is Q
        self.evecs_inv  = torch.Tensor(np.linalg.inv(np.linalg.eigh(self.cov)[1])) #save them to avoid doing eigen decomposition evertime of calling

        self.output_dim = OUTPUT_DIM

        self.n_bin            = 30 #Number of Tomographic bin
        self.theta_bin        = 26
        self.dropout          = 0.3
        self.learningrate     = 5e-4
        self.reduce_lr        = True
        self.loss_vali_goal   = 0.05
        self.gpu_parallel     = False
        self.boundary_removal = False #Force points at the boundary of the box to have chi2=0
        
    
            
        if model == 'simply_connected':
            print("Using simply connected NN...")
            print("model output_dim: ", OUTPUT_DIM)
            self.model = nn.Sequential(
                                nn.Linear(N_DIM, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(512, 512), #additional laryer
                                nn.ReLU(), #additional laryer
                                nn.Dropout(self.dropout), #additional laryer
                                nn.Linear(512, OUTPUT_DIM),
                                Affine()
                                )            
        elif(model=='resnet'):
            print("Using resnet model...")
            print("model output_dim: ", OUTPUT_DIM)
            self.model = nn.Sequential(
                    nn.Linear(N_DIM, 128),
                    ResBlock(128, 256),
                    nn.Dropout(self.dropout),
                    ResBlock(256, 256),
                    nn.Dropout(self.dropout),
                    ResBlock(256, 256),
                    nn.Dropout(self.dropout),
                    ResBlock(256, 512),
                    nn.Dropout(self.dropout),
                    ResBlock(512, 512),
                    nn.Dropout(self.dropout),
                    ResBlock(512, 512),
                    nn.Dropout(self.dropout),
                    ResBlock(512, 1024),
                    nn.Dropout(self.dropout),
                    ResBlock(1024, 1024),
                    nn.Dropout(self.dropout),
                    ResBlock(1024, 1024),
                    nn.Dropout(self.dropout),
                    ResBlock(1024, 1024),
                    nn.Dropout(self.dropout),
                    ResBlock(1024, 1024),
                    Affine(),
                    nn.PReLU(),
                    nn.Linear(1024, OUTPUT_DIM),
                    Affine()
                )
        elif(model=='resnet_small_LSST'):
            print("Using resnet_samll_LSST model...")
            print("model output_dim: ", OUTPUT_DIM)

            # #this works for LSST Cosmic Shear LHS only, 800k training samples
            # self.model = nn.Sequential(
            #     nn.Linear(N_DIM, 512),
            #     ResBlock(512, 1024),
            #     nn.Dropout(self.dropout),
            #     nn.PReLU(),
            #     nn.Linear(1024, OUTPUT_DIM),
            #     Affine()
            #     )

            self.model = nn.Sequential(
                nn.Linear(N_DIM, 2048),
                ResBottle(2048, 4),
                nn.Linear(2048, OUTPUT_DIM),
                Affine()
            )
            #TESTING: for LSST 2x2 LHS only, 800k training samples
            # self.model = nn.Sequential(
            #     nn.Linear(N_DIM, 256),
            #     DenseBlock(256, 4),
            #     DenseBlock(512, 4),
            #     DenseBlock(1024, 4),
            #     nn.Linear(2048, OUTPUT_DIM),
            #     Affine()
            # )

        elif(model=='resnet_small_DES'):
            print("Using resnet_samll_DES model...")
            print("model output_dim: ", OUTPUT_DIM)
            self.model = nn.Sequential(
                nn.Linear(N_DIM, 512),
                ResBlock(512, 1024),
                nn.Dropout(self.dropout),
                nn.PReLU(),
                nn.Linear(1024, OUTPUT_DIM),
                Affine()
                )
            # self.model = nn.Sequential(
            #     nn.Linear(N_DIM, 1024),
            #     ResBottle(1024, 4),
            #     ResBottle(1024, 4),
            #     ResBottle(1024, 4),
            #     nn.Linear(1024, OUTPUT_DIM),
            #     Affine()
            # )
            # self.model = nn.Sequential(
            #     nn.Linear(N_DIM, 2048),
            #     ResBottle(2048, 4),
            #     nn.Linear(2048, OUTPUT_DIM),
            #     Affine()
            # )

            # DenseBlock layers: DEBUGGING
            # self.model = nn.Sequential(
            #     nn.Linear(N_DIM, 64),
            #     DenseBlock(64, 4),
            #     # DenseBlock(128, 4),
            #     # DenseBlock(256, 4),
            #     nn.Linear(64, OUTPUT_DIM),
            #     Affine()
            # )

        
        ###use multi gpu with nn.DataParallel
        if self.gpu_parallel==True:
            print("training with gpu-parallel")
            self.model= nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.to(torch.float64)
        if self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learningrate)
        if self.reduce_lr == True:
            print('Reduce LR on plateu: ',self.reduce_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min')
        
    def train(self, X, y, X_validation, y_validation, test_split=None, batch_size=32, n_epochs=100):
        if not self.trained:
            self.X_mean = torch.Tensor(X.mean(axis=0, keepdims=True)).double()
            self.X_std  = torch.Tensor(X.std(axis=0,  keepdims=True)).double()
            self.X_max, self.idx  = (torch.max(torch.Tensor(X), dim=0))
            self.X_min, self.idx  = (torch.min(torch.Tensor(X), dim=0))
            #del self.idx
        ###
        print("Model summary:", summary(self.model))
    
        epoch_range = tqdm(range(n_epochs))

        losses_train = []
        losses_vali = []
        loss = 100.
        
        tmp_dv_max        = self.dv_max.to(self.device).double()
        tmp_dv_mean       = self.dv_mean.to(self.device).double()
        tmp_dv_std        = self.dv_std.to(self.device).double()
        tmp_cov           = self.cov.to(self.device).double()
        tmp_cov_inv       = self.cov_inv.to(self.device).double()
        tmp_X_mean        = self.X_mean.to(self.device).double()
        tmp_X_std         = self.X_std.to(self.device).double()
        tmp_X_max         = self.X_max.to(self.device).double()
        tmp_X_min         = self.X_min.to(self.device).double()
        tmp_X_validation  = X_validation.to(self.device).double()
        tmp_Y_validation  = y_validation.to(self.device).double()
        tmp_L_inv            = self.L_inv.to(self.device).double() # in the new basis where Cov = QLQ^(-1)

        #X_train     = ((X.double() - self.X_mean)/self.X_std)
        #X_train     = ((X.double() - self.X_min)/(self.X_max - self.X_min))
        X_train     = X.double()
        y_train     = y
        trainset    = torch.utils.data.TensorDataset(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

        for epoch in epoch_range:
            for i, data in enumerate(trainloader):
                X       = data[0].to(self.device).double()         
                X_norm  = ((data[0]- self.X_min)/(self.X_max - self.X_min)).to(self.device).double() # This is input
                Y_batch = data[1].to(self.device).double()         
                                                # This is label
                Y_pred  = self.model.train()(X_norm) * tmp_dv_std #technically should add y_fid back, back loss=chi2 is the difference so they are the same
                
                ### TEST: turn on boundary_removal only after 100 epoch
                # self.boundary_removal = False
                # if epoch ==150 and i==0:
                #     print("Turning on Boundary Removal: force these regions to have loss=0")
                #     self.boundary_removal = True
                
                if self.boundary_removal:
                    # Make a mask for boundary_removal, such that points near boundary don't contribute to the total chi2
                    boundary_mask= []
                    for k in range(len(X)):
                        boundary_mask.append(boundary_check(X[k], self.lhs_minmax, rg=0.1))
                    boundary_mask = torch.Tensor(boundary_mask)==False #This gives False for those in the boundary. Opposite to the origional function
                    
                    loss = torch.mean(torch.diagonal( torch.matmul( torch.matmul((Y_batch - Y_pred),tmp_L_inv), torch.t(Y_batch - Y_pred)) )[boundary_mask] )
                    loss = torch.nan_to_num(loss) # prevent nan, should not happen with reasonable training set
                else:
                    loss = torch.mean(torch.diagonal( torch.matmul( torch.matmul((Y_batch - Y_pred),tmp_L_inv), torch.t(Y_batch - Y_pred)) ) )

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            losses_train.append(loss.cpu().detach().numpy())
            ###Validation loss
            with torch.no_grad():
                #Y_pred = self.model.eval()((tmp_X_validation - tmp_X_mean)/tmp_X_std) * tmp_dv_std 
                Y_pred = self.model.eval()((tmp_X_validation - tmp_X_min)/(tmp_X_max - tmp_X_min)) * tmp_dv_std 


            loss_vali = torch.mean(torch.diagonal( torch.matmul( torch.matmul((tmp_Y_validation - Y_pred),tmp_L_inv), torch.t(tmp_Y_validation - Y_pred)) ) )
            #loss_vali = torch.mean(torch.abs(tmp_Y_validation - Y_pred)) * 1000000

            losses_vali.append(loss_vali.cpu().detach().numpy())

            if self.reduce_lr:
                self.scheduler.step(loss_vali)
               
            epoch_range.set_description('Loss: {0}, Loss_validation: {1}'.format(loss, loss_vali))

            if loss_vali < self.loss_vali_goal:
                continue

        #print("Model summary:", summary(self.model))
        #torch.set_printoptions(precision=15)
        torch.set_printoptions(precision=7)

        np.savetxt("losses.txt", [losses_train,losses_vali], fmt='%s')
        #save last dv from validation just for plotting
        np.savetxt("test_dv.txt", np.array( [tmp_Y_validation.cpu().detach().numpy().astype(np.float64)[-1], Y_pred.cpu().detach().numpy().astype(np.float64)[-1] ]), fmt='%s')
        self.trained = True

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            X_mean = self.X_mean.clone().detach().to(self.device).double()
            X_std  = self.X_std.clone().detach().to(self.device).double()
            X_max  = self.X_max.clone().detach().to(self.device).double()
            X_min  = self.X_min.clone().detach().to(self.device).double()

            #=== mean/std normalization
            # X_norm = (X.to(self.device) - X_mean) / X_std

            #=== max/min normalization
            X_norm = (X.to(self.device) - X_min) / (X_max - X_min)
            X_norm = np.reshape(X_norm, (1, len(X_norm)))

            y_pred = self.model.eval()(X_norm).double().cpu() * self.dv_std #normalization

        y_pred = y_pred @ torch.Tensor(np.transpose(self.evecs)) + self.dv_mean #change of basis
        return y_pred.double().numpy()


    def save(self, filename):
        #NOTE KZ: when training with GPU-parallel, should only save the weights instead, then loading on cpu is easier.
        torch.save(self.model, filename)
        with h5.File(filename  + '.h5', 'w') as f:
            f['X_mean']        = self.X_mean
            f['X_std']         = self.X_std
            f['X_max']         = self.X_max
            f['X_min']         = self.X_min
            f['dv_fid']        = self.dv_fid
            f['dv_std']        = self.dv_std
            f['dv_max']        = self.dv_max
            f['dv_mean']       = self.dv_mean
            f['cov']           = self.cov
            f['evecs']         = self.evecs
            f['evecs_inv']     = self.evecs_inv
        
    def load(self, filename, map_location):
        self.trained = True
        self.model = torch.load(filename, map_location)
        ###
        print("Model summary:", summary(self.model))
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean  = torch.Tensor(f['X_mean'][:]).double()
            self.X_std   = torch.Tensor(f['X_std'][:]).double()
            self.X_max   = torch.Tensor(f['X_max'][:]).double()
            self.X_min   = torch.Tensor(f['X_min'][:]).double()
            self.dv_fid  = torch.Tensor(f['dv_fid'][:]).double()
            self.dv_std  = torch.Tensor(f['dv_std'][:]).double()
            self.dv_max  = torch.Tensor(f['dv_max'][:]).double()
            self.dv_mean = torch.Tensor(f['dv_mean'][:]).double()
            self.cov     = torch.Tensor(f['cov'][:]).double()
            self.evecs   = torch.Tensor(f['evecs'][:]).double()
            self.evecs_inv  = torch.Tensor(f['evecs_inv'][:]).double()



