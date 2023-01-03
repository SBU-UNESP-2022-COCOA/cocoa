import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5
from torchvision import models
from torchinfo import summary


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

    
class NNEmulator:
    def __init__(self, N_DIM, OUTPUT_DIM, dv_fid, dv_std, cov, dv_max, device, model='resnet_small', optim=None):

        torch.set_default_dtype(torch.float32)
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
        self.L_inv      = torch.Tensor(np.diag(1/np.linalg.eig(self.cov)[0]))  # in the new basis where Cov = QLQ^(-1)
        self.evecs      = torch.Tensor(np.linalg.eig(self.cov)[1]) #this is Q
        self.evecs_inv  = torch.Tensor(np.linalg.inv(np.linalg.eig(self.cov)[1])) #save them to avoid doing eigen decomposition evertime of calling

        self.output_dim = OUTPUT_DIM

        self.n_bin          = 30 #Number of Tomographic bin
        self.theta_bin      = 26
        self.dropout        = 0.5
        self.learningrate   = 5e-4
        self.reduce_lr      = True
        self.loss_vali_goal = 0.05
       
        
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
        elif(model=='resnet_small'):
            print("Using resnet_samll model...")
            print("model output_dim: ", OUTPUT_DIM)
            # self.model = nn.Sequential(
            #     nn.Linear(N_DIM, 128),
            #     ResBlock(128, 256),
            #     nn.Dropout(0.3),
            #     ResBlock(256, 512),
            #     nn.Dropout(0.3),
            #     ResBlock(512, 2048),
            #     nn.Dropout(0.3),
            #     nn.PReLU(),
            #     nn.Linear(2048, OUTPUT_DIM),
            #     Affine()
            #     )
            # self.model = nn.Sequential(
            #     nn.Linear(N_DIM, 128),
            #     ResBlock(128, 256),
            #     nn.Dropout(0.3),
            #     ResBlock(256, 512),
            #     nn.Dropout(0.3),
            #     ResBlock(512, 1024),
            #     nn.Dropout(0.3),
            #     nn.PReLU(),
            #     nn.Linear(1024, OUTPUT_DIM),
            #     Affine()
            #     )
            self.model = nn.Sequential(
                nn.Linear(N_DIM, 512),
                ResBlock(512, 1024),
                nn.Dropout(0.3),
                nn.PReLU(),
                nn.Linear(1024, OUTPUT_DIM),
                Affine()
                )
        ###
        self.model.to(self.device)
        self.model.to(torch.float32)
        if self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learningrate)
        if self.reduce_lr == True:
            print('Reduce LR on plateu: ',self.reduce_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min')
        
    def train(self, X, y, X_validation, y_validation, test_split=None, batch_size=32, n_epochs=100):
        if not self.trained:
            self.X_mean = torch.Tensor(X.mean(axis=0, keepdims=True)).float()
            self.X_std  = torch.Tensor(X.std(axis=0, keepdims=True)).float()
    
        epoch_range = tqdm(range(n_epochs))

        losses_train = []
        losses_vali = []
        loss = 100.
        
        tmp_dv_max       = self.dv_max.to(self.device).float()
        tmp_dv_std       = self.dv_std.to(self.device).float()
        tmp_cov          = self.cov.to(self.device).float()
        tmp_cov_inv      = self.cov_inv.to(self.device).float()
        tmp_X_mean       = self.X_mean.to(self.device).float()
        tmp_X_std        = self.X_std.to(self.device).float()
        tmp_X_validation = X_validation.to(self.device).float()
        tmp_Y_validation = y_validation.to(self.device).float()
        tmp_L_inv            = self.L_inv.to(self.device).float() # in the new basis where Cov = QLQ^(-1)

        X_train     = ((X.float() - self.X_mean)/self.X_std)
        y_train     = y
        trainset    = torch.utils.data.TensorDataset(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    
        for _ in epoch_range:
            for i, data in enumerate(trainloader):    
                X       = data[0].to(self.device).float()                  # This is input
                Y_batch = data[1].to(self.device).float()                  # This is label

                Y_pred  = self.model.train()(X) * tmp_dv_std #technically should add y_fid back, back loss=chi2 is the difference so they are the same

                loss = torch.mean(torch.diagonal( torch.matmul( torch.matmul((Y_batch - Y_pred),tmp_L_inv), torch.t(Y_batch - Y_pred)) ) )

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            losses_train.append(loss.cpu().detach().numpy())
            ###Validation loss
            with torch.no_grad():
                Y_pred = self.model.eval()((tmp_X_validation - tmp_X_mean)/tmp_X_std) * tmp_dv_std 

            loss_vali = torch.mean(torch.diagonal( torch.matmul( torch.matmul((tmp_Y_validation - Y_pred),tmp_L_inv), torch.t(tmp_Y_validation - Y_pred)) ) )

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
        np.savetxt("test_dv.txt", np.array( [tmp_Y_validation.cpu().detach().numpy().astype(np.float32)[-1], Y_pred.cpu().detach().numpy().astype(np.float32)[-1] ]), fmt='%s')
        self.trained = True

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            X_mean = self.X_mean.clone().detach().to(self.device).float()
            X_std  = self.X_std.clone().detach().to(self.device).float()

            y_pred = self.model.eval()((X.to(self.device) - X_mean) / X_std).float().cpu() * self.dv_std #normalization
        
        y_pred = y_pred @ torch.Tensor(np.transpose(self.evecs)) + self.dv_fid #change of basis
        return y_pred.float().numpy()

    def save(self, filename):
        torch.save(self.model, filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['X_mean']       = self.X_mean
            f['X_std']        = self.X_std
            f['dv_fid']       = self.dv_fid
            f['dv_std']       = self.dv_std
            f['dv_max']       = self.dv_max
            f['cov']          = self.cov
            f['evecs']        = self.evecs
            f['evecs_inv']    = self.evecs_inv
        
    def load(self, filename):
        self.trained = True
        self.model = torch.load(filename)
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:]).float()
            self.X_std  = torch.Tensor(f['X_std'][:]).float()
            self.dv_fid = torch.Tensor(f['dv_fid'][:]).float()
            self.dv_std = torch.Tensor(f['dv_std'][:]).float()
            self.dv_max = torch.Tensor(f['dv_max'][:]).float()
            self.cov    = torch.Tensor(f['cov'][:]).float()
            self.evecs  = torch.Tensor(f['evecs'][:]).float()
            self.evecs_inv  = torch.Tensor(f['evecs_inv'][:]).float()

