import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5

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

        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2

    
class NNEmulator:
    def __init__(self, N_DIM, OUTPUT_DIM, dv_fid, dv_std, cov_inv, dv_max, device, model='resnet', optim=None):
        self.N_DIM = N_DIM
        self.model = model
        self.optim = optim
        self.device = device
        self.trained = False
        
        self.dv_fid  = torch.Tensor(dv_fid)
        self.dv_std  = torch.Tensor(dv_std)
        self.cov_inv = torch.Tensor(cov_inv)        
        self.dv_max  = torch.Tensor(dv_max)

        self.output_dim = OUTPUT_DIM
        
        if model is None:
            print("Using simply connected NN...")
            self.model = nn.Sequential(
                                nn.Linear(N_DIM, 512),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(512, 512), #additional laryer
                                nn.ReLU(), #additional laryer
                                nn.Dropout(0.3), #additional laryer
                                nn.Linear(512, OUTPUT_DIM),
                                Affine()
                                )            
        elif(model=='resnet'):
            print("Using resnet model...")
            self.model = nn.Sequential(
                    nn.Linear(N_DIM, 128),
                    ResBlock(128, 256),
                    nn.Dropout(0.3),
                    ResBlock(256, 256),
                    nn.Dropout(0.3),
                    ResBlock(256, 256),
                    nn.Dropout(0.3),
                    ResBlock(256, 512),
                    nn.Dropout(0.3),
                    ResBlock(512, 512),
                    nn.Dropout(0.3),
                    ResBlock(512, 512),
                    nn.Dropout(0.3),
                    ResBlock(512, 1024),
                    nn.Dropout(0.3),
                    ResBlock(1024, 1024),
                    nn.Dropout(0.3),
                    ResBlock(1024, 1024),
                    nn.Dropout(0.3),
                    ResBlock(1024, 1024),
                    nn.Dropout(0.3),
                    ResBlock(1024, 1024),
                    Affine(),
                    nn.PReLU(),
                    nn.Linear(1024, OUTPUT_DIM),
                    Affine()
                )

        self.model.to(self.device)
        

        if self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        
    def train(self, X, y, X_validation, y_validation, test_split=None, batch_size=32, n_epochs=100):
        if not self.trained:
            self.X_mean = torch.Tensor(X.mean(axis=0, keepdims=True))
            self.X_std  = torch.Tensor(X.std(axis=0, keepdims=True))
            self.y_mean = self.dv_fid
            self.y_std  = self.dv_std
    
        epoch_range = tqdm(range(n_epochs))

        losses = []
        losses_vali = []
        loss = 100.

        tmp_dv_max       = self.dv_max.to(self.device)
        tmp_cov_inv      = self.cov_inv.to(self.device)
        tmp_X_mean       = self.X_mean.to(self.device)
        tmp_X_std        = self.X_std.to(self.device)
        tmp_X_validation = X_validation.to(self.device)
        tmp_Y_validation = y_validation.to(self.device)

        X_train     = ((X - self.X_mean)/self.X_std)
        y_train     = y
        trainset    = torch.utils.data.TensorDataset(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=3500, shuffle=True, drop_last=True, num_workers=1)
    
        for _ in epoch_range:
            for i, data in enumerate(trainloader):    
                X       = data[0].to(self.device)                  # This is input
                Y_batch = data[1].to(self.device) * tmp_dv_max     # This is label
                Y_pred  = self.model(X) * tmp_dv_max
                    
                loss = torch.mean(torch.abs(((Y_batch - Y_pred) @ tmp_cov_inv) @ torch.t(Y_batch - Y_pred)))
                                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            ###Test loss
            Y_pred = self.model((tmp_X_validation - tmp_X_mean)/tmp_X_std) * tmp_dv_max

            loss_vali = torch.mean(torch.abs(((tmp_Y_validation - Y_pred) @ tmp_cov_inv) @ torch.t(tmp_Y_validation - Y_pred)))
 
            losses_vali.append(loss_vali.cpu().detach().numpy())
               
            epoch_range.set_description('Loss: {0}, Loss_validation: {1}'.format(loss, loss_vali))
        
        np.savetxt("losses.txt", losses_vali, fmt='%s')
        np.savetxt("test_dv.txt", np.array( [y_validation.detach().numpy()[-1], y_vali_pred.detach().numpy()[-1]] ), fmt='%s')
        self.trained = True

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            X_mean = self.X_mean.clone().detach()
            X_std  = self.X_std.clone().detach()

            X_norm = (X - X_mean) / X_std
            y_pred = self.model.eval()(X_norm).cpu()
            
        y_pred = y_pred* self.dv_max

        return y_pred.numpy()

    def save(self, filename):
        torch.save(self.model, filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['X_mean'] = self.X_mean
            f['X_std']  = self.X_std
            f['Y_mean'] = self.y_mean
            f['Y_std']  = self.y_std
            f['dv_fid'] = self.dv_fid
            f['dv_std'] = self.dv_std
            f['dv_max'] = self.dv_max
        
    def load(self, filename):
        self.trained = True
        self.model = torch.load(filename)
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:])
            self.X_std  = torch.Tensor(f['X_std'][:])
            self.y_mean = torch.Tensor(f['Y_mean'][:])
            self.y_std  = torch.Tensor(f['Y_std'][:])
            self.dv_fid = torch.Tensor(f['dv_fid'][:])
            self.dv_std = torch.Tensor(f['dv_std'][:])
            self.dv_max = torch.Tensor(f['dv_max'][:])