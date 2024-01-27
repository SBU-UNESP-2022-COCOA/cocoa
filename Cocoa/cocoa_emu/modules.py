import torch
from torch import nn
import torch.nn.functional as F
import random, math
import numpy as np

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


### From https://github.com/pbloem/former/blob/b438731ceeaf6c468f8b961bb07c2adde3b54a9f/former/modules.py#L10
class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            #nn.ReLU(),
            #nn.Tanh(),
            nn.PReLU(),
            #nn.Linear(emb, ff_hidden_mult * emb),
            #nn.PReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        #KZ
        #x = torch.stack([x,x,x,x],dim=2)

        attended = self.attention(x)

        #KZ
        #x = torch.stack([x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x],dim=2)
        #print("TEST",x.size(), attended.size() )
        x = self.norm1(attended + x)

        x = self.drop(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.drop(x)
        
        #KZ
        #print("TEST", x.size())
        # tmp1,tmp2,tmp3 = x.size()
        # x = torch.flatten(x)
        # x = torch.reshape(x,(tmp1,tmp2*tmp3))
        return x

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

    #KZ
        #tmp1, tmp2 = x.size()
        #x = torch.stack([x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x],dim=2)
        #print(x.size())
        b, t, e = x.size()

        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        #KZ
        #assert not util.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)


#Expand a tensor of size(batch, t) to 3d(batch, t, k)
class Expand2D(nn.Module):
    def __init__(self,t, k, mask=False):

        super().__init__()

        self.t = t
        self.k = k 
        self.layer = nn.Linear(t, t*self.k)

    def forward(self, x):

        tmp1, tmp2 = x.size()
        assert tmp2 == self.t, f'Input embedding dim ({tmp2}) should match layer embedding dim ({self.t})'
        
        x = self.layer(x)
        x = torch.reshape(x,(tmp1,tmp2,self.k))

        return x

class Squeeze(nn.Module):
    def __init__(self,t, k, mask=False):

        super().__init__()

        self.t = t
        self.k = k

    def forward(self, x):
        tmp1,tmp2,tmp3 = x.size()
        assert tmp2 == self.t, f'Input embedding dim ({tmp2}) should match layer embedding dim ({self.t})'
        assert tmp3 == self.k, f'Input embedding dim ({tmp3}) should match layer embedding dim ({self.k})'

        x = torch.flatten(x)
        x = torch.reshape(x,(tmp1,tmp2*tmp3))

        return x

class Attention_EV(nn.Module):
    def __init__(self, in_size ,n_partitions,device):
        super(Attention_EV, self).__init__()
        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)
        self.act          = nn.Softmax(dim=2)
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions
        self.norm         = torch.nn.BatchNorm1d(in_size)
        self.device       = device
        
    def forward(self, x):
        batchsize = x.shape[0]

        self.Q = torch.empty((batchsize,self.embed_dim,self.n_partitions)).to(self.device)
        self.K = torch.empty((batchsize,self.embed_dim,self.n_partitions)).to(self.device)
        self.V = torch.empty((batchsize,self.embed_dim,self.n_partitions)).to(self.device)

        # stack the input to find Q,K,V
        for i in range(self.n_partitions):
            qi = self.WQ(self.norm(x)[:,i*self.embed_dim:(i+1)*self.embed_dim])
            ki = self.WK(self.norm(x)[:,i*self.embed_dim:(i+1)*self.embed_dim])
            vi = self.WV(self.norm(x)[:,i*self.embed_dim:(i+1)*self.embed_dim])

            self.Q[:,:,i] = qi
            self.K[:,:,i] = ki
            self.V[:,:,i] = vi

        # compute weighted dot product
        dot_product = torch.bmm(self.Q,self.K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,self.V)

        #concat results of each head
        out = torch.cat(tuple([prod[:,i] for i in range(self.embed_dim)]),dim=1)+x

        return out


class Transformer_EV(nn.Module):
    def __init__(self, n_heads, int_dim,device):
        super(Transformer_EV, self).__init__()  
    
        self.int_dim     = int_dim
        self.n_heads     = n_heads
        self.module_list = nn.ModuleList([nn.Linear(int_dim,int_dim) for i in range(n_heads)])
        self.act         = nn.Tanh()#nn.SiLU()
        self.norm        = torch.nn.BatchNorm1d(int_dim*n_heads)
        self.device       = device

    def forward(self,x):
        # init result array
        batchsize = x.shape[0]
        
        results = torch.empty((batchsize,self.int_dim,self.n_heads))

        # do mlp for each head
        for i,layer in enumerate(self.module_list):
            o = self.norm(x)[:,i*self.int_dim:(i+1)*self.int_dim]
            o = self.act(layer(o))
            results[:,:,i] = o

        # concat heads
        out = torch.cat(tuple([results[:,i] for i in range(self.int_dim)]),dim=1).to(self.device)

        return out+x

# Transformer Block using the pytorch default self-attention function, nn.MultiheadAttention
class TransformerBlock_NN(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.0):
        super(TransformerBlock_NN, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# 1D-CNN test
class CNN1D(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(CNN1D, self).__init__()

        self.layer1 = nn.Conv1d(in_size, out_size, kernel_size, stride=1)
        self.act1   = nn.Tanh()
    def forward(self, x):
        # Reshape for 1D convolution
        x = x.view(x.size(0), 1, -1)
        x = self.layer1(x)
        return self.act1(x)

class Simple1DCNN(nn.Module):
    def __init__(self, in_size, out_size):
        super(Simple1DCNN, self).__init__()
        
        # Input linear transformation
        self.input_layer = nn.Linear(in_size, 300) #n_in = linear transform cosmological parameters to a longer tensor
        
        # 1D Convolution layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        
        # You might need to adjust the size here based on the exact architecture.
        # This number (128*86) depends on the lengths of your sequences after each convolution.
        # input of this layer is 128*N, where N = (( (n_in - 4)/2-4 ) / 2 -4 )/2
        # First minus the (kernel-1) for convolv dim reduction; then devide by kernel size of pool layer
        self.fc1 = nn.Linear(256 * 17, 1024)
        self.fc2 = nn.Linear(1024, 1024) 
        self.fc3 = nn.Linear(1024, 1024)  
        self.fc4 = nn.Linear(1024, out_size)
        
        self.act = nn.PReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.skip = nn.Identity()
        
    def forward(self, x):
        # Expanding input dimensions
        #x = self.act(self.input_layer(x)) # with act
        x = self.input_layer(x) # without act, just linear transform
        # Reshape for 1D convolution
        x = x.view(x.size(0), 1, -1)
        
        # 1D Convolutions
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.conv3(x))
        x = self.pool(x)
        
        # Flattening the tensor
        x = x.view(x.size(0), -1)
        
        # # Fully connected layers
        # x = self.act(self.fc1(x))
        # x = self.fc4(x)

        # TEST Residual Network
        x = self.act(self.fc1(x))
        xskip = self.skip(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x + xskip
        x = self.fc4(x)
        
        return x

