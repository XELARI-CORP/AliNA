import math
import torch
import torch.nn as nn
from torch.nn import functional as F



def pos_enc(seq, dim):
      
    pos_enc = torch.FloatTensor([
        [pos / (10000**(2 * (j // 2) / dim)) for j in range(dim)]  for pos in range(seq)
    ])

    pos_enc[1:, 0::2] = torch.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = torch.cos(pos_enc[1:, 1::2]) # dim 2i+1

    return pos_enc


class ConvLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.b = torch.nn.parameter.Parameter(torch.zeros((dim,1,1)), requires_grad=True)
        self.a = torch.nn.parameter.Parameter(torch.ones((dim,1,1)), requires_grad=True)
        self.eps = 1e-6
        
    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        dif = x - mean
        var = dif.pow(2).mean(1, keepdim=True)
        x = dif / torch.sqrt(var + self.eps)
        x = self.a*x + self.b
        
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, do):
        super().__init__()
        
        self.H = nn.Conv2d(in_ch, out_ch, kernel_size=(1,5), stride=1, padding=(0,2))
        self.W = nn.Conv2d(out_ch, out_ch, kernel_size=(5,1), stride=1, padding=(2,0))
        torch.nn.init.kaiming_uniform_(self.H.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.W.weight, nonlinearity='relu')
        
        self.l1 = nn.Conv2d(out_ch, out_ch*4, 
                            kernel_size=(1,1), stride=1, padding='valid')
        self.l2 = nn.Conv2d(out_ch*4, out_ch, 
                            kernel_size=(1,1), stride=1, padding='valid')
        
        torch.nn.init.kaiming_uniform_(self.l1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=1.0)
        self.drop = nn.Dropout(do)
        self.norm = ConvLayerNorm(out_ch)
        
        
    def forward(self, x):
        x = self.H(x)
        x = F.relu(x)

        x = self.W(x)
        a = F.relu(x)

        x = self.l1(a)
        x = F.relu(x)
        x = self.l2(x)
        
        x = self.drop(x)
        x = x+a
        x = self.norm(x)
        
        return x
        
        
class MHAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        
        self.heads = heads
        self.dim = dim
        self.depth = dim//heads
        self.norm = math.sqrt(self.depth)
        
        self.Q = torch.nn.parameter.Parameter(torch.empty(dim, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.Q, gain=1.0)
        
        self.K = torch.nn.parameter.Parameter(torch.empty(dim, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.K, gain=1.0)
        
        self.V = torch.nn.parameter.Parameter(torch.empty(dim, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.V, gain=1.0)
        
        self.O = torch.nn.parameter.Parameter(torch.empty(dim, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.O, gain=1.0)
        
        
    def forward(self, args):
        q, k, v, mask = args
        
        q = torch.matmul(q, self.Q)
        k = torch.matmul(k, self.K)
        v = torch.matmul(v, self.V)

        # batch, seq, heads, dim
        q = q.view(-1, q.shape[-2], self.heads, self.depth)
        k = k.view(-1, k.shape[-2], self.heads, self.depth)
        v = v.view(-1, v.shape[-2], self.heads, self.depth)

        # batch, heads, seq, dim
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        
        #att
        g = torch.matmul(q, k)
        g /= self.norm
        if mask is not None:
            g -= (mask*1e9)
        A = F.softmax(g, dim=-1)

        att = torch.matmul(A, v)# b,h,s,d

        att = att.permute(0, 2, 1, 3)# b,s,h,d
        att = torch.reshape(att, (att.shape[0], att.shape[-3], self.dim))
        att = torch.matmul(att, self.O)
        
        return att
    
    
class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, do):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.do = do

        self.Att = MHAttention(self.dim, self.heads) 

        self.drop1 = nn.Dropout(self.do)
        self.drop2 = nn.Dropout(self.do)

        self.LN1 = nn.LayerNorm(normalized_shape=dim)
        self.LN2 = nn.LayerNorm(normalized_shape=dim)

        self.FC1 = nn.Linear(dim, dim*4)
        self.FC2 = nn.Linear(dim*4, dim)
        torch.nn.init.kaiming_uniform_(self.FC1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.FC2.weight, gain=1.0)


    def forward(self, args):
        q, k, v, mask = args

        att = self.Att([q, k, v, mask])
        att = self.drop1(att)
        
        x = att + q
        x = self.LN1(x)

        d = F.relu(self.FC1(x))
        d = self.FC2(d)
        d = self.drop2(d)

        x = d + x
        x = self.LN2(x)

        return x