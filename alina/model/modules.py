import math
import torch
import torch.nn as nn


        
class MHAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        
        self.heads = heads
        self.dim = dim
        self.depth = dim//heads
        self.norm = math.sqrt(self.depth)
        
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.O = nn.Linear(dim, dim)

        for l in [self.Q, self.K, self.V, self.O]:
            torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
            torch.nn.init.zeros_(l.bias)    

        
    def forward(self, q, k, v, mask):
        batch, seq = q.size(0), q.size(1)

        # batch, seq, heads, dim
        q = self.Q(q).view(batch, seq, self.heads, self.depth)
        k = self.K(k).view(batch, seq, self.heads, self.depth)
        v = self.V(v).view(batch, seq, self.heads, self.depth)

        # batch, heads, seq, dim
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        
        #att
        g = torch.matmul(q, k)
        g /= self.norm
        A = g - mask*1e7
        A = torch.nn.functional.softmax(A, dim=-1)

        att = torch.matmul(A, v) # b,h,s,d

        att = att.permute(0, 2, 1, 3) # b,s,h,d
        att = att.reshape(batch, seq, self.dim)
        att = self.O(att)
        
        return att, g
    
    
class FFSwiglu(nn.Module):
    def __init__(self, dim: int, do: float):
        super().__init__()
        h = round( (8*dim/3)/16 )*16 #hidden layer size
        self.gate_linear = nn.Linear(dim, h)
        self.l1 = nn.Linear(dim, h)
        self.l2 = nn.Linear(h, dim)
        self.drop = nn.Dropout(do)

        torch.nn.init.kaiming_uniform_(self.gate_linear.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=1.0)

        torch.nn.init.zeros_(self.gate_linear.bias)
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.zeros_(self.l2.bias)
        
    def forward(self, x):
        gate = self.gate_linear(x)
        gate = torch.nn.functional.silu(gate)

        x = gate * self.l1(x)
        x = self.drop(x)
        x = self.l2(x)

        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, do: float, norm_layer):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.Att = MHAttention(self.dim, self.heads)
        self.drop = nn.Dropout(do)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.FF = FFSwiglu(dim, do)


    def forward(self, x, mask):

        att = self.norm1(x)
        att, scores = self.Att(att, att, att, mask)
        att = self.drop(att)
        x += att

        ff = self.norm2(x)
        ff = self.FF(ff)
        x += ff

        return x, scores