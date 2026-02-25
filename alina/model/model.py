import torch
import torch.nn as nn
from torch.nn import functional as F

from .modules import pos_enc, ConvLayerNorm, ConvBlock, MHAttention, EncoderLayer

#  заглушка, чтобы не чинить импорты во всех файлах
class Model(nn.Module):
    def __init__():
        super().__init__()

class MSABody(nn.Module):
    def __init__(self, 
                 vocab: int, 
                 emb: int, 
                 dim: int, 
                 layers: int, 
                 heads: int, 
                 channels: int, 
                 convdrop: float
                ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab, emb)
        
        # ENCODER
        self.pool = nn.MaxPool2d(2, stride=2)
        self.EncCVBlock1 = ConvBlock(emb, channels[0], convdrop)
        self.EncCVBlock2 = ConvBlock(channels[0], channels[1], convdrop)
        self.EncCVBlock3 = ConvBlock(channels[1], channels[2], convdrop)
        self.EncCVBlock4 = ConvBlock(channels[2], channels[3], convdrop)
        
        # MIDDLE
        self.postpool = nn.Linear(channels[3], dim)
        torch.nn.init.xavier_uniform_(self.postpool.weight, gain=1.0)
        
        self.PE = torch.nn.parameter.Parameter(pos_enc(seq=256, dim=dim), requires_grad=False)
        self.encoders_list = nn.ModuleList()
        for i in range(layers):
            self.encoders_list.append(EncoderLayer(dim=dim, heads=heads, do=0.1))

        # DECODER
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.DecCVBlock1 = ConvBlock((dim+channels[3]), channels[-1], convdrop)
        self.DecCVBlock2 = ConvBlock(channels[-2]+channels[-1], channels[-2], convdrop)
        self.DecCVBlock3 = ConvBlock(channels[-3]+channels[-2], channels[-3], convdrop)
        
    def get_pad_mask(self, x):
        x = x.view(-1, 16, 16, 256)
        x = torch.permute(x, (0, 1, 3, 2)).contiguous() # b, Hpatch, patch_dim, W -> b, Hpatch, W, patch_dim
        x = x.view(-1, 16**2, 16**2) # b, Hpatch, W, patch_dim -> b, Hpatch*Wpatch, patch_dim*patch_dim
        
        x = torch.sum(x, dim=-1, keepdim=False) # b, Hpatch*Wpatch
        return (x==0.).float()
        
            
    def forward(self, x):
        mask = self.get_pad_mask(x) # b, seq, seq
        att_mask = torch.unsqueeze(torch.unsqueeze(mask, 1), 1) # b, seq(1), heads(1), seq
        
        # EMB
        x = self.embedding(x) # b, 256H, 256W, emb
        x = torch.permute(x, (0, 3, 1, 2)).contiguous() # b, dim, 256H, 256W
        
        # ENCODER
        x1 = self.EncCVBlock1(x) # b, 16, 256, 256 -> b, 32, 256, 256
        x = self.pool(x1) # b, 32, 256, 256 -> b, 32, 128, 128

        x2 = self.EncCVBlock2(x) # b, 32, 128, 128 -> b, 64, 128, 128
        x = self.pool(x2) # b, 64, 128, 128 -> b, 64, 64, 64

        x3 = self.EncCVBlock3(x) # b, 64, 64, 64 -> b, 128, 64, 64
        x = self.pool(x3) # b, 128, 64, 64 -> b, 128, 32, 32

        x4 = self.EncCVBlock4(x) # b, 128, 32, 32 -> b, 256, 32, 32
        x = self.pool(x4) # b, 256, 32, 32 -> b, 256, 16, 16
        
        # MIDDLE
        x = torch.permute(x, (0, 2, 3, 1)).contiguous() # b, dim, 16, 16 -> b, 16, 16, dim
        x = x.view(-1, 256, x.shape[-1]) # b, 16, 16, dim -> b, seq(256), dim
        x = self.postpool(x)
        x += self.PE
        
        for l in self.encoders_list:
            x = l([x, x, x, att_mask])
            
        x = x.view(-1, 16, 16, x.shape[-1])
        x = torch.permute(x, (0, 3, 1, 2)).contiguous() # b, H, W, dim -> b, dim, H, W
            
        # DECODER
        x = self.upsample(x) # b, 256, 16, 16 -> # b, 256, 32, 32
        x = torch.cat((x, x4), 1) # b, 256, 32, 32 + b, 256, 32, 32 -> b, 512, 32, 32
        x = self.DecCVBlock1(x) # b, 512, 32, 32 -> b, 256, 32, 32

        x = self.upsample(x) # b, 256, 32, 32 -> b, 256, 64, 64
        x = torch.cat((x, x3), 1) # b, 256, 64, 64 + b, 128, 64, 64 -> b, 384, 64, 64
        x = self.DecCVBlock2(x) # b, 384, 64, 64 -> b, 128, 64, 64

        x = self.upsample(x) # b, 128, 64, 64 -> b, 128, 128, 128
        x = torch.cat((x, x2), 1) # b, 128, 128, 128 + b, 64, 128, 128 -> b, 192, 128, 128
        x = self.DecCVBlock3(x) # b, 192, 128, 128 -> b, 64, 128, 128

        x = self.upsample(x) # b, 64, 128, 128 -> b, 64, 256, 256
        x = torch.cat((x, x1), 1) # b, 64, 256, 256 + b, 32, 256, 256 -> b, 96, 256, 256
        
        return x

class MSAClassifier(nn.Module):
    def __init__(self,
                 body_parameters: dict,
                 head_dim: int = 128):
        super().__init__()
        self.body = MSABody(**body_parameters)
        self.head_dim = head_dim

        mlp_input_dim = body_parameters['channels'][-3] + body_parameters['channels'][0] 
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, self.head_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.head_dim, self.head_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.head_dim, self.head_dim)
        )
        torch.nn.init.kaiming_uniform_(self.mlp[0].weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.mlp[3].weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_( self.mlp[6].weight, gain=1.0)

        torch.nn.init.zeros_(self.mlp[0].bias)
        torch.nn.init.zeros_(self.mlp[3].bias)
        torch.nn.init.zeros_(self.mlp[6].bias)

        self.head = nn.Sequential(
            nn.Linear(self.head_dim+1, 2*self.head_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(2*self.head_dim, 2*self.head_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(2*self.head_dim, 1)
        )
        torch.nn.init.kaiming_uniform_(self.head[0].weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.head[3].weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_( self.head[6].weight, gain=1.0)

        torch.nn.init.zeros_(self.head[0].bias)
        torch.nn.init.zeros_(self.head[3].bias)
        torch.nn.init.zeros_(self.head[6].bias)
        
    def forward(self, msa, adj): # MSA, RNA 2d adj. matrix

        msa = self.body(msa) # b, dim, H, W
        msa = torch.permute(msa, (0, 2, 3, 1)).contiguous() # b, H, W, dim
        msa = self.mlp(msa) # b, H, W, head_dim

        msa = msa.sum(axis=0).squeeze() # H, W, head_dim

        adj = adj.unsqueeze(-1).to(msa.dtype) # H, W, 1
        x = torch.cat((msa, adj), dim=-1) # H, W, head_dim + 1

        x = self.head(x).squeeze(-1) # H, W
        x = torch.sigmoid(x)

        return x
        
        
        
        
















        