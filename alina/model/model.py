import torch
import torch.nn as nn

from .modules import EncoderLayer, PirwiseHead


class Model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 dim: int, 
                 conv_layers: int,
                 encoder_layers: int,
                 heads: int,
                 convdrop: float,
                 conv_activation,
                 norm_layer
                ):
        super().__init__()
        self.dim = dim
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.conv_model = nn.Sequential()
        for i in range(conv_layers):
            if i>0:
                self.conv_model.append(nn.Dropout(convdrop))

            layer = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding="same")
            if (i+1)<conv_layers:
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                self.conv_model.append(layer)
                self.conv_model.append(conv_activation())
            else:
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                self.conv_model.append(layer)
        
        self.encoders_list = nn.ModuleList()
        for i in range(encoder_layers):
            self.encoders_list.append(EncoderLayer(dim=dim, heads=heads, do=0.1, norm_layer=norm_layer))

        self.final_norm = norm_layer(dim)
        self.final_project = nn.Linear(dim, dim//4)
        torch.nn.init.xavier_uniform_(self.final_project.weight, gain=1.0)
        self.head = PirwiseHead(dim//4)


    @torch.compile
    def pos_enc(self, seq: int, dim: int):
        div  = 10000**(2 * (torch.arange(dim, dtype=torch.float32) // 2) / dim)
        pos_enc = torch.arange(seq, dtype=torch.float32)
        pos_enc = pos_enc.view(seq, 1).repeat(1, dim) / div
        
        pos_enc[1:, 0::2] = torch.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = torch.cos(pos_enc[1:, 1::2]) # dim 2i+1

        return pos_enc
        
            
    def forward(self, seq): 
        
        x = self.embedding(seq) # b, seq, dim
        x = torch.transpose(x, 1, 2) # -> b, dim, seq
        x = self.conv_model(x)
        x = torch.transpose(x, 1, 2) # -> b, seq, dim

        x += self.pos_enc(x.size(1), x.size(2)).to(x.dtype).to(x.device)

        att_mask = (seq==0).to(x.dtype).view(seq.size(0), 1, 1, seq.size(1)) # b, seq -> b, 1, 1, seq
        for l in self.encoders_list:
            x = l(x, att_mask)
            
        x = self.final_norm(x)
        x = self.final_project(x)
        x = self.head(x)
        x = torch.sigmoid(x)

        diag_mask = 1. - torch.diag(
            torch.ones(x.size(1), dtype=x.dtype, device=x.device)
        ).unsqueeze(0)
        x = x * diag_mask
        
        return x


















        