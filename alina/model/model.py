from typing import List
import math
import torch
import torch.nn as nn

from .modules import ComplementaryLayer, EncoderLayer


class Model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 dim: int, 
                 conv_layers: int,
                 encoders_order: int | List[int],
                 heads: int,
                 convdrop: float,
                 conv_activation,
                 norm_layer
                ):
        super().__init__()
        self.dim = dim
        self.final_dot_norm = math.sqrt(dim)
        
        # Embedding
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

        # Secondary structure
        self.complementary_layer = ComplementaryLayer(dim)
        
        # Transformer
        if isinstance(encoders_order, int):
            encoders_order = list(range(encoders_order))

        assert 0 not in encoders_order, "Encoder indices must start from 1, not 0"
        assert set(encoders_order) == set(range(1, 1+max(encoders_order))), "All layers must be used"

        self.encoders_list = nn.ModuleList()
        for i in range(max(encoders_order)):
            self.encoders_list.append(EncoderLayer(dim=dim, heads=heads, do=0.1, norm_layer=norm_layer))

        self.encoders_order = [i-1 for i in encoders_order]
        
        # Head
        self.final_norm = norm_layer(dim)
        self.DotW = torch.nn.Parameter(torch.rand((dim, dim)), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.DotW, gain=1.0)
        self.final_bias = torch.nn.Parameter(torch.tensor(-3.0), requires_grad=True)


    @torch.compile
    def pos_enc(self, seq: int, dim: int):
        div  = 10000**(2 * (torch.arange(dim, dtype=torch.float32) // 2) / dim)
        pos_enc = torch.arange(seq, dtype=torch.float32)
        pos_enc = pos_enc.view(seq, 1).repeat(1, dim) / div
        
        pos_enc[1:, 0::2] = torch.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = torch.cos(pos_enc[1:, 1::2]) # dim 2i+1

        return pos_enc
        
            
    def forward(self, seq: torch.Tensor, struct_vec: torch.Tensor): 
        
        x = self.embedding(seq) # b, seq, dim
        x = torch.transpose(x, 1, 2) # -> b, dim, seq
        x = self.conv_model(x)
        x = torch.transpose(x, 1, 2) # -> b, seq, dims

        x += self.pos_enc(x.size(1), x.size(2)).to(x.dtype).to(x.device)
        x = self.complementary_layer(x, struct_vec)

        att_mask = (seq==0).to(x.dtype).view(seq.size(0), 1, 1, seq.size(1)) # b, seq -> b, 1, 1, seq
        for li in self.encoders_order:
            l = self.encoders_list[li]
            x, scores = l(x, att_mask)
            
        x = self.final_norm(x)
        x = torch.matmul(
            torch.matmul(x, self.DotW), # -> b, seq, dim
            torch.transpose(x, 1, 2) # @ b, dim, seq -> b, seq, seq
        )
        x = x / self.final_dot_norm
        x = x + self.final_bias
        x = x - 1e7 * torch.diag(torch.ones(x.size(1), dtype=x.dtype, device=x.device)).unsqueeze(0)

        nbn, m = x[:, 0], x[:, 1:] # b, seq | b, seq-1, seq
        nbn = torch.sigmoid(nbn)
        m = torch.nn.functional.softmax(m, dim=-1)
        
        return nbn, m


















        