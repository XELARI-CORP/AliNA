from typing import List
import math
import torch
import torch.nn as nn

from .modules import ComplementaryLayer, EncoderLayer, MSATransformer, make_transformer_block


class Model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 dim: int, 
                 seq_len: int,
                 conv_layers: int,
                 msa_transformers_order: int | List[int],
                 encoders_order: int | List[int],
                 heads: int,
                 convdrop: float,
                 conv_activation,
                 norm_layer
                ):
        super().__init__()
        self.dim = dim
        self.final_dot_norm = math.sqrt(dim) # ???
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, dim)
        self.nb_token = torch.nn.Parameter(torch.rand(1,1,dim), requires_grad=True) # NoBound token 

        # Conv Block
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
        
        # MSA Transformer Block
        self.msa_transformers_block, self.msa_transformers_order = make_transformer_block(
            msa_transformers_order, MSATransformer,
            (dim, heads, seq_len, 0.1, norm_layer))
        
        # Encoder Block
        self.encoders_block, self.encoders_order = make_transformer_block(
            encoders_order, EncoderLayer,
            (dim, heads, seq_len, 0.1, norm_layer))

        self._init_pos_enc(seq_len+1, dim)
        
        # Head
        self.final_norm = norm_layer(dim)
        self.DotW = torch.nn.Parameter(torch.rand((dim, dim)), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.DotW, gain=1.0)
        self.final_bias = torch.nn.Parameter(torch.tensor(-3.0), requires_grad=True)

    def _init_pos_enc(self, seq: int, dim: int):
        div  = 10000**(2 * (torch.arange(dim, dtype=torch.float32) // 2) / dim)
        pos_enc = torch.arange(seq, dtype=torch.float32)
        pos_enc = pos_enc.view(seq, 1).repeat(1, dim) / div
        
        pos_enc[1:, 0::2] = torch.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = torch.cos(pos_enc[1:, 1::2]) # dim 2i+1
        
        self.register_buffer('pe', pos_enc)

    # @torch.compile
    # def pos_enc(self, seq: int, dim: int):
    #     div  = 10000**(2 * (torch.arange(dim, dtype=torch.float32) // 2) / dim)
    #     pos_enc = torch.arange(seq, dtype=torch.float32)
    #     pos_enc = pos_enc.view(seq, 1).repeat(1, dim) / div
        
    #     pos_enc[1:, 0::2] = torch.sin(pos_enc[1:, 0::2]) # dim 2i
    #     pos_enc[1:, 1::2] = torch.cos(pos_enc[1:, 1::2]) # dim 2i+1

    #     return pos_enc
        
            
    def forward(self, seq: torch.Tensor, struct_vec: torch.Tensor): 
        batch, msa_len, seq_len = seq.shape
        x = self.embedding(seq) # b, msa, seq, dim
        
        x = x.view(batch*msa_len, seq_len, self.dim).transpose(1,2) # b+msa, dim, seq
        x = self.conv_model(x)
        x = x.transpose(1,2).reshape(batch, msa_len, seq_len, self.dim) # b, msa, seq, dim

        seq_mask = (seq==0).to(x.dtype).view(-1, 1, 1, seq_len) # b, msa, seq -> b+msa, 1, 1, seq
        msa_mask = (seq.sum(dim=-1) == 0).to(x.dtype) # b, msa
        # b, msa -> b, 1, msa -> b, seq, msa -> b*seq, 1, 1, msa
        msa_mask = msa_mask.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch * seq_len, 1, 1, msa_len)
        
        for li in self.msa_transformers_order:
            l = self.msa_transformers_block[li]
            x = l(x, msa_mask, seq_mask)

        x = x[:,0,:,:]   # batch, msa, seq, dim -> batch, seq, dim
        # add nb token
        nb_token = self.nb_token.expand(batch, 1, -1) # 1,1,dim -> b, 1, dim
        x = torch.cat((nb_token, x), dim=1) # b, seq+1, dim

        seq = seq[:,0,:] # batch, msa, seq -> b, seq
        nb_pad = torch.ones((batch, 1), dtype=seq.dtype, device=seq.device) # b,1
        seq = torch.cat((nb_pad, seq), dim=1) # b, seq+1

        #x += self.pos_enc(x.size(1), x.size(2)).to(x.dtype).to(x.device)
        x = x + self.pe.to(x.dtype)
        x = self.complementary_layer(x, struct_vec)

        att_mask = (seq==0).to(x.dtype).view(seq.size(0), 1, 1, seq.size(1)) # b, seq -> b, 1, 1, seq
        for li in self.encoders_order:
            l = self.encoders_block[li]
            x, _ = l(x, att_mask) # b,seq+1,dim

        x = self.final_norm(x)
        x = torch.matmul(
            torch.matmul(x, self.DotW), # @ dim,dim -> b, seq, dim
            torch.transpose(x, 1, 2) # @ b, dim, seq -> b, seq, seq
        )
        x = x / self.final_dot_norm
        x = x + self.final_bias
        x = x - 1e7 * torch.eye(x.size(1), dtype=x.dtype, device=x.device).unsqueeze(0)

        nbn, m = x[:, 0], x[:, 1:] # b, seq | b, seq-1, seq
        nbn = torch.sigmoid(nbn)
        m = torch.nn.functional.softmax(m, dim=-1)
        
        return nbn, m


















        