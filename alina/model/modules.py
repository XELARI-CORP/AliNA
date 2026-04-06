import math
import torch
import torch.nn as nn

def make_transformer_block(transformer_layers_order, transformer_layer,
                           transformer_layer_hparams):

        dim, heads, seq_len, do, norm = transformer_layer_hparams
    
        if isinstance(transformer_layers_order, int):
            transformer_layers_order = list(range(1, transformer_layers_order+1))

        assert 0 not in transformer_layers_order, "Layer indices must start from 1, not 0"
        assert set(transformer_layers_order) == set(range(1, 1+max(transformer_layers_order))), "All layers must be used"

        transformers_list = nn.ModuleList()
        for _ in range(max(transformer_layers_order)):
            transformers_list.append(transformer_layer(dim=dim, heads=heads,
                                                       seq_len=seq_len, do=do,
                                                       norm_layer=norm))

        transformer_layers_order = [i-1 for i in transformer_layers_order]

        return transformers_list, transformer_layers_order
        
class ComplementaryLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
        self.l1 = nn.Linear(dim+2, dim)
        torch.nn.init.kaiming_uniform_(self.l1.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.l1.bias)

        self.l2 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=1.0)
        torch.nn.init.zeros_(self.l2.bias)
        
        self.drop = torch.nn.Dropout(0.1)


    def take_compl_embeds(self, x: torch.Tensor, struct_vec: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        
        idx = torch.where(struct_vec!=0, struct_vec, torch.arange(seq, device=x.device))
        idx = idx + seq*torch.arange(batch, dtype=torch.int32, device=idx.device).unsqueeze(1)
        assert idx.size(0) == batch and idx.size(1) == seq

        return x.view(batch*seq, dim)[idx]


    def forward(self, x: torch.Tensor, struct_vec: torch.Tensor):
        free_nts = (struct_vec==0).to(x.dtype) # b, seq
        compl_nts = 1.0 - free_nts

        x = torch.cat([x, free_nts.unsqueeze(2), compl_nts.unsqueeze(2)], dim=-1) # b, seq, dim+2
        x = self.l1(x)
        compl_x = self.take_compl_embeds(x, struct_vec)
        x = x + compl_x
        x = torch.nn.functional.silu(x)
        x = self.drop(x)
        x = self.l2(x)

        return x


class MHAttention(nn.Module):
    def __init__(self, dim: int, heads: int,
                 seq_len: int, use_rope: bool):
        super().__init__()
        
        self.heads = heads
        self.dim = dim
        self.depth = dim//heads
        self.norm = math.sqrt(self.depth)

        self.use_rope = use_rope
        self.seq_len = seq_len
        
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.O = nn.Linear(dim, dim)

        if self.use_rope:
            self.prepare_rope()
        
        for l in [self.Q, self.K, self.V, self.O]:
            torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
            torch.nn.init.zeros_(l.bias)

    def prepare_rope(self):
        i = torch.arange(0, self.depth, 2) #0,2...,depth-2
        dim_freq = 10_000 ** (-i / self.depth)
        pos  = torch.arange(self.seq_len) 

        freq = torch.outer(pos, dim_freq) # seq_len, depth//2
        freq = torch.cat((freq, freq), dim=-1) # seq_len, depth
        
        self.rope_sin = freq.sin()[None,None,:,:] # 1,1,seq_len,depth
        self.rope_cos = freq.cos()[None,None,:,:]

    def swap_dims(self,x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2,x1), dim=-1)
    
    
    def forward(self, q, k, v, mask):
        # if common att, then q.shape() = (b,seq,dim)
        # if msa att, then q.shape() = (b,_,n,dim) where n ~ msa or seq
        orig_shape = q.shape
        batch, n = orig_shape[0], orig_shape[-2]

        # virt batch (0), n (1), heads (2), depth (3)
        q = self.Q(q).view(-1, n, self.heads, self.depth)
        k = self.K(k).view(-1, n, self.heads, self.depth)
        v = self.V(v).view(-1, n, self.heads, self.depth)

        q = q.permute(0, 2, 1, 3) # b, h, n, d
        k = k.permute(0, 2, 1, 3) # b(0), h(1), n(2), d(3)
        v = v.permute(0, 2, 1, 3) # b, h, n, d

        if self.use_rope:
            cos = self.rope_cos.to(q.dtype).to(q.device)
            sin = self.rope_sin.to(q.dtype).to(q.device)
            
            q = (q * cos) + (self.swap_dims(q) * sin)
            k = (k * cos) + (self.swap_dims(k) * sin)
            
        k = k.permute(0, 1, 3, 2) # b, h, d, n 
        
        #att
        g = torch.matmul(q, k) # b, h, n, n
        g /= self.norm
        A = g - mask*1e7
        A = torch.nn.functional.softmax(A, dim=-1)

        att = torch.matmul(A, v) # b, h, n, d

        att = att.permute(0, 2, 1, 3).contiguous() # b, n, h, d
        att = att.view(*orig_shape) # virt batch, n, head, depth -> batch, _, n, dim
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

class MSATransformer(nn.Module):
    def __init__(self, dim: int, heads: int, seq_len: int, do: float, norm_layer):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.SeqAtt = MHAttention(dim, heads, seq_len, True)
        self.MSAAtt = MHAttention(dim, heads, seq_len, False)
        self.drop = nn.Dropout(do)

        self.norm_layers = nn.ModuleList([norm_layer(dim) for _ in range(3)])

        self.FF = FFSwiglu(dim, do)

    def forward(self, x, msa_mask, seq_mask):
        # x.shape: b(0), msa(1), seq(2), dim(3)
        # seq-wise att
        att = self.norm_layers[0](x)
        att, _ = self.SeqAtt(att, att, att, seq_mask)
        att = self.drop(att)
        x += att

        x = x.permute(0, 2, 1, 3).contiguous() # b, seq, msa, dim

        #print(x.shape, msa_mask.shape) # !!!
        att = self.norm_layers[1](x)
        att, _ = self.MSAAtt(att, att, att, msa_mask)
        att = self.drop(att)
        x += att

        x = x.permute(0, 2, 1, 3).contiguous() # b, msa, seq, dim
        
        ff = self.norm_layers[2](x)
        ff = self.FF(ff)
        x += ff

        return x
        
        

class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, seq_len: int, do: float, norm_layer):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.Att = MHAttention(dim, heads, seq_len, False)
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