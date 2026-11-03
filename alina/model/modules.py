import math
import torch
import torch.nn as nn
from typing import List

def make_encoder_block(layers_order: int | List[int],
                       dim: int,
                       heads: int,
                       do: float,
                       encoder_layer: nn.Module,
                       norm_layer: nn.Module,
                       skip_msa_att: bool = False):
            
        if isinstance(layers_order, int):
            layers_order = list(range(1, layers_order+1))

        assert 0 not in layers_order, "Layer indices must start from 1, not 0"
        assert set(layers_order) == set(range(1, 1+max(layers_order))), "All layers must be used"

        encoders_list = nn.ModuleList()
        for _ in range(max(layers_order)):
            encoders_list.append(encoder_layer(dim=dim, heads=heads, do=do,
                                               norm_layer=norm_layer,
                                               skip_att=skip_msa_att))

        layers_order = [i-1 for i in layers_order]

        return encoders_list, layers_order
        
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
        
        idx = torch.where(struct_vec!=-1, struct_vec, torch.arange(seq, device=x.device))
        idx = idx + seq*torch.arange(batch, dtype=torch.int32, device=idx.device).unsqueeze(1)
        assert idx.size(0) == batch and idx.size(1) == seq

        return x.view(batch*seq, dim)[idx]


    def forward(self, x: torch.Tensor, struct_vec: torch.Tensor):
        free_nts = (struct_vec==-1).to(x.dtype) # b, seq
        compl_nts = 1.0 - free_nts

        x = torch.cat([x, free_nts.unsqueeze(2), compl_nts.unsqueeze(2)], dim=-1) # b, seq, dim+2
        x = self.l1(x)
        compl_x = self.take_compl_embeds(x, struct_vec)
        x = x + compl_x
        x = torch.nn.functional.silu(x)
        x = self.drop(x)
        x = self.l2(x)

        return x

class FakeRoPE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k):
        return q, k

class RoPE(nn.Module):
    def __init__(self, depth):
        super().__init__()

        self.depth = depth
        
    def swap_dims(self,x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2,x1), dim=-1)
        
    @torch.compile
    def prepare_rope(self, seq_len, device, dtype):
        
        i = torch.arange(0, self.depth, 2, device=device, dtype=dtype) #0,2...,depth-2
        dim_freq = 10_000 ** (-i / self.depth)
        pos  = torch.arange(seq_len, device=device, dtype=dtype) 

        freq = torch.outer(pos, dim_freq) # seq_len, depth//2
        freq = torch.cat((freq, freq), dim=-1) # seq_len, depth
        
        sin = freq.sin().view(1, 1, seq_len, self.depth) # 1,1,seq_len,depth
        cos = freq.cos().view(1, 1, seq_len, self.depth) # 1,1,seq_len,depth

        return sin, cos
         
    def forward(self, q, k):
        # q.shape = b,h,seq_len,depth
        seq_len = q.size(-2)

        sin, cos = self.prepare_rope(seq_len, q.device, q.dtype)
        
        q = (q * cos) + (self.swap_dims(q) * sin)
        k = (k * cos) + (self.swap_dims(k) * sin)

        return q, k
        

    
class MHAttention(nn.Module):
    def __init__(self, dim: int, heads: int, use_rope: bool):
        super().__init__()
        
        self.heads = heads
        self.dim = dim
        self.depth = dim//heads
        self.norm = math.sqrt(self.depth)
        
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.O = nn.Linear(dim, dim)

        if use_rope:
            self.rope = RoPE(self.depth)
        else:
            self.rope = FakeRoPE()
        
        for l in [self.Q, self.K, self.V, self.O]:
            torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
            torch.nn.init.zeros_(l.bias)
    
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

        q, k = self.rope(q, k)
            
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

class FakeMHAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, mask):
        x = x.permute(0, 2, 1, 3).contiguous() # return dims back
        return x
    
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
    def __init__(self, dim: int, heads: int, do: float,
                 norm_layer: nn.Module, skip_att: bool):
        super().__init__()

        self.SeqAttBlock = MHAttentionBlock(dim, heads, do,
                                            norm_layer, use_rope=True)
        self.MSAAttBlock = MHAttentionBlock(dim, heads, do,
                                            norm_layer, use_rope=False) \
                            if not skip_att else FakeMHAttention()
        
        self.ff_norm_layer = norm_layer(dim)
        self.FF = FFSwiglu(dim, do)

    def forward(self, x, msa_mask, seq_mask):
        # x.shape: b(0), msa(1), seq(2), dim(3)
        # seq-wise att
        x = self.SeqAttBlock(x, seq_mask)

        # msa-wise att
        x = self.MSAAttBlock(x, msa_mask)

        # feed-forward
        ff = self.ff_norm_layer(x)
        ff = self.FF(ff)
        x = x + ff

        return x

class MHAttentionBlock(nn.Module):
    def __init__(self, dim, heads, do, norm_layer, use_rope):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = MHAttention(dim, heads, use_rope)
        self.drop = nn.Dropout(do)

    def forward(self, x, mask):
        att = self.norm(x)
        att, _ = self.attn(att, att, att, mask)
        att = self.drop(att)
        x = x + att
        x = x.permute(0, 2, 1, 3).contiguous()

        return x
        
class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, do: float,
                 norm_layer: nn.Module, skip_att: bool):
        # skip_att is a placeholder
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.Att = MHAttention(dim, heads, False)
        self.drop = nn.Dropout(do)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.FF = FFSwiglu(dim, do)


    def forward(self, x, mask):

        att = self.norm1(x)
        att, scores = self.Att(att, att, att, mask)
        att = self.drop(att)
        x = x + att

        ff = self.norm2(x)
        ff = self.FF(ff)
        x = x + ff

        return x, scores