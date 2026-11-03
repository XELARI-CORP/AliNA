from typing import List
import math
import torch
import torch.nn as nn

from .modules import EncoderLayer, MSATransformer, make_encoder_block


class Model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 dim: int,
                 conv_layers: int,
                 seq_encoders_order: int | List[int],
                 struct_encoders_order: int | List[int],
                 heads: int,
                 convdrop: float,
                 encdrop: float,
                 conv_activation: nn.Module,
                 norm_layer: nn.Module,
                 skip_msa_att: bool
                ):
        super().__init__()

        self.dim = dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, dim)

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

        # Sequence Transformer Block (MSA layers, used with a single-row "MSA")
        self.seq_encoders_block, self.seq_encoders_order = make_encoder_block(
            seq_encoders_order, dim, heads, encdrop,
            MSATransformer, norm_layer, skip_msa_att)

        # Encoder Block
        self.struct_encoders_block, self.struct_encoders_order = make_encoder_block(
            struct_encoders_order, dim, heads, encdrop,
            EncoderLayer, norm_layer)

        # Head: per-nucleotide reactivity logit
        self.final_norm = norm_layer(dim)
        self.head = nn.Linear(dim, 1)
        torch.nn.init.xavier_uniform_(self.head.weight, gain=1.0)
        torch.nn.init.constant_(self.head.bias, 0.0)

    @staticmethod
    def to_prediction(raw: torch.Tensor) -> torch.Tensor:
        """
        Raw head output -> reactivity prediction: reactivity is non-negative, so negative outputs are clipped to 0
        (no upper bound: reactivity is not bounded by 1). Not used in the loss: clamp has zero gradient below 0.
        """
        return raw.float().clamp(min=0.0)


    @torch.compile
    def pos_enc(self, seq: int, dim: int, start_idx: int, device: torch.device):
        div = torch.arange(dim, dtype=torch.float32, device=device)
        div = 10000**(2 * (div//2) / dim)

        pos_enc = torch.arange(start_idx, seq+start_idx, dtype=torch.float32, device=device) # 1,...,seq if start_idx=1
        pos_enc = pos_enc.view(seq, 1).repeat(1, dim) / div

        pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2]) # dim 2i
        pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2]) # dim 2i+1

        return pos_enc


    def forward(self, seq: torch.Tensor, raw: bool = False):
        """
        Args:
            seq: (b, L) int tokens, 0 = pad
            raw: return the raw (unclipped) head output, which is what the loss is computed on
        Returns:
            (b, L) reactivity prediction clipped to >= 0 (or the raw head output)
        """
        # assert seq.dim() == 2, f"expected seq of shape (b, L), got {tuple(seq.shape)}"
        batch, seq_len = seq.shape

        # pad positions are zeroed at every conv input, otherwise the k=3 convs leak
        # pad embeddings into the last real nucleotides and outputs depend on batch padding
        pad = (seq!=0).unsqueeze(1) # b, 1, seq
        x = self.embedding(seq).transpose(1,2) # b, dim, seq
        x = x * pad.to(x.dtype)
        for layer in self.conv_model:
            x = layer(x)
            x = x * pad.to(x.dtype)
        x = x.transpose(1,2) # b, seq, dim

        seq_mask = (seq==0).to(x.dtype).view(batch, 1, 1, seq_len) # b, seq -> b, 1, 1, seq

        # seq encoders work on (b, msa, seq, dim); here msa = 1
        x = x.unsqueeze(1) # b, 1, seq, dim
        msa_mask = torch.zeros((batch * seq_len, 1, 1, 1), dtype=x.dtype, device=x.device)
        for li in self.seq_encoders_order:
            l = self.seq_encoders_block[li]
            x = l(x, msa_mask, seq_mask)

        x = x[:,0,:,:]   # batch, 1, seq, dim -> batch, seq, dim
        x = x + self.pos_enc(x.size(1), x.size(2), start_idx=1, device=x.device).to(x.dtype)

        # encoders
        for li in self.struct_encoders_order:
            l = self.struct_encoders_block[li]
            x, _ = l(x, seq_mask) # b, seq, dim

        x = self.final_norm(x)
        out = self.head(x).squeeze(-1) # b, seq, dim -> b, seq

        if raw:
            return out
        return self.to_prediction(out)
