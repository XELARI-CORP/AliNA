from typing import List
import pickle
from dataclasses import dataclass
from collections.abc import Callable
import torch
import naskit as nsk



@dataclass
class AlinaDataPoint:
    seq: torch.Tensor
    inp_struct: torch.Tensor
    out_struct: torch.Tensor
    len: int

    def __len__(self):
        return self.seq.size(1) # seq dim

    def to(self, device: str | torch.device):
        device = torch.device(device)
        return AlinaDataPoint(seq=self.seq.to(device), 
                              inp_struct=self.inp_struct.to(device),
                              out_struct=self.out_struct.to(device), 
                              len=self.len)
    

    def compress(self):
        return AlinaDataPoint(seq=self.seq.to(torch.int8), 
                              inp_struct=self.inp_struct.to(torch.int16),
                              out_struct=self.out_struct.to(torch.int16), 
                              len=self.len)
    
    def decompress(self):
        return AlinaDataPoint(seq=self.seq.to(torch.int32), 
                              inp_struct=self.inp_struct.to(torch.int32),
                              out_struct=self.out_struct.to(torch.int32), 
                              len=self.len)


@dataclass
class AlinaBatch:
    seq: torch.Tensor
    inp_struct: torch.Tensor
    out_struct: torch.Tensor
    lens: List[int]

    def __len__(self):
        return self.seq.size(0)
    
    def to(self, device: str | torch.device):
        device = torch.device(device)
        return AlinaBatch(seq=self.seq.to(device),
                          inp_struct=self.inp_struct.to(device),
                          out_struct=self.out_struct.to(device),
                          lens=self.lens)
    


class BasePreprocessor:
    def __call__(self, na: nsk.NucleicAcid) -> torch.Tensor:
        inp_struct = torch.full((len(na),), -1, dtype=torch.int32)
        for i, j in na.pairs:
            inp_struct[i] = j
            inp_struct[j] = i

        return inp_struct
        
class EmptyPreprocessor:
    def __call__(self, na: nsk.NucleicAcid) -> None: 
        return torch.full((len(na),), -1, dtype=torch.int32)


class AlinaDataset:
    NT_MAP = {"A":1, "U":2, "G":3, "C":4, "N":5}
    
    def __init__(self,
                 nas: List[nsk.NucleicAcid],
                 cache: bool = True,
                 inp_struct_source: str | None = "struct",
                 out_struct_source: str | None = None
                 ):

        for na in nas:
            assert inp_struct_source in {None, "struct", *na.meta}
            assert out_struct_source in {None, "struct", *na.meta}
        
        self.nas = nas
        self.inp_preprocessor = EmptyPreprocessor() if (inp_struct_source is None) else BasePreprocessor()
        self.inp_struct_source = inp_struct_source
        self.out_preprocessor = EmptyPreprocessor() if (out_struct_source is None) else BasePreprocessor()
        self.out_struct_source = out_struct_source
        self.cache = cache
        self.X: List[AlinaDataPoint | None] = [None]*len(nas)
    
    def __len__(self):
        return len(self.nas)

    
    def __getitem__(self, n: int) -> AlinaDataPoint:
        x: AlinaDataPoint | None = self.X[n]
        if isinstance(x, AlinaDataPoint):
            return x.decompress()
        
        na: nsk.NucleicAcid = self.nas[n]

        if self.inp_struct_source not in (None, "struct"):
            ss = nsk.NA(na.meta[self.inp_struct_source]) # e.g., take "coev_struct" and create pairs via NA class
            inp_struct = self.inp_preprocessor(ss)
        else:
            inp_struct = self.inp_preprocessor(na)

        if self.out_struct_source not in (None, "struct"):
            ss = nsk.NA(na.meta[self.out_struct_source])
            out_struct = self.out_preprocessor(ss)
        else: 
            out_struct = self.out_preprocessor(na)

        if "msa" in na.meta.keys():
            msa = na.meta["msa"]
            msa_len = len(msa)
            if msa[0]!=na.seq:
                msa = [na.seq] + msa # prepend target seq to msa
                msa_len+=1
                
            maxl = max([len(seq) for seq in msa])
            seq_tensor = torch.zeros((msa_len, maxl), dtype=torch.int32)
            for i, seq in enumerate(msa):
                t = torch.IntTensor([self.NT_MAP[nt] for nt in seq])
                seq_tensor[i,:len(seq)] = t
        else: 
            maxl = len(na)
            seq_tensor =torch.zeros((1, maxl), dtype=torch.int32)
            seq_tensor[0] = torch.IntTensor([self.NT_MAP[nt] for nt in na.seq])
        
        dp: AlinaDataPoint = AlinaDataPoint(seq=seq_tensor,
                                            inp_struct=inp_struct,
                                            out_struct=out_struct,
                                            len=maxl)
        if self.cache:
            self.X[n] = dp.compress()
        
        return dp

    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({"nas":self.nas, "X":self.X}, f)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        ds = cls(data["nas"])
        ds.X = data["X"]
        return ds


    def __add__(self, other):
        self.nas += other.nas
        self.X += other.X
        return self
        

def collate_fn(dps: List[AlinaDataPoint]) -> AlinaBatch:
    
    N: int = len(dps) # batch
    max_seq_len: int = max([dp.len for dp in dps])
    max_msa_len: int = max([dp.seq.size(0) for dp in dps])
    
    seq = torch.zeros((N, max_msa_len, max_seq_len), dtype=torch.int32)
    inp_struct = torch.zeros((N, max_seq_len), dtype=torch.int32)
    out_struct = torch.zeros((N, max_seq_len), dtype=torch.int32)
    lens = []
    
    for i, dp in enumerate(dps):
        msa_len, seq_len = dp.seq.shape
        seq[i,:msa_len,:seq_len] = dp.seq
        inp_struct[i, :seq_len] = dp.inp_struct
        out_struct[i, :seq_len]   = dp.out_struct
        lens.append(dp.len)
    
    return AlinaBatch(
        seq=seq,
        inp_struct=inp_struct,
        out_struct=out_struct,
        lens=lens
    )









