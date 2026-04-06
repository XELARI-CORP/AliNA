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
    

class BaseInpPreprocessor:
    def __call__(self, na: nsk.NucleicAcid) -> torch.Tensor:
        inp_struct = torch.zeros(len(na)+1, dtype=torch.int32)
        for i, j in na.pairs:
            inp_struct[i+1] = j+1
            inp_struct[j+1] = i+1

        return inp_struct

class BaseOutPreprocessor:
    def __call__(self, na: nsk.NucleicAcid) -> torch.Tensor:
        inp_struct = torch.zeros(len(na), dtype=torch.int32)
        for i, j in na.pairs:
            inp_struct[i] = j
            inp_struct[j] = i

        return inp_struct

class EmptyInpPreprocessor:
    def __call__(self, na: nsk.NucleicAcid) -> torch.Tensor:
        return torch.zeros(len(na)+1, dtype=torch.int32)
        
class EmptyOutPreprocessor:
    def __call__(self, na: nsk.NucleicAcid) -> None: 
        return torch.zeros(len(na), dtype=torch.int32)


class AlinaDataset:
    NT_MAP = {"A":2, "U":3, "G":4, "C":5, "N":6}
    MAX_MSA_LEN = 128
    
    def __init__(self,
                 nas: List[nsk.NucleicAcid],
                 cache: bool = True,
                 inp_struct_source: str | None = "struct",
                 out_struct_source: str | None = None
                 ):
        
        assert inp_struct_source in {None, "struct", *nas[0].meta}
        assert out_struct_source in {None, "struct", *nas[0].meta}

        self.nas = nas
        self.inp_preprocessor = EmptyInpPreprocessor() if (inp_struct_source is None) else BaseInpPreprocessor()
        self.inp_struct_source = inp_struct_source
        self.out_preprocessor = EmptyOutPreprocessor() if (out_struct_source is None) else BaseOutPreprocessor()
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
            maxl = max([len(seq) for seq in msa])
            seq_tensor = torch.zeros((self.MAX_MSA_LEN, maxl), dtype=torch.int32)
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
    MAX_MSA_LEN: int = 128
    N: int = len(dps)
    #maxl: int = max([dp.len for dp in dps])
    maxl = 256
    
    seq = torch.zeros((N, MAX_MSA_LEN, maxl), dtype=torch.int32)
    inp_struct = torch.zeros((N, maxl+1), dtype=torch.int32)
    out_struct = torch.zeros((N, maxl), dtype=torch.int32)
    lens = []
    
    for i, dp in enumerate(dps):
        n = len(dp)
        seq[i,:,:n] = dp.seq
        inp_struct[i, :n+1] = dp.inp_struct
        out_struct[i, :n]   = dp.out_struct
        lens.append(dp.len)
    
    return AlinaBatch(
        seq=seq,
        inp_struct=inp_struct,
        out_struct=out_struct,
        lens=lens
    )









