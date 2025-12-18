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
        return self.seq.size(0)

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
    

class BaseInputPreprocessor:
    def __call__(self, na: nsk.NucleicAcid) -> torch.Tensor:
        inp_struct = torch.zeros(len(na)+1, dtype=torch.int32)
        for i, j in na.pairs:
            inp_struct[i+1] = j+1
            inp_struct[j+1] = i+1

        return inp_struct


class AlinaDataset:
    NT_MAP = {"No_Bond": 1, "A":2, "U":3, "G":4, "C":5}

    def __init__(self,
                 nas: List[nsk.NucleicAcid],
                 input_preprocessor: Callable[[nsk.NucleicAcid], torch.Tensor] = BaseInputPreprocessor(),
                 cache: bool = True
                 ):
        
        self.nas = nas
        self.input_preprocessor = input_preprocessor
        self.cache = cache
        self.X: List[AlinaDataPoint | None] = [None]*len(nas)

    
    def __len__(self):
        return len(self.nas)

    
    def __getitem__(self, n: int) -> AlinaDataPoint:
        x: AlinaDataPoint | None = self.X[n]
        if isinstance(x, AlinaDataPoint):
            return x.decompress()
        
        na: nsk.NucleicAcid = self.nas[n]
        seq = torch.IntTensor([1] + [self.NT_MAP[nt] for nt in na.seq])
        
        inp_struct = self.input_preprocessor(na)
        out_struct = torch.zeros(seq.size(0), dtype=torch.int32)
        for i, j in na.pairs:
            out_struct[i+1] = j+1
            out_struct[j+1] = i+1
        
        dp: AlinaDataPoint = AlinaDataPoint(seq=seq,
                                            inp_struct=inp_struct,
                                            out_struct=out_struct,
                                            len=len(na)+1)
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
    N: int = len(dps)
    maxl: int = max([len(dp) for dp in dps])
    
    seq = torch.zeros((N, maxl), dtype=torch.int32)
    inp_struct = torch.zeros((N, maxl), dtype=torch.int32)
    out_struct = torch.zeros((N, maxl), dtype=torch.int32)
    lens = []
    
    for i, dp in enumerate(dps):
        n = len(dp)
        seq[i, :n] = dp.seq
        inp_struct[i, :n] = dp.inp_struct
        out_struct[i, :n] = dp.out_struct
        lens.append(dp.len)
    
    return AlinaBatch(
        seq=seq,
        inp_struct=inp_struct,
        out_struct=out_struct,
        lens=lens
    )









