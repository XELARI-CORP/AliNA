from typing import List
import pickle
from dataclasses import dataclass
import torch
import naskit as nsk



@dataclass
class AlinaDataPoint:
    seq: torch.Tensor
    adj: torch.Tensor
    len: int

    def __len__(self):
        return self.seq.size(0)

    def to(self, device: str | torch.device):
        device = torch.device(device)
        return AlinaDataPoint(seq=self.seq.to(device), adj=self.adj.to(device), len=self.len)


@dataclass
class AlinaBatch:
    seq: torch.Tensor
    adj: torch.Tensor
    lens: List[int]

    def __len__(self):
        return self.seq.size(0)
    
    def to(self, device: str | torch.device):
        device = torch.device(device)
        return AlinaBatch(seq=self.seq.to(device), adj=self.adj.to(device), lens=self.lens)
    

class AlinaDataset:
    NT_MAP = {"No_Bond": 1, "A":2, "U":3, "G":4, "C":5}

    def __init__(self, nas: List[nsk.NucleicAcid]):
        
        self.nas = nas
        self.X: List[AlinaDataPoint | None] = [None]*len(nas)

    
    def __len__(self):
        return len(self.nas)

    
    def __getitem__(self, n: int) -> AlinaDataPoint:
        x: AlinaDataPoint | None = self.X[n]
        if isinstance(x, AlinaDataPoint):
            return x
        
        na: nsk.NucleicAcid = self.nas[n]
        seq = torch.IntTensor([1] + [self.NT_MAP[nt] for nt in na.seq])
        
        adj = torch.zeros((len(na)+1, len(na)+1), dtype=torch.float32)
        for i in range(len(na)):
            b: int | None = na.complnb(i)
            if b is None:
                adj[i+1][0] = 1
                adj[0][i+1] = 1
            else:
                adj[i+1][b+1] = 1

        dp: AlinaDataPoint = AlinaDataPoint(seq=seq, adj=adj, len=len(na)+1)
        self.X[n] = dp
        
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
    
    X = torch.zeros((N, maxl), dtype=torch.int32)
    Y = torch.zeros((N, maxl, maxl), dtype=torch.float32)
    lens = []
    
    for i, dp in enumerate(dps):
        n = len(dp)
        X[i, :n] = dp.seq
        Y[i, :n, :n] = dp.adj
        Y[i, n:, 0] = 1.
        Y[i, 0, n:] = 1.
        lens.append(dp.len)
    
    return AlinaBatch(
        seq=X,
        adj=Y,
        lens=lens
    )









