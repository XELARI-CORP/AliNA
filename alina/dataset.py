from typing import List
import pickle
from dataclasses import dataclass
from collections.abc import Callable
import torch
import naskit as nsk
import math



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
    

@dataclass
class ReactDataPoint:
    seq: torch.Tensor          # (L,) tokens
    reactivity: torch.Tensor   # (L,) float, NaN = not measured
    len: int

    def __len__(self):
        return self.seq.size(0)

    def to(self, device: str | torch.device):
        device = torch.device(device)
        return ReactDataPoint(seq=self.seq.to(device),
                              reactivity=self.reactivity.to(device),
                              len=self.len)

    def compress(self):
        return ReactDataPoint(seq=self.seq.to(torch.int8),
                              reactivity=self.reactivity.to(torch.float32),
                              len=self.len)

    def decompress(self):
        return ReactDataPoint(seq=self.seq.to(torch.int32),
                              reactivity=self.reactivity.to(torch.float32),
                              len=self.len)


@dataclass
class ReactBatch:
    seq: torch.Tensor          # (b, L) tokens, 0 = pad
    reactivity: torch.Tensor   # (b, L) float, NaN = not measured or pad
    lens: List[int]

    def __len__(self):
        return self.seq.size(0)

    def to(self, device: str | torch.device):
        device = torch.device(device)
        return ReactBatch(seq=self.seq.to(device),
                          reactivity=self.reactivity.to(device),
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
    NT_MAP = {"A":1, "U":2, "G":3, "C":4, "N":5, "T":2} # T is mapped to the U token
    
    def __init__(self,
                 nas: List[nsk.NucleicAcid],
                 msa_sample_fraction: float = 1.0,
                 msa_sample_max_size: int = 100,
                 cache: bool = True,
                 inp_struct_source: str | None = "struct",
                 out_struct_source: str | None = None,
                 valid: bool = False,
                 task: str = "s",
                 reactivity_source: str = "reactivity"
                 ):
        """
        task: "s" - structural tasks (AlinaDataPoint / AlinaBatch): seq, MSA, input/target structures
              "r" - reactivity (ReactDataPoint / ReactBatch): seq and reactivity (target) are used
        """
        assert task in {"s", "r"}, f"unknown task: expected 's' or 'r', got {task}"

        for na in nas:
            if task == "r":
                assert reactivity_source in na.meta, f"{na.name}: no '{reactivity_source}' in meta"
            else:
                assert inp_struct_source in {None, "struct", *na.meta}
                assert out_struct_source in {None, "struct", *na.meta}
        
        self.nas = nas
        self.msa_sample_fraction = msa_sample_fraction
        self.msa_sample_max_size = msa_sample_max_size

        self._valid = valid
        self.task = task
        self.reactivity_source = reactivity_source
        
        self.inp_preprocessor = EmptyPreprocessor() if (inp_struct_source is None) else BasePreprocessor()
        self.inp_struct_source = inp_struct_source
        self.out_preprocessor = EmptyPreprocessor() if (out_struct_source is None) else BasePreprocessor()
        self.out_struct_source = out_struct_source
        self.cache = cache
        self.X: List[AlinaDataPoint | ReactDataPoint | None] = [None]*len(nas)
    
    def __len__(self):
        return len(self.nas)

        
    def _make_react_dp(self, na: nsk.NucleicAcid) -> ReactDataPoint:
        seq_len = len(na.seq)
        seq = torch.IntTensor([self.NT_MAP[nt] for nt in na.seq])

        react = na.meta[self.reactivity_source]
        if isinstance(react, str):
            react = [float(v) for v in react.strip().strip("[]").split(",")]
        react = torch.tensor(react, dtype=torch.float32)

        if react.numel() != seq_len:
            raise ValueError(f"{na.name}: reactivity length {react.numel()} != sequence length {seq_len}")

        return ReactDataPoint(seq=seq, reactivity=react, len=seq_len)

    def _make_struct_dp(self, na: nsk.NucleicAcid) -> AlinaDataPoint:
        
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

        seqs_list, seqs_set = [na.seq], {na.seq}
        seq_len = len(na.seq)
        
        if "msa" in na.meta.keys():
            for seq in na.meta["msa"]:
                if (seq not in seqs_set) and (len(seq)==seq_len):
                    #assert len(seq)==seq_len, f"Sequence length mismatch: {len(seq)} != {seq_len}"
                    seqs_set.add(seq)
                    seqs_list.append(seq)
                    
            # assert len(seqs_list)<=1, "empty or invalid MSA"
            
        msa_len = len(seqs_list)
        seq_tensor = torch.zeros((msa_len, seq_len), dtype=torch.int32)
        
        for i, seq in enumerate(seqs_list):
            seq_tensor[i,:] = torch.IntTensor([self.NT_MAP[nt] for nt in seq])
            
        return AlinaDataPoint(seq=seq_tensor,
                              inp_struct=inp_struct,
                              out_struct=out_struct,
                              len=seq_len)

    def __getitem__(self, n: int) -> AlinaDataPoint | ReactDataPoint:
        
        x: AlinaDataPoint | ReactDataPoint | None = self.X[n]
        if x is not None:
            dp = x.decompress()
        else:

            if self.task == "s":
                dp = self._make_struct_dp(self.nas[n])
            else: # self.task == "r"
                dp = self._make_react_dp(self.nas[n])

            if self.cache:
                self.X[n] = dp.compress()

        # MSA is sampled on every access (cached or not); the cache stores the full MSA
        if self.task == "s":
            dp.seq = self.make_msa_sample(dp.seq)

        return dp
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({"nas":self.nas, "X":self.X,
                         "msa_sample_fraction":self.msa_sample_fraction,
                         "msa_sample_max_size":self.msa_sample_max_size,
                         "valid": self._valid,
                         "task": self.task,
                         "reactivity_source": self.reactivity_source},
                        f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        ds = cls(nas=data["nas"],
                 msa_sample_fraction=data["msa_sample_fraction"],
                 msa_sample_max_size=data["msa_sample_max_size"],
                 valid=data["valid"],
                 task=data.get("task", "s"),
                 reactivity_source=data.get("reactivity_source", "reactivity"))
        
        ds.X = data["X"]
        return ds


    def __add__(self, other):
        assert self.task == other.task, f"cannot merge datasets with different tasks: {self.task}, {other.task}"
        assert self.task == "s" or self.reactivity_source == other.reactivity_source, \
            "cannot merge datasets with different reactivity sources"
        self.nas += other.nas
        self.X += other.X
        return self

    def make_msa_sample(self, msa: torch.Tensor):
        
        n = msa.size(0)
        device = msa.device
        
        k = int(math.ceil(n * self.msa_sample_fraction))
        k = min(k, self.msa_sample_max_size)
        k = max(1, k)

        
        if self._valid:
            idxs = torch.arange(k, device=device)
        else:
            if torch.rand(1).item() < 0.5:
                first_idx = torch.tensor([0], device=device)
                other_idxs = torch.randperm(n-1, device=device)[:k-1] + 1 
                idxs = torch.cat([first_idx, other_idxs])
            else:
                idxs = torch.tensor([0], device=device)

        return msa[idxs,:]
        

def collate_react(dps: List[ReactDataPoint]) -> ReactBatch:
    N: int = len(dps)
    max_seq_len: int = max([dp.len for dp in dps])

    seq = torch.zeros((N, max_seq_len), dtype=torch.int32)
    reactivity = torch.full((N, max_seq_len), float("nan"), dtype=torch.float32)
    lens = []

    for i, dp in enumerate(dps):
        seq[i, :dp.len] = dp.seq
        reactivity[i, :dp.len] = dp.reactivity
        lens.append(dp.len)

    return ReactBatch(seq=seq, reactivity=reactivity, lens=lens)

def collate_struct(dps: List[AlinaDataPoint]) -> AlinaBatch:

    N: int = len(dps) # batch
    max_seq_len: int = max([dp.len for dp in dps])
    max_msa_len: int = max([dp.seq.size(0) for dp in dps])
    
    seq = torch.zeros((N, max_msa_len, max_seq_len), dtype=torch.int32)
    inp_struct = torch.full((N, max_seq_len), -1, dtype=torch.int32)
    out_struct = torch.full((N, max_seq_len), -1, dtype=torch.int32)
    lens = []
    
    for i, dp in enumerate(dps):
        msa_len, seq_len = dp.seq.shape
        seq[i,:msa_len,:seq_len] = dp.seq
        inp_struct[i, :seq_len]  = dp.inp_struct
        out_struct[i, :seq_len]  = dp.out_struct
        lens.append(dp.len)
    
    return AlinaBatch(
        seq=seq,
        inp_struct=inp_struct,
        out_struct=out_struct,
        lens=lens
    )



def collate_fn(dps: List[AlinaDataPoint | ReactDataPoint]) -> AlinaBatch | ReactBatch:

    assert all(isinstance(dp, ReactDataPoint) for dp in dps) or \
           all(isinstance(dp, AlinaDataPoint) for dp in dps), "mixed data point types in a batch"
    
    if isinstance(dps[0], ReactDataPoint):
        return collate_react(dps)
    else:
        return collate_struct(dps)








