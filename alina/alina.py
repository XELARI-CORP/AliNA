import warnings
import sys
import importlib.resources
from typing import Union, Optional, Tuple, Iterable
from pathlib import Path

import tqdm
import numpy as np
import torch
import naskit as nsk

from .dataset import AlinaDataset, make_collate
from .model import Model, pretrained_model_parameters



def get_package_path():
    if sys.version_info.minor>=9:
        path = importlib.resources.files('alina')
    else:
        path = next(importlib.resources.path("alina", "").gen)
        
    return path


class SequenceError(ValueError):
    pass

    
class AliNA(Model):
    
    def __init__(self,
                 model_parameters: dict,
                 dimer_embeddings: bool,
                 center_pad: bool
                ):
        super().__init__(**model_parameters)
        self.__model_params = model_parameters
        self.__dimer_embeddings = dimer_embeddings
        self.__center_pad = center_pad
        self.__device = torch.device("cpu")
    
    @property
    def model_params(self):
        return self.__model_params
    
    @property
    def dimer_embeddings(self):
        return self.__dimer_embeddings
    
    @property
    def center_pad(self):
        return self.__center_pad

    @property
    def device(self):
        return self.__device

    @property
    def state(self):
        return {
            "model_state_dict":self.state_dict(),
            "model_params":self.__model_params,
            "dimer_embeddings":self.__dimer_embeddings,
            "center_pad":self.__center_pad
        }


    @torch.compiler.disable(recursive=True)
    def to(self, device: Union[str, torch.device]):
        device = torch.device(device)
        super().to(device)
        self.__device = device
        return self

        
    @classmethod
    @torch.compiler.disable(recursive=True)
    def load(cls, *,
             model: Optional[str] = None,
             path: Optional[Union[str, Path]] = None
            ):

        if path is None:
            package_path = get_package_path()
            if model == "pretrained_augmented":
                path = package_path/"model"/"Pretrained_augmented.pth"
            else:
                raise ValueError(f"Unknown model name {model}. Choose from: ['pretrained_augmented']")
        
        state = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(
            model_parameters = state["model_params"],
            dimer_embeddings = state["dimer_embeddings"],
            center_pad = state["center_pad"]
        )
        model.load_state_dict(state["model_state_dict"])
        return model


    @torch.compiler.disable(recursive=False)
    def _prepare_data(self, nas):
        for na in nas:
            if len(na)>256 or len(na)==0:
                raise SequenceError(f'Sequence length must be in range (0, 256], got {len(na)}')

            seq = na if isinstance(na, str) else na.seq
            rn = set(seq) - {'A', 'U', 'G', 'C'}
            if len(rn)!=0:
                raise SequenceError(f'Sequence contains unknown symbols: {tuple(rn)}')

        data = [nsk.NA(na) for na in nas]
        return data
    

    @torch.compiler.disable(recursive=False)
    def fold(self, 
             data: Union[str, nsk.NucleicAcid, Iterable[Union[str, nsk.NucleicAcid]]],
             threshold: float = 0.5,
             with_probs: bool = False,
             batch_size: int = 8,
             verbose: bool = False
            ):
        
        if threshold>1 or threshold<0:
            raise ValueError(f'Threshold value must be in the range [0, 1], got {threshold}')

        if not isinstance(data, (list, tuple)):
            data = [data]

        data = self._prepare_data(data)
        dataset = AlinaDataset(data, self.dimer_embeddings, with_adjacency=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, drop_last=False,
                                             collate_fn=make_collate(256, center_pad = self.center_pad))
            
        preds, ls, sls = [], [], []
        self.eval()
        iter_wrapper = tqdm.tqdm if verbose else iter
        with torch.no_grad():
            for x, _, l, sl in iter_wrapper(loader):
                x = x.to(self.device)
                pred = self(x)
                preds.append(pred.cpu())
                ls += l
                sls += sl

        preds = preds[0] if len(preds)==0 else torch.cat(preds, dim=0)
        preds = [p[l:(l+sl), l:(l+sl)] for p, l, sl in zip(preds, ls, sls)]
        adjs = [self.quantize_matrix(p, threshold).numpy() for p in preds]
        nas = [nsk.NucleicAcid.from_adjacency(adj, seq=na.seq, name=na.name, meta=na.meta) for adj, na in zip(adjs, data)]

        if len(nas)==1:
            out = (nas[0], preds[0]) if with_probs else nas[0]
            return out

        out = (nas, preds) if with_probs else nas
        return out

    
    @torch.compiler.disable(recursive=True)
    def quantize_matrix(self, M, threshold = 0.5):
        seq_length = M.shape[-1]
        diag = 1 - torch.diag(torch.ones(seq_length))
        fm = torch.zeros((seq_length, seq_length), dtype=torch.int32)
        
        thmask = M>threshold
        s = M*diag*thmask
    
        while torch.sum(s)>0:
            m = int(s.argmax())
            r = m//seq_length
            c = m%seq_length
    
            fm[r,c] = 1
            s[r] = 0
            s[c] = 0
            s[:, r] = 0
            s[:, c] = 0
    
        x = (fm + fm.T)
        return x
        
        
        
        
        
        