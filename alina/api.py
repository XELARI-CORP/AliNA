import warnings
import sys
warnings.simplefilter('always', UserWarning)
from typing import Union, Optional, Tuple
from pathlib import Path

if sys.version_info>=(3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources

import numpy as np
import torch

from .utils import seq2matrix, pad_bounds, quantize_matrix, matrix2struct, SequenceError
from .model import Alina, pretrained_model_parameters



class AliNA:
    
    PRETRAINED_WEIGHTS = 'Pretrained_augmented.pth'
    
    def __init__(self,
                skip_error_data : bool = False,
                warn : bool = True,
                gpu : bool = False,
                device: Optional[str] = None,
                weights_path : Union[str, Path, None] = None,
                model_parameters : Union[dict, None] = None
                ):
        
        self.skip_error_data = skip_error_data
        self.warn = warn
        
        self.device = 'cpu'
        if gpu:
            if torch.cuda.is_available():
                self.device = device if device is not None else 'cuda'
            else:
                warnings.warn('Cuda is not available, AliNA will run on cpu!')
        
        self.model = self.load_model(weights_path, model_parameters)
        
        
    def fold(self, 
             seq : Union[str, list],
             threshold : float = 0.5,
             with_probs : bool = False
            ):
        
        if isinstance(seq, list):
            for s in seq:
                if not isinstance(s, str):
                    raise TypeError(f'Input list must contain only strings, got {type(s)}')
                    
        elif not isinstance(seq, str):
            raise TypeError(f'Input data must be string or list of strings, got {type(seq)}')

        if threshold>1 or threshold<0:
            raise ValueError(f'Threshold value must be in the range [0, 1], got {threshold}')
        
        single_sequence = False
        if isinstance(seq, str):
            single_sequence = True
            seq = [seq]
            
        results = []
        for n, s in enumerate(seq):
            try:
                batch = self.prepare_data(s)
            except Exception as e: 
                if self.skip_error_data:
                    if self.warn: warnings.warn(str(e))
                    results.append(None)
                    continue
                else:
                    raise e
                    
            struct, probs = self.predict(batch, threshold)
            if with_probs:
                results.append((struct, probs))
            else:
                results.append(struct)
            
        if single_sequence:
            return results[0]
        return results
        
        
    @staticmethod
    def prepare_data(seq: str):
        if not isinstance(seq, str):
            raise SequenceError(f'Sequence must be a string, got {type(seq)}')

        seq = seq.upper()
        if len(seq)>256 or len(seq)==0:
            raise SequenceError(f'Sequence length must be in range (0, 256], got {len(seq)}')

        rn = set(seq) - {'A', 'U', 'G', 'C'}
        if len(rn)!=0:
            raise SequenceError(f'Sequence contains unknown symbols: {tuple(rn)}')
        
        l, r = pad_bounds(seq)
        padded_seq = ''.join(['N'*l, seq, 'N'*r])
        batch = seq2matrix(padded_seq)
    
        return batch, l, r
    
    
    def predict(self, 
                batch: Tuple[np.ndarray, int, int], 
                threshold: float = 0.5):
        
        batch, lpad, rpad = batch
        batch = torch.tensor(batch, dtype=torch.int32, device=self.device)[None, ...]
            
        with torch.no_grad():
            pred = self.model(batch).view((256, 256))
        
        if self.device!='cpu':
            pred = pred.to('cpu')
        pred = pred.numpy()
        rpad = None if rpad==0 else -rpad
        pred = pred[lpad:rpad, lpad:rpad]
        
        M = quantize_matrix(pred, threshold = threshold)
        struct = matrix2struct(M)
        
        return struct, pred
        
        
    def load_model(self, weights_path, parameters):
        if parameters is None:
            model_parameters = pretrained_model_parameters
        else:
            model_parameters = parameters
            
        model = Alina(**model_parameters)
        
        if weights_path is None:
            model_pkg = pkg_resources.files("alina.model")
            weights_path = model_pkg.joinpath(self.PRETRAINED_WEIGHTS)
        
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state['model_state_dict'])
        
        if self.device!='cpu':
            model = model.to(self.device)
        model.eval()

        return model
        
        
        
        
        