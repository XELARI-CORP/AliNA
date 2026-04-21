import os
from pathlib import Path
import torch
import numpy as np
import pynvml
pynvml.nvmlInit()
print(f"Driver Version: {pynvml.nvmlSystemGetDriverVersion()}; N devices: {pynvml.nvmlDeviceGetCount()}")



def Loss(pred_m, coev_struct, true_struct):
    # pred_m.shape     : b, seq, seq+1
    # coev_struct.shape: b, seq
    # true_struct.shape: b, seq
    batch, seq, seq1p = pred_m.shape
    true_adj = torch.nn.functional.one_hot((true_struct+1).long(), num_classes=seq1p)
    true_adj = true_adj.float().to(pred_m.device) # b,seq,seq+1

    mask = (coev_struct!=-1).float().to(pred_m.device) # b,seq

    logs = - true_adj*torch.log(pred_m + 1e-7)
    logs = logs.sum(dim=-1) # b,seq,seq+1 -> b,seq

    logs = logs*mask
    
    CE = logs.sum() / mask.sum()
    
    return CE

def AlinaMetrics(pred_m, y, inp, TH=0.5):
    """
    pred: (batch, seq, seq+1)  - probs
    y:    (batch, seq)         - ground truth (-1 for unbound/pad)
    inp:  (batch, seq)         - coev ss (-1 for unbound/pad)
    """
    # print((inp==y).int().sum())
    p = pred_m.view(-1, pred_m.size(2)) # b*seq, seq+1
    y = (y.view(-1)).long() #-1,...
    inp = inp.view(-1).long()
    
    mask = (inp != -1) # b*seq, mask pad and unbound, eg positive mask
    
    maxr = torch.max(p, dim=-1) # b*seq
    vals = maxr.values
    pred = maxr.indices - 1 # -1,...,

    # eq: correct bound with high confidence 
    eq = (pred==y) & (vals>=TH)
    # tp: mask AND real bound AND correct bounds
    tp = mask & (y!=-1) & eq
    tp = tp.float().sum()
    # метрики для связей, которые нужно поменять / оставить
    # т.е. связь есть в обоих вторичках
    recall = tp / ( ((y!=-1) & mask ).float().sum() + 1e-7 )
    prec   = tp / ( ((pred!=-1) & mask ).float().sum() + 1e-7 )
    # метрики относительно коэволюционной вторички
    tp_over_inp = mask & (inp==y)
    preserved_tp_ratio = (tp_over_inp & eq).float().sum() / ( tp_over_inp.float().sum() + 1e-7 )

    fp_over_inp = mask & (inp!=y)
    fixed_fp_ratio = ( fp_over_inp & (vals>=TH) & ((pred==y) | (pred==-1)) ).float().sum() / ( fp_over_inp.float().sum() + 1e-7 )
    # метрики для связей, которые нужно удалить
    # remove accuracy - accuracy of correctly removed bonds
    fmask = mask & (y==-1)  # fake bonds mask, present in inp AND not in target
    racc = (eq & fmask).float().sum() / ( fmask.float().sum() + 1e-7 )
    
    recall, prec, racc = float(recall), float(prec), float(racc)
    preserved_tp_ratio = float(preserved_tp_ratio)
    fixed_fp_ratio     = float(fixed_fp_ratio)
    
    Fscore = 2*preserved_tp_ratio*fixed_fp_ratio / (preserved_tp_ratio + fixed_fp_ratio + 1e-7)

    return recall, prec, preserved_tp_ratio, fixed_fp_ratio, Fscore

def get_cexplr_scheduler(warmup=200, peak=5e-4, c=4e-4, 
                         min_lr=5.e-5, max_lr=1.e-3,
                         rules=[]
                        ):
    
    def lr_scheduler(n):
        for sth, v in rules[::-1]:
            if n>=sth:
                return v

        if n<warmup:
            lr = ((peak)/warmup)*n
        else:
            ampl = peak - min_lr
            step = (n-warmup)
            step = step * (0.25 + 0.75*np.exp(-step))
            lr = (ampl*0.5)*np.cos(c*step) + (ampl*0.5 + min_lr)
        
        if lr>max_lr: lr=max_lr
        if n>warmup and (step*c >= np.pi): lr=min_lr

        return lr
    return lr_scheduler

class GpuWatch:
    def __init__(self, idx: int):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

    def __call__(self):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return int(mem_info.used / (1024**2)), util_info.gpu

class Checkpointer:
    def __init__(self, dir_path, model, optim):
        self.dir_path = Path(dir_path)
        self.model = model
        self.optim = optim
        self.best_value = -1 * float("inf")
        self.best_model_name = None
    
    def __call__(self, name):
        if not self.dir_path.is_dir():
            os.mkdir(self.dir_path)

        state = self.model.state
        state["optim_state_dict"] = self.optim.state_dict()
        torch.save(state, self.dir_path/f"{name}.pth")
        
    def save_by_metric(self, name, value):
        if value <= self.best_value:
            return
            
        best_name = f"best_val={value:.4f}_{name}"
        if self.best_model_name is not None:
            os.remove(self.dir_path/f"{self.best_model_name}.pth")
        self.best_value = value
        self.best_model_name = best_name
        self(self.best_model_name)


