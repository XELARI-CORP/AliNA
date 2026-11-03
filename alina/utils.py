import os
from pathlib import Path
import torch
import numpy as np
import pynvml
pynvml.nvmlInit()
print(f"Driver Version: {pynvml.nvmlSystemGetDriverVersion()}; N devices: {pynvml.nvmlDeviceGetCount()}")



def MaskedCELoss(pred_m, coev_struct, true_struct):
    '''
    Summary: для классификации TP/FP пар из коев. вторички
    Args:
        pred_m.shape     : b, seq, seq+1
        coev_struct.shape: b, seq
        true_struct.shape: b, seq
    '''
    batch, seq, seq1p = pred_m.shape
    true_adj = torch.nn.functional.one_hot((true_struct+1).long(), num_classes=seq1p)
    true_adj = true_adj.float().to(pred_m.device) # b,seq,seq+1

    mask = (coev_struct!=-1).float().to(pred_m.device) # b,seq

    logs = - true_adj*torch.log(pred_m + 1e-7)
    logs = logs.sum(dim=-1) # b,seq,seq+1 -> b,seq

    logs = logs*mask
    
    CE = logs.sum() / mask.sum()
    
    return CE

def WeightedMaskedCELoss(pred_m, coev_struct, true_struct, pos_weight=2.0):
    '''
    Summary: как MaskedCELoss, но с усиленным штрафом за истинные пары (true_struct != -1),
    чтобы модель не убирала их
    Args:
        pred_m.shape     : b, seq, seq+1
        coev_struct.shape: b, seq
        true_struct.shape: b, seq
        pos_weight        : множитель штрафа для позиций с true_struct != -1
    '''
    batch, seq, seq1p = pred_m.shape
    true_adj = torch.nn.functional.one_hot((true_struct+1).long(), num_classes=seq1p)
    true_adj = true_adj.float().to(pred_m.device) # b,seq,seq+1

    mask = (coev_struct!=-1).float().to(pred_m.device) # b,seq

    logs = - true_adj*torch.log(pred_m + 1e-7)
    logs = logs.sum(dim=-1) # b,seq,seq+1 -> b,seq

    weights = torch.where(true_struct!=-1, pos_weight, 1.0).float().to(pred_m.device) # b,seq
    logs = logs*mask*weights

    CE = logs.sum() / mask.sum()

    return CE

def CELoss(pred_m, coev_struct, true_struct):
    '''
    Summary: для обычного предсказания вторички
    Args:
        pred_m.shape     : b, seq, seq+1
        coev_struct.shape: b, seq
        true_struct.shape: b, seq
    '''
    batch, seq, seq1p = pred_m.shape
    true_adj = torch.nn.functional.one_hot((true_struct+1).long(), num_classes=seq1p)
    true_adj = true_adj.float().to(pred_m.device) # b,seq,seq+1

    logs = - true_adj*torch.log(pred_m + 1e-7)
    logs = logs.sum(dim=-1) # b,seq,seq+1 -> b,seq
    
    CE = logs.mean()
    
    return CE

def MaskingSoftBCELoss(logits, y):
    '''
    Summary: BCE with soft labels for reactivity; positions with NaN in y (not measured / pad) are masked
    Args:
        logits.shape: b, seq (raw model output, before sigmoid)
        y.shape     : b, seq (reactivity in [0, 1], NaN = masked)
    '''
    y = y.to(logits.device)
    valid = ~torch.isnan(y) # b,seq
    y = torch.nan_to_num(y, nan=0.0).float() # NaN * 0 is still NaN, so replace it before the product

    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
    loss = loss * valid

    return loss.sum() / valid.sum().clamp(min=1)

def MaskingMSELoss(pred, y):
    '''
    Summary: MSE for (unbounded) reactivity regression; positions with NaN in y (not measured / pad) are masked
    Args:
        pred.shape: b, seq (raw model output, i.e. forward(..., raw=True), NO sigmoid, NOT clipped)
        y.shape   : b, seq (reactivity, any range, NaN = masked)
    '''
    y = y.to(pred.device)
    valid = ~torch.isnan(y) # b,seq
    y = torch.nan_to_num(y, nan=0.0).float() # NaN * 0 is still NaN, so replace it before the product

    loss = (pred.float() - y) ** 2
    loss = loss * valid

    return loss.sum() / valid.sum().clamp(min=1)

def MaskingMAELoss(pred, y):
    '''
    Summary: MAE (L1) for (unbounded) reactivity regression; positions with NaN in y (not measured / pad) are masked
    Args:
        pred.shape: b, seq (raw model output, i.e. forward(..., raw=True), NOT clipped)
        y.shape   : b, seq (reactivity, any range, NaN = masked)
    '''
    y = y.to(pred.device)
    valid = ~torch.isnan(y) # b,seq
    y = torch.nan_to_num(y, nan=0.0).float() # NaN * 0 is still NaN, so replace it before the product

    loss = (pred.float() - y).abs()
    loss = loss * valid

    return loss.sum() / valid.sum().clamp(min=1)

class LossFn:
    '''
    Summary: унифицированная обёртка над функциями потерь, чтобы train loop
    всегда вызывал loss_fn(pred_m, coev_struct, true_struct) независимо от
    гиперпараметров конкретной функции (напр. pos_weight у WeightedMaskedCELoss);
    для реактивности вызывается как loss_fn(raw_output, reactivity)
    '''
    _REGISTRY = {
        "CELoss": CELoss,
        "MaskedCELoss": MaskedCELoss,
        "WeightedMaskedCELoss": WeightedMaskedCELoss,
        "MaskingSoftBCELoss": MaskingSoftBCELoss,
        "MaskingMSELoss": MaskingMSELoss,
        "MaskingMAELoss": MaskingMAELoss,
    }

    def __init__(self, name: str, **params):
        if name not in self._REGISTRY:
            raise ValueError(f"Unknown loss_fn_type: {name}, expected one of {list(self._REGISTRY)}")
        self.name = name
        self.fn = self._REGISTRY[name]
        self.params = params

    def __call__(self, *args):
        return self.fn(*args, **self.params)

def ClasMetrics(pred_m, inp, y, TH=0.5):
    """
    Summary: для классификации
    Args:
        pred: (batch, seq, seq+1)  - probs
        y:    (batch, seq)         - ground truth (-1 for unbound/pad)
        inp:  (batch, seq)         - coev ss (-1 for unbound/pad)
    """
    p = pred_m.view(-1, pred_m.size(2)) # b*seq, seq+1
    y = (y.view(-1)).long() # b*seq; -1,...
    inp = inp.view(-1).long() # b*seq
    
    mask = (inp != -1) # b*seq, mask pad and unbound, e.g. positive mask
    
    maxr = torch.max(p, dim=-1) # b*seq
    vals = maxr.values
    pred = maxr.indices - 1 # 0 > -1,...

    # eq: correct bound with high confidence 
    eq = (pred==y) & (vals>=TH)
    
    # метрики относительно коэволюционной вторички
    tp_over_inp = mask & (inp==y)
    preserved_tp_ratio = (tp_over_inp & eq).float().sum() / ( tp_over_inp.float().sum() + 1e-7 )

    fp_over_inp = mask & (inp!=y)
    fixed_fp_ratio = ( fp_over_inp & (vals>=TH) & ((pred==y) | (pred==-1)) ).float().sum() / ( fp_over_inp.float().sum() + 1e-7 )

    preserved_tp_ratio, fixed_fp_ratio = float(preserved_tp_ratio), float(fixed_fp_ratio)
    
    Fscore = 2*preserved_tp_ratio*fixed_fp_ratio / (preserved_tp_ratio + fixed_fp_ratio + 1e-7)

    metrics = {"preserved_tp_ratio" : preserved_tp_ratio,
               "fixed_fp_ratio" : fixed_fp_ratio,
               "Fscore" : Fscore}

    return metrics

def PredMetrics(pred_m, inp, y, TH=0.5):
    """
    Summary: для предсказания вторички
    Args:
        pred: (batch, seq, seq+1)  - probs
        y:    (batch, seq)         - ground truth (-1 for unbound/pad)
        inp:  (batch, seq)         - coev ss (-1 for unbound/pad)
    """
    p = pred_m.view(-1, pred_m.size(2)) # b*seq, seq+1
    y = (y.view(-1)).long() # b*seq; -1,...
    inp = inp.view(-1).long() # b*seq
    
    maxr = torch.max(p, dim=-1) # b*seq
    vals = maxr.values
    pred = maxr.indices - 1 # 0 > -1,...,

    # eq: matches with high confidence 
    eq = (pred==y) & (vals>=TH)
    # tp: real bound AND correct bounds
    tp = (y!=-1) & eq
    tp = tp.float().sum()
    
    recall = tp / ( (y!=-1).float().sum() + 1e-7 )
    prec   = tp / ( (pred!=-1).float().sum() + 1e-7 )

    # remove accuracy - accuracy of correctly removed bonds
    # fmask = (inp!=-1) & (y==-1)  # fake bonds mask, present in inp AND not in target
    # racc = (eq & fmask).float().sum() / ( fmask.float().sum() + 1e-7 )
    
    # recall, prec, racc = float(recall), float(prec), float(racc)
    recall, prec = float(recall), float(prec)
    
    Fscore = 2*prec*recall / (prec + recall + 1e-7)

    metrics = {"recall" : recall, "precision" : prec, "Fscore" : Fscore}

    return metrics

def _rankdata(x):
    """average ranks (1-based), ties get the mean of their ranks"""
    sorted_x, idx = torch.sort(x)
    _, inverse, counts = torch.unique_consecutive(sorted_x, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0)
    avg = (2*ends - counts + 1).to(torch.float64) / 2 # (start + end) / 2, start = end - count + 1
    ranks = torch.empty_like(x, dtype=torch.float64)
    ranks[idx] = avg[inverse]
    return ranks

def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = torch.sqrt((a*a).sum() * (b*b).sum())
    if den == 0:
        return float("nan") # constant input, correlation is undefined
    return float((a*b).sum() / den)

def ReactMetrics(pred, y):
    """
    Summary: для предсказания реактивности; считается по позициям, где y не NaN
    Args:
        pred: (batch, seq)  - probs in [0, 1]
        y:    (batch, seq)  - ground truth (NaN for not measured/pad)
    Returns: dict with MAE, RMSE, Pearson, Spearman (NaN if undefined)
    """
    y = y.to(pred.device)
    valid = ~torch.isnan(y)
    p, t = pred[valid].detach().to(torch.float64), y[valid].to(torch.float64)

    if p.numel() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"),
                "Pearson": float("nan"), "Spearman": float("nan")}

    err = p - t
    metrics = {"MAE" : float(err.abs().mean()),
               "RMSE": float(torch.sqrt((err**2).mean())),
               "Pearson" : float("nan"),
               "Spearman": float("nan")}

    if p.numel() > 1:
        metrics["Pearson"]  = _pearson(p, t)
        metrics["Spearman"] = _pearson(_rankdata(p), _rankdata(t))

    return metrics

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
    def __init__(self, dir_path, model, optim, mode="max"):
        assert mode in ("max", "min"), f"mode must be 'max' or 'min', got {mode}"
        self.dir_path = Path(dir_path)
        self.model = model
        self.optim = optim
        self.mode = mode
        self.best_value = -1 * float("inf") if mode == "max" else float("inf")
        self.best_model_name = None
    
    def __call__(self, name):
        if not self.dir_path.is_dir():
            os.mkdir(self.dir_path)

        state = self.model.state
        state["optim_state_dict"] = self.optim.state_dict()
        torch.save(state, self.dir_path/f"{name}.pth")
        
    def save_by_metric(self, name, value):
        if (self.mode == "max" and value <= self.best_value) or \
           (self.mode == "min" and value >= self.best_value):
            return
            
        best_name = f"best_val{value:.4f}_{name}"
        if self.best_model_name is not None:
            os.remove(self.dir_path/f"{self.best_model_name}.pth")
        self.best_value = value
        self.best_model_name = best_name
        self(self.best_model_name)


