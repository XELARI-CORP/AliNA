### IMPORT PACKAGES ###
import os
import time
import tqdm
import mlflow
from loguru import logger

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.nn import Module
from pathlib import Path
from typing import Any

from alina.utils import ClasMetrics, PredMetrics, ReactMetrics, LossFn, get_cexplr_scheduler, Checkpointer, GpuWatch
from alina.dataset import AlinaDataset, collate_fn
from alina.model import Model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### FUNCTIONS ###
def setup_model_optimizer(
    model_class: type[Module],
    config: dict[str, Any],
    checkpoint: Path | str | None = None,
    device: torch.device | str = 'cpu'):

    weight_decay = config["const"]["WEIGHT_DECAY"]
    freeze_layers  = config["special_params"]["freeze_layers"] 

    compile_model = config["special_params"]["compile_model"]
    
    if checkpoint:
        logger.info(f"Load checkpoint: {str(checkpoint)}")
        # load model
        model = model_class.load(path=checkpoint).to(device)
        if freeze_layers:
            freezed_layers = []
            for param_name, param in model.named_parameters():
                if any(layer_name in param_name for layer_name in freeze_layers):
                    param.requires_grad = False
                    freezed_layers.append(param_name)
                else:
                    param.requires_grad = True
            logger.info(f"Freezed layers: {freezed_layers}")

        logger.info("Update optimizer state")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        optim = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=1)
        optim.load_state_dict(state["optim_state_dict"])

        # for state in optim.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)
                    
        for param_group in optim.param_groups:
            param_group['weight_decay'] = weight_decay
        
    else:
        logger.info("Setup fresh model and optimizer")
        hparams = config["hparams"]
        model = model_class(hparams).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=1, weight_decay=weight_decay)
                                  
    if compile_model:
        model = torch.compile(model)
        
    _, _ = model.sanity_check()
    logger.success("Sanity check is successful!")
                
    return model, optim
                                
def setup_train_modules(
    model: Module,
    config: dict[str, Any],
    work_dir: Path | str,
    train_data: Dataset,
    valid_data: Dataset | dict[Dataset],
    optim: Optimizer):
    ### unpack config:
    lr_params    = config['lr_params']
    batch_size   = config['const']['BATCH_SIZE']
    model_name   = config['run_name']
    model_task   = config['special_params']['model_task']
    loss_fn_type   = config['special_params']['loss_fn']
    loss_fn_params = config['special_params'].get('loss_fn_params', {})
    
    assert model_task in ["p","c","r"], f"unknown model task: expected 'p', 'c' or 'r', got {model_task}"
    if model_task in ["p","c"]:
        raise NotImplementedError(
            f"model_task='{model_task}' is unsupported on the `react` branch: the model takes a sequence only "
            "and predicts reactivity, use model_task='r'")

    datasets = {"train": train_data, **valid_data}
    for label, ds in datasets.items():
        ds_task = getattr(ds, "task", "s")
        assert ds_task == "r", f"model_task='r' needs datasets with task='r', but '{label}' has task='{ds_task}'"
    
    work_dir = Path(work_dir)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, drop_last=True,
                              collate_fn=collate_fn)
    valid_loaders = {}
    for label, ds in valid_data.items():
        vloader = DataLoader(ds, batch_size=batch_size,
                             shuffle=False, drop_last=False,
                             collate_fn=collate_fn)
        valid_loaders[label] = vloader
                                  
    lr_func   =  get_cexplr_scheduler(**lr_params)
        
    ### setup train modules 
    modules = dict()
    assert (
        (model_task=="c") and \
        (loss_fn_type in ["MaskedCELoss", "WeightedMaskedCELoss"])
        ) or (
            (model_task=="p") and (loss_fn_type in ["CELoss"])
        ) or (
            (model_task=="r") and (loss_fn_type in ["MaskingSoftBCELoss", "MaskingMSELoss", "MaskingMAELoss"])
        ), f"Invalid 'model_task'/'loss_fn_type' combination, got: {model_task}, {loss_fn_type}"

    loss_fn = LossFn(loss_fn_type, **loss_fn_params)

    if model_task == "c":
        metrics_fn = ClasMetrics
    elif model_task == "p":
        metrics_fn = PredMetrics
    elif model_task == "r":
        metrics_fn = ReactMetrics

    # torch.amp.GradScaler appeared in torch 2.3
    scaler = torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
        
    modules['optim']         = optim
    modules['scaler']        = scaler
    modules['log_fn']        = mlflow.log_metric
    modules['loss_fn']       = loss_fn
    modules['metrics_fn']    = metrics_fn
    modules['train_loader']  = train_loader
    # reactivity: the best model has the lowest MAE, structure tasks: the highest Fscore
    modules["checkpointer"]  = Checkpointer(work_dir / model_name, model, optim,
                                            mode="min" if model_task=="r" else "max")
    modules['lr_scheduler']  = LambdaLR(optim, lr_lambda=lr_func)
    modules['valid_loaders'] = valid_loaders

    return modules 


def react_prediction(out: torch.Tensor, loss_fn: Any) -> torch.Tensor:
    """
    Raw model output (raw=True) -> reactivity prediction used by the metrics:
    BCE treats the output as a logit (sigmoid -> [0, 1]), MSE/MAE regress the raw output, which is clipped
    to >= 0 exactly like the model does at inference (Model.to_prediction)
    """
    out = out.detach().float()
    return torch.sigmoid(out) if loss_fn.name == "MaskingSoftBCELoss" else Model.to_prediction(out)

def validate_react(model: Module,
                   loader: DataLoader,
                   loss_fn: Any,
                   metrics_fn: Any,
                   verbose: bool = False):
    """
    Loss is averaged over valid positions, metrics are computed once over all valid positions of the loader
    """
    device = model.device
    iterf = tqdm.tqdm if verbose else iter

    loss_sum, n_valid = 0, 0
    preds, ys = [], []

    model.eval()
    with torch.no_grad():
        for b in iterf(loader):
            b = b.to(device)
            raw_out = model(b.seq, raw=True)

            valid = ~torch.isnan(b.reactivity)
            loss_sum += loss_fn(raw_out, b.reactivity).item() * valid.sum().item()
            n_valid += valid.sum().item()

            preds.append(react_prediction(raw_out, loss_fn)[valid].cpu())
            ys.append(b.reactivity[valid].cpu())

    model.train()

    metrics = metrics_fn(torch.cat(preds), torch.cat(ys))
    loss = loss_sum / max(n_valid, 1)

    return loss, metrics

def validate_struct(
             model: Module,
             loader: DataLoader,
             loss_fn: Any,
             metrics_fn: Any,
             verbose: bool = False):
             #model_task: str = "c"):

    # if model_task == "r":
    #     return validate_react(model, loader, loss_fn, metrics_fn, verbose)
    
    device = model.device
    iterf = tqdm.tqdm if verbose else iter
    
    loss_sum, c = 0, 0
    metrics_sum = {}
    
    model.eval()
    with torch.no_grad():
        for b in iterf(loader):
            b = b.to(device)
            _, pred = model(b.seq, b.inp_struct)
            
            loss_batch = loss_fn(pred, b.inp_struct, b.out_struct)
            loss_sum += loss_batch.item()
            
            metrics_batch = metrics_fn(pred, b.inp_struct, b.out_struct)
            
            if c == 0:
                metrics_sum = {k:v for k, v in metrics_batch.items()}
            else:
                for k, v in metrics_batch.items():
                    metrics_sum[k] += v
                
            c+=1
            
    model.train()

    metrics = {k: v / c for k, v in metrics_sum.items()}
    loss = loss_sum / c
    
    return loss, metrics

@logger.catch(reraise=True)
def train(model: Module,
          modules: dict[str, Any],
          config:  dict[str, Any],
          device: torch.device | str = 'cpu',
          verbose: bool | None = False):

    train_const = config.get('const', {})
    # unpack training modules
    optim         = modules['optim']
    scaler        = modules['scaler']
    log_fn        = modules['log_fn']
    loss_fn       = modules['loss_fn']
    metrics_fn    = modules['metrics_fn'] 
    lr_scheduler  = modules['lr_scheduler']
    train_loader  = modules['train_loader']
    checkpointer  = modules["checkpointer"]
    valid_loaders = modules['valid_loaders']
    # unpack training parameters
    MAX_TRAIN_STEPS = train_const['MAX_TRAIN_STEPS']
    VALID_EVERY     = train_const['VALID_EVERY']
    LOG_EVERY       = train_const['LOG_EVERY']
    SAVE_EVERY      = train_const['SAVE_EVERY']
    CLIP_GRAD       = train_const['CLIP_GRAD']
    DEVICE_IDX      = train_const['DEVICE_IDX']
    BATCH_SIZE      = train_const['BATCH_SIZE']
    grad_acum       = train_const['GRAD_ACUM']


    model_task = config["special_params"]["model_task"]
    is_react = (model_task == "r")
    # metric of the best checkpoint (see Checkpointer mode in setup_train_modules)
    save_metric = "MAE" if is_react else "Fscore"
    validate_fn = validate_react if is_react else validate_struct

    save_by = config["special_params"]["save_by"]
    vds_keys = list(valid_loaders.keys())
    assert save_by in vds_keys, f"v.loaders: {vds_keys}, but got save_by = {save_by}"
    # counters
    global_step = 0
    train_step = 0
    ep = 0
    
    step_start_time = time.time()
    iter_start_time = time.time()
    
    gw = GpuWatch(DEVICE_IDX)
    # prevalidation
    for label, vloader in valid_loaders.items():
        vloss, vmetrics = validate_fn(model, vloader, loss_fn, metrics_fn, False)
        vmetrics_str = ", ".join(f"{k}={v:.2f}" for k, v in vmetrics.items())
        logger.info(f"{label} preval.: loss={vloss:.2f}, {vmetrics_str}")
    
    #mloss, recall, precision, tp_ratio, fp_ratio, Fscore = 0, 0, 0, 0, 0, 0
    mloss = 0
    metrics, metrics_cnt = {}, {} # NaN metrics (e.g., undefined correlation) are skipped, so counted separately
    # main train loop: 1 iteration = 1 epoach 
    while (train_step<MAX_TRAIN_STEPS):
        ep+=1
        for i, b in enumerate(train_loader):
            global_step += 1
            b = b.to(device)

            if is_react:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    raw_out = model(b.seq, raw=True)

                loss = loss_fn(raw_out, b.reactivity)
                metrics_batch = metrics_fn(react_prediction(raw_out, loss_fn), b.reactivity)
            else:
                seq, inp, y = b.seq, b.inp_struct, b.out_struct
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, pred_m = model(seq, inp)
        
                loss = loss_fn(pred_m, inp, y)
                metrics_batch = metrics_fn(pred_m, inp, y)

            mloss += loss.item()
            loss = loss / grad_acum
            
            for k, v in metrics_batch.items():
                if v == v: # skip NaN
                    metrics[k] = metrics.get(k, 0) + v
                    metrics_cnt[k] = metrics_cnt.get(k, 0) + 1
                    
            scaler.scale(loss).backward() # calculate gradient

            # training step
            if global_step % grad_acum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=CLIP_GRAD)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                lr_scheduler.step()

                stepps = 1/(time.time() - step_start_time)
                step_start_time = time.time()
                
                train_step += 1
                if train_step%LOG_EVERY==0:
                    
                    log_fn('Loss', mloss / grad_acum, train_step)

                    for k, v in metrics.items():
                        log_fn(k, v / metrics_cnt[k], train_step)
                        
                    log_fn('Lr', lr_scheduler.get_last_lr()[0], train_step)
                    log_fn('step/s', stepps, train_step)

                    mem, util = gw()
                    log_fn('GPUMem', mem, train_step)
                    log_fn('GPUUtil', util, train_step)
                    
                if train_step%VALID_EVERY==0:
                    print(f"\rvalidation ...", end='')
                    for label, vloader in valid_loaders.items():
                        
                        vloss, vmetrics = validate_fn(model, vloader, loss_fn, metrics_fn, False)
                        log_fn(f"{label}_valid_Loss", float(vloss), train_step)

                        for k, v in vmetrics.items():
                            if v == v: # skip NaN
                                log_fn(f"{label}_valid_{k}", v, train_step)
                            
                        if label == save_by and vmetrics[save_metric] == vmetrics[save_metric]:
                            checkpointer.save_by_metric(f"{label}_step{train_step}", vmetrics[save_metric])

                if SAVE_EVERY is not None and train_step%SAVE_EVERY==0:
                    checkpointer(f"step{train_step}")

                metrics, metrics_cnt = {}, {}
                mloss = 0

            itps = 1/(time.time() - iter_start_time)
            iter_start_time = time.time()
            if verbose:
                msg = (f"\rEp: {ep} | {i+1} / {len(train_loader)} | {itps:.2f} it/s | Loss: {loss:.6f}")
                print(msg, end='')
                
            if train_step>=MAX_TRAIN_STEPS:
                break
                
    checkpointer(f"finished")
    print()

def clean_string(s: str) -> str:
    mapper = str.maketrans({"[":"",
                            "]":"",
                            " ":"",
                            ",":""
                           })

    return s.translate(mapper)