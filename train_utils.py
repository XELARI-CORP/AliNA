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

from alina.utils import AlinaMetrics, Loss, get_cexplr_scheduler, Checkpointer, GpuWatch
from alina.dataset import AlinaDataset, collate_fn

### FUNCTIONS ###
def setup_model_optimizer(
    model_class: type[Module],
    config: dict[str, Any],
    checkpoint: Path | str | None = None,
    device: torch.device | str = 'cpu',
    compile_model: bool = False):

    weight_decay = config["const"]["WEIGHT_DECAY"]
    
    if checkpoint:
        logger.info(f"Load checkpoint: {str(checkpoint)}")
        # load model
        model = model_class.load(checkpoint).to(device)

        logger.info("Update optimizer state")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        optim = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=1)
        optim.load_state_dict(state["optim_state_dict"])

        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
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
    valid_data: Dataset,
    optim: Optimizer):
    ### unpack config:
    lr_params    = config['lr_params']
    batch_size   = config['const']['BATCH_SIZE']
    model_name   = config['model_name']
    
    work_dir = Path(work_dir)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, drop_last=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=batch_size,
                              shuffle=False, drop_last=False,
                              collate_fn=collate_fn)
                                  
    lr_func   =  get_cexplr_scheduler(**lr_params)
        
    ### setup train modules 
    modules = dict()
    modules['optim'] = optim
    modules['scaler'] = torch.amp.GradScaler("cuda")
    modules['log_fn'] = mlflow.log_metric
    modules['loss_fn'] = Loss
    modules['train_loader'] = train_loader
    modules['valid_loader'] = valid_loader
    modules["checkpointer"] = Checkpointer(work_dir / model_name, model, optim)
    modules['lr_scheduler'] = LambdaLR(optim, lr_lambda=lr_func)

    return modules 


def validate(model: Module,
             loader: DataLoader,
             loss_fn: Any,
             verbose: bool = False):
    
    device = model.device
    iterf = tqdm.tqdm if verbose else iter
    loss, recall, prec, tp_ratio, fp_ratio, Fscore = 0, 0, 0, 0, 0, 0
    c = 0
    model.eval()
    with torch.no_grad():
        for b in iterf(loader):
            b = b.to(device)
            _, pred = model(b.seq, b.inp_struct)
            pred = pred.cpu()
            y = b.out_struct.cpu()
            inp = b.inp_struct.cpu()

            loss += loss_fn(pred, inp, y)
            _recall, _prec, _tp_ratio, _fp_ratio, _Fscore = AlinaMetrics(pred, y, inp)
            
            recall += _recall
            prec += _prec
            tp_ratio += _tp_ratio
            fp_ratio += _fp_ratio
            Fscore += _Fscore
            c+=1
    model.train()
    
    return loss/c, recall/c, prec/c, tp_ratio/c, fp_ratio/c, Fscore/c

@logger.catch
def train(model: Module,
          modules: dict[str, Any],
          train_const:  dict[str, Any],
          device: torch.device | str = 'cpu',
          verbose: bool | None = False):
    # unpack training modules
    optim        = modules['optim']
    scaler       = modules['scaler']
    log_fn       = modules['log_fn']
    loss_fn      = modules['loss_fn']
    lr_scheduler = modules['lr_scheduler']
    train_loader = modules['train_loader']
    valid_loader = modules['valid_loader']
    checkpointer = modules["checkpointer"]
    # unpack training parameters
    MAX_TRAIN_STEPS = train_const['MAX_TRAIN_STEPS']
    VALID_EVERY     = train_const['VALID_EVERY']
    LOG_EVERY       = train_const['LOG_EVERY']
    CLIP_GRAD       = train_const['CLIP_GRAD']
    DEVICE_IDX      = train_const['DEVICE_IDX']
    BATCH_SIZE      = train_const['BATCH_SIZE']
    grad_acum       = train_const['GRAD_ACUM']
    
    # counters
    global_step = 0
    train_step = 0
    ep = 0
    
    step_start_time = time.time()
    iter_start_time = time.time()
    
    gw = GpuWatch(DEVICE_IDX)
    # prevalidation
    vloss, _, _, _, _, Fscore = validate(model, valid_loader, loss_fn, False)
    logger.info(f"Prevalidation: loss={vloss:.2f}, Fscore={Fscore:.2f} ")
    mloss, recall, precision, tp_ratio, fp_ratio, Fscore = 0, 0, 0, 0, 0, 0
    # main train loop: 1 iteration = 1 epoach 
    while (train_step<MAX_TRAIN_STEPS):
        ep+=1
        for i, b in enumerate(train_loader):
            global_step += 1
            b = b.to(device)

            seq, inp, y = b.seq, b.inp_struct, b.out_struct
            
            #with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, pred_m = model(seq, inp)

            loss = loss_fn(pred_m, inp, y)
            _recall, _precision, _tp_ratio, _fp_ratio, _Fscore = AlinaMetrics(pred_m, y, inp)
                
            mloss += float(loss)
            recall += _recall
            precision += _precision
            tp_ratio += _tp_ratio
            fp_ratio += _fp_ratio
            Fscore += _Fscore
                    
            scaler.scale(loss).backward() #calculate gradient

            # training step
            if (global_step + 1)%grad_acum==0:
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
                    
                    log_fn('Loss', mloss, train_step)
                    log_fn('recall', recall / grad_acum, train_step)
                    log_fn('precision', precision / grad_acum, train_step)
                    log_fn('preserved_tp_ratio', tp_ratio / grad_acum, train_step)
                    log_fn('fixed_fp_ratio', fp_ratio / grad_acum, train_step)
                    log_fn('Fscore', Fscore / grad_acum, train_step)
                    log_fn('Lr', lr_scheduler.get_last_lr()[0], train_step)
                    log_fn('step/s', stepps, train_step)

                    mem, util = gw()
                    log_fn('GPUMem', mem, train_step)
                    log_fn('GPUUtil', util, train_step)
                    
                if train_step%VALID_EVERY==0:
                    print(f"\rvalidation ...", end='')
                    vloss, recall, precision, tp_ratio, fp_ratio, Fscore = validate(model, valid_loader, loss_fn, False)
                    log_fn('valid_Loss', float(vloss), train_step)
                    log_fn('valid_recall', recall, train_step)
                    log_fn('valid_precision', precision, train_step)
                    log_fn('valid_preserved_tp_ratio', tp_ratio, train_step)
                    log_fn('valid_fixed_fp_ratio', fp_ratio, train_step)
                    log_fn('valid_Fscore', Fscore, train_step)
                    
                    checkpointer.save_by_metric(f"step={train_step}", Fscore)
                    
                mloss, recall, precision, tp_ratio, fp_ratio, Fscore = 0, 0, 0, 0, 0, 0

            itps = 1/(time.time() - iter_start_time)
            iter_start_time = time.time()
            if verbose:
                msg = (f"\rEp: {ep} | {i+1} / {len(train_loader)} | {itps:.2f} it/s | Loss: {loss:.6f}")
                print(msg, end='')
                
            if train_step>=MAX_TRAIN_STEPS:
                break
                
    checkpointer(f"finished")
    print()