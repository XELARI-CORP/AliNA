import sys
sys.path.append('/usr/local/lib/python3.12/dist-packages')
sys.path.append('/home/pavel/repos/AliNA')

# import json
# import fire
import torch
import random
import mlflow
import numpy as np
from pathlib import Path
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf

from alina import AliNA
from alina.dataset import AlinaDataset
import warnings

from train_utils import (
    clean_string,
    setup_model_optimizer,
    setup_train_modules,
    train )

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

logger.remove()
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
logger.add(sys.stderr, format=fmt)

@hydra.main(version_base=None, config_path="configs", config_name="config")
@logger.catch
def main(cfg: DictConfig):
    """
    Main entry point for training
    """
    config = OmegaConf.to_container(cfg, resolve=True)
    # 1. Setup paths
    train_data_path = config["data"]["train_data_path"]
    valid_data_path = config["data"]["valid_data_path"].split(":")
    checkpoint_path = config["checkpoint_path"]
    
    work_dir        = Path(config["work_dir"])
    
    work_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Start Training ===")
    model_name = config["model_name"]
    model_name = clean_string(model_name)
    config["model_name"] = model_name
    
    config["hparams"]["conv_activation"] = torch.nn.SiLU
    config["hparams"]["norm_layer"]      = torch.nn.LayerNorm

    # Determine device (using the IDX from your config if available)
    #train_const = config.get('const', {})
    device_idx = config.get('const', {}).get('DEVICE_IDX', 0) #!!!
    device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Use device: {device}")

    # 3. Load Datasets
    logger.info("Initialize datasets...")
    train_dataset = AlinaDataset.load(train_data_path)
    
    valid_data = {}
    for label_path in valid_data_path:
        label, path = label_path.split("|")
        ds = AlinaDataset.load(path)
        valid_data[label] = ds
        
    logger.success(f"Loaded train dataset ({len(train_dataset)}) and "
                   f"{len(valid_data)} valid dataset(s) ({", ".join([str(len(ds)) for ds in valid_data.values()])})")

    # 4. Setup Model & Optimizer
    model, optim = setup_model_optimizer(
        model_class=AliNA, 
        config=config,
        checkpoint=checkpoint_path,
        device=device,
        compile_model=config.get("compile_model", False)
    )
    
    # 5. Setup Train Modules (DataLoaders, Schedulers, Loss)
    logger.info("Configurate training modules and DataLoaders")
    modules = setup_train_modules(
        model=model,
        config=config,
        work_dir=work_dir,
        train_data=train_dataset,
        valid_data=valid_data,
        optim=optim
    )

    # 6. Run Training
    mlflow.set_tracking_uri(uri="http://127.0.0.1:31420") 
    mlflow.set_experiment(config.get("experiment_name", "AliNA"))
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(config)

    checkpointer = modules['checkpointer']
    try:
        logger.info("Start train loop")
        train(
            model=model,
            modules=modules,
            config=config,
            device=device,
            verbose=True
        )
        print()
        logger.success("Training finished successfully")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
        checkpointer("stopped")
        
    except Exception as e:
        logger.error(f"Training failed due to error: {e}")
        checkpointer("error")
        raise e
        
    finally:
        mlflow.end_run()
        logger.info("MLflow session closed")

if __name__ == '__main__':
    main()