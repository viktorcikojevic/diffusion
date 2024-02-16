from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

import diffusion.core.dataset as datasets


def build_train_val_dataloaders(
	cfg: DictConfig,
) -> (DataLoader, DataLoader):
    
    
    dataset_type = cfg["dataset"]["type"]
    dataset = getattr(datasets, dataset_type)()
    
    # train test split 
    train_pct = cfg["train_pct"]
    train_len = int(len(dataset) * train_pct)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    print(f"Train data count: {train_len}")
    print(f"Validation data count: {val_len}")
    
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["apparent_batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2*cfg["apparent_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader