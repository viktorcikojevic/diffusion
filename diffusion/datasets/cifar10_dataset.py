from pathlib import Path
import numpy as np
import json
from torchvision.datasets import CIFAR10
from PIL import Image

from diffusion.envs.environments import DATA_DIR

class CIFAR10DiffusionDataset(CIFAR10):
    
    def __init__(
            self,    
            root=DATA_DIR,
            img_key: str = "img",
            download=True,
            cfg_fold: dict = None,
            is_val: bool = False,
    ) -> None:
        
        super().__init__(root=root, download=download)
        self.img_key = img_key
        self.cfg_fold = cfg_fold
        self.is_val = is_val
        
        
        shuffle_seed = cfg_fold["shuffle_seed"]
        train_pct = cfg_fold["train_pct"]
        
        # shuffle the dataset according to the seed, and take only first train_pct if not val
        np.random.seed(shuffle_seed)
        idxs = np.random.permutation(len(self.data))
        if is_val:
            idxs = idxs[int(train_pct * len(idxs)):]
        else:
            idxs = idxs[:int(train_pct * len(idxs))]
            
        self.data = self.data[idxs]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        img_np = self.data[i]
        img_np = np.transpose(img_np, axes=(2, 0, 1))
        
        # normalize from 0-255 to -1 to 1
        img_np = img_np / 127.5 - 1
        
        out = {
            self.img_key: img_np.astype(np.float32),
        }
        
        return out