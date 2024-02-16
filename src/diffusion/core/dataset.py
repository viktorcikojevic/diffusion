from pathlib import Path
import numpy as np
import json
from torchvision.datasets import CIFAR10
from diffusion.environments.environments import DATA_DIR
from PIL import Image

class CIFAR10DiffusionDataset(CIFAR10):
    
    def __init__(
            self,    
            root=DATA_DIR,
            download=True,
    ) -> None:
        
        super().__init__(root=root, download=download)
        self.img_width_height=(32,32)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        item = self.data[i]
        img_pil = Image.fromarray(item)
        
        # convert to numpy array
        img_np = np.array(img_pil) # ()
        
        # go from (h, w, 3) to (3, h, w)
        img_np = np.transpose(img_np, axes=(2, 0, 1))
        
        # normalize from 0-255 to -1 to 1
        img_np = img_np / 127.5 - 1
        
        out = dict(
            img=img_np,
        )
        
        return out
        
        
class CelebHQDataset(CIFAR10):
    
    def __init__(
            self,    
            root="/home/viktor/Documents/diffusion/data/celeb-hq",
    ) -> None:
        
        self.img_width_height=(256, 256)
        
        # list all the files
        files = list(Path(root).glob("*.jpg"))
        self.file_paths = [f"{root}/{f.name}" for f in files]
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, i):
        
        file_path = self.file_paths[i]
        # load jpg img to numpy array
        img_pil = Image.open(file_path)
        
        # convert to numpy array
        img_np = np.array(img_pil) 
        
        # go from (h, w, 3) to (3, h, w)
        img_np = np.transpose(img_np, axes=(2, 0, 1))
        
        # normalize from 0-255 to -1 to 1
        img_np = img_np / 127.5 - 1
        
        out = dict(
            img=img_np,
        )
        
        return out
        