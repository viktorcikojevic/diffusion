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
            download=True
    ) -> None:
        
        super().__init__(root=root, download=download)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        item = self.data[i]
        img_pil = Image.fromarray(item[0])
        
        # convert to numpy array
        img_np = np.array(img_pil)
        
        out = dict(
            image=img_np,
            label=item[1]
        )
        
        return out
        
        