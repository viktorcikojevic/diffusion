from diffusion.custom_modules.models.base_model import BaseDenoiser, DenoiserOutput
from diffusion.custom_modules.models import layers
from diffusion.custom_modules.models.unet import UNetModel
from diffusion.environments.constants import PRETRAINED_DIR
import segmentation_models_pytorch as smp
from typing import Union, Optional
from pathlib import Path
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F



class WrappedUNetModel(BaseDenoiser):
    def __init__(self, version: str, **kw):
        BaseDenoiser.__init__(self)
        self.version = kw.pop('version', 'unet')
        self.model = UNetModel(**kw)
        
    
    def get_name(self) -> str:
        return f"WrappedUNetModel"

    
    def predict(self, img: torch.Tensor, timesteps: torch.Tensor) -> DenoiserOutput:
        
        model_out = self.model(img, timesteps)
        
        out = DenoiserOutput(
            pred=model_out
        )
        
        return out
