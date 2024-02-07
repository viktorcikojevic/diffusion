from sennet.custom_modules.models.unet3d import model as unet_model
from sennet.custom_modules.models.base_model import BaseDenoiser, SegmentorOutput
from sennet.custom_modules.models import medical_net_resnet3d as resnet3ds
from sennet.custom_modules.models import layers
from sennet.environments.constants import PRETRAINED_DIR
# from super_image import EdsrModel, EdsrConfig
import segmentation_models_pytorch as smp
from line_profiler_pycharm import profile
from typing import Union, Optional
from pathlib import Path
import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from collections import OrderedDict
import torch.nn.functional as F
from sennet.custom_modules.models.unetr import UNETR




class SMPModelUpsampleBy2(BaseDenoiser):
    def __init__(self, version: str, **kw):
        BaseDenoiser.__init__(self)
        self.version = version
        if 'freeze_bn_layers' in kw:
            freeze_bn_layers = kw.pop('freeze_bn_layers') if kw['freeze_bn_layers'] is not None else False
        else: 
            freeze_bn_layers = False
        self.freeze_bn_layers = freeze_bn_layers 
        self.kw = kw
        self.upsampler = layers.PixelShuffleUpsample(in_channels=1, upscale_factor=2)
        constructor = getattr(smp, self.version)
        self.model = constructor(**kw)
        self.downscale_layer = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        
        if self.freeze_bn_layers:
            self.freeze_bn(self.model)

    def freeze_bn(self, module):
        for child_name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                child.eval()
            else:
                self.freeze_bn(child)

    def get_name(self) -> str:
        return f"SMP_{self.version}_{self.kw['encoder_name']}_{self.kw['encoder_weights']}"

    @profile
    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        assert img.shape[1] == 1, f"{self.__class__.__name__} works in 1 channel images only (for now), expected to have c=1, got {img.shape=}"
        B, _, C, H, W = img.shape
        img = img.reshape(B*C, 1, H, W)
        img_upsampled = self.upsampler(img) # (b*c, 1, h, w) -> (b*c, 1, h*2, w*2)
        img_upsampled = img_upsampled.reshape(B, C, H*2, W*2)
        model_out = self.model(img_upsampled) # (b, c, h*2, w*2) -> (b, c, h*2, w*2)
        model_out = model_out.reshape(B*C, 1, H*2, W*2)
        model_out = self.downscale_layer(model_out) # (b*c, 1, h*2, w*2) -> (b*c, 1, h, w)
        model_out = model_out.reshape(B, C, H, W)
        return SegmentorOutput(
            pred=model_out,
            take_indices_start=0,
            take_indices_end=C,
        )
