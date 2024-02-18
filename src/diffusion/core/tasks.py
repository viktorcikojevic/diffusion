from diffusion.core.noise_scheduler import DDPMScheduler
from diffusion.environments.constants import PROCESSED_DATA_DIR, TMP_SUB_MMAP_DIR
from diffusion.custom_modules.models import BaseDenoiser
from diffusion.core.fid_evaluation import FIDEvaluation
import pytorch_lightning as pl
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.nn as nn
import torch.optim
import json
import numpy as np
from tempfile import TemporaryDirectory
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm 
from einops import rearrange, repeat
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.fid_score import calculate_frechet_distance

class EMA(nn.Module):
    def __init__(self, model, momentum=0.00001):
        # https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/429060
        # https://github.com/Lightning-AI/pytorch-lightning/issues/10914
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.momentum = momentum
        self.decay = 1 - self.momentum

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class DenoisingTask(pl.LightningModule):
    def __init__(
            self,
            model: BaseDenoiser,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer_spec: Dict[str, Any],
            experiment_name: str,
            noise_scheduler: DDPMScheduler,
            ema_momentum: float | None = None,
            scheduler_spec: dict[str, Any] = None,
            accumulate_grad_batches: int = 1,
            val_check_epochs: int = 1,
            val_n_imgs: int = 1000,
            **kwargs
    ):
        pl.LightningModule.__init__(self)
        print(f"unused kwargs: {kwargs}")
        self.model = model
        self.ema_momentum = ema_momentum
        if self.ema_momentum is not None:
            print(f"{ema_momentum=} is given, evaluations will be done using ema")
            self.ema_model = EMA(self.model, self.ema_momentum)
        else:
            print(f"{ema_momentum=} not given, evaluations will be done using the model")
            self.ema_model = None
        self.val_loader = val_loader
        self.optimizer_spec = optimizer_spec
        self.scheduler_spec = scheduler_spec
        self.noise_scheduler = noise_scheduler
        self.experiment_name = experiment_name
        
        self.train_loader = train_loader
        self.accumulate_grad_batches = accumulate_grad_batches
        self.val_check_epochs = val_check_epochs
        self.val_n_imgs = val_n_imgs
            
        self.total_val_loss = 0.
        self.val_count = 0.
        self.best_val_loss = 9999.
        self.height, self.width = train_loader.dataset.dataset.img_width_height
        self.batch_size = train_loader.batch_size
        
        inception_block_idx = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(self.device)
        
        
    def generate_noise(self, img: torch.Tensor, time: int) -> torch.Tensor:
        # generate label noise
        return img + time
        
    def calculate_loss(self, noise_pred: torch.Tensor, noise: torch.Tensor, time: int, weights: torch.Tensor) -> torch.Tensor:
        
        # calculate loss
        loss_per_sample = torch.nn.L1Loss(reduction="none")(noise_pred, noise)
        while len(weights.shape) < len(loss_per_sample.shape):
            weights = weights[..., None]
        loss = loss_per_sample * weights / weights.mean()
        loss = loss.mean()
        
        return loss
    
    def training_step(self, batch: Dict, batch_idx: int):
        
        self.model = self.model.train()
        
        # generate label noise
        img = batch["img"]
        img_noised, noise, timesteps, weights = self.noise_scheduler.generate_denoising_data(img)
        
        # forward pass
        noise_pred = self.model.predict(img_noised, timesteps).pred

        loss = self.calculate_loss(noise_pred, noise, timesteps, weights)
        
        current_lr = self.optimizers().optimizer.param_groups[0]['lr']
        self.log_dict({
            "train_loss": loss,
            "lr": current_lr,
        }, prog_bar=True)
        return loss

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)
        return pl.LightningModule.backward(self, loss, *args, **kwargs)

    def _get_eval_model(self):
        if self.ema_model is not None:
            model = self.ema_model.module
        else:
            model = self.model.eval()
        return model

    def validation_step(self, batch: Dict, batch_idx: int):
        
        with torch.no_grad():
            model = self._get_eval_model()
            
            # generate label noise
            img = batch["img"]
            img_noised, noise, timesteps, weights = self.noise_scheduler.generate_denoising_data(img)
            
            # forward pass
            noise_pred = model.predict(img_noised, timesteps).pred
            loss = self.calculate_loss(noise_pred, noise, timesteps, weights)
            
            self.total_val_loss += loss.cpu().item()
            self.val_count += 1
            
    @torch.no_grad()
    def generate_single_batch(self):
        
        diffusion_steps = self.noise_scheduler.num_diffusion_timesteps
        diffusion_steps_vals = np.arange(diffusion_steps)[::-1]
        model = self._get_eval_model()
        # get current device
        device = self.device
        img = torch.randn(self.batch_size, 3, self.width, self.height).to(device)
        
        for step in diffusion_steps_vals:
            timestep = torch.Tensor([step]).to(device)
            z_img = torch.randn(self.batch_size, 3, self.width, self.height).to(device)
            if step == 0:
                sigma_t = 0.
            else:
                sigma_t = self.noise_scheduler.sigma_t[step]
            alpha_t = self.noise_scheduler.alpha_t[step]
            alpha_t_bar = self.noise_scheduler.alpha_t_bar[step]
            
            with torch.no_grad():
                model_out = model.predict(img, timestep).pred
        
            img = (1/np.sqrt(alpha_t)) * (img - (1-alpha_t)/np.sqrt(1-alpha_t_bar) * model_out) + sigma_t * z_img

        return img
    
    @torch.no_grad()
    def generate_imgs(self):
        
        imgs = []
        n_iters = self.val_n_imgs // self.batch_size
        for _ in tqdm(range(n_iters), total=n_iters, desc="generating images for validation"):
            img = self.generate_single_batch()
            imgs.append(img)
        
        imgs = torch.stack(imgs, dim=0).view(-1, 3, self.width, self.height)
        
        return imgs
    
    def calculate_inception_features(self, samples):
        
        self.inception_v3.eval()
        self.inception_v3 = self.inception_v3.to(self.device)
        
        features = self.inception_v3(samples.unsqueeze(0).float())[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features
    
    def calculate_fid_score(self):
        
        real_images = []
        collecting = True

        for batch in self.train_loader:
            if not collecting:
                break
            for image in batch["img"]:
                real_images.append(image)
                if len(real_images) >= self.val_n_imgs:
                    collecting = False
                    break
        
        fake_images = self.generate_imgs().to(self.device)
        real_images = torch.stack(real_images).to(self.device)
        
        # dataset images are from -1 to 1. Convert to 0-1
        fake_images = (fake_images + 1) / 2
        real_images = (real_images + 1) / 2
        
        # clip to 0-1
        fake_images = torch.clamp(fake_images, 0, 1)
        real_images = torch.clamp(real_images, 0, 1)
        
        fid = FrechetInceptionDistance(normalize=True).to(self.device)
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)
        fid_score = float(fid.compute())
        
        
        return fid_score
        
        
    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            val_loss = self.total_val_loss / (self.val_count + 1e-6)
            
            # get epoch number
            epoch = self.current_epoch
            if epoch % self.val_check_epochs == 0:
                fid_score = self.calculate_fid_score()
            else:
                fid_score = None
            
            print("val_loss:")
            print(f"{val_loss = }")
            print(f"{fid_score = }")
            print("--------------------------------")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            self.total_val_loss = 0.
            self.val_count = 0.
            if fid_score is not None:
                self.log_dict({
                    "val_loss": val_loss,
                    "fid_score": fid_score,
                })
            else:
                self.log_dict({
                    "val_loss": val_loss,
                })

    def configure_optimizers(self):
        if self.optimizer_spec["kwargs"]["lr"] is None:
            self.optimizer_spec["kwargs"]["lr"] = 10 ** self.optimizer_spec["log_lr"]
        optimizer_class = getattr(torch.optim, self.optimizer_spec["type"])
        optimizer = optimizer_class(self.model.parameters(), **self.optimizer_spec["kwargs"])
        print(f"{optimizer = }")
        ret_val = {
            "optimizer": optimizer,
        }
        if self.scheduler_spec is not None and "type" in self.scheduler_spec:
            scheduler_kwargs = self.scheduler_spec["kwargs"]
            if "override_total_steps" in self.scheduler_spec:
                key = self.scheduler_spec["override_total_steps"]["key"]
                num_epochs = self.scheduler_spec["override_total_steps"]["num_epochs"]
                train_loader = self.train_loader
                scheduler_kwargs[key] = int(num_epochs * len(train_loader) / self.accumulate_grad_batches) + 1
                print(f"scheduler override_total_steps given as {num_epochs}, now set to {scheduler_kwargs[key]}")
            scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler_spec["type"])
            print(f"{scheduler_kwargs = }")
            scheduler = scheduler_class(
                optimizer=optimizer,
                **scheduler_kwargs,
            )
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",
            }
            print(f"{scheduler = }")
            ret_val["lr_scheduler"] = scheduler_dict
        else:
            print("no scheduler")
        return ret_val

