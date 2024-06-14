import pytorch_lightning as pl
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.nn as nn
import torch.optim

from diffusion.core.noise_scheduler import DDPMScheduler

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
        with torch.inference_mode():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class DenoisingTask(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer_spec: dict[str, any],
            noise_scheduler: DDPMScheduler,
            ema_momentum: float | None = None,
            scheduler_specs: dict[str, any] = None,
            accumulate_grad_batches: int = 1,
            val_check_epochs: int = 1,
            noise_key: str= "noise",
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
        self.scheduler_specs = scheduler_specs
        self.noise_scheduler = noise_scheduler
        
        self.train_loader = train_loader
        self.accumulate_grad_batches = accumulate_grad_batches
        self.val_check_epochs = val_check_epochs
            
        self.total_val_loss = 0.
        self.val_count = 0.
        self.best_val_loss = 9999.
        
        self.noise_key = noise_key
        
        self.loss_fn = torch.nn.MSELoss(reduction="mean")


    def calculate_mean_loss(self, noise_pred: torch.Tensor, batch: dict) -> torch.Tensor:
        
        # calculate loss
        noise = batch[self.noise_key]
        loss = self.loss_fn(noise_pred, noise)
        return loss
    
    def training_step(self, batch: dict, batch_idx: int):
        
        self.model = self.model.train()
        batch = self.noise_scheduler.sample_forward_process(batch)
        noise_pred = self.model(batch)
        loss = self.calculate_mean_loss(noise_pred, batch)
        loss_to_log = loss.detach().cpu().item()
        
        current_lr = self.optimizers().optimizer.param_groups[0]['lr']
        self.log_dict({
            "train_loss": loss_to_log,
            "lr": current_lr,
        }, prog_bar=True)
        return loss

    def backward(self, loss: torch.Tensor, *args: any, **kwargs: any) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)
        return pl.LightningModule.backward(self, loss, *args, **kwargs)

    def _get_eval_model(self):
        if self.ema_model is not None:
            model = self.ema_model.module
        else:
            model = self.model.eval()
        return model

    def validation_step(self, batch: dict, batch_idx: int):
        
        with torch.inference_mode():
            model = self._get_eval_model()
            batch = self.noise_scheduler.sample_forward_process(batch)
            noise_pred = model(batch)
            loss = self.calculate_mean_loss(noise_pred, batch)
            
            self.total_val_loss += loss.detach().cpu().item()
            self.val_count += 1
            

        
    def on_validation_epoch_end(self) -> None:
        with torch.inference_mode():
            val_loss = self.total_val_loss / (self.val_count + 1e-6)
        
            print("val_loss:")
            print(f"{val_loss = }")
            print("--------------------------------")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            self.total_val_loss = 0.
            self.val_count = 0.
            self.log_dict({
                "val_loss": val_loss,
            })

    def configure_optimizers(self):
        optimiser = self.optimizer_spec(self.model.parameters())
        print(f"{optimiser=}")
        ret_val = {
            "optimizer": optimiser,
        }
        if self.scheduler_specs is not None:
            schedulers = [ss(optimiser) for ss in self.scheduler_specs["schedulers"]]
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=optimiser,
                schedulers=schedulers,
                milestones=self.scheduler_specs["milestones"],
            )
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",
            }
            print(f"{scheduler=}")
            ret_val["lr_scheduler"] = scheduler_dict
        else:
            print("no scheduler")
        return ret_val
