from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from diffusion.environments.constants import MODEL_OUT_DIR, PRETRAINED_DIR
from diffusion.core.tasks import DenoisingTask
from diffusion.core.dataset import CIFAR10DiffusionDataset
from diffusion.custom_modules.models.base_model import BaseDenoiser
from diffusion.core.noise_scheduler import DDPMScheduler
# from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader, TensorDataset
import diffusion.custom_modules.models as models
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict
import hydra
import torch
import json


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    
    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # ---------------------------------------
    model_class = getattr(models, cfg_dict["model"]["type"])
    model: BaseDenoiser = model_class(**cfg_dict["model"]["kwargs"])
    if "pretrained" in cfg_dict["model"] and cfg_dict["model"]["pretrained"] is not None:
        ckpt = torch.load(PRETRAINED_DIR / cfg_dict["model"]["pretrained"])
        load_res = model.load_state_dict(ckpt["model_state_dict"])
        print(f"{load_res = }")
    else:
        print("no pretrained model given")
    # ---------------------------------------

    experiment_name = (
        f"{model.get_name()}"
        f"-bs{cfg.batch_size}"
        f"-abs{cfg.apparent_batch_size}"
        f"-llr{cfg.optimizer.log_lr}"
        f"-ema{bool(cfg.task.kwargs.ema_momentum if 'ema_momentum' in cfg.task.kwargs else False)}"
        f"-{time_now}"
    )
    model_out_dir = MODEL_OUT_DIR / experiment_name
    model_out_dir.mkdir(exist_ok=True, parents=True)
    print(f"{model_out_dir = }")

    dataset = CIFAR10DiffusionDataset()
    
    # train test split 
    train_pct = cfg.train_pct
    train_len = int(len(dataset) * train_pct)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    print(f"Train data count: {train_len}")
    print(f"Validation data count: {val_len}")
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.apparent_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2*cfg.apparent_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    OmegaConf.save(cfg, model_out_dir / "config.yaml", resolve=True)

    criterion = torch.nn.L1Loss()

    
    accumulate_grad_batches = max(1, int(cfg.batch_size / cfg.apparent_batch_size))
    print(f"{accumulate_grad_batches = }")
    
    noise_scheduler_kwargs = cfg_dict["noise_scheduler"]["kwargs"]
    print("noise_scheduler_kwargs")
    print(json.dumps(noise_scheduler_kwargs, indent=4))
    noise_scheduler = DDPMScheduler(**noise_scheduler_kwargs)
    
    task = DenoisingTask(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_spec=cfg_dict["optimizer"],
        experiment_name=experiment_name,
        criterion=criterion,
        noise_scheduler=noise_scheduler,
        scheduler_spec=cfg_dict["scheduler"],
        accumulate_grad_batches=accumulate_grad_batches,
        **cfg_dict["task"]["kwargs"],
    )
    callbacks = [
        pl.callbacks.RichModelSummary(max_depth=3),
    ]
    if cfg.dry_logger:
        logger = None
    else:
        logger = WandbLogger(project=cfg.exp_name, name=experiment_name)
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
        logger.experiment.config["experiment_name"] = experiment_name
        logger.experiment.config["model_full"] = str(model)
        callbacks.append(pl.callbacks.LearningRateMonitor())
    callbacks += [
        pl.callbacks.ModelCheckpoint(
            dirpath=model_out_dir,
            save_top_k=-1,
            filename=f"{cfg.model.type}" + "-{epoch:02d}",
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_out_dir,
            save_top_k=1,
            monitor="val_loss" if cfg.task.type == "DenoisingTask" else "val_loss",
            mode="min",
            filename=f"{cfg.model.type}" + "-{epoch:02d}-{val_loss:.2f}",
        ),
    ]
    
    # the weird adjustment is because the original val check interval was designed for apparent batch size of 2
    adjusted_val_check_interval = float(cfg.val_check_interval * (2.0 / cfg.apparent_batch_size))
    print(f"{adjusted_val_check_interval = }")
    val_check_interval = min(adjusted_val_check_interval / len(train_loader), 1.0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        logger=logger,
        val_check_interval=val_check_interval,
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        benchmark=True,
        log_every_n_steps=20,
        # gradient_clip_val=2.0,
        # gradient_clip_algorithm="norm",
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        # strategy=DeepSpeedStrategy(
        #     stage=3,
        #     offload_optimizer=True,
        #     offload_parameters=True,
        # ),
        devices=-1,
    )
    trainer.fit(
        model=task,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    if not cfg.dry_logger:
        logger.experiment.config["best_val_loss"] = task.best_val_loss
        logger.experiment.finish()
    return task.best_val_loss


if __name__ == "__main__":
    main()
