from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict
import hydra
import torch
import json
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import Logger


from diffusion.envs.constants import MODEL_OUT_DIR



@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    
    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    print("Config: ")
    print(json.dumps(cfg_dict, indent=4))
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print(f"{time_now=}")
    
    # ---------------------------------------
    
    
    model = hydra.utils.instantiate(cfg.model)
    if cfg.compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    model_name = model.__class__.__name__
    
    experiment_name = (
        f"{model_name}"
        f"-{time_now}"
    )
    
    
    
    model_out_dir = MODEL_OUT_DIR / experiment_name
    model_out_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg, model_out_dir / "config.yaml", resolve=True)
    print(f"{model_out_dir=}")
    
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    
    

    train_dataset = hydra.utils.instantiate(cfg.dataset, cfg_fold=cfg.fold)
    val_dataset = hydra.utils.instantiate(cfg.dataset, cfg_fold=cfg.fold)
    
    train_loader = DataLoader(dataset=train_dataset, **cfg.train_dataloader_kwargs)
    val_loader = DataLoader(dataset=val_dataset, **cfg.val_dataloader_kwargs)


    task = hydra.utils.instantiate(cfg.task, model=model, noise_scheduler=noise_scheduler, train_loader=train_loader, val_loader=val_loader)

    print(f"{len(train_dataset)=}, {len(val_dataset)=}")
    print(f"{len(train_loader)=}, {len(val_loader)=}")

    
    logger: Logger = None if cfg.logger is None else hydra.utils.instantiate(cfg.logger, name=experiment_name)
    if logger is not None:
        logger.experiment.config.update(cfg_dict)
        logger.experiment.config["experiment_name"] = experiment_name
    print(f"{logger=}")
    
    
    callbacks = [
        pl.callbacks.RichModelSummary(max_depth=3),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_out_dir,
            save_top_k=cfg.model_checkpoint.save_top_k,
            monitor=cfg.model_checkpoint.monitor,
            mode=cfg.model_checkpoint.mode,
            filename=f"{model_name}" + "-{epoch:02d}-{" + cfg.model_checkpoint.monitor + ":.2f}",
            # save_last=True,
            save_last=False,
        ),
    ]
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer_kwargs,
    )
    
    
    
    trainer.fit(
        model=task,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.load_from_ckpt_path,
    )
    
    
    # if cfg.validate:
    #     trainer.validate(
    #         model=task,
    #         dataloaders=val_loader,
    #         ckpt_path=cfg.load_from_ckpt_path,
    #     )
    # else:
    #     trainer.fit(
    #         model=task,
    #         train_dataloaders=train_loader,
    #         val_dataloaders=val_loader,
    #         ckpt_path=cfg.load_from_ckpt_path,
    #     )
    # try:
    #     logger.experiment.config["target_metric"] = task.target_metric
    #     logger.experiment.config["target_metric_best"] = task.target_metric_best
    #     logger.experiment.finish()
    # except Exception as e:
    #     print(f"can't call experiment.finish(): {repr(e)}, skipping I guess")
    # return task.target_metric_best

    
    
    
    

if __name__ == "__main__":
    main()
