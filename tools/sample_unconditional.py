from omegaconf import DictConfig, OmegaConf
from typing import Dict
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import click
import hydra
from torch.utils.data import DataLoader


from diffusion.envs.constants import MODEL_OUT_DIR
from diffusion.core.utils import load_config_from_dir, load_model_from_dir

@click.command()
@click.option("--experiment_name", default="UNetModel-2024-06-14-13-47-19", type=str)
@click.option("--num_timesteps", default=1000, type=int)
@click.option("--n_samples", default=8, type=int)
@click.option("--compile", is_flag=True, default=False)
@click.option("--out_dir", default="samples", type=str)
@click.option("--save_all", is_flag=True)
def main(
        experiment_name: str,
        num_timesteps: int,
        n_samples: int,
        compile: bool,
        out_dir: str,
        save_all: bool
    ):
    
    
    print(f"Inputs: {experiment_name=}, {num_timesteps=}, {n_samples=}, {compile=}")


    
    # Load the configs and the model
    model_dir = MODEL_OUT_DIR / experiment_name
    cfg = load_config_from_dir(model_dir)
    # turn to DictConfig
    cfg = OmegaConf.create(cfg)
    
    print(f"Loaded config from {model_dir}: {cfg}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_from_dir(model_dir).eval().to(device)
    if compile:
        model = torch.compile(model, mode="reduce-overhead")
    
    cfg.noise_scheduler.num_diffusion_timesteps = num_timesteps
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    
    train_dataset = hydra.utils.instantiate(cfg.dataset, cfg_fold=cfg.fold)
    
    # Start the sampling
    height, width = train_dataset.img_height, train_dataset.img_width
    n_channels = train_dataset.n_channels
    z_start = torch.randn(n_samples, n_channels, height, width).to(device)
    timesteps = torch.tensor([num_timesteps-1] * n_samples).to(device) # all timesteps are the same
    timesteps = timesteps.long()
    batch = {
        "img_noised": z_start,
        "timesteps": timesteps
    }
    model.img_key = "img_noised"
    model.timesteps_key = "timesteps"
    
    # make the out_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with torch.inference_mode():
        for step in tqdm(range(num_timesteps-1), total=num_timesteps-1):
            noise_pred = model(batch)
            xt_minus_one = noise_scheduler.get_xt_minus_one(batch["img_noised"], noise_pred, batch["timesteps"])
            
            # update batch elements
            batch["timesteps"] = batch["timesteps"] - 1
            batch["img_noised"] = xt_minus_one
            
            if save_all:
                # create a directory for all samples
                (out_dir / "all_samples").mkdir(exist_ok=True, parents=True)
                
                for i in range(n_samples):
                    img = xt_minus_one[i].detach().cpu().numpy().transpose(1, 2, 0)
                    img = (img + 1) / 2
                    # go to 0 to 1
                    img = (img - img.min()) / (img.max() - img.min())
                    img = Image.fromarray((img * 255).astype(np.uint8))
                    img.save(out_dir / "all_samples" / f"sample_{i}_step_{step}.png")

    
    # save the last samples to out_dir/final_samples
    (out_dir / "final_sample").mkdir(exist_ok=True, parents=True)
    for i in range(n_samples):
        img = xt_minus_one[i].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(out_dir / "final_sample" / f"sample_{i}.png")
    


if __name__ == "__main__":
    main()
