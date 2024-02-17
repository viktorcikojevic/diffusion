from omegaconf import DictConfig, OmegaConf
from typing import Dict
import hydra
import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip


from diffusion.environments.constants import MODEL_OUT_DIR, PRETRAINED_DIR
from diffusion.core.tasks import DenoisingTask
import diffusion.core.dataset as datasets
from diffusion.custom_modules.models.base_model import BaseDenoiser
from diffusion.core.noise_scheduler import DDPMScheduler
import diffusion.custom_modules.models as models
from diffusion.core.sample_utils import load_model_from_dir
from diffusion.core.dataset_factory import build_train_val_dataloaders

@hydra.main(config_path="../configs", config_name="sample", version_base="1.2")
def main(cfg: DictConfig):
    
    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    
    # prepare the model
    model_dir = Path(cfg_dict["predictors"]["model_dir"])
    model_dir = MODEL_OUT_DIR / model_dir
    cfg, model = load_model_from_dir(model_dir)
    
    # prepare the noise scheduler
    num_diffusion_timesteps = cfg["noise_scheduler"]["kwargs"]["num_diffusion_timesteps"]
    noise_scheduler = cfg["noise_scheduler"]["kwargs"]["noise_scheduler"]
    noise_scheduler = DDPMScheduler(num_diffusion_timesteps, noise_scheduler)
    
    # prepare the dataloaders
    cfg["batch_size"] = 1
    dataset_type = cfg["dataset"]["type"]
    dataset = getattr(datasets, dataset_type)()
    width, height = dataset.img_width_height
    
    
    
    # prepare the output directory
    save_dir = Path(cfg_dict["output"]["save_dir"])
    save_all = cfg_dict["output"]["save_all"]
    save_last = cfg_dict["output"]["save_last"]
    batch_size = cfg_dict["sampler"]["batch_size"]
    
    existing_dirs = list(save_dir.glob("*"))
    if len(existing_dirs) > 0:
        save_dir = save_dir / f"{len(existing_dirs) + 1}"
    else:
        save_dir = save_dir / "1"
    
    save_dir.mkdir(exist_ok=True, parents=True)
    (save_dir / "imgs").mkdir(exist_ok=True, parents=True)
    
    # start the backward pass
    diffusion_steps_vals = np.arange(num_diffusion_timesteps)[::-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval().to(device)
    img = torch.randn(batch_size, 3, height, width).to(device)
    
    
    for step in tqdm(diffusion_steps_vals, total=num_diffusion_timesteps):
        timestep = torch.Tensor([step]).to(device)
        z_img = torch.randn(batch_size, 3, height, width).to(device)
        if step == 0:
            sigma_t = 0.
        else:
            sigma_t = noise_scheduler.sigma_t[step]
        alpha_t = noise_scheduler.alpha_t[step]
        alpha_t_bar = noise_scheduler.alpha_t_bar[step]
        
        with torch.no_grad():
            model_out = model.predict(img, timestep).pred
    
        img = (1/np.sqrt(alpha_t)) * (img - (1-alpha_t)/np.sqrt(1-alpha_t_bar) * model_out) + sigma_t * z_img
        
        if save_all:
            img_np = img.squeeze().cpu().numpy()
            # convert to pil and save
            img_np = 255 * (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_np = np.transpose(img_np, (1, 2, 0))
            img_pil = Image.fromarray(np.uint8(img_np))
            step_process = diffusion_steps - step
            img_pil.save(save_dir / "imgs" / f"step_{step_process}.png")
        if save_last:
            if step == 0:
                img_np = img.cpu().numpy()
                # convert to pil and save
                img_np = 255 * (img_np - np.min(img_np, axis=(1, 2, 3), keepdims=True)) / (np.max(img_np, axis=(1, 2, 3), keepdims=True) - np.min(img_np, axis=(1, 2, 3), keepdims=True))
                
                for img_idx in range(img_np.shape[0]):
                    img_np_t = np.transpose(img_np[img_idx], (1, 2, 0))
                    img_pil = Image.fromarray(np.uint8(img_np_t))
                    img_pil.save(save_dir / f"output_{img_idx}.png")
            
            
    if save_all:
        # convert all the images to a gif
        # Ensure the images are sorted by their step number
        img_paths = sorted((save_dir / "imgs").glob("*.png"), key=lambda x: int(x.stem.split('_')[1]))

        # Load the images
        imgs = [Image.open(img_path) for img_path in img_paths]

        # Save the images as a GIF
        gif_path = save_dir / "output.gif"
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], optimize=False, duration=100, loop=0)
        
        # Convert img_paths to string paths because ImageSequenceClip expects paths
        img_paths_str = [str(img_path) for img_path in img_paths]
        
        # Calculate frame rate to make the video last 15 seconds
        total_images = len(img_paths)
        video_duration = 30  # seconds
        frame_rate = total_images / video_duration
        
        # Create a video clip
        clip = ImageSequenceClip(img_paths_str, fps=frame_rate)
        
        # Save the video as an MP4 file
        mp4_path = str(save_dir / "output.mp4")
        clip.write_videofile(mp4_path, codec='libx264')



if __name__ == "__main__":
    main()
